"""Orbbec Gemini 2 相机驱动。"""
from __future__ import annotations

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple

from .base import CameraDriver


class OrbbecGemini2(CameraDriver):
    """Orbbec Gemini 2 RGBD 相机驱动。

    Args:
        width, height, fps: 分辨率与帧率（颜色流；深度流尝试相同分辨率）
        calib_dir: 标定目录路径；含 intrinsics.npz 时从中读取畸变系数
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        calib_dir: Optional[str] = None,
    ) -> None:
        self._w = width
        self._h = height
        self._fps = fps
        self._calib_dir = Path(calib_dir) if calib_dir else None

        self._pipeline = None
        self._K: Optional[np.ndarray] = None
        self._D: Optional[np.ndarray] = None
        self._aruco = None
        self._OBFormat = None  # cached SDK enum, set in open()

    # ── 生命周期 ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        """初始化相机管线，压制 SDK C++ 日志。"""
        # 先 import（不压制，避免 C 库加载错误被吞掉）。设备上 SDK 模块名可能
        # 是 pyorbbecsdk2（新）或 pyorbbecsdk（旧），都试一遍。
        _sdk = None
        for _name in ("pyorbbecsdk2", "pyorbbecsdk"):
            try:
                _sdk = __import__(_name)
                break
            except ImportError:
                continue
        if _sdk is None:
            raise RuntimeError(
                "未安装 pyorbbecsdk2 / pyorbbecsdk，请在设备上编译安装 Orbbec SDK"
            )
        Pipeline = _sdk.Pipeline
        Config = _sdk.Config
        OBSensorType = _sdk.OBSensorType
        OBFormat = _sdk.OBFormat
        OBAlignMode = _sdk.OBAlignMode
        Context = _sdk.Context
        # Cache the enum for per-frame format dispatch (get_frame is hot).
        self._OBFormat = OBFormat
        self._ob_sdk_name = _name

        # 仅在 C++ SDK 初始化时压制 stderr（避免时间戳异常等日志刷屏）
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved = os.dup(2)
        os.dup2(devnull, 2)
        os.close(devnull)

        try:
            try:
                # OBLogSeverity 在部分版本中不存在，动态尝试
                OBLogSeverity = _sdk.OBLogSeverity
                Context().set_logger_severity(OBLogSeverity.FATAL)
            except Exception:
                pass

            try:
                self._pipeline = Pipeline()
            except Exception as e:
                raise RuntimeError(
                    f"Orbbec 相机未找到: {e}\n"
                    "  可能原因: 未插入 / USB 接口松动 / udev 权限未配置\n"
                    "  配置权限: sudo chmod a+rw /dev/bus/usb/*/*"
                ) from e

            cfg = Config()

            # 颜色流
            plist = self._pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            cp = None
            for fmt in (OBFormat.MJPG, OBFormat.RGB):
                try:
                    cp = plist.get_video_stream_profile(self._w, self._h, fmt, self._fps)
                    break
                except Exception:
                    pass
            if cp is None:
                cp = plist.get_default_video_stream_profile()
            cfg.enable_stream(cp)

            # 深度流
            dplist = self._pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            try:
                dp = dplist.get_video_stream_profile(self._w, self._h, OBFormat.Y16, self._fps)
            except Exception:
                dp = dplist.get_default_video_stream_profile()
            cfg.enable_stream(dp)

            cfg.set_align_mode(OBAlignMode.HW_MODE)
            self._pipeline.start(cfg)

            # Prime the stream before returning. On a fresh start the first
            # ~0.2-1s of frames arrive slower than the 500ms `get_frame` wait,
            # so the first high-level get_frame() (and warm_up, which just loops
            # it) returns None until the depth/color sync settles — making the
            # first grasp after camera-open spuriously fail with
            # "camera returned no frame". Block here on a long timeout until a
            # synced color+depth frame lands so steady-state 500ms reads succeed.
            for _ in range(30):
                try:
                    fs = self._pipeline.wait_for_frames(2000)
                except Exception:
                    fs = None
                if (fs is not None
                        and fs.get_color_frame() is not None
                        and fs.get_depth_frame() is not None):
                    break

            # 从 SDK 读取内参
            intr = self._pipeline.get_camera_param().rgb_intrinsic
            self._K = np.array([
                [intr.fx, 0,       intr.cx],
                [0,       intr.fy, intr.cy],
                [0,       0,       1      ],
            ], dtype=np.float64)

            # 畸变系数
            self._D = self._load_distortion()

        finally:
            os.dup2(saved, 2)
            os.close(saved)

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None

    # ── 帧获取 ───────────────────────────────────────────────────────────────

    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._pipeline is None:
            return None, None
        try:
            OBFormat = self._OBFormat
            frames = self._pipeline.wait_for_frames(500)
            if frames is None:
                return None, None

            color_bgr = None
            cf = frames.get_color_frame()
            if cf is not None:
                w, h = cf.get_width(), cf.get_height()
                raw = np.asanyarray(cf.get_data(), dtype=np.uint8)
                fmt = cf.get_format()
                try:
                    if fmt == OBFormat.MJPG:
                        color_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                    elif fmt == OBFormat.RGB:
                        color_bgr = cv2.cvtColor(raw.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
                    else:
                        color_bgr = raw.reshape(h, w, 3)
                except Exception:
                    pass

            depth_mm = None
            df = frames.get_depth_frame()
            if df is not None:
                dw, dh = df.get_width(), df.get_height()
                depth_mm = np.frombuffer(df.get_data(), dtype=np.uint16).reshape(dh, dw)

            return color_bgr, depth_mm
        except Exception:
            return None, None

    # ── 内参 ─────────────────────────────────────────────────────────────────

    @property
    def K(self) -> np.ndarray:
        if self._K is None:
            raise RuntimeError("相机未打开，请先调用 open()")
        return self._K

    @property
    def D(self) -> np.ndarray:
        if self._D is None:
            raise RuntimeError("相机未打开，请先调用 open()")
        return self._D

    # ── 内部 ─────────────────────────────────────────────────────────────────

    def _load_distortion(self) -> np.ndarray:
        """从标定目录加载畸变系数，k1 异常时降级为零畸变。"""
        if self._calib_dir is not None:
            npz_path = self._calib_dir / "intrinsics.npz"
            if npz_path.exists():
                try:
                    data = np.load(str(npz_path))
                    D = data["dist_coeffs"].flatten()
                    if abs(D[0]) > 5.0:
                        print(f"[OrbbecGemini2] 畸变系数 k1={D[0]:.2f} 偏大，改用零畸变")
                        return np.zeros((1, 5), dtype=np.float64)
                    return D.reshape(1, -1)
                except Exception:
                    pass
        return np.zeros((1, 5), dtype=np.float64)
