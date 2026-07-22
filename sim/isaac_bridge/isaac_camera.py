"""IsaacCameraDriver — sim-backed CameraDriver duck-type.

Renders RGB + distance-to-image-plane depth from an Isaac camera prim whose K is
set to match the real Orbbec intrinsics, mounted so that the optical-frame
extrinsic equals a supplied T_cam2base (base <- camera, optical convention:
+Z forward into scene, +X image-right, +Y image-down).

USD/Isaac cameras look down -Z (OpenGL: +X right, +Y up, -Z forward). The vision
pipeline uses optical convention (+Z forward, +Y down). Conversion:
    R_usd = R_opt @ R_opt2usd ,  R_opt2usd = diag(1, -1, -1)
i.e. flip Y and Z to go optical->USD camera basis.
"""
import numpy as np

# optical (cv) -> USD/OpenGL camera basis: X same, Y flip, Z flip
_OPT2USD = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


class IsaacCameraDriver:
    def __init__(self, camera, K_real, width=1280, height=720, D=None):
        """camera: isaacsim.sensors.camera.Camera (already initialized).
        K_real: 3x3 real intrinsics. The camera focal/aperture were set at build
        time to match; we just expose K_real as .K for the pipeline.
        """
        self._cam = camera
        self._K = np.asarray(K_real, dtype=np.float64)
        self._D = np.zeros((1, 5), dtype=np.float64) if D is None else np.asarray(D, np.float64)
        self.width = width
        self.height = height

    # ── lifecycle ──
    def open(self):
        pass

    def close(self):
        pass

    # ── pose ──
    def set_extrinsic(self, T_cam2base):
        """Place the camera so its OPTICAL frame == T_cam2base (base<-camera)."""
        T = np.asarray(T_cam2base, dtype=np.float64)
        R_opt = T[:3, :3]
        t = T[:3, 3]
        R_usd = R_opt @ _OPT2USD
        from isaacsim.core.utils.rotations import rot_matrix_to_quat
        quat = rot_matrix_to_quat(R_usd)  # (w,x,y,z)
        # camera_axes="usd": the supplied orientation is the USD camera basis
        # (+Y up, -Z forward), which is exactly R_usd. (Default "world" basis is
        # +X forward / +Z up and would mis-aim the camera.)
        self._cam.set_world_pose(position=t.astype(np.float32),
                                 orientation=np.asarray(quat, dtype=np.float32),
                                 camera_axes="usd")

    # ── frame ──
    def get_frame(self):
        """Return (color_bgr uint8 HxWx3, depth_mm uint16 HxW). 0 = invalid."""
        rgba = self._cam.get_rgba()  # HxWx4 uint8
        if rgba is None or rgba.size == 0:
            return None, None
        rgb = np.asarray(rgba)[:, :, :3].astype(np.uint8)
        bgr = rgb[:, :, ::-1].copy()
        depth_m = self._cam.get_depth()  # distance-to-image-plane, metres, float
        if depth_m is None:
            return bgr, None
        depth_m = np.asarray(depth_m, dtype=np.float64)
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        depth_m[depth_m < 0] = 0.0
        # Isaac returns inf for "no hit" which nan_to_num already zeroed.
        depth_mm = np.clip(np.round(depth_m * 1000.0), 0, 65535).astype(np.uint16)
        return bgr, depth_mm

    @property
    def K(self):
        return self._K.copy()

    @property
    def D(self):
        return self._D.copy()
