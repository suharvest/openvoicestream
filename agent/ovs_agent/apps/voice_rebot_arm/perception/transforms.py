"""坐标变换工具函数。

Vendored 自 reBot 仓。未使用的 GraspNet / quaternion / retreat 路径已裁。
"""
from typing import Optional

import numpy as np


_ROT_X_PI = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)


def _nearest_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """Project a near-rotation matrix onto SO(3)."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"rotation matrix must be (3, 3), got {R.shape}")

    if not np.all(np.isfinite(R)):
        raise ValueError("rotation matrix contains non-finite values")

    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0.0:
        U[:, -1] *= -1.0
        R_ortho = U @ Vt
    return R_ortho.astype(np.float64)


def pose6d_to_mat4(x, y, z, rx, ry, rz, degrees=False) -> np.ndarray:
    """
    将 6D 位姿 (平移 + ZYX 内旋欧拉角) 转换为 4×4 齐次变换矩阵。

    Args:
        x, y, z: 平移 (米)
        rx, ry, rz: 欧拉角，ZYX 内旋约定 (roll=rx around X, pitch=ry around Y, yaw=rz around Z)
        degrees: True 时输入为度，False 时为弧度

    Returns:
        T: (4, 4) numpy array
    """
    if degrees:
        rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)

    # 绕 X 轴
    Rx = np.array([
        [1,          0,           0],
        [0,  np.cos(rx), -np.sin(rx)],
        [0,  np.sin(rx),  np.cos(rx)],
    ])
    # 绕 Y 轴
    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [          0, 1,          0],
        [-np.sin(ry), 0, np.cos(ry)],
    ])
    # 绕 Z 轴
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [         0,           0, 1],
    ])

    # ZYX 内旋 = R = Rz @ Ry @ Rx
    R = Rz @ Ry @ Rx

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = [x, y, z]
    return T


def mat4_to_pose6d(T: np.ndarray) -> tuple:
    """
    将 4×4 齐次变换矩阵转换为 (x, y, z, rx, ry, rz)，ZYX 内旋约定，弧度。
    """
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    rpy = rotation_matrix_to_euler_zyx(T[:3, :3])
    return float(x), float(y), float(z), float(rpy[0]), float(rpy[1]), float(rpy[2])


def rotation_matrix_to_euler_zyx(R: np.ndarray) -> np.ndarray:
    """将旋转矩阵转换为 ZYX 内旋欧拉角 (roll, pitch, yaw)。"""
    R = _nearest_rotation_matrix(R)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0.0
    return np.array([rx, ry, rz], dtype=np.float64)


def canonicalize_parallel_gripper_tcp_rotation(R: np.ndarray) -> np.ndarray:
    """规范并联夹爪 TCP 姿态的等效 180 度扭转。

    对于对称的并联夹爪，沿工具 X 轴旋转 180 度通常是抓取等效的。
    这里在 ``R`` 和 ``R @ Rx(pi)`` 之间选择 roll 绝对值更小的那一支，
    让输出的 RPY 更贴近人的直觉，也更方便调试。
    """
    R = _nearest_rotation_matrix(R)
    alt = R @ _ROT_X_PI

    roll = float(rotation_matrix_to_euler_zyx(R)[0])
    alt_roll = float(rotation_matrix_to_euler_zyx(alt)[0])
    return alt if abs(alt_roll) < abs(roll) else R


def grasp_axes_to_rebot_tcp_rotation(
    grip_axis: np.ndarray,
    open_axis: np.ndarray,
    approach_axis: np.ndarray,
) -> np.ndarray:
    """将抓取坐标系映射到 reBotArm 的 TCP 坐标系。

    视觉抓取结果约定：
      - X = grip_axis
      - Y = open_axis
      - Z = approach_axis

    reBotArm 末端期望：
      - X = 工具前向 / 接近方向
      - Y = 夹爪开合方向
      - Z = 由右手系补齐
    """
    grip = np.asarray(grip_axis, dtype=np.float64)
    open_vec = np.asarray(open_axis, dtype=np.float64)
    approach = np.asarray(approach_axis, dtype=np.float64)

    grip /= max(np.linalg.norm(grip), 1e-8)
    open_vec /= max(np.linalg.norm(open_vec), 1e-8)
    approach /= max(np.linalg.norm(approach), 1e-8)

    # tcp_x = tool-forward = approach direction pointing INTO the object (downward in base).
    # plane.normal points toward the camera (upward), so negate it here.
    tcp_x = -approach
    tcp_y = open_vec - float(np.dot(open_vec, tcp_x)) * tcp_x
    tcp_y /= max(np.linalg.norm(tcp_y), 1e-8)
    tcp_z = np.cross(tcp_x, tcp_y)
    tcp_z /= max(np.linalg.norm(tcp_z), 1e-8)

    # 期望 tcp_z 与 grip 同向（取反后方向一致）。
    if float(np.dot(tcp_z, grip)) < 0.0:
        tcp_y = -tcp_y
        tcp_z = -tcp_z

    R = np.column_stack([tcp_x, tcp_y, tcp_z]).astype(np.float64)
    if np.linalg.det(R) < 0.0:
        R[:, 2] *= -1.0
    return R


def _make_grasp_base_transform(
    position_cam: np.ndarray,
    tcp_rotation_cam: np.ndarray,
    T_cam2base: np.ndarray,
) -> np.ndarray:
    T_grasp_cam = np.eye(4, dtype=np.float64)
    T_grasp_cam[:3, :3] = np.asarray(tcp_rotation_cam, dtype=np.float64)
    T_grasp_cam[:3, 3] = np.asarray(position_cam, dtype=np.float64)

    T_grasp_base = np.asarray(T_cam2base, dtype=np.float64) @ T_grasp_cam
    T_grasp_base[:3, :3] = canonicalize_parallel_gripper_tcp_rotation(T_grasp_base[:3, :3])
    return T_grasp_base


def _offset_along_axis(
    T: np.ndarray, axis_base: np.ndarray, offset_m: float
) -> np.ndarray:
    T_offset = T.copy()
    T_offset[:3, 3] = T[:3, 3] - np.asarray(axis_base, dtype=np.float64) * float(offset_m)
    return T_offset


def transform_grasp_pose_to_base(
    position_cam: np.ndarray,
    tcp_rotation_cam: np.ndarray,
    T_cam2base: np.ndarray,
    pregrasp_offset_m: float,
    insertion_depth_m: float = 0.0,
    offset_axis_cam: Optional[np.ndarray] = None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """将相机坐标系下的夹取位姿转换为 base 坐标系下的夹取/预夹取 6D 位姿。

    ``offset_axis_cam`` (camera frame, will be normalised) decouples the
    insertion/pregrasp TRANSLATION direction from the TCP orientation. By
    default the offsets run along the tool-X (= -approach), i.e. the historical
    behaviour. The grasp builder now re-aims the approach AZIMUTH to align the
    jaw with an angled box (2026-06-16), which ALSO rotates tool-X — so a fixed
    insertion depth landed the jaw centre up to ~1 cm to the side and it stopped
    enclosing the object. Passing the camera→object ray here keeps the
    insertion along the ORIGINAL (un-re-aimed) direction, so the jaw centre
    lands exactly where it did before the orientation fix while the gripper
    still turns to face the box.
    """
    T_grasp_base = _make_grasp_base_transform(position_cam, tcp_rotation_cam, T_cam2base)
    if offset_axis_cam is not None:
        axis = np.asarray(offset_axis_cam, dtype=np.float64)
        n = float(np.linalg.norm(axis))
        if n > 1e-9:
            axis_base = np.asarray(T_cam2base, dtype=np.float64)[:3, :3] @ (axis / n)
        else:
            axis_base = T_grasp_base[:3, 0]
    else:
        axis_base = T_grasp_base[:3, 0]
    T_grasp_base = _offset_along_axis(T_grasp_base, axis_base, -insertion_depth_m)
    T_pregrasp_base = _offset_along_axis(T_grasp_base, axis_base, pregrasp_offset_m)
    return mat4_to_pose6d(T_grasp_base), mat4_to_pose6d(T_pregrasp_base)
