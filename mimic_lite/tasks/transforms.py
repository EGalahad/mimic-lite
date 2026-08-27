import torch

from active_adaptation.utils.math import (
    batchify,
    matrix_from_quat,
    quat_angle_magnitude,
    quat_conjugate,
    quat_from_euler_xyz as _quat_from_euler_xyz,
    quat_from_yaw,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
    sample_uniform as _sample_uniform,
)


quat_apply_inverse = batchify(quat_rotate_inverse)


def projected_yaw_quat(
    quat: torch.Tensor, x_axis_xy_threshold: float = 0.1
) -> torch.Tensor:
    """Build a level yaw quaternion from horizontal axis projections.

    This keeps the returned frame aligned with world-up and chooses its heading from:
    1. the anchor x-axis projection when that projection is significant, or
    2. the anchor z-axis projection when the x-axis is close to vertical.

    The z-axis fallback is sign-adjusted so the heading stays continuous when the
    anchor x-axis crosses between pointing upward and downward.

    Args:
        quat: The orientation in (w, x, y, z). Shape is (..., 4).
        x_axis_xy_threshold: Minimum horizontal norm for using the projected x-axis.

    Returns:
        A quaternion with only a world-up yaw component.
    """
    shape = quat.shape
    quat_flat = quat.reshape(-1, 4)

    basis_x = torch.zeros(quat_flat.shape[0], 3, device=quat.device, dtype=quat.dtype)
    basis_x[:, 0] = 1.0
    basis_z = torch.zeros_like(basis_x)
    basis_z[:, 2] = 1.0

    x_axis_w = quat_rotate(quat_flat, basis_x)
    z_axis_w = quat_rotate(quat_flat, basis_z)

    x_axis_xy = x_axis_w[:, :2]
    z_axis_xy = z_axis_w[:, :2]
    x_axis_xy_norm = torch.linalg.norm(x_axis_xy, dim=-1, keepdim=True)

    z_axis_heading_xy = torch.where(x_axis_w[:, 2:3] < 0.0, z_axis_xy, -z_axis_xy)
    heading_xy = torch.where(
        x_axis_xy_norm > x_axis_xy_threshold,
        x_axis_xy,
        z_axis_heading_xy,
    )

    yaw = torch.atan2(heading_xy[:, 1], heading_xy[:, 0])
    return quat_from_yaw(yaw).view(shape)


def sample_uniform(low, high, size, device):
    return _sample_uniform(size=size, low=low, high=high, device=device)


def quat_from_euler_xyz(roll, pitch, yaw):
    return _quat_from_euler_xyz(torch.stack([roll, pitch, yaw], dim=-1))


def _body_pose_in_anchor_frame(
    anchor_pos_w: torch.Tensor,
    anchor_quat_w: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    anchor_pos_w_z0 = anchor_pos_w.clone()
    anchor_pos_w_z0[..., 2] = 0.0
    anchor_yaw_quat_w = projected_yaw_quat(anchor_quat_w)
    return (
        quat_apply_inverse(anchor_yaw_quat_w, body_pos_w - anchor_pos_w_z0),
        quat_mul(quat_conjugate(anchor_yaw_quat_w).expand_as(body_quat_w), body_quat_w),
    )


@torch.compile(mode="max-autotune-no-cudagraphs")
def _compute_current_tracking_state(
    ref_anchor_pos_w: torch.Tensor,
    ref_anchor_quat_w: torch.Tensor,
    robot_anchor_pos_w: torch.Tensor,
    robot_anchor_quat_w: torch.Tensor,
    ref_body_pos_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
    robot_body_link_pos_w: torch.Tensor,
    robot_body_link_quat_w: torch.Tensor,
):
    ref_body_pos_local, ref_body_quat_local = _body_pose_in_anchor_frame(
        ref_anchor_pos_w[:, None],
        ref_anchor_quat_w[:, None],
        ref_body_pos_w,
        ref_body_quat_w,
    )
    robot_body_pos_local, robot_body_quat_local = _body_pose_in_anchor_frame(
        robot_anchor_pos_w[:, None],
        robot_anchor_quat_w[:, None],
        robot_body_link_pos_w,
        robot_body_link_quat_w,
    )
    return ref_body_pos_local, ref_body_quat_local, robot_body_pos_local, robot_body_quat_local


@torch.compile(mode="max-autotune-no-cudagraphs")
def _compute_tracking_errors(
    ref_body_pos_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
    ref_body_lin_vel_w: torch.Tensor,
    ref_body_ang_vel_w: torch.Tensor,
    ref_body_pos_local: torch.Tensor,
    ref_body_quat_local: torch.Tensor,
    robot_body_link_pos_w: torch.Tensor,
    robot_body_link_quat_w: torch.Tensor,
    robot_body_lin_vel_w: torch.Tensor,
    robot_body_ang_vel_w: torch.Tensor,
    robot_body_pos_local: torch.Tensor,
    robot_body_quat_local: torch.Tensor,
    ref_joint_pos: torch.Tensor,
    ref_joint_vel: torch.Tensor,
    robot_joint_pos: torch.Tensor,
    robot_joint_vel: torch.Tensor,
):
    body_pos_error = (ref_body_pos_w - robot_body_link_pos_w).norm(dim=-1)
    body_pos_error_local = (ref_body_pos_local - robot_body_pos_local).norm(dim=-1)

    body_quat_diff = quat_mul(
        quat_conjugate(ref_body_quat_w),
        robot_body_link_quat_w,
    )
    body_ori_error = quat_angle_magnitude(body_quat_diff)

    body_quat_local_diff = quat_mul(
        quat_conjugate(ref_body_quat_local),
        robot_body_quat_local,
    )
    body_ori_error_local = quat_angle_magnitude(body_quat_local_diff)

    body_lin_vel_error = (ref_body_lin_vel_w - robot_body_lin_vel_w).norm(dim=-1)
    body_ang_vel_error = (ref_body_ang_vel_w - robot_body_ang_vel_w).norm(dim=-1)

    joint_pos_error = (ref_joint_pos - robot_joint_pos).abs()
    joint_vel_error = (ref_joint_vel - robot_joint_vel).abs()

    return (
        body_pos_error,
        body_pos_error_local,
        body_ori_error,
        body_ori_error_local,
        body_lin_vel_error,
        body_ang_vel_error,
        joint_pos_error,
        joint_vel_error,
    )


@torch.compile(mode="max-autotune-no-cudagraphs")
def _compute_root_diff_obs(
    robot_root_pos_w: torch.Tensor,
    robot_root_quat_w: torch.Tensor,
    ref_root_pos_future_w: torch.Tensor,
    ref_root_quat_future_w: torch.Tensor,
):
    robot_root_pos_w_expand = robot_root_pos_w[:, None, :]
    robot_root_quat_w_expand = robot_root_quat_w[:, None, :]
    robot_root_quat_w_expand_inv = quat_conjugate(robot_root_quat_w_expand)
    ref_root_pos_future_b = quat_apply_inverse(
        robot_root_quat_w_expand,
        ref_root_pos_future_w - robot_root_pos_w_expand,
    )
    ref_root_quat_future_b = quat_mul(
        robot_root_quat_w_expand_inv.expand_as(ref_root_quat_future_w),
        ref_root_quat_future_w,
    )
    ref_root_mat_future_b = matrix_from_quat(ref_root_quat_future_b)
    return ref_root_pos_future_b, ref_root_mat_future_b


@torch.compile(mode="max-autotune-no-cudagraphs")
def _compute_motion_local_obs(
    ref_anchor_pos_w: torch.Tensor,
    ref_anchor_quat_w: torch.Tensor,
    ref_body_pos_future_w: torch.Tensor,
    ref_body_quat_future_w: torch.Tensor,
):
    ref_body_pos_future_local, ref_body_quat_future_local = _body_pose_in_anchor_frame(
        ref_anchor_pos_w[:, None, None],
        ref_anchor_quat_w[:, None, None],
        ref_body_pos_future_w,
        ref_body_quat_future_w,
    )
    ref_body_ori_future_local_matrix = matrix_from_quat(ref_body_quat_future_local)

    return ref_body_pos_future_local, ref_body_ori_future_local_matrix


@torch.compile(mode="max-autotune-no-cudagraphs")
def _compute_body_diff_obs(
    # anchor pose
    ref_anchor_pos_w: torch.Tensor,
    ref_anchor_quat_w: torch.Tensor,
    robot_anchor_pos_w: torch.Tensor,
    robot_anchor_quat_w: torch.Tensor,
    # body pose
    ref_body_pos_future_w: torch.Tensor,
    ref_body_quat_future_w: torch.Tensor,
    robot_body_link_pos_w: torch.Tensor,
    robot_body_link_quat_w: torch.Tensor,
):
    ref_body_pos_future_local, ref_body_quat_future_local = _body_pose_in_anchor_frame(
        ref_anchor_pos_w[:, None, None],
        ref_anchor_quat_w[:, None, None],
        ref_body_pos_future_w,
        ref_body_quat_future_w,
    )
    robot_body_pos_local, robot_body_quat_local = _body_pose_in_anchor_frame(
        robot_anchor_pos_w[:, None],
        robot_anchor_quat_w[:, None],
        robot_body_link_pos_w,
        robot_body_link_quat_w,
    )
    robot_body_quat_local_conj = quat_conjugate(robot_body_quat_local)

    diff_body_quat_future = quat_mul(
        robot_body_quat_local_conj.unsqueeze(1).expand_as(ref_body_quat_future_local),
        ref_body_quat_future_local,
    )
    diff_body_ori_future_local_matrix = matrix_from_quat(diff_body_quat_future)

    return (
        ref_body_pos_future_local - robot_body_pos_local.unsqueeze(1),
        diff_body_ori_future_local_matrix,
    )
