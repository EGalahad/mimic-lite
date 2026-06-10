from active_adaptation.envs.mdp.commands.base import Command
from active_adaptation.envs.utils import find_bodies, find_joints
from mimic_lite.tasks.motion import MotionData, create_dataset_from_path
from mimic_lite.tasks.multi_dataset import (
    MotionDatasetConfig,
    load_motion_dataset_collection,
    normalize_motion_cfgs,
)

from dataclasses import dataclass
from typing import List, Dict, Tuple, TYPE_CHECKING, Literal, Mapping
import copy
import importlib
import json
import os

if TYPE_CHECKING:
    from mjlab.viewer.viser import ViserMujocoScene

import torch
import numpy as np

from active_adaptation.utils.math import (
    sample_uniform as _sample_uniform,
    quat_from_euler_xyz as _quat_from_euler_xyz,
    quat_rotate_inverse as quat_apply_inverse,
    quat_mul,
    quat_conjugate,
    quat_angle_magnitude,
    matrix_from_quat,
    quat_rotate,
    quat_from_yaw,
    batchify,
)
from active_adaptation.utils.profiling import ScopedTimer
from tensordict import TensorDict

PROFILE_SYNC_TIMERS = os.environ.get("AA_PROFILE_SYNC_TIMERS", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

def projected_yaw_quat(quat: torch.Tensor, x_axis_xy_threshold: float = 0.1) -> torch.Tensor:
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



quat_apply_inverse = batchify(quat_apply_inverse)

_DESIRED_FRAME_COLORS = (
    (0.9, 0.3, 0.3, 0.9),
    (0.3, 0.9, 0.3, 0.9),
    (0.3, 0.3, 0.9, 0.9),
)


def sample_uniform(low, high, size, device):
    return _sample_uniform(size=size, low=low, high=high, device=device)


def quat_from_euler_xyz(roll, pitch, yaw):
    return _quat_from_euler_xyz(torch.stack([roll, pitch, yaw], dim=-1))


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
    ref_anchor_pos_w_z0 = ref_anchor_pos_w.clone()
    ref_anchor_pos_w_z0[..., 2] = 0.0
    robot_anchor_pos_w_z0 = robot_anchor_pos_w.clone()
    robot_anchor_pos_w_z0[..., 2] = 0.0

    ref_anchor_yaw_quat_w = projected_yaw_quat(ref_anchor_quat_w)
    robot_anchor_yaw_quat_w = projected_yaw_quat(robot_anchor_quat_w)
    
    ref_anchor_yaw_quat_conj_w = quat_conjugate(ref_anchor_yaw_quat_w)
    robot_anchor_yaw_quat_conj_w = quat_conjugate(robot_anchor_yaw_quat_w)

    ref_body_pos_local = quat_apply_inverse(
        ref_anchor_yaw_quat_w[:, None],
        ref_body_pos_w - ref_anchor_pos_w_z0[:, None],
    )
    ref_body_quat_local = quat_mul(
        ref_anchor_yaw_quat_conj_w[:, None].expand_as(ref_body_quat_w),
        ref_body_quat_w,
    )

    robot_body_pos_local = quat_apply_inverse(
        robot_anchor_yaw_quat_w[:, None],
        robot_body_link_pos_w - robot_anchor_pos_w_z0[:, None],
    )
    robot_body_quat_local = quat_mul(
        robot_anchor_yaw_quat_conj_w[:, None].expand_as(robot_body_link_quat_w),
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
    ref_anchor_pos_w_z0 = ref_anchor_pos_w.clone()
    ref_anchor_pos_w_z0[..., 2] = 0.0
    ref_anchor_pos_w_z0_future = ref_anchor_pos_w_z0[:, None, None, :]

    ref_anchor_yaw_quat_w = projected_yaw_quat(ref_anchor_quat_w)
    ref_anchor_yaw_quat_w_future = ref_anchor_yaw_quat_w[:, None, None, :]
    ref_anchor_yaw_quat_conj_w_future = quat_conjugate(ref_anchor_yaw_quat_w_future)

    ref_body_pos_future_local = quat_apply_inverse(
        ref_anchor_yaw_quat_w_future,
        ref_body_pos_future_w - ref_anchor_pos_w_z0_future,
    )
    ref_body_quat_future_local = quat_mul(
        ref_anchor_yaw_quat_conj_w_future.expand_as(ref_body_quat_future_w),
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
    ref_anchor_pos_w_z0 = ref_anchor_pos_w.clone()
    ref_anchor_pos_w_z0[..., 2] = 0.0
    robot_anchor_pos_w_z0 = robot_anchor_pos_w.clone()
    robot_anchor_pos_w_z0[..., 2] = 0.0
    ref_anchor_pos_w_z0_future = ref_anchor_pos_w_z0[:, None, None, :]
    robot_anchor_pos_w_z0_body = robot_anchor_pos_w_z0[:, None, :]

    ref_anchor_yaw_quat_w = projected_yaw_quat(ref_anchor_quat_w)
    robot_anchor_yaw_quat_w = projected_yaw_quat(robot_anchor_quat_w)
    ref_anchor_yaw_quat_w_future = ref_anchor_yaw_quat_w[:, None, None, :]
    robot_anchor_yaw_quat_w_body = robot_anchor_yaw_quat_w[:, None, :]
    ref_anchor_yaw_quat_conj_w_future = quat_conjugate(ref_anchor_yaw_quat_w_future)
    robot_anchor_yaw_quat_conj_w_body = quat_conjugate(robot_anchor_yaw_quat_w_body)

    ref_body_pos_future_local = quat_apply_inverse(
        ref_anchor_yaw_quat_w_future,
        ref_body_pos_future_w - ref_anchor_pos_w_z0_future,
    )
    ref_body_quat_future_local = quat_mul(
        ref_anchor_yaw_quat_conj_w_future.expand_as(ref_body_quat_future_w),
        ref_body_quat_future_w,
    )

    robot_body_pos_local = quat_apply_inverse(
        robot_anchor_yaw_quat_w_body,
        robot_body_link_pos_w - robot_anchor_pos_w_z0_body,
    )
    robot_body_quat_local = quat_mul(
        robot_anchor_yaw_quat_conj_w_body.expand_as(robot_body_link_quat_w),
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


@dataclass
class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    # mode: Literal["ghost", "frames"] = "frames"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)


@dataclass
class ResetProfileStats:
    calls: int = 0
    envs_total: int = 0
    total_reset_t: int = 0
    max_reset_envs_one_call: int = 0
    max_reset_t_seen: int = 0


class RobotTracking(Command, namespace="mimic_lite"):
    def __init__(
        self,
        env,
        motion_cfgs: Mapping[str, object],
        tracking_body_names: List[str],
        tracking_joint_names: List[str],
        obs_body_names: List[str] | None = None,
        # reset parameters
        # will be offloaded to a dedicated randomization module in the future
        root_body_name: str = "pelvis",
        pose_range: Dict[str, Tuple[float, float]] = {
            "x": (-0.0, 0.0),
            "y": (-0.0, 0.0),
            "z": (-0.0, 0.0),
            "roll": (-0.0, 0.0),
            "pitch": (-0.0, 0.0),
            "yaw": (-0.0, 0.0),
        },
        velocity_range: Dict[str, Tuple[float, float]] = {
            "x": (-0.0, 0.0),
            "y": (-0.0, 0.0),
            "z": (-0.0, 0.0),
            "roll": (-0.0, 0.0),
            "pitch": (-0.0, 0.0),
            "yaw": (-0.0, 0.0),
        },
        init_joint_pos_noise: float = 0.0,
        init_joint_vel_noise: float = 0.0,
        # observation parameters
        future_steps: List[int] = [1, 2, 8, 16],
        diff_future_steps: List[int] = [0, 1],
        anchor_body_name: str = "torso_link",
        windowed_next_window_device: str | None = "current",
        windowed_pin_window_load: bool = True,
        call_update: bool = True,
        replay_motion: bool = False,
        record_motion: bool = False,
        start_from_zero: bool = False,
        rewind_prob: float = 0.0,
        rewind_steps_range: Tuple[int, int] = (25, 125),
        viz: VizCfg | Dict | None = None,
    ):
        for module_name in (".observations", ".rewards", ".terminations"):
            importlib.import_module(module_name, package=__package__)

        super().__init__(env)
        self.motion_cfgs: list[MotionDatasetConfig] = normalize_motion_cfgs(motion_cfgs)

        # Resolve the exact asset names before loading motion data so large
        # windowed datasets can skip body/joint fields this task never reads.
        tracking_body_indices_asset, self.tracking_body_names = find_bodies(
            self.asset, tracking_body_names
        )
        if obs_body_names is None:
            obs_body_names = self.tracking_body_names
        _, self.obs_body_names = find_bodies(self.asset, obs_body_names)
        tracking_joint_indices_asset, self.tracking_joint_names = find_joints(
            self.asset, tracking_joint_names
        )

        env_next_window_device = os.environ.get("ANY4HDMI_NEXT_WINDOW_DEVICE")
        if env_next_window_device:
            windowed_next_window_device = env_next_window_device
        if os.environ.get("ANY4HDMI_PIN_WINDOW_LOAD", "0") == "1":
            windowed_pin_window_load = True

        motion_body_names = list(dict.fromkeys([
            *self.tracking_body_names,
            root_body_name,
            anchor_body_name,
        ]))
        motion_joint_names = list(self.asset.joint_names)

        self.dataset = load_motion_dataset_collection(
            self.motion_cfgs,
            create_dataset_fn=create_dataset_from_path,
            target_fps=int(1 / self.env.step_dt),
            num_envs=self.num_envs,
            body_names=motion_body_names,
            joint_names=motion_joint_names,
            windowed_next_window_device=windowed_next_window_device,
            windowed_pin_window_load=windowed_pin_window_load,
        ).to(self.device)
        print(
            "[mimic_lite][motion_dataset]"
            f" pruned bodies={len(self.dataset.body_names)}"
            f" joints={len(self.dataset.joint_names)}"
        )

        # Set tracking body and joint names for observation and termination
        self.tracking_body_indices_motion = [
            self.dataset.body_names.index(name) for name in self.tracking_body_names
        ]
        self.tracking_body_indices_asset = list(tracking_body_indices_asset)

        self.obs_body_indices_tracking = torch.tensor(
            [self.tracking_body_names.index(name) for name in self.obs_body_names],
            dtype=torch.long,
            device=self.device,
        )

        self.tracking_joint_indices_motion = [
            self.dataset.joint_names.index(name) for name in self.tracking_joint_names
        ]
        self.tracking_joint_indices_asset = list(tracking_joint_indices_asset)

        self.num_tracking_bodies = len(self.tracking_body_indices_asset)
        self.num_tracking_joints = len(self.tracking_joint_indices_asset)
        self.num_future_steps = len(future_steps)

        future_steps = sorted(future_steps)
        assert 0 in future_steps, "future_steps must include 0 to compute current observation"
        assert 1 in future_steps, "future_steps must include 1 to compute current reward"
        self.obs_current_step_index = future_steps.index(0)
        self.reward_current_step_index = future_steps.index(1)
        diff_future_steps = sorted(diff_future_steps)
        for step in diff_future_steps:
            assert step in future_steps, (
                f"diff_future_steps must be a subset of future_steps, got step={step}"
            )

        self.anchor_body_name = anchor_body_name
        self.anchor_body_idx_motion = self.dataset.body_names.index(anchor_body_name)
        self.anchor_body_idx_asset = self.asset.body_names.index(anchor_body_name)

        if self.env.backend == "mjlab":
            indexing = self.asset.data.indexing
            tracking_body_indices_asset = torch.as_tensor(
                self.tracking_body_indices_asset,
                dtype=torch.long,
                device=indexing.body_ids.device,
            )
            tracking_joint_indices_asset = torch.as_tensor(
                self.tracking_joint_indices_asset,
                dtype=torch.long,
                device=indexing.joint_q_adr.device,
            )
            self._mjlab_tracking_body_ids = torch.index_select(
                indexing.body_ids, 0, tracking_body_indices_asset
            )
            self._mjlab_tracking_joint_q_adr = torch.index_select(
                indexing.joint_q_adr, 0, tracking_joint_indices_asset
            )
            self._mjlab_tracking_joint_v_adr = torch.index_select(
                indexing.joint_v_adr, 0, tracking_joint_indices_asset
            )
            self._mjlab_root_body_id = indexing.root_body_id
            self._mjlab_anchor_body_id = int(
                indexing.body_ids[self.anchor_body_idx_asset].item()
            )

        with torch.device(self.device):
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)

        with torch.device(self.dataset.device):
            self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long)
            self.motion_len = torch.zeros(self.num_envs, dtype=torch.long)
            self.t = torch.zeros(self.num_envs, dtype=torch.long)
            self.future_steps = torch.tensor(future_steps)
            self.diff_future_steps = torch.tensor(diff_future_steps)
            self.future_one_step = torch.zeros(1, dtype=torch.long)
            self.diff_future_step_indices = torch.tensor(
                [future_steps.index(step) for step in diff_future_steps],
                dtype=torch.long,
            )

        # get root body and joint indices in motion for reset
        self.root_body_name = root_body_name
        self.root_body_idx_motion = self.dataset.body_names.index(root_body_name)
        self.asset_joint_idx_motion = [
            self.dataset.joint_names.index(joint_name)
            for joint_name in self.asset.joint_names
        ]

        pose_range_list = [
            pose_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        self.pose_range = torch.tensor(pose_range_list, device=self.device)
        velocity_range_list = [
            velocity_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        self.velocity_range = torch.tensor(velocity_range_list, device=self.device)

        self.init_joint_pos_noise = init_joint_pos_noise
        self.init_joint_vel_noise = init_joint_vel_noise

        self.rewind_prob = rewind_prob
        self.rewind_steps_range: Tuple[int, int] = tuple(rewind_steps_range)
        assert self.rewind_steps_range[0] >= 0
        assert self.rewind_steps_range[1] > self.rewind_steps_range[0]

        self.first_sample_motion = True
        self.record_motion = record_motion
        self.replay_motion = replay_motion
        self.start_from_zero = start_from_zero
        self._profile_resets = os.environ.get("MIMIC_LITE_PROFILE_RESETS", "0") == "1"
        self._profile_resets_print_every = max(
            1, int(os.environ.get("MIMIC_LITE_PROFILE_RESETS_PRINT_EVERY", "10"))
        )
        self._reset_profile_stats = ResetProfileStats()

        self.all_env_ids = torch.arange(self.num_envs, device=self.device)

        if self.record_motion:
            assert self.num_envs == 1, "record_motion only supports num_envs=1"
            self.pose_range.fill_(0.0)
            self.init_joint_pos_noise = 0.0
            self.init_joint_vel_noise = 0.0

        if call_update:
            self._read_current_robot_state()
            self._refresh_future_buffers()
            self.update()
            if self.record_motion:
                self.motion_frames = []

        # TODO: simplify viz config
        if isinstance(viz, dict):
            viz = VizCfg(**viz)
        self.viz = viz or VizCfg()
        self._ghost_model = None

    def _sample_motions(self, env_ids: torch.Tensor) -> None:
        terminated_t = self.t[env_ids]
        rewind_mask = torch.rand(len(env_ids), device=self.dataset.device) < self.rewind_prob

        # do not rewind when motion is about to finish
        finish_mask = terminated_t >= self.motion_len[env_ids] - 50
        rewind_mask &= ~finish_mask

        if self.first_sample_motion:
            rewind_mask.fill_(False)
        rewind_steps = torch.randint(
            *self.rewind_steps_range,
            (len(env_ids),),
            device=self.dataset.device,
        )
        sampled_motion = self.dataset.sample_motion(
            env_ids,
            terminated_t=terminated_t,
            rewind_mask=rewind_mask,
            rewind_steps=rewind_steps,
        )
        if self.start_from_zero or self.replay_motion:
            sampled_motion.start_t.fill_(0)
        self.motion_ids[env_ids] = sampled_motion.motion_id
        self.motion_len[env_ids] = sampled_motion.motion_len
        self.t[env_ids] = sampled_motion.start_t
        self.first_sample_motion = False

    def sample_init(self, env_ids: torch.Tensor) -> None:
        if self._profile_resets:
            self._record_reset_profile(env_ids)
        self._sample_motions(env_ids)

        # reset root state and joint position/velocity from motion
        self._motion_reset: MotionData = self.dataset.get_slice(
            self.motion_ids[env_ids],
            self.t[env_ids],
            self.future_one_step,
            profile_name="motion_reset",
        ).to(self.device).squeeze(1)
        # shape: [len(env_ids), num_bodies/num_joints, 3/4/...]

        motion = self._motion_reset
        init_root_pos = motion.body_pos_w[:, self.root_body_idx_motion]
        init_root_quat = motion.body_quat_w[:, self.root_body_idx_motion]
        init_root_lin_vel = motion.body_lin_vel_w[:, self.root_body_idx_motion]
        init_root_ang_vel = motion.body_ang_vel_w[:, self.root_body_idx_motion]

        # poses
        pose_rand_samples = sample_uniform(
            self.pose_range[:, 0],
            self.pose_range[:, 1],
            (len(env_ids), 6),
            device=self.device,
        )
        if not self.env.training or self.replay_motion:
            pose_rand_samples.fill_(0.0)
        positions = (
            init_root_pos
            + self.env.scene.env_origins.to(self.device)[env_ids]
            + pose_rand_samples[:, 0:3]
        )
        orientations_delta = quat_from_euler_xyz(
            pose_rand_samples[:, 3], pose_rand_samples[:, 4], pose_rand_samples[:, 5]
        )
        orientations = quat_mul(init_root_quat, orientations_delta)

        # velocities
        vel_rand_samples = sample_uniform(
            self.velocity_range[:, 0],
            self.velocity_range[:, 1],
            (len(env_ids), 6),
            device=self.device,
        )
        if not self.env.training or self.replay_motion:
            vel_rand_samples.fill_(0.0)
        velocities = (
            torch.cat([init_root_lin_vel, init_root_ang_vel], dim=-1) + vel_rand_samples
        )

        self.asset.write_root_link_pose_to_sim(
            torch.cat([positions, orientations], dim=-1), env_ids=env_ids
        )
        self._write_root_com_velocity(velocities, env_ids)
        # self.asset.write_root_com_velocity_to_sim(velocities, env_ids=env_ids)

        init_joint_pos = motion.joint_pos[:, self.asset_joint_idx_motion]
        init_joint_vel = motion.joint_vel[:, self.asset_joint_idx_motion]

        joint_pos_noise = sample_uniform(
            -1, 1, init_joint_pos.shape, device=self.device
        )
        joint_vel_noise = sample_uniform(
            -1, 1, init_joint_vel.shape, device=self.device
        )

        init_joint_pos += joint_pos_noise * self.init_joint_pos_noise
        init_joint_vel += joint_vel_noise * self.init_joint_vel_noise

        # joint_pos_limits = self.asset.data.soft_joint_pos_limits[env_ids]
        # init_joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
        # if hasattr(self.asset.data, "soft_joint_vel_limits"):
        #     joint_vel_limits = self.asset.data.soft_joint_vel_limits[env_ids]
        #     init_joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

        self.asset.write_joint_state_to_sim(
            init_joint_pos, init_joint_vel, env_ids=env_ids
        )

        if self.record_motion:
            if len(self.motion_frames) > 0:
                self._save_motion()
                self.motion_frames = []

    def reset(self, env_ids: torch.Tensor) -> None:
        pass

    def _record_reset_profile(self, env_ids: torch.Tensor) -> None:
        reset_count = int(len(env_ids))
        if reset_count == 0:
            return
        reset_t = self.t[env_ids.to(self.dataset.device)]
        total_reset_t = int(reset_t.sum().item())
        max_reset_t = int(reset_t.max().item())
        stats = self._reset_profile_stats
        stats.calls += 1
        stats.envs_total += reset_count
        stats.total_reset_t += total_reset_t
        stats.max_reset_envs_one_call = max(stats.max_reset_envs_one_call, reset_count)
        stats.max_reset_t_seen = max(stats.max_reset_t_seen, max_reset_t)

        if stats.calls % self._profile_resets_print_every != 0:
            return

        avg_envs = stats.envs_total / stats.calls
        avg_reset_t = stats.total_reset_t / stats.envs_total if stats.envs_total > 0 else 0.0
        dataset_kind = getattr(self.dataset, "dataset_kind", None)
        if dataset_kind is None:
            dataset_kind = (
                "online_any4hdmi"
                if self.dataset.__class__.__module__.startswith("any4hdmi.")
                else "legacy_npz"
            )
        print(
            "[mimic_lite][reset_profile]"
            f" dataset={dataset_kind}"
            f" calls={stats.calls}"
            f" envs_total={stats.envs_total}"
            f" avg_envs_per_call={avg_envs:.2f}"
            f" avg_reset_t={avg_reset_t:.2f}"
            f" max_reset_envs_one_call={stats.max_reset_envs_one_call}"
            f" max_reset_t_seen={stats.max_reset_t_seen}"
        )

    def _write_root_com_velocity(
        self, root_com_velocity: torch.Tensor, env_ids: torch.Tensor
    ) -> None:
        if self.env.backend == "isaaclab":
            self.asset.write_root_com_velocity_to_sim(
                root_com_velocity, env_ids=env_ids
            )
        elif self.env.backend == "mjlab":
            asset_data = self.asset.data
            quat_w = asset_data.data.qpos[
                env_ids[:, None], asset_data.indexing.free_joint_q_adr[3:7]
            ]
            com_offset_b = asset_data.model.body_ipos[
                env_ids, asset_data.indexing.root_body_id
            ]
            com_offset_w = quat_rotate(quat_w, com_offset_b)

            ang_vel_w = root_com_velocity[:, 3:]
            lin_vel_link = root_com_velocity[:, :3] - torch.cross(
                ang_vel_w, com_offset_w, dim=-1
            )
            link_velocity = torch.cat([lin_vel_link, ang_vel_w], dim=-1)
            self.asset.write_root_link_velocity_to_sim(link_velocity, env_ids=env_ids)

    def _save_motion(self):
        motion_data: TensorDict = torch.cat(self.motion_frames, dim=0)
        motion_data = motion_data[25:].numpy()
        moton_meta = {
            "joint_names": self.asset.joint_names,
            "body_names": self.asset.body_names,
            "fps": int(1 / self.env.step_dt),
        }
        save_dir = "record_motion"
        motion_data_path = f"{save_dir}/motion.npz"
        motion_meta_path = f"{save_dir}/meta.json"

        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(motion_data_path, **motion_data)
        with open(motion_meta_path, "w") as f:
            json.dump(moton_meta, f, indent=4)
        print(f"Saved recorded motion to {motion_data_path} and {motion_meta_path}")
        breakpoint()

    def _read_current_robot_state(self):
        if self.env.backend == "mjlab":
            self._read_current_robot_state_mjlab()
            return

        self.robot_body_link_pos_w = self.asset.data.body_link_pos_w[
            :, self.tracking_body_indices_asset
        ]
        self.robot_body_lin_vel_w = self.asset.data.body_com_lin_vel_w[
            :, self.tracking_body_indices_asset
        ]
        self.robot_body_link_quat_w = self.asset.data.body_link_quat_w[
            :, self.tracking_body_indices_asset
        ]
        self.robot_body_ang_vel_w = self.asset.data.body_com_ang_vel_w[
            :, self.tracking_body_indices_asset
        ]

        self.robot_joint_pos = self.asset.data.joint_pos[
            :, self.tracking_joint_indices_asset
        ]
        self.robot_joint_vel = self.asset.data.joint_vel[
            :, self.tracking_joint_indices_asset
        ]

        self.robot_root_pos_w = self.asset.data.root_link_pos_w
        self.robot_root_quat_w = self.asset.data.root_link_quat_w

        self.robot_anchor_pos_w = self.asset.data.body_link_pos_w[
            :, self.anchor_body_idx_asset
        ]
        self.robot_anchor_quat_w = self.asset.data.body_link_quat_w[
            :, self.anchor_body_idx_asset
        ]

    def _read_current_robot_state_mjlab(self):
        asset_data = self.asset.data
        sim_data = asset_data.data

        body_ids = self._mjlab_tracking_body_ids
        root_body_id = self._mjlab_root_body_id
        anchor_body_id = self._mjlab_anchor_body_id

        body_cvel = sim_data.cvel[:, body_ids]

        self.robot_body_link_pos_w = sim_data.xpos[:, body_ids]
        self.robot_body_lin_vel_w = body_cvel[..., 3:6]
        self.robot_body_link_quat_w = sim_data.xquat[:, body_ids]
        self.robot_body_ang_vel_w = body_cvel[..., 0:3]

        self.robot_joint_pos = sim_data.qpos[:, self._mjlab_tracking_joint_q_adr]
        self.robot_joint_vel = sim_data.qvel[:, self._mjlab_tracking_joint_v_adr]

        self.robot_root_pos_w = sim_data.xpos[:, root_body_id]
        self.robot_root_quat_w = sim_data.xquat[:, root_body_id]

        self.robot_anchor_pos_w = sim_data.xpos[:, anchor_body_id]
        self.robot_anchor_quat_w = sim_data.xquat[:, anchor_body_id]

    def _refresh_future_buffers(self):
        # `self.t` anchors the future-motion buffer used by observations.
        self.obs_motion_t = self.t.clone()
        self.future_ref_motion = self.dataset.get_slice(
            self.motion_ids,
            self.t,
            steps=self.future_steps,
            profile_name="future_ref_motion",
        )
        env_origins = self.env.scene.env_origins

        self.ref_body_pos_future_w = (
            self.future_ref_motion.body_pos_w[..., self.tracking_body_indices_motion, :]
            + env_origins[:, None, None, :]
        )
        self.ref_body_lin_vel_future_w = self.future_ref_motion.body_lin_vel_w[
            ..., self.tracking_body_indices_motion, :
        ]
        self.ref_body_quat_future_w = self.future_ref_motion.body_quat_w[
            ..., self.tracking_body_indices_motion, :
        ]
        self.ref_body_ang_vel_future_w = self.future_ref_motion.body_ang_vel_w[
            ..., self.tracking_body_indices_motion, :
        ]

        self.ref_joint_pos_future_ = self.future_ref_motion.joint_pos[
            ..., self.tracking_joint_indices_motion
        ]
        self.ref_joint_vel_future_ = self.future_ref_motion.joint_vel[
            ..., self.tracking_joint_indices_motion
        ]

        self.ref_root_pos_future_w = (
            self.future_ref_motion.body_pos_w[..., self.root_body_idx_motion, :]
            + env_origins[:, None, :]
        )
        self.ref_root_quat_future_w = self.future_ref_motion.body_quat_w[
            ..., self.root_body_idx_motion, :
        ]

        self.ref_anchor_pos_future_w = (
            self.future_ref_motion.body_pos_w[..., self.anchor_body_idx_motion, :]
            + env_origins[:, None, :]
        )
        self.ref_anchor_quat_future_w = self.future_ref_motion.body_quat_w[
            ..., self.anchor_body_idx_motion, :
        ]

        # root_diff_obs
        (
            self.ref_root_pos_future_b,
            self.ref_root_ori_future_b_matrix,
        ) = _compute_root_diff_obs(
            self.robot_root_pos_w,
            self.robot_root_quat_w,
            self.ref_root_pos_future_w,
            self.ref_root_quat_future_w,
        )

        # motion_local_obs
        (
            self.ref_body_pos_future_local,
            self.ref_body_ori_future_local_matrix,
        ) = _compute_motion_local_obs(
            self.ref_anchor_pos_future_w[:, self.obs_current_step_index],
            self.ref_anchor_quat_future_w[:, self.obs_current_step_index],
            self.ref_body_pos_future_w[:, :, self.obs_body_indices_tracking],
            self.ref_body_quat_future_w[:, :, self.obs_body_indices_tracking],
        )

        # body_local_diff_obs
        (
            self.diff_body_pos_future_local,
            self.diff_body_ori_future_local_matrix,
        ) = _compute_body_diff_obs(
            self.ref_anchor_pos_future_w[:, self.obs_current_step_index],
            self.ref_anchor_quat_future_w[:, self.obs_current_step_index],
            self.robot_anchor_pos_w,
            self.robot_anchor_quat_w,
            self.ref_body_pos_future_w[:, self.diff_future_step_indices],
            self.ref_body_quat_future_w[:, self.diff_future_step_indices],
            self.robot_body_link_pos_w,
            self.robot_body_link_quat_w,
        )
        self.diff_body_lin_vel_future = (
            self.ref_body_lin_vel_future_w[:, self.diff_future_step_indices] - self.robot_body_lin_vel_w.unsqueeze(1)
        )
        self.diff_body_ang_vel_future = (
            self.ref_body_ang_vel_future_w[:, self.diff_future_step_indices] - self.robot_body_ang_vel_w.unsqueeze(1)
        )

    def step(self):
        self._refresh_future_buffers()
        self.t += 1

    def update(self):
        if self.replay_motion:
            # Set the full robot state to the reference frame used for replay.
            env_ids = self.all_env_ids
            time_index = self.obs_current_step_index
            self.asset.write_root_link_pose_to_sim(
                torch.cat(
                    [
                        self.ref_root_pos_future_w[:, time_index],
                        self.ref_root_quat_future_w[:, time_index],
                    ],
                    dim=-1,
                ),
                env_ids=env_ids,
            )
            self._write_root_com_velocity(
                torch.cat(
                    [
                        self.future_ref_motion.body_lin_vel_w[
                            :, time_index, self.root_body_idx_motion
                        ],
                        self.future_ref_motion.body_ang_vel_w[
                            :, time_index, self.root_body_idx_motion
                        ],
                    ],
                    dim=-1,
                ),
                env_ids=env_ids,
            )
            self.asset.write_joint_state_to_sim(
                self.future_ref_motion.joint_pos[
                    :, time_index, self.asset_joint_idx_motion
                ],
                self.future_ref_motion.joint_vel[
                    :, time_index, self.asset_joint_idx_motion
                ],
                env_ids=env_ids,
            )
            if self.env.backend == "mjlab":
                self.env.sim.forward()

        if hasattr(self, "motion_frames"):
            with ScopedTimer("command_update.record_motion", sync=False):
                motion_frame = {}
                motion_frame["body_pos_w"] = self.asset.data.body_link_pos_w.cpu()
                motion_frame["body_quat_w"] = self.asset.data.body_link_quat_w.cpu()
                motion_frame["body_lin_vel_w"] = self.asset.data.body_com_lin_vel_w.cpu()
                motion_frame["body_ang_vel_w"] = self.asset.data.body_com_ang_vel_w.cpu()
                motion_frame["joint_pos"] = self.asset.data.joint_pos.cpu()
                motion_frame["joint_vel"] = self.asset.data.joint_vel.cpu()
                self.motion_frames.append(TensorDict(motion_frame, batch_size=[1]))

        with ScopedTimer("command_update.read_current_robot_state", sync=False):
            self._read_current_robot_state()

        # Reward / termination: consume the current frame from the previously
        # prepared future-motion buffer.
        with ScopedTimer("command_update.select_current_reference", sync=False):
            self.current_ref_motion = self.future_ref_motion[:, self.reward_current_step_index]
            self.ref_body_pos_w = self.ref_body_pos_future_w[:, self.reward_current_step_index]
            self.ref_body_lin_vel_w = self.ref_body_lin_vel_future_w[:, self.reward_current_step_index]
            self.ref_body_quat_w = self.ref_body_quat_future_w[:, self.reward_current_step_index]
            self.ref_body_ang_vel_w = self.ref_body_ang_vel_future_w[:, self.reward_current_step_index]
            self.ref_joint_pos = self.ref_joint_pos_future_[:, self.reward_current_step_index]
            self.ref_joint_vel = self.ref_joint_vel_future_[:, self.reward_current_step_index]
            self.ref_anchor_pos_w = self.ref_anchor_pos_future_w[:, self.reward_current_step_index]
            self.ref_anchor_quat_w = self.ref_anchor_quat_future_w[:, self.reward_current_step_index]
        with ScopedTimer(
            "command_update.current_tracking_state", sync=PROFILE_SYNC_TIMERS
        ):
            (
                self.ref_body_pos_local,
                self.ref_body_quat_local,
                self.robot_body_pos_local,
                self.robot_body_quat_local,
            ) = _compute_current_tracking_state(
                self.ref_anchor_pos_w,
                self.ref_anchor_quat_w,
                self.robot_anchor_pos_w,
                self.robot_anchor_quat_w,
                self.ref_body_pos_w,
                self.ref_body_quat_w,
                self.robot_body_link_pos_w,
                self.robot_body_link_quat_w,
            )

        with ScopedTimer("command_update.tracking_errors", sync=PROFILE_SYNC_TIMERS):
            (
                self.body_pos_error,
                self.body_pos_error_local,
                self.body_ori_error,
                self.body_ori_error_local,
                self.body_lin_vel_error,
                self.body_ang_vel_error,
                self.joint_pos_error,
                self.joint_vel_error,
            ) = _compute_tracking_errors(
                self.ref_body_pos_w,
                self.ref_body_quat_w,
                self.ref_body_lin_vel_w,
                self.ref_body_ang_vel_w,
                self.ref_body_pos_local,
                self.ref_body_quat_local,
                self.robot_body_link_pos_w,
                self.robot_body_link_quat_w,
                self.robot_body_lin_vel_w,
                self.robot_body_ang_vel_w,
                self.robot_body_pos_local,
                self.robot_body_quat_local,
                self.ref_joint_pos,
                self.ref_joint_vel,
                self.robot_joint_pos,
                self.robot_joint_vel,
            )

    def debug_draw(self):
        if not hasattr(self, "current_ref_motion"):
            return

        viewer = getattr(self.env.sim, "viewer", None)
        if viewer is None:
            return
        scene: "ViserMujocoScene" | None = getattr(viewer, "scene", None)
        if scene is None:
            return

        if self.viz.mode == "ghost":
            if self._ghost_model is None:
                self._ghost_model = copy.deepcopy(self.env.sim.mj_model)
                self._ghost_model.geom_rgba[:] = self.viz.ghost_color

            indexing = self.asset.indexing
            free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
            joint_q_adr = indexing.joint_q_adr.cpu().numpy()

            if scene.show_all_envs or self.num_envs == 1:
                env_ids = range(self.num_envs)
            else:
                env_ids = [int(scene.env_idx)]

            for env_idx in env_ids:
                qpos = np.zeros(self.env.sim.mj_model.nq)
                # for time_index in [self.obs_current_step_index, -1]:
                for time_index in [self.obs_current_step_index]:
                    qpos[free_joint_q_adr[0:3]] = (
                        self.ref_root_pos_future_w[env_idx, time_index].cpu().numpy()
                    )
                    qpos[free_joint_q_adr[3:7]] = (
                        self.ref_root_quat_future_w[env_idx, time_index].cpu().numpy()
                    )
                    qpos[joint_q_adr] = (
                        self.future_ref_motion.joint_pos[
                            env_idx, time_index, self.asset_joint_idx_motion
                        ]
                        .cpu()
                        .numpy()
                    )

                    scene.add_ghost_mesh(
                        qpos,
                        model=self._ghost_model,
                        label=f"env_{env_idx}",
                    )
        elif self.viz.mode == "frames":
            for env_idx in range(self.num_envs):
                desired_body_pos = self.ref_body_pos_w[env_idx].cpu().numpy()
                desired_body_quat = self.ref_body_quat_w[env_idx]
                desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

                current_body_pos = self.robot_body_link_pos_w[env_idx].cpu().numpy()
                current_body_quat = self.robot_body_link_quat_w[env_idx]
                current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

                for i, body_name in enumerate(self.tracking_body_names):
                    scene.add_frame(
                        position=desired_body_pos[i],
                        rotation_matrix=desired_body_rotm[i],
                        scale=0.08,
                        label=f"desired_{body_name}_env_{env_idx}",
                        axis_colors=_DESIRED_FRAME_COLORS,
                    )
                    scene.add_frame(
                        position=current_body_pos[i],
                        rotation_matrix=current_body_rotm[i],
                        scale=0.12,
                        label=f"current_{body_name}_env_{env_idx}",
                    )
