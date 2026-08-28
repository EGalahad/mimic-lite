from active_adaptation.envs.mdp.commands.base import Command
from active_adaptation.envs.utils import find_bodies, find_joints
from mimic_lite.tasks.motion import MotionData, create_dataset_from_path
from mimic_lite.tasks.multi_dataset import (
    MotionDatasetConfig,
    load_motion_dataset_collection,
    normalize_motion_cfgs,
)
from mimic_lite.tasks.transforms import (
    _compute_body_diff_obs,
    _compute_current_tracking_state,
    _compute_motion_local_obs,
    _compute_root_diff_obs,
    _compute_tracking_errors,
    quat_from_euler_xyz,
    sample_uniform,
)

from dataclasses import dataclass
from typing import List, Dict, Tuple, TYPE_CHECKING, Literal, Mapping
import copy
import importlib
import os

if TYPE_CHECKING:
    from mjlab.viewer.viser import ViserMujocoScene

import torch
import numpy as np

from active_adaptation.utils.math import (
    matrix_from_quat,
    quat_mul,
    quat_rotate,
)
from active_adaptation.utils.profiling import ScopedTimer
from tensordict import TensorDictBase

PROFILE_SYNC_TIMERS = os.environ.get("AA_PROFILE_SYNC_TIMERS", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

_DESIRED_FRAME_COLORS = (
    (0.9, 0.3, 0.3, 0.9),
    (0.3, 0.9, 0.3, 0.9),
    (0.3, 0.3, 0.9, 0.9),
)


@dataclass
class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    # mode: Literal["ghost", "frames"] = "frames"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)


class RobotTracking(Command, namespace="mimic_lite"):
    def __init__(
        self,
        motion_cfgs: Mapping[str, object],
        tracking_body_names: List[str],
        tracking_joint_names: List[str],
        obs_body_names: List[str] | None = None,
        extra_motion_body_names: List[str] | None = None,
        extra_motion_joint_names: List[str] | None = None,
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
        start_from_zero: bool = False,
        rewind_prob: float = 0.0,
        rewind_steps_range: Tuple[int, int] = (25, 125),
        viz: VizCfg | Dict | None = None,
    ):
        for module_name in (".observations", ".rewards", ".terminations"):
            importlib.import_module(module_name, package=__package__)
        super().__init__()
        self.motion_cfgs: list[MotionDatasetConfig] = normalize_motion_cfgs(motion_cfgs)
        self._tracking_body_names_cfg = list(tracking_body_names)
        self._tracking_joint_names_cfg = list(tracking_joint_names)
        self._obs_body_names_cfg = None if obs_body_names is None else list(obs_body_names)
        self._extra_motion_body_names = list(extra_motion_body_names or ())
        self._extra_motion_joint_names = list(extra_motion_joint_names or ())
        self._windowed_next_window_device = windowed_next_window_device
        self._windowed_pin_window_load = windowed_pin_window_load
        self._call_update = call_update

        # Resolve the exact asset names before loading motion data so large
        # windowed datasets can skip body/joint fields this task never reads.
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
        self._future_steps_cfg = list(future_steps)
        self._diff_future_steps_cfg = list(diff_future_steps)

        # get root body and joint indices in motion for reset
        self.root_body_name = root_body_name

        range_keys = ("x", "y", "z", "roll", "pitch", "yaw")
        self._pose_range_cfg = [pose_range.get(key, (0.0, 0.0)) for key in range_keys]
        self._velocity_range_cfg = [
            velocity_range.get(key, (0.0, 0.0)) for key in range_keys
        ]

        self.init_joint_pos_noise = init_joint_pos_noise
        self.init_joint_vel_noise = init_joint_vel_noise

        self.rewind_prob = rewind_prob
        self.rewind_steps_range: Tuple[int, int] = tuple(rewind_steps_range)
        assert self.rewind_steps_range[0] >= 0
        assert self.rewind_steps_range[1] > self.rewind_steps_range[0]

        self.first_sample_motion = True
        self.replay_motion = replay_motion
        self.start_from_zero = start_from_zero

        if isinstance(viz, dict):
            viz = VizCfg(**viz)
        self.viz = viz or VizCfg()
        self._ghost_model = None

    def _initialize(self, env) -> None:
        super()._initialize(env)
        tracking_body_indices_asset, self.tracking_body_names = find_bodies(
            self.asset, self._tracking_body_names_cfg
        )
        obs_body_names = self._obs_body_names_cfg or self.tracking_body_names
        _, self.obs_body_names = find_bodies(self.asset, obs_body_names)
        tracking_joint_indices_asset, self.tracking_joint_names = find_joints(
            self.asset, self._tracking_joint_names_cfg
        )

        motion_body_names = list(
            dict.fromkeys(
                [
                    *self.tracking_body_names,
                    *self._extra_motion_body_names,
                    self.root_body_name,
                    self.anchor_body_name,
                ]
            )
        )
        motion_joint_names = list(
            dict.fromkeys([*self.asset.joint_names, *self._extra_motion_joint_names])
        )
        self.dataset = load_motion_dataset_collection(
            self.motion_cfgs,
            create_dataset_fn=create_dataset_from_path,
            target_fps=int(1 / self.env.step_dt),
            num_envs=self.num_envs,
            body_names=motion_body_names,
            joint_names=motion_joint_names,
            windowed_next_window_device=self._windowed_next_window_device,
            windowed_pin_window_load=self._windowed_pin_window_load,
        ).to(self.device)
        print(
            "[mimic_lite][motion_dataset]"
            f" pruned bodies={len(self.dataset.body_names)}"
            f" joints={len(self.dataset.joint_names)}"
        )

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
        self.anchor_body_idx_motion = self.dataset.body_names.index(self.anchor_body_name)
        self.anchor_body_idx_asset = self.asset.body_names.index(self.anchor_body_name)
        self.root_body_idx_motion = self.dataset.body_names.index(self.root_body_name)
        self.asset_joint_idx_motion = [
            self.dataset.joint_names.index(joint_name)
            for joint_name in self.asset.joint_names
        ]
        self.pose_range = torch.tensor(self._pose_range_cfg, device=self.device)
        self.velocity_range = torch.tensor(self._velocity_range_cfg, device=self.device)

        if self.env.backend == "mjlab":
            indexing = self.asset.data.indexing
            body_indices = torch.as_tensor(
                self.tracking_body_indices_asset,
                dtype=torch.long,
                device=indexing.body_ids.device,
            )
            joint_indices = torch.as_tensor(
                self.tracking_joint_indices_asset,
                dtype=torch.long,
                device=indexing.joint_q_adr.device,
            )
            self._mjlab_tracking_body_ids = torch.index_select(
                indexing.body_ids, 0, body_indices
            )
            self._mjlab_tracking_joint_q_adr = torch.index_select(
                indexing.joint_q_adr, 0, joint_indices
            )
            self._mjlab_tracking_joint_v_adr = torch.index_select(
                indexing.joint_v_adr, 0, joint_indices
            )
            self._mjlab_root_body_id = indexing.root_body_id
            self._mjlab_anchor_body_id = int(
                indexing.body_ids[self.anchor_body_idx_asset].item()
            )

        with torch.device(self.device):
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)
            self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long)
            self.motion_len = torch.zeros(self.num_envs, dtype=torch.long)
            self.t = torch.zeros(self.num_envs, dtype=torch.long)
            self.future_steps = torch.tensor(self._future_steps_cfg)
            self.diff_future_steps = torch.tensor(self._diff_future_steps_cfg)
            self.future_one_step = torch.zeros(1, dtype=torch.long)
            self.diff_future_step_indices = torch.tensor(
                [
                    self._future_steps_cfg.index(step)
                    for step in self._diff_future_steps_cfg
                ],
                dtype=torch.long,
            )
            self.all_env_ids = torch.arange(self.num_envs, device=self.device)

        if self._call_update:
            self._read_current_robot_state()
            self._refresh_future_buffers()
            self.update()

    def _sample_motions(
        self,
        env_ids: torch.Tensor,
        *,
        terminated: torch.Tensor | None = None,
    ) -> None:
        terminated_t = self.t[env_ids]
        rewind_mask = torch.rand(len(env_ids), device=self.device) < self.rewind_prob
        if terminated is None:
            terminated_mask = torch.zeros(
                len(env_ids),
                dtype=torch.bool,
                device=self.device,
            )
        else:
            terminated_mask = terminated.to(
                device=self.device,
                dtype=torch.bool,
            ).reshape(-1)
        rewind_mask &= terminated_mask

        # do not rewind when motion is about to finish
        finish_mask = terminated_t >= self.motion_len[env_ids] - 50
        rewind_mask &= ~finish_mask

        if self.first_sample_motion:
            rewind_mask.fill_(False)
        rewind_steps = torch.randint(
            *self.rewind_steps_range,
            (len(env_ids),),
            device=self.device,
        )
        sampled_motion = self.dataset.sample_motion(
            env_ids,
            terminated_t=terminated_t,
            rewind_mask=rewind_mask,
            rewind_steps=rewind_steps,
        )
        if self.start_from_zero or self.replay_motion:
            sampled_motion.start_t.fill_(1)
        self.motion_ids[env_ids] = sampled_motion.motion_id
        self.motion_len[env_ids] = sampled_motion.motion_len
        self.t[env_ids] = sampled_motion.start_t
        self.first_sample_motion = False

    def sample_init(
        self,
        env_ids: torch.Tensor,
        reset_td: TensorDictBase | None = None,
    ) -> None:
        terminated = reset_td.get("terminated", None) if reset_td is not None else None
        self._sample_motions(
            env_ids,
            terminated=terminated,
        )

        # reset root state and joint position/velocity from motion
        motion_reset: MotionData = self.dataset.get_slice(
            self.motion_ids[env_ids],
            self.t[env_ids],
            self.future_one_step,
        ).to(self.device).squeeze(1)
        # shape: [len(env_ids), num_bodies/num_joints, 3/4/...]

        init_root_pos = motion_reset.body_pos_w[:, self.root_body_idx_motion]
        init_root_quat = motion_reset.body_quat_w[:, self.root_body_idx_motion]
        init_root_lin_vel = motion_reset.body_lin_vel_w[:, self.root_body_idx_motion]
        init_root_ang_vel = motion_reset.body_ang_vel_w[:, self.root_body_idx_motion]

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
        init_joint_pos = motion_reset.joint_pos[:, self.asset_joint_idx_motion]
        init_joint_vel = motion_reset.joint_vel[:, self.asset_joint_idx_motion]

        joint_pos_noise = sample_uniform(
            -1, 1, init_joint_pos.shape, device=self.device
        )
        joint_vel_noise = sample_uniform(
            -1, 1, init_joint_vel.shape, device=self.device
        )

        init_joint_pos += joint_pos_noise * self.init_joint_pos_noise
        init_joint_vel += joint_vel_noise * self.init_joint_vel_noise

        self.asset.write_joint_state_to_sim(
            init_joint_pos, init_joint_vel, env_ids=env_ids
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

    def _read_current_robot_state(self):
        if self.env.backend == "mjlab":
            self._read_current_robot_state_mjlab()
            return

        data = self.asset.data
        self.robot_body_link_pos_w = data.body_link_pos_w[
            :, self.tracking_body_indices_asset
        ]
        self.robot_body_lin_vel_w = data.body_com_lin_vel_w[
            :, self.tracking_body_indices_asset
        ]
        self.robot_body_link_quat_w = data.body_link_quat_w[
            :, self.tracking_body_indices_asset
        ]
        self.robot_body_ang_vel_w = data.body_com_ang_vel_w[
            :, self.tracking_body_indices_asset
        ]

        self.robot_joint_pos = data.joint_pos[:, self.tracking_joint_indices_asset]
        self.robot_joint_vel = data.joint_vel[:, self.tracking_joint_indices_asset]

        self.robot_root_pos_w = data.root_link_pos_w
        self.robot_root_quat_w = data.root_link_quat_w

        self.robot_anchor_pos_w = data.body_link_pos_w[:, self.anchor_body_idx_asset]
        self.robot_anchor_quat_w = data.body_link_quat_w[:, self.anchor_body_idx_asset]

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
        with ScopedTimer("command_step.get_slice", sync=PROFILE_SYNC_TIMERS):
            self.future_ref_motion = self.dataset.get_slice(
                self.motion_ids,
                self.t,
                steps=self.future_steps,
            )
        motion = self.future_ref_motion
        body_indices = self.tracking_body_indices_motion
        joint_indices = self.tracking_joint_indices_motion
        env_origins = self.env.scene.env_origins

        self.ref_body_pos_future_w = (
            motion.body_pos_w[..., body_indices, :]
            + env_origins[:, None, None, :]
        )
        self.ref_body_lin_vel_future_w = motion.body_lin_vel_w[..., body_indices, :]
        self.ref_body_quat_future_w = motion.body_quat_w[..., body_indices, :]
        self.ref_body_ang_vel_future_w = motion.body_ang_vel_w[..., body_indices, :]

        self.ref_joint_pos_future_ = motion.joint_pos[..., joint_indices]
        self.ref_joint_vel_future_ = motion.joint_vel[..., joint_indices]

        self.ref_root_pos_future_w = (
            motion.body_pos_w[..., self.root_body_idx_motion, :]
            + env_origins[:, None, :]
        )
        self.ref_root_quat_future_w = motion.body_quat_w[
            ..., self.root_body_idx_motion, :
        ]

        self.ref_anchor_pos_future_w = (
            motion.body_pos_w[..., self.anchor_body_idx_motion, :]
            + env_origins[:, None, :]
        )
        self.ref_anchor_quat_future_w = motion.body_quat_w[
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
            self.ref_body_lin_vel_future_w[:, self.diff_future_step_indices]
            - self.robot_body_lin_vel_w.unsqueeze(1)
        )
        self.diff_body_ang_vel_future = (
            self.ref_body_ang_vel_future_w[:, self.diff_future_step_indices]
            - self.robot_body_ang_vel_w.unsqueeze(1)
        )

    def step(self):
        self._refresh_future_buffers()
        self.t += 1

    def update(self):
        motion = self.future_ref_motion
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
                        motion.body_lin_vel_w[:, time_index, self.root_body_idx_motion],
                        motion.body_ang_vel_w[:, time_index, self.root_body_idx_motion],
                    ],
                    dim=-1,
                ),
                env_ids=env_ids,
            )
            self.asset.write_joint_state_to_sim(
                motion.joint_pos[:, time_index, self.asset_joint_idx_motion],
                motion.joint_vel[:, time_index, self.asset_joint_idx_motion],
                env_ids=env_ids,
            )
            if self.env.backend == "mjlab":
                self.env.sim.forward()

        with ScopedTimer("command_update.read_current_robot_state", sync=False):
            self._read_current_robot_state()

        # Reward / termination: consume the current frame from the previously
        # prepared future-motion buffer.
        with ScopedTimer("command_update.select_current_reference", sync=False):
            time_index = self.reward_current_step_index
            self.current_ref_motion = motion[:, time_index]
            self.ref_body_pos_w = self.ref_body_pos_future_w[:, time_index]
            self.ref_body_lin_vel_w = self.ref_body_lin_vel_future_w[:, time_index]
            self.ref_body_quat_w = self.ref_body_quat_future_w[:, time_index]
            self.ref_body_ang_vel_w = self.ref_body_ang_vel_future_w[:, time_index]
            self.ref_joint_pos = self.ref_joint_pos_future_[:, time_index]
            self.ref_joint_vel = self.ref_joint_vel_future_[:, time_index]
            self.ref_anchor_pos_w = self.ref_anchor_pos_future_w[:, time_index]
            self.ref_anchor_quat_w = self.ref_anchor_quat_future_w[:, time_index]
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

        sim = self.env.sim
        viewer = getattr(sim, "viewer", None)
        if viewer is None:
            return
        scene: "ViserMujocoScene" | None = getattr(viewer, "scene", None)
        if scene is None:
            return

        if self.viz.mode == "ghost":
            if self._ghost_model is None:
                self._ghost_model = copy.deepcopy(sim.mj_model)
                robot_body_ids = set(
                    np.asarray(
                        self.asset.data.indexing.body_ids.cpu().numpy()
                    ).reshape(-1)
                )
                for geom_id in range(self._ghost_model.ngeom):
                    body_id = int(self._ghost_model.geom_bodyid[geom_id])
                    group_id = int(self._ghost_model.geom_group[geom_id])
                    visible = (
                        body_id in robot_body_ids
                        and group_id < len(scene.geom_groups_visible)
                        and scene.geom_groups_visible[group_id]
                    )
                    if visible:
                        self._ghost_model.geom_rgba[geom_id] = self.viz.ghost_color
                    else:
                        self._ghost_model.geom_rgba[geom_id, 3] = 0.0

            indexing = self.asset.indexing
            free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
            joint_q_adr = indexing.joint_q_adr.cpu().numpy()

            if scene.show_all_envs or self.num_envs == 1:
                env_ids = range(self.num_envs)
            else:
                env_ids = [int(scene.env_idx)]

            motion = self.future_ref_motion
            time_index = self.obs_current_step_index
            for env_idx in env_ids:
                qpos = np.zeros(sim.mj_model.nq)
                qpos[free_joint_q_adr[0:3]] = (
                    self.ref_root_pos_future_w[env_idx, time_index].cpu().numpy()
                )
                qpos[free_joint_q_adr[3:7]] = (
                    self.ref_root_quat_future_w[env_idx, time_index].cpu().numpy()
                )
                qpos[joint_q_adr] = (
                    motion.joint_pos[
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
