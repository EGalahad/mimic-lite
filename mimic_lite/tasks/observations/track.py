from mimic_lite.tasks.command import RobotTracking
from mimic_lite.tasks.actions import JointPosition

from active_adaptation.utils.string import resolve_matching_names
from active_adaptation.utils.math import matrix_from_quat
from mimic_lite.tasks.deferred import DeferredObservation as BaseObservation
from mimic_lite.tasks.transforms import (
    _body_pose_in_anchor_frame,
    _spatial_motion_from_local_poses,
)

import torch
from typing import cast, List

TrackObservation = BaseObservation[RobotTracking]


def _select_available_body_names(
    available_body_names: list[str],
    body_names: List[str] | str,
) -> tuple[list[int], list[str]]:
    _, selected_body_names = resolve_matching_names(body_names, available_body_names)
    if not selected_body_names:
        raise ValueError("No tracking body matched for observation.")
    selected_body_indices = [
        available_body_names.index(body_name) for body_name in selected_body_names
    ]
    return selected_body_indices, selected_body_names


class _tracking_future_step_observation(TrackObservation):
    def _initialize_impl(
        self,
        future_steps: List[int] | int | None = None,
        **kwargs,
    ):
        if future_steps is None:
            future_steps = self.command_manager.future_steps.tolist()
        elif isinstance(future_steps, int):
            future_steps = [future_steps]

        available_future_steps = [
            int(step) for step in self.command_manager.future_steps.tolist()
        ]
        future_step_indices = []
        for step in future_steps:
            step = int(step)
            if step not in available_future_steps:
                raise ValueError(
                    f"future step {step} not in command.future_steps={available_future_steps}"
                )
            future_step_indices.append(available_future_steps.index(step))

        self.future_step_indices = torch.as_tensor(
            future_step_indices, dtype=torch.long, device=self.device
        )

    def _select_future_steps(self, x: torch.Tensor) -> torch.Tensor:
        return torch.index_select(x, 1, self.future_step_indices)


class ref_joint_pos_future(_tracking_future_step_observation, namespace="mimic_lite"):
    def _initialize_impl(self, noise_std=0.0, **kwargs):
        super()._initialize_impl(**kwargs)
        self.noise_std = noise_std
        
    def compute(self):
        joint_pos = self._select_future_steps(
            self.command_manager.ref_joint_pos_future_
        ).reshape(self.num_envs, -1)
        if self.noise_std > 0.0:
            joint_pos += (
                torch.randn_like(joint_pos).clamp(-3.0, 3.0) * self.noise_std
            )
        return joint_pos


class ref_joint_vel_future(_tracking_future_step_observation, namespace="mimic_lite"):
    def compute(self):
        return self._select_future_steps(
            self.command_manager.ref_joint_vel_future_
        ).reshape(self.num_envs, -1)


class ref_joint_action(TrackObservation, namespace="mimic_lite"):
    def _initialize_impl(self, **kwargs):
        action_manager = cast(JointPosition, self.env.action_manager)
        self.action_joint_ids = action_manager.joint_ids
        self.action_indices_motion = [
            self.command_manager.dataset.joint_names.index(joint_name)
            for joint_name in action_manager.joint_names
        ]

        self.action_scaling = action_manager.action_scaling
        self.default_joint_pos = action_manager.default_joint_pos[
            :, self.action_joint_ids
        ]

    def compute(self):
        ref_joint_pos = self.command_manager.current_ref_motion.joint_pos[
            :, self.action_indices_motion
        ]
        ref_joint_action = (
            ref_joint_pos - self.default_joint_pos
        ) / self.action_scaling
        return ref_joint_action

# root_diff_obs

class ref_root_pos_future_b(
    _tracking_future_step_observation, namespace="mimic_lite"
):
    """
    Reference root position in robot root frame
    """

    def compute(self):
        return self._select_future_steps(
            self.command_manager.ref_root_pos_future_b
        ).reshape(self.num_envs, -1)


class ref_root_ori_future_b(_tracking_future_step_observation, namespace="mimic_lite"):
    """
    Reference root orientation in robot root frame
    """

    def _initialize_impl(self, noise_std=0.0, **kwargs):
        super()._initialize_impl(**kwargs)
        self.noise_std = noise_std

    def compute(self):
        ref_root_ori_future_b = self._select_future_steps(
            self.command_manager.ref_root_ori_future_b_matrix
        )
        if self.noise_std > 0.0:
            ref_root_ori_future_b = ref_root_ori_future_b.clone()
            ref_root_ori_future_b += (
                torch.randn_like(ref_root_ori_future_b).clamp(-3.0, 3.0) * self.noise_std
            )
        return ref_root_ori_future_b[:, :, :2, :].reshape(self.num_envs, -1)


# motion_local_obs

class _tracking_body_future_observation(TrackObservation):
    available_body_names_attr = "tracking_body_names"
    available_future_steps_attr = "future_steps"

    def _initialize_impl(
        self,
        body_names: List[str] | str | None = None,
        future_steps: List[int] | int | None = None,
        **kwargs,
    ):
        available_body_names = list(
            getattr(self.command_manager, self.available_body_names_attr)
        )
        if body_names is None:
            body_names = available_body_names
        if future_steps is None:
            future_steps = getattr(
                self.command_manager, self.available_future_steps_attr
            ).tolist()
        elif isinstance(future_steps, int):
            future_steps = [future_steps]

        body_indices_tracking, matched_body_names = _select_available_body_names(
            available_body_names,
            body_names,
        )

        available_future_steps = [
            int(step)
            for step in getattr(
                self.command_manager, self.available_future_steps_attr
            ).tolist()
        ]
        future_step_indices = []
        for step in future_steps:
            step = int(step)
            if step not in available_future_steps:
                raise ValueError(
                    f"future step {step} not in command.{self.available_future_steps_attr}={available_future_steps}"
                )
            future_step_indices.append(available_future_steps.index(step))

        self.body_indices_tracking = torch.as_tensor(
            body_indices_tracking, dtype=torch.long, device=self.device
        )
        self.future_step_indices = torch.as_tensor(
            future_step_indices, dtype=torch.long, device=self.device
        )

    def _select_body_future(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.index_select(x, 1, self.future_step_indices)
        return torch.index_select(x, 2, self.body_indices_tracking)


class _motion_local_body_future_observation(_tracking_body_future_observation):
    available_body_names_attr = "obs_body_names"


class ref_body_pos_future_local(
    _motion_local_body_future_observation, namespace="mimic_lite"
):
    """
    Reference body position in the projected-yaw anchor frame.
    """
    def _initialize_impl(self, noise_std=0.0, **kwargs):
        super()._initialize_impl(**kwargs)
        self.noise_std = noise_std

    def compute(self):
        ref_body_pos_future_local = self._select_body_future(
            self.command_manager.ref_body_pos_future_local
        ).reshape(self.num_envs, -1)
        if self.noise_std > 0.0:
            ref_body_pos_future_local += (
                torch.randn_like(ref_body_pos_future_local).clamp(-3.0, 3.0) * self.noise_std
            )
        return ref_body_pos_future_local


class ref_body_ori_future_local(
    _motion_local_body_future_observation, namespace="mimic_lite"
):
    """
    Reference body orientation in the projected-yaw anchor frame.
    """

    def compute(self):
        return self._select_body_future(
            self.command_manager.ref_body_ori_future_local_matrix
        )[:, :, :, :2, :].reshape(self.num_envs, -1)


class body_spatial_motion_local(
    _tracking_body_future_observation, namespace="mimic_lite"
):
    """Per-body reference spatial motion in the current reference anchor."""

    def compute(self):
        command = self.command_manager
        current_index = command.obs_current_step_index
        body_indices = self.body_indices_tracking

        future_position_w = self._select_body_future(command.ref_body_pos_future_w)
        future_quaternion_w = self._select_body_future(
            command.ref_body_quat_future_w
        )
        current_position_w = torch.index_select(
            command.ref_body_pos_future_w[:, current_index], 1, body_indices
        )
        current_quaternion_w = torch.index_select(
            command.ref_body_quat_future_w[:, current_index], 1, body_indices
        )
        anchor_position_w = command.ref_anchor_pos_future_w[:, current_index]
        anchor_quaternion_w = command.ref_anchor_quat_future_w[:, current_index]

        future_position, future_quaternion = _body_pose_in_anchor_frame(
            anchor_position_w[:, None, None],
            anchor_quaternion_w[:, None, None],
            future_position_w,
            future_quaternion_w,
        )
        current_position, current_quaternion = _body_pose_in_anchor_frame(
            anchor_position_w[:, None],
            anchor_quaternion_w[:, None],
            current_position_w,
            current_quaternion_w,
        )
        spatial_position, spatial_rotation = _spatial_motion_from_local_poses(
            future_position,
            matrix_from_quat(future_quaternion),
            current_position,
            matrix_from_quat(current_quaternion),
        )
        rotation_6d = spatial_rotation[..., :2, :].reshape(
            self.num_envs,
            len(self.future_step_indices),
            len(self.body_indices_tracking),
            6,
        )
        return torch.cat([spatial_position, rotation_6d], -1).reshape(
            self.num_envs, -1
        )


class body_spatial_error_local(
    _tracking_body_future_observation, namespace="mimic_lite"
):
    """Per-body reference-to-actual spatial correction in robot-local frames."""

    def compute(self):
        command = self.command_manager
        body_indices = self.body_indices_tracking
        current_index = command.obs_current_step_index

        future_position_w = self._select_body_future(command.ref_body_pos_future_w)
        future_quaternion_w = self._select_body_future(command.ref_body_quat_future_w)
        future_position, future_quaternion = _body_pose_in_anchor_frame(
            command.ref_anchor_pos_future_w[:, current_index, None, None],
            command.ref_anchor_quat_future_w[:, current_index, None, None],
            future_position_w,
            future_quaternion_w,
        )
        actual_position = torch.index_select(
            command.robot_body_pos_local, 1, body_indices
        )
        actual_quaternion = torch.index_select(
            command.robot_body_quat_local, 1, body_indices
        )
        spatial_position, spatial_rotation = _spatial_motion_from_local_poses(
            future_position,
            matrix_from_quat(future_quaternion),
            actual_position,
            matrix_from_quat(actual_quaternion),
        )
        rotation_6d = spatial_rotation[..., :2, :].reshape(
            self.num_envs,
            len(self.future_step_indices),
            len(self.body_indices_tracking),
            6,
        )
        return torch.cat([spatial_position, rotation_6d], -1).reshape(
            self.num_envs, -1
        )

# body_local_diff_obs

class _diff_body_future_observation(_tracking_body_future_observation):
    available_future_steps_attr = "diff_future_steps"


class diff_body_pos_future_local(
    _diff_body_future_observation, namespace="mimic_lite"
):
    """
    Reference body position in the projected-yaw anchor frame minus robot body position in the projected-yaw anchor frame.
    """

    def compute(self):
        return self._select_body_future(
            self.command_manager.diff_body_pos_future_local
        ).reshape(self.num_envs, -1)


class diff_body_lin_vel_future(
    _diff_body_future_observation, namespace="mimic_lite"
):
    """
    Reference body linear velocity minus robot body linear velocity.
    """

    def compute(self):
        return self._select_body_future(
            self.command_manager.diff_body_lin_vel_future
        ).reshape(self.num_envs, -1)


class diff_body_ori_future_local(
    _diff_body_future_observation, namespace="mimic_lite"
):
    """
    Reference body orientation in the projected-yaw anchor frame minus robot body orientation in the projected-yaw anchor frame.
    """

    def compute(self):
        return self._select_body_future(
            self.command_manager.diff_body_ori_future_local_matrix
        )[:, :, :, :2, :].reshape(self.num_envs, -1)


class diff_body_ang_vel_future(
    _diff_body_future_observation, namespace="mimic_lite"
):
    """
    Reference body angular velocity minus robot body angular velocity.
    """

    def compute(self):
        return self._select_body_future(
            self.command_manager.diff_body_ang_vel_future
        ).reshape(self.num_envs, -1)


class ref_motion_phase(TrackObservation, namespace="mimic_lite"):
    def compute(self):
        return (self.command_manager.obs_motion_t / self.command_manager.motion_len).unsqueeze(1)


class motion_length(TrackObservation, namespace="mimic_lite"):
    def compute(self):
        return self.command_manager.motion_len.to(torch.float32).unsqueeze(1)
