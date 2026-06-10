from mimic_lite.tasks.command import RobotTracking

from active_adaptation.envs.mdp.rewards.base import Reward as BaseReward
from active_adaptation.envs.utils import find_bodies, find_joints, find_sensor_bodies

from typing import List, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.sensor import ContactSensor

TrackReward = BaseReward[RobotTracking]


def _select_tracking_body_names(
    command_manager: RobotTracking,
    body_names: List[str] | str,
) -> tuple[list[int], list[str]]:
    available_body_names = list(command_manager.tracking_body_names)
    _, matched_body_names = find_bodies(command_manager.asset, body_names)
    matched_name_set = set(matched_body_names)
    selected_body_names = [
        body_name for body_name in available_body_names if body_name in matched_name_set
    ]
    assert selected_body_names, "No body names matched in tracking_body_names"
    selected_body_indices = [
        available_body_names.index(body_name) for body_name in selected_body_names
    ]
    return selected_body_indices, selected_body_names


def _select_tracking_joint_names(
    command_manager: RobotTracking,
    joint_names: List[str] | str,
) -> tuple[list[int], list[str]]:
    available_joint_names = list(command_manager.tracking_joint_names)
    _, matched_joint_names = find_joints(command_manager.asset, joint_names)
    matched_name_set = set(matched_joint_names)
    selected_joint_names = [
        joint_name for joint_name in available_joint_names if joint_name in matched_name_set
    ]
    assert selected_joint_names, "No joint names matched in tracking_joint_names"
    selected_joint_indices = [
        available_joint_names.index(joint_name) for joint_name in selected_joint_names
    ]
    return selected_joint_indices, selected_joint_names


class _tracking_body(TrackReward, namespace="mimic_lite"):
    def __init__(
        self,
        env,
        body_names: List[str] | str | None = None,
        sigma: float = 0.03,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        if body_names is None:
            body_names = self.command_manager.tracking_body_names

        self.sigma = sigma
        body_indices_tracking, matched_body_names = _select_tracking_body_names(
            self.command_manager,
            body_names,
        )
        self.body_indices_tracking = list(body_indices_tracking)
        self.body_names = list(matched_body_names)
        self.num_bodies = len(self.body_names)

    def _compute(self):
        raise NotImplementedError


class body_pos_exp(_tracking_body, namespace="mimic_lite"):
    def _compute(self):
        error = self.command_manager.body_pos_error[:, self.body_indices_tracking]
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class body_pos_local_exp(_tracking_body, namespace="mimic_lite"):
    def _compute(self):
        error = self.command_manager.body_pos_error_local[:, self.body_indices_tracking]
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class body_pos_error(_tracking_body, namespace="mimic_lite"):
    def _compute(self):
        error = self.command_manager.body_pos_error[:, self.body_indices_tracking]
        return error.mean(dim=1).unsqueeze(1)


class body_pos_error_local(_tracking_body, namespace="mimic_lite"):
    def _compute(self):
        error = self.command_manager.body_pos_error_local[:, self.body_indices_tracking]
        return error.mean(dim=1).unsqueeze(1)


class body_ori_exp(_tracking_body, namespace="mimic_lite"):
    def _compute(self):
        error = self.command_manager.body_ori_error[:, self.body_indices_tracking]
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class body_ori_local_exp(_tracking_body, namespace="mimic_lite"):
    def _compute(self):
        error = self.command_manager.body_ori_error_local[:, self.body_indices_tracking]
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class body_ori_error(_tracking_body, namespace="mimic_lite"):
    def _compute(self):
        error = self.command_manager.body_ori_error[:, self.body_indices_tracking]
        return error.mean(dim=1).unsqueeze(1)


class body_ori_error_local(_tracking_body, namespace="mimic_lite"):
    def _compute(self):
        error = self.command_manager.body_ori_error_local[:, self.body_indices_tracking]
        return error.mean(dim=1).unsqueeze(1)


class body_lin_vel_exp(_tracking_body, namespace="mimic_lite"):
    def _compute(self):
        error = self.command_manager.body_lin_vel_error[:, self.body_indices_tracking]
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class body_ang_vel_exp(_tracking_body, namespace="mimic_lite"):
    def _compute(self):
        error = self.command_manager.body_ang_vel_error[:, self.body_indices_tracking]
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class _tracking_joint(TrackReward, namespace="mimic_lite"):
    def __init__(
        self,
        env,
        joint_names: List[str] | str | None = None,
        sigma: float = 0.03,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        if joint_names is None:
            joint_names = self.command_manager.tracking_joint_names

        self.sigma = sigma
        joint_indices_tracking, matched_joint_names = _select_tracking_joint_names(
            self.command_manager,
            joint_names,
        )
        self.joint_indices_tracking = list(joint_indices_tracking)
        self.joint_names = list(matched_joint_names)

    def _compute(self):
        raise NotImplementedError

class joint_pos_tracking_product(_tracking_joint, namespace="mimic_lite"):
    def _compute(self):
        error = self.command_manager.joint_pos_error[:, self.joint_indices_tracking]
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class joint_pos_error(_tracking_joint, namespace="mimic_lite"):
    def _compute(self):
        error = self.command_manager.joint_pos_error[:, self.joint_indices_tracking]
        return error.mean(dim=1).unsqueeze(1)


class joint_vel_tracking_product(_tracking_joint, namespace="mimic_lite"):
    def _compute(self):
        error = self.command_manager.joint_vel_error[:, self.joint_indices_tracking]
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class feet_air_time_ref(TrackReward, namespace="mimic_lite"):
    def __init__(self, env, body_names: List[str] | str, thres: float, **kwargs):
        super().__init__(env, **kwargs)
        self.thres = thres
        self.asset = self.command_manager.asset
        self.contact_sensor: "ContactSensor" = self.env.scene["feet_ground_contact"]

        body_indices_tracking, matched_body_names = _select_tracking_body_names(
            self.command_manager,
            body_names,
        )
        self.body_indices_tracking = list(body_indices_tracking)
        sensor_ids, sensor_names = find_sensor_bodies(
            self.asset,
            self.contact_sensor,
            matched_body_names,
        )
        if set(sensor_names) != set(matched_body_names):
            missing = sorted(set(matched_body_names) - set(sensor_names))
            raise RuntimeError(
                f"feet_air_time_ref: missing feet in contact sensor: {missing}"
            )
        self.sensor_body_ids = torch.tensor(sensor_ids, device=self.device)

        num_bodies = len(matched_body_names)
        self.reward_time = torch.zeros(self.num_envs, num_bodies, device=self.device)
        self.last_contact = torch.zeros(
            self.num_envs, num_bodies, dtype=bool, device=self.device
        )

        self.h_low, self.h_high = 0.035, 0.12
        self.c_low, self.c_high = 0.5, 2.0
        self.exp_log_c_ratio = torch.log(
            torch.tensor(self.c_high / self.c_low, device=self.device)
        )

    def reset(self, env_ids):
        self.reward_time[env_ids] = 0.0
        self.last_contact[env_ids] = False

    def _compute(self):
        current_contact = (
            self.contact_sensor.data.current_contact_time[:, self.sensor_body_ids] > 0.0
        )
        first_contact = (~self.last_contact) & current_contact
        self.last_contact[:] = current_contact

        ref_vel = self.command_manager.ref_body_lin_vel_w[:, self.body_indices_tracking]
        ref_pos = self.command_manager.ref_body_pos_w[:, self.body_indices_tracking]
        ref_feet_standing = (ref_vel.norm(dim=-1) < 0.2) & (ref_pos[..., 2] < 0.15)

        feet_height = self.command_manager.robot_body_link_pos_w[
            :, self.body_indices_tracking, 2
        ]
        t = (feet_height - self.h_low) / (self.h_high - self.h_low)
        t = torch.clamp(t, 0.0, 1.0)
        feet_height_coef = self.c_low * torch.exp(self.exp_log_c_ratio * t)

        contact_diff = ref_feet_standing ^ current_contact
        self.reward_time = self.reward_time + torch.where(
            contact_diff, -self.env.step_dt, self.env.step_dt * feet_height_coef
        )

        reward = torch.sum(
            (self.reward_time - self.thres).clamp_max(0.0) * first_contact,
            dim=1,
            keepdim=True,
        )

        self.reward_time = self.reward_time * (~current_contact)
        return reward
