from mimic_lite.tasks.command import RobotTracking
from active_adaptation.envs.mdp.terminations.base import Termination as BaseTermination

import torch
from typing import List
try:
    from isaaclab.utils.string import resolve_matching_names
except ModuleNotFoundError:
    from mjlab.utils.lab_api.string import resolve_matching_names


class _cum_error_mixin:
    def __init__(self, env, min_steps: int = 1, threshold: float = 0.25, **kwargs):
        super().__init__(env, **kwargs)
        self.min_steps = min_steps
        self.threshold = threshold

        with torch.device(self.device):
            self.error = torch.zeros(self.num_envs)
            self.__exceeded = torch.zeros(self.num_envs, dtype=torch.bool)
            self.__cum_steps = torch.zeros(self.num_envs, dtype=torch.int32)

    def update(self):
        self.__exceeded = self.error >= self.threshold
        self.__cum_steps[self.__exceeded] += 1
        self.__cum_steps[~self.__exceeded] = 0

    def reset(self, env_ids):
        self.__cum_steps[env_ids] = 0

    def compute(self, termination: torch.Tensor):
        return (self.__cum_steps >= self.min_steps).unsqueeze(-1)


RobotTrackTermination = BaseTermination[RobotTracking]


class motion_timeout(RobotTrackTermination):
    """
    Terminates when the motion clip is consumed (or always true in replay mode).
    """

    def __init__(self, env, is_timeout: bool = True, **kwargs):
        super().__init__(env, is_timeout=is_timeout, **kwargs)

    def compute(self, termination: torch.Tensor):
        timeout = (self.command_manager.t >= self.command_manager.motion_len)
        return timeout.to(self.device).unsqueeze(1)


class cum_body_pos_error(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, env, body_names: str | List[str] = ".*", **kwargs):
        super().__init__(env, **kwargs)
        self.body_names = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )[1]
        self.body_indices_tracking = [
            self.command_manager.tracking_body_names.index(name)
            for name in self.body_names
        ]

    def update(self):
        body_pos_error = self.command_manager.body_pos_error[:, self.body_indices_tracking]
        self.error[:] = body_pos_error.max(dim=1).values
        super().update()


class cum_body_z_error(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, env, body_names: str | List[str] = ".*", **kwargs):
        super().__init__(env, **kwargs)
        self.body_names = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )[1]
        self.body_indices_tracking = [
            self.command_manager.tracking_body_names.index(name)
            for name in self.body_names
        ]

    def update(self):
        body_pos_error = (
            self.command_manager.ref_body_pos_w
            - self.command_manager.robot_body_link_pos_w
        )[:, self.body_indices_tracking, 2].abs()
        self.error[:] = body_pos_error.max(dim=1).values
        super().update()


class cum_body_ori_error(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, env, body_names: str | List[str] = ".*", **kwargs):
        super().__init__(env, **kwargs)
        self.body_names = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )[1]
        self.body_indices_tracking = [
            self.command_manager.tracking_body_names.index(name)
            for name in self.body_names
        ]

    def update(self):
        body_ori_error = self.command_manager.body_ori_error[:, self.body_indices_tracking]
        self.error[:] = body_ori_error.max(dim=1).values
        super().update()


class cum_body_lin_vel_error(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, env, body_names: str | List[str] = ".*", **kwargs):
        super().__init__(env, **kwargs)
        self.body_names = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )[1]
        self.body_indices_tracking = [
            self.command_manager.tracking_body_names.index(name)
            for name in self.body_names
        ]

    def update(self):
        body_lin_vel_error = self.command_manager.body_lin_vel_error[
            :, self.body_indices_tracking
        ]
        self.error[:] = body_lin_vel_error.max(dim=1).values
        super().update()


class cum_body_ang_vel_error(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, env, body_names: str | List[str] = ".*", **kwargs):
        super().__init__(env, **kwargs)
        self.body_names = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )[1]
        self.body_indices_tracking = [
            self.command_manager.tracking_body_names.index(name)
            for name in self.body_names
        ]

    def update(self):
        body_ang_vel_error = self.command_manager.body_ang_vel_error[
            :, self.body_indices_tracking
        ]
        self.error[:] = body_ang_vel_error.max(dim=1).values
        super().update()


class cum_body_pos_error_local(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, env, body_names: str | List[str] = ".*", **kwargs):
        super().__init__(env, **kwargs)
        self.body_names = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )[1]
        self.body_indices_tracking = [
            self.command_manager.tracking_body_names.index(name)
            for name in self.body_names
        ]

    def update(self):
        body_pos_error = self.command_manager.body_pos_error_local[:, self.body_indices_tracking]
        self.error[:] = body_pos_error.max(dim=1).values
        super().update()

class cum_body_ori_error_local(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, env, body_names: str | List[str] = ".*", **kwargs):
        super().__init__(env, **kwargs)
        self.body_names = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )[1]
        self.body_indices_tracking = [
            self.command_manager.tracking_body_names.index(name)
            for name in self.body_names
        ]

    def update(self):
        body_ori_error = self.command_manager.body_ori_error_local[:, self.body_indices_tracking]
        self.error[:] = body_ori_error.max(dim=1).values
        super().update()
        # print("body ori error local, value:", self.error)
        # print("body ori error local,  step:", self._cum_error_mixin__cum_steps)
        # print("body ori error local, max indices:", [self.body_names[i.item()] for i in body_ori_error.argmax(dim=1)])


class cum_joint_pos_error(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, env, joint_names: str | List[str] = ".*", **kwargs):
        super().__init__(env, **kwargs)
        self.joint_names = resolve_matching_names(
            joint_names, self.command_manager.tracking_joint_names
        )[1]
        self.joint_indices_tracking = [
            self.command_manager.tracking_joint_names.index(name)
            for name in self.joint_names
        ]

    def update(self):
        joint_pos_error = self.command_manager.joint_pos_error[:, self.joint_indices_tracking]
        self.error[:] = joint_pos_error.max(dim=1).values
        super().update()
