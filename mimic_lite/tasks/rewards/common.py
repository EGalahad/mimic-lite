"""Common reward aliases for mimic-lite tasks."""

from active_adaptation.envs.mdp.rewards.base import Reward as BaseReward
from active_adaptation.envs.utils import find_joints
from typing import List, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from mjlab.sensor import ContactSensor as MJLabContactSensor
    from isaaclab.sensors import ContactSensor as IsaacContactSensor


class joint_pos_limits(BaseReward, namespace="mimic_lite"):
    def __init__(
        self,
        env,
        weight: float,
        joint_names: List[str] | str = ".*",
        soft_factor: float = 0.9,
        **kwargs,
    ):
        super().__init__(env, weight=weight, **kwargs)
        self.asset = self.env.scene.articulations["robot"]
        joint_ids, self.joint_names = find_joints(self.asset, joint_names)
        self.joint_ids = torch.as_tensor(joint_ids, device=self.device)
        jpos_limits = self.asset.data.joint_pos_limits[:, self.joint_ids]
        jpos_mean = (jpos_limits[..., 0] + jpos_limits[..., 1]) / 2
        jpos_range = jpos_limits[..., 1] - jpos_limits[..., 0]
        self.soft_limits = torch.zeros_like(jpos_limits)
        self.soft_limits[..., 0] = jpos_mean - 0.5 * jpos_range * soft_factor
        self.soft_limits[..., 1] = jpos_mean + 0.5 * jpos_range * soft_factor

    def _compute(self):
        jpos = self.asset.data.joint_pos[:, self.joint_ids]
        violation_min = (self.soft_limits[..., 0] - jpos).clamp_min(0.0)
        violation_max = (jpos - self.soft_limits[..., 1]).clamp_min(0.0)
        return -(violation_min + violation_max).sum(dim=1, keepdim=True)


# class joint_torque_limits(BaseReward, namespace="mimic_lite"):
#     def __init__(
#         self,
#         env,
#         weight: float,
#         joint_names: List[str] | str = ".*",
#         soft_factor: float = 0.9,
#         **kwargs,
#     ):
#         super().__init__(env, weight=weight, **kwargs)
#         self.asset = self.env.scene.articulations["robot"]
#         _, matched_joint_names = resolve_matching_names(
#             joint_names, self.asset.joint_names
#         )
#         self.joint_names = to_simulation_joint_order(matched_joint_names, self.asset.cfg)
#         self.joint_ids = torch.as_tensor(
#             [self.asset.joint_names.index(name) for name in self.joint_names],
#             device=self.device,
#         )
#         self.soft_limits = (
#             self.asset.data.joint_effort_limits[:, self.joint_ids] * soft_factor
#         )

#     def compute(self):
#         if hasattr(self.asset.data, "actuator_force"):
#             applied_torque = self.asset.data.actuator_force[:, self.joint_ids]
#         else:
#             applied_torque = self.asset.data.applied_torque[:, self.joint_ids]
#         violation_high = (applied_torque / self.soft_limits - 1.0).clamp_min(0.0)
#         violation_low = (-applied_torque / self.soft_limits - 1.0).clamp_min(0.0)
#         return -(violation_high + violation_low).sum(dim=1, keepdim=True)


class self_collisions(BaseReward, namespace="mimic_lite"):
    def __init__(
        self,
        env,
        weight: float,
        sensor_name: str = "self_collision",
        force_threshold: float = 10.0,
        **kwargs,
    ):
        super().__init__(env, weight=weight, **kwargs)
        self.sensor_name = sensor_name
        self.force_threshold = force_threshold
        self.contact_sensor: "MJLabContactSensor" | "IsaacContactSensor" = self.env.scene.sensors[sensor_name]

    def _compute(self):
        data = self.contact_sensor.data
        if hasattr(data, "force_history") and data.force_history is not None:
            # force_history: [B, N, H, 3]
            force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
            hit = (force_mag > self.force_threshold).any(dim=1).any(dim=-1)  # [B]
            return hit.unsqueeze(1).float()  # [B, 1]

        if hasattr(data, "net_forces_w") and data.net_forces_w is not None:
            # net_forces_w: [B, N, 3]
            force_mag = torch.norm(data.net_forces_w, dim=-1)  # [B, N]
            hit = (force_mag > self.force_threshold).any(dim=1)  # [B]
            return hit.unsqueeze(1).float()  # [B, 1]

        raise ValueError(
            f"Contact sensor {self.sensor_name} does not have force data to compute self-collision reward."
        )
