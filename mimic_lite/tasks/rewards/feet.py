"""Feet-related reward aliases for mimic-lite tasks."""

import active_adaptation as aa
from mimic_lite.tasks.command import RobotTracking
from active_adaptation.envs.mdp.rewards.base import Reward as BaseReward
from active_adaptation.envs.utils import find_bodies, find_sensor_bodies
from typing import TYPE_CHECKING, cast
import torch

if aa.get_backend() == "isaaclab":
    from isaaclab.sensors import ContactSensor as IsaacContactSensor
elif aa.get_backend() == "mjlab":
    from mjlab.sensor import ContactSensor as MJLabContactSensor

if TYPE_CHECKING:
    from mjlab.viewer.viser import ViserMujocoScene

TrackReward = BaseReward[RobotTracking]


def _select_tracking_body_names(
    command_manager: RobotTracking,
    body_names: str | list[str],
) -> list[str]:
    available_body_names = list(command_manager.tracking_body_names)
    _, matched_body_names = find_bodies(command_manager.asset, body_names)
    matched_name_set = set(matched_body_names)
    selected_body_names = [
        body_name for body_name in available_body_names if body_name in matched_name_set
    ]
    if not selected_body_names:
        raise RuntimeError("No matched feet in tracking bodies.")
    return selected_body_names


def _current_in_contact(contact_sensor, body_ids: torch.Tensor) -> torch.Tensor:
    """Return [N, B] in-contact mask using the configured history window."""
    if aa.get_backend() == "isaaclab":
        contact_sensor = cast(IsaacContactSensor, contact_sensor)
    elif aa.get_backend() == "mjlab":
        contact_sensor = cast(MJLabContactSensor, contact_sensor)

    data = contact_sensor.data

    # mjlab ContactSensor: [N, B, H, 3]
    force_history = getattr(data, "force_history", None)
    if force_history is not None:
        force_mag = force_history[:, body_ids].norm(dim=-1)  # [N, B, H]
        return force_mag.gt(0.0).any(dim=-1)

    # isaac ContactSensor: commonly [N, H, B, 3], but keep robust to [N, B, H, 3].
    current_contact_time = getattr(data, "current_contact_time", None)
    if current_contact_time is not None:
        return current_contact_time[:, body_ids] > 1e-6

    raise RuntimeError("Contact sensor does not expose usable contact fields.")


class feet_slip(TrackReward, namespace="mimic_lite"):
    def __init__(
        self,
        env,
        body_names: str,
        weight: float,
        tolerance: float = 0.0,
        **kwargs,
    ):
        super().__init__(env, weight=weight, **kwargs)
        self.asset = self.env.scene.articulations["robot"]
        self.contact_sensor = self.env.scene.sensors["contact_forces"]
        articulation_body_ids, self.body_names = find_bodies(self.asset, body_names)
        self.articulation_body_ids = torch.as_tensor(
            articulation_body_ids,
            device=self.device,
        )
        sensor_ids, sensor_names = find_sensor_bodies(
            self.asset,
            self.contact_sensor,
            self.body_names,
        )
        if set(sensor_names) != set(self.body_names):
            missing = sorted(set(self.body_names) - set(sensor_names))
            raise RuntimeError(
                f"feet_slip: missing feet in contact sensor: {missing}"
            )
        self.sensor_body_ids = torch.as_tensor(sensor_ids, device=self.device)
        self.tolerance = tolerance

    def _compute(self):
        in_contact_step = _current_in_contact(self.contact_sensor, self.sensor_body_ids)
        feet_vel = self.asset.data.body_com_lin_vel_w[:, self.articulation_body_ids, :2]
        feet_vel = (feet_vel.norm(dim=-1) - self.tolerance).clamp(min=0.0, max=1.0)
        slip = (in_contact_step * feet_vel).sum(dim=1, keepdim=True)
        return -slip


class feet_air_time(TrackReward, namespace="mimic_lite"):
    supported_backends = ("isaaclab", "mjlab")

    def __init__(
        self,
        env,
        body_names: str | list[str],
        thres: float,
        weight: float,
        height_range: tuple[float, float] | list[float] = (0.035, 0.155),
        time_factor_range: tuple[float, float] | list[float] = (0.2, 2.0),
        body2_names: str | list[str] | None = None,
        debug_first_contact_color: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
        debug_air_color: tuple[float, float, float, float] = (0.1, 0.4, 1.0, 1.0),
        **kwargs,
    ):
        super().__init__(env, weight=weight, **kwargs)
        self.asset = self.env.scene.articulations["robot"]
        self.contact_sensor: "IsaacContactSensor" | "MJLabContactSensor" = self.env.scene.sensors["contact_forces"]
        body_indices, matched_body_names = find_bodies(self.asset, body_names)
        self.body_indices = torch.as_tensor(
            body_indices, dtype=torch.long, device=self.device
        )
        if body2_names is None:
            self.body2_indices = self.body_indices
        else:
            body2_indices, _ = find_bodies(self.asset, body2_names)
            self.body2_indices = torch.as_tensor(
                body2_indices, dtype=torch.long, device=self.device
            )
            if len(self.body2_indices) != len(self.body_indices):
                raise ValueError(
                    "body2_names must match body_names length for feet_air_time."
                )

        sensor_ids, sensor_names = find_sensor_bodies(
            self.asset,
            self.contact_sensor,
            matched_body_names,
        )
        if set(sensor_names) != set(matched_body_names):
            missing = sorted(set(matched_body_names) - set(sensor_names))
            raise RuntimeError(
                f"feet_air_time: missing feet in contact sensor: {missing}"
            )

        self.sensor_body_ids = torch.as_tensor(sensor_ids, device=self.device)
        self.current_contact = torch.zeros(
            self.num_envs, len(self.sensor_body_ids), dtype=torch.bool, device=self.device
        )
        self.prev_contact = torch.zeros_like(self.current_contact)
        self.is_first_contact = torch.zeros_like(self.current_contact)

        self.current_air_time = torch.zeros_like(self.current_contact, dtype=torch.float32)
        self.reward_time = torch.zeros_like(self.current_air_time)
        self.air_ratio = torch.zeros_like(self.current_air_time)

        self.thres = thres
        if len(height_range) != 2:
            raise ValueError("height_range must have exactly two values for feet_air_time.")
        if len(time_factor_range) != 2:
            raise ValueError("time_factor_range must have exactly two values for feet_air_time.")
        self.air_h_low = float(height_range[0])
        self.air_h_high = float(height_range[1])
        self.air_h_span = max(self.air_h_high - self.air_h_low, 1e-6)
        self.air_time_factor_low = float(time_factor_range[0])
        self.air_time_factor_high = float(time_factor_range[1])
        self.air_time_factor_span = self.air_time_factor_high - self.air_time_factor_low

        self.debug_first_contact_color = debug_first_contact_color
        self.debug_air_color = debug_air_color
        self.debug_first_contact_size = 0.2
        self.debug_air_size = 0.2

    def reset(self, env_ids):
        self.current_contact[env_ids] = False
        self.prev_contact[env_ids] = False
        self.is_first_contact[env_ids] = False
        self.current_air_time[env_ids] = 0.0
        self.reward_time[env_ids] = 0.0
        self.air_ratio[env_ids] = 0.0

    def update(self):
        self.prev_contact[:] = self.current_contact
        self.current_contact[:] = _current_in_contact(
            self.contact_sensor, self.sensor_body_ids
        )
        self.is_first_contact[:] = (~self.prev_contact) & self.current_contact

        feet_height = torch.minimum(
            self.asset.data.body_link_pos_w[:, self.body_indices, 2],
            self.asset.data.body_link_pos_w[:, self.body2_indices, 2],
        )
        air_ratio = ((feet_height - self.air_h_low) / self.air_h_span).clamp(0.0, 1.0)
        self.air_ratio.copy_(air_ratio)
        # print(feet_height[0])
        # print(air_ratio[0])
        air_time_factor = self.air_time_factor_low + air_ratio * self.air_time_factor_span

        self.current_air_time += self.env.step_dt * air_time_factor
        self.reward_time.copy_(self.current_air_time)
        self.current_air_time.masked_fill_(self.current_contact, 0.0)

    def _compute(self):
        reward = torch.sum(
            (self.reward_time - self.thres).clamp_max(0.0) * self.is_first_contact,
            dim=1,
            keepdim=True,
        )
        reward *= ~self.command_manager.is_standing_env
        return reward

    def debug_draw(self):
        positions = self.asset.data.body_link_pos_w[:, self.body_indices].clone()
        first_contact = self.is_first_contact
        in_air = ~self.current_contact
        air_ratio = self.air_ratio

        if aa.get_backend() == "isaaclab":
            debug_draw = getattr(self.env, "debug_draw", None)
            if debug_draw is None:
                return

            positions = positions.detach().cpu()
            first_contact = first_contact.detach().cpu()
            in_air = in_air.detach().cpu()
            air_ratio = air_ratio.detach().cpu()

            first_contact_points = positions[first_contact]
            if first_contact_points.numel() > 0:
                debug_draw.point(
                    first_contact_points,
                    color=self.debug_first_contact_color,
                    size=self.debug_first_contact_size,
                )

            air_points = positions[in_air]
            air_sizes = (air_ratio[in_air] * self.debug_air_size).tolist()
            if air_points.numel() > 0 and air_sizes:
                debug_draw._draw.draw_points(
                    air_points.reshape(-1, 3).tolist(),
                    [self.debug_air_color] * len(air_sizes),
                    air_sizes,
                )
            return

        viewer = getattr(self.env.sim, "viewer", None)
        if viewer is None:
            return
        scene: "ViserMujocoScene" | None = getattr(viewer, "scene", None)
        if scene is None:
            return

        positions = positions.detach().cpu()
        first_contact = first_contact.detach().cpu()
        in_air = in_air.detach().cpu()
        air_ratio = air_ratio.detach().cpu()

        if scene.show_all_envs or self.num_envs == 1:
            env_ids = range(self.num_envs)
        else:
            env_ids = [int(scene.env_idx)]

        for env_idx in env_ids:
            for body_idx in range(len(self.body_indices)):
                if first_contact[env_idx, body_idx]:
                    # print(f"env {env_idx} body {body_idx} first contact, size {self.debug_first_contact_size}")
                    scene.add_sphere(
                        positions[env_idx, body_idx],
                        self.debug_first_contact_size,
                        self.debug_first_contact_color,
                    )
                if in_air[env_idx, body_idx]:
                    # print(f"env {env_idx} body {body_idx} in air, size {self.debug_air_size}")
                    scene.add_sphere(
                        positions[env_idx, body_idx],
                        self.debug_air_size * float(air_ratio[env_idx, body_idx]),
                        self.debug_air_color,
                    )


class feet_contact_count(TrackReward, namespace="mimic_lite"):
    supported_backends = ("isaaclab", "mjlab")

    def __init__(self, env, body_names: str, weight: float, enabled: bool = True):
        super().__init__(env, weight=weight, enabled=enabled)
        self.asset = self.env.scene.articulations["robot"]
        self.contact_sensor: "IsaacContactSensor" | "MJLabContactSensor" = self.env.scene.sensors["contact_forces"]

        body_ids, self.body_names = find_sensor_bodies(
            self.asset,
            self.contact_sensor,
            body_names,
        )
        self.body_ids = torch.as_tensor(body_ids, device=self.env.device)
        self.current_contact = torch.zeros(
            self.num_envs, len(self.body_ids), dtype=torch.bool, device=self.device
        )
        self.prev_contact = torch.zeros_like(self.current_contact)
        self.is_first_contact = torch.zeros_like(self.current_contact)

    def reset(self, env_ids):
        self.current_contact[env_ids] = False
        self.prev_contact[env_ids] = False
        self.is_first_contact[env_ids] = False

    def update(self):
        self.prev_contact[:] = self.current_contact
        self.current_contact[:] = _current_in_contact(
            self.contact_sensor, self.body_ids
        )
        self.is_first_contact[:] = (~self.prev_contact) & self.current_contact

    def _compute(self):
        return self.is_first_contact.float().mean(1, keepdim=True)


class feet_contact_duration(TrackReward, namespace="mimic_lite"):
    supported_backends = ("isaaclab", "mjlab")

    def __init__(self, env, body_names: str, weight: float, enabled: bool = True):
        super().__init__(env, weight=weight, enabled=enabled)
        self.asset = self.env.scene.articulations["robot"]
        self.contact_sensor: "IsaacContactSensor" | "MJLabContactSensor" = self.env.scene.sensors["contact_forces"]

        body_ids, self.body_names = find_sensor_bodies(
            self.asset,
            self.contact_sensor,
            body_names,
        )
        self.body_ids = torch.as_tensor(body_ids, device=self.env.device)
        self.current_contact = torch.zeros(
            self.num_envs, len(self.body_ids), dtype=torch.bool, device=self.device
        )

    def reset(self, env_ids):
        self.current_contact[env_ids] = False

    def update(self):
        self.current_contact[:] = _current_in_contact(
            self.contact_sensor, self.body_ids
        )

    def _compute(self):
        return self.current_contact.float().mean(1, keepdim=True)

