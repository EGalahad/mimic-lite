"""Common observation aliases for mimic-lite tasks."""

import torch
from typing import cast
import active_adaptation as aa
from active_adaptation.envs.mdp.observations.base import Observation as BaseObservation
from active_adaptation.envs.utils import find_bodies, find_joints
from active_adaptation.utils.math import quat_rotate_inverse
from mimic_lite.tasks.actions import JointPosition

if aa.get_backend() == "isaaclab":
    from isaaclab.assets import ArticulationData
elif aa.get_backend() == "mjlab":
    from mjlab.entity import EntityData


def random_noise(x: torch.Tensor, std: float):
    return x + torch.randn_like(x).clamp(-3.0, 3.0) * std


def _get_simulation_joint_selection(asset, joint_names: str, device: torch.device):
    joint_ids, joint_names = find_joints(asset, joint_names)
    return torch.as_tensor(joint_ids, device=device), joint_names


def _get_simulation_body_selection(asset, body_names: str, device: torch.device):
    body_ids, body_names = find_bodies(asset, body_names)
    return torch.as_tensor(body_ids, device=device), body_names


class root_ang_vel_history(BaseObservation, namespace="mimic_lite"):
    def __init__(self, env, noise_std: float = 0.0, history_steps: list[int] = [1]):
        super().__init__(env)
        self.asset = self.env.scene.articulations["robot"]
        self.noise_std = noise_std
        self.history_steps = history_steps
        self.buffer_size = max(history_steps) + 1
        self.history_offsets = torch.as_tensor(history_steps, device=self.device)
        self.head = 0
        self.buffer = torch.zeros((self.num_envs, self.buffer_size, 3), device=self.device)
        self.reset(torch.arange(self.num_envs, device=self.device))

    def reset(self, env_ids):
        value = self.asset.data.root_com_ang_vel_b[env_ids]
        value = value.unsqueeze(1).expand(-1, self.buffer_size, -1)
        if self.noise_std > 0:
            value = random_noise(value, self.noise_std)
        self.buffer[env_ids] = value

    def update(self):
        value = self.asset.data.root_com_ang_vel_b
        if self.noise_std > 0:
            value = random_noise(value, self.noise_std)
        self.head = (self.head - 1) % self.buffer_size
        self.buffer[:, self.head] = value

    def compute(self) -> torch.Tensor:
        indices = (self.history_offsets + self.head) % self.buffer_size
        return self.buffer[:, indices].reshape(self.num_envs, -1)


class projected_gravity_history(BaseObservation, namespace="mimic_lite"):
    def __init__(self, env, noise_std: float = 0.0, history_steps: list[int] = [1]):
        super().__init__(env)
        self.asset = self.env.scene.articulations["robot"]
        self.noise_std = noise_std
        self.history_steps = history_steps
        self.buffer_size = max(history_steps) + 1
        self.history_offsets = torch.as_tensor(history_steps, device=self.device)
        self.head = 0
        self.buffer = torch.zeros((self.num_envs, self.buffer_size, 3), device=self.device)
        self.reset(torch.arange(self.num_envs, device=self.device))

    def reset(self, env_ids):
        value = self.asset.data.projected_gravity_b[env_ids]
        value = value.unsqueeze(1).expand(-1, self.buffer_size, -1)
        if self.noise_std > 0:
            value = random_noise(value, self.noise_std)
            value = value / value.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.buffer[env_ids] = value

    def update(self):
        value = self.asset.data.projected_gravity_b
        if self.noise_std > 0:
            value = random_noise(value, self.noise_std)
            value = value / value.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.head = (self.head - 1) % self.buffer_size
        self.buffer[:, self.head] = value

    def compute(self):
        indices = (self.history_offsets + self.head) % self.buffer_size
        return self.buffer[:, indices].reshape(self.num_envs, -1)


class joint_pos_history(BaseObservation, namespace="mimic_lite"):
    def __init__(
        self,
        env,
        joint_names: str = ".*",
        history_steps: list[int] = [0],
        noise_std: float = 0.0,
    ):
        super().__init__(env)
        self.history_steps = history_steps
        self.buffer_size = max(history_steps) + 1
        self.history_offsets = torch.as_tensor(history_steps, device=self.device)
        self.head = 0
        self.noise_std = max(noise_std, 0.0)

        self.asset = self.env.scene.articulations["robot"]
        self.joint_ids, self.joint_names = _get_simulation_joint_selection(
            self.asset,
            joint_names,
            self.device,
        )

        self.num_joints = len(self.joint_ids)
        self.buffer = torch.zeros(
            (self.num_envs, self.buffer_size, self.num_joints), device=self.device
        )
        self.action_manager = cast(JointPosition, self.env.input_managers["action"])

    def reset(self, env_ids):
        value = self.asset.data.joint_pos[
            env_ids.unsqueeze(1), self.joint_ids.unsqueeze(0)
        ]
        self.buffer[env_ids] = value.unsqueeze(1)

    def update(self):
        self.head = (self.head - 1) % self.buffer_size
        value = self.asset.data.joint_pos[:, self.joint_ids]
        if self.noise_std > 0:
            value = random_noise(value, self.noise_std)
        self.buffer[:, self.head] = value

    def compute(self):
        # joint_pos = self.buffer - self.asset.data.encoder_bias[
        #     :, self.joint_ids
        # ].unsqueeze(1)
        joint_pos = self.buffer - self.action_manager.offset[
            :, self.joint_ids
        ].unsqueeze(1)
        indices = (self.history_offsets + self.head) % self.buffer_size
        joint_pos_selected = joint_pos[:, indices]
        return joint_pos_selected.reshape(self.num_envs, -1)

class joint_vel_history(BaseObservation, namespace="mimic_lite"):
    def __init__(
        self,
        env,
        joint_names: str = ".*",
        history_steps: list[int] = [0],
        noise_std: float = 0.0,
    ):
        super().__init__(env)
        self.history_steps = history_steps
        self.buffer_size = max(history_steps) + 1
        self.history_offsets = torch.as_tensor(history_steps, device=self.device)
        self.head = 0
        self.noise_std = max(noise_std, 0.0)
        self.asset = self.env.scene.articulations["robot"]
        self.joint_ids, self.joint_names = _get_simulation_joint_selection(
            self.asset,
            joint_names,
            self.device,
        )
        self.num_joints = len(self.joint_ids)
        self.buffer = torch.zeros(
            (self.num_envs, self.buffer_size, self.num_joints), device=self.device
        )
        self.action_manager = cast(JointPosition, self.env.input_managers["action"])

    def reset(self, env_ids):
        value = self.asset.data.joint_vel[
            env_ids.unsqueeze(1), self.joint_ids.unsqueeze(0)
        ]
        self.buffer[env_ids] = value.unsqueeze(1)

    def update(self):
        self.head = (self.head - 1) % self.buffer_size
        value = self.asset.data.joint_vel[:, self.joint_ids]
        if self.noise_std > 0:
            value = random_noise(value, self.noise_std)
        self.buffer[:, self.head] = value

    def compute(self):
        joint_vel = self.buffer
        indices = (self.history_offsets + self.head) % self.buffer_size
        joint_vel_selected = joint_vel[:, indices]
        return joint_vel_selected.reshape(self.num_envs, -1)

class applied_action(BaseObservation, namespace="mimic_lite"):
    def __init__(self, env):
        super().__init__(env)
        self.action_manager = cast(JointPosition, self.env.input_managers["action"])

    def compute(self):
        return self.action_manager.applied_action


class prev_actions(BaseObservation, namespace="mimic_lite"):
    def __init__(self, env, key: str = "action", steps: int = 1):
        super().__init__(env)
        self.steps = steps
        self.action_manager = cast(JointPosition, self.env.input_managers["action"])

    def compute(self):
        action_buf = self.action_manager.action_buf[:, : self.steps]
        return action_buf.reshape(self.num_envs, -1)


class body_pos_b(BaseObservation, namespace="mimic_lite"):
    def __init__(self, env, body_names: str):
        super().__init__(env)
        self.asset = self.env.scene.articulations["robot"]
        self.body_indices, self.body_names = _get_simulation_body_selection(
            self.asset,
            body_names,
            self.device,
        )
        self.update()

    def update(self):
        self.root_link_pos_w = self.asset.data.root_link_pos_w.unsqueeze(1).clone()
        self.root_link_quat_w = self.asset.data.root_link_quat_w.unsqueeze(1)
        self.root_link_pos_w[..., 2] = 0.0

        self.body_link_pos_w = self.asset.data.body_link_pos_w[:, self.body_indices]

    def compute(self):
        body_pos_b = quat_rotate_inverse(
            self.root_link_quat_w, self.body_link_pos_w - self.root_link_pos_w
        )
        return body_pos_b.reshape(self.num_envs, -1)


class body_vel_b(BaseObservation, namespace="mimic_lite"):
    def __init__(self, env, body_names: str):
        super().__init__(env)
        self.asset = self.env.scene.articulations["robot"]
        self.body_indices, self.body_names = _get_simulation_body_selection(
            self.asset,
            body_names,
            self.device,
        )
        self.update()

    def update(self):
        self.root_link_quat_w = self.asset.data.root_link_quat_w.unsqueeze(1)
        self.body_com_lin_vel_w = self.asset.data.body_com_lin_vel_w[:, self.body_indices]

    def compute(self):
        body_lin_vel_b = quat_rotate_inverse(
            self.root_link_quat_w, self.body_com_lin_vel_w
        )
        return body_lin_vel_b.reshape(self.num_envs, -1)


class applied_torque(BaseObservation, namespace="mimic_lite"):
    def __init__(self, env, joint_names: str = ".*"):
        super().__init__(env)
        self.asset = self.env.scene.articulations["robot"]
        self.joint_ids, self.joint_names = _get_simulation_joint_selection(
            self.asset,
            joint_names,
            self.device,
        )

    def compute(self):
        if aa.get_backend() == "isaaclab":
            asset_data = cast(ArticulationData, self.asset.data)
            applied_torque = asset_data.applied_torque
        else:
            asset_data = cast(EntityData, self.asset.data)
            applied_torque = asset_data.actuator_force
        return applied_torque[:, self.joint_ids]
