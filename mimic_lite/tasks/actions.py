from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch

from active_adaptation.envs.mdp.actions.base import Action
from active_adaptation.utils.symmetry import joint_space_symmetry

try:
    import isaaclab.utils.string as string_utils
except ModuleNotFoundError:
    from mjlab.utils.lab_api import string as string_utils


class JointPosition(Action, namespace="mimic_lite"):
    """mimic-lite-style joint position controller with delay + smoothing."""

    def __init__(
        self,
        action_scaling: float | Dict[str, float] = 0.5,
        min_delay: int = 0,
        max_delay: int = 0,
        alpha: float | Sequence[float] | None = None,
        alpha_range: Tuple[float, float] = (0.5, 1.0),
    ):
        super().__init__()
        self._action_scaling_cfg = action_scaling
        self.min_delay = int(min_delay) if min_delay is not None else 0
        self.max_delay = int(max_delay) if max_delay is not None else 0
        if alpha is not None:
            if isinstance(alpha, (float, int)):
                self.alpha_range = (float(alpha), float(alpha))
            else:
                self.alpha_range = (float(alpha[0]), float(alpha[1]))
        else:
            self.alpha_range = (float(alpha_range[0]), float(alpha_range[1]))

    def _initialize(self, env) -> None:
        super()._initialize(env)
        action_scaling = self._action_scaling_cfg
        if isinstance(action_scaling, float):
            action_scaling = {".*": float(action_scaling)}
        _, self.joint_names, scaling = string_utils.resolve_matching_names_values(
            dict(action_scaling), self.asset.cfg.joint_names_simulation
        )
        self.joint_ids = torch.tensor(
            [self.asset.joint_names.index(name) for name in self.joint_names],
            device=self.device,
        )
        self.action_scaling = torch.tensor(scaling, device=self.device)
        self.names = list(self.joint_names)
        self.default_joint_pos = self.asset.data.default_joint_pos.clone()
        delay_hist = max((self.max_delay - 1) // self.env.decimation + 1, 3)
        with torch.device(self.device):
            self.action_buf = torch.zeros(self.num_envs, delay_hist, self.action_dim)
            self.applied_action = torch.zeros(self.num_envs, self.action_dim)
            self.alpha = torch.ones(self.num_envs, 1)
            self.delay = torch.zeros(self.num_envs, 1, dtype=torch.long)
            self.offset = torch.zeros(self.num_envs, len(self.asset.joint_names))

    @property
    def action_dim(self) -> int:
        return len(self.joint_ids)

    def reset(self, env_ids: torch.Tensor):
        self.action_buf[env_ids] = 0
        self.applied_action[env_ids] = 0

        self.delay[env_ids] = torch.randint(
            self.min_delay,
            self.max_delay + 1,
            (len(env_ids), 1),
            device=self.device,
        )
        self.alpha[env_ids] = torch.empty(len(env_ids), 1, device=self.device).uniform_(
            self.alpha_range[0], self.alpha_range[1]
        )

    def process_action(self, action: torch.Tensor):
        self.action_buf = self.action_buf.roll(1, dims=1)
        self.action_buf[:, 0] = action

    def apply_action(self, substep: int):
        delay_idx = (
            self.delay - substep + self.env.decimation - 1
        ) // self.env.decimation
        delay_idx = delay_idx.clamp_(0, self.action_buf.shape[1] - 1)
        delayed_action = torch.gather(
            self.action_buf,
            1,
            delay_idx[:, :, None].expand(-1, 1, self.action_dim),
        ).squeeze(1)

        self.applied_action.lerp_(delayed_action, self.alpha)

        # pos_target = self.default_joint_pos + self.asset.data.encoder_bias
        pos_target = self.default_joint_pos + self.offset
        pos_target[:, self.joint_ids] += self.applied_action * self.action_scaling
        self.asset.set_joint_position_target(pos_target)
        # self.asset.set_joint_position_target(self.asset.data.joint_pos)

    def symmetry_transform(self):
        return joint_space_symmetry(self.asset, self.joint_names)
