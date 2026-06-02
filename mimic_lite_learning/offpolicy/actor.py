from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase

from active_adaptation.learning.modules.common import MLP
from active_adaptation.learning.modules.vecnorm import VecNorm
from active_adaptation.learning.ppo.common import ACTION_KEY


ACTOR_INPUT_KEY = "_actor_input"


class TanhActor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        *,
        hidden_dim: int = 512,
        log_std_max: float = 0.0,
        log_std_min: float = -5.0,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        hidden_dims = [hidden_dim, hidden_dim // 2, hidden_dim // 4]
        layer_norm = "pre" if use_layer_norm else None
        self.net = MLP([input_dim, *hidden_dims], nn.SiLU, layer_norm=layer_norm)
        last_dim = hidden_dims[-1]
        self.fc_mu = nn.Linear(last_dim, action_dim)
        self.fc_logstd = nn.Linear(last_dim, action_dim)
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min

        nn.init.constant_(self.fc_mu.weight, 0.0)
        nn.init.constant_(self.fc_mu.bias, 0.0)
        nn.init.constant_(self.fc_logstd.weight, 0.0)
        nn.init.constant_(self.fc_logstd.bias, 0.0)

    def forward(self, actor_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(actor_input)
        loc = self.fc_mu(hidden)
        log_std = self.fc_logstd(hidden)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)
        scale = log_std.exp()
        return loc, scale


class WarmupUniformRolloutPolicy:
    def __init__(self, policy, actor_rollout_policy: TensorDictModuleBase) -> None:
        object.__setattr__(self, "_policy", policy)
        self.actor_rollout_policy = actor_rollout_policy

    def __call__(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self.forward(tensordict)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        policy = self._policy
        if len(policy.replay_buffer) >= policy.warmup_transition_threshold:
            return self.actor_rollout_policy(tensordict)

        action = torch.rand(
            (*tensordict.batch_size, policy.action_dim),
            device=policy.action_min.device,
            dtype=policy.action_min.dtype,
        )
        action = policy.action_min + action * (policy.action_max - policy.action_min)
        tensordict.set(ACTION_KEY, action)
        tensordict.set("loc", torch.zeros_like(action))
        return tensordict


class RolloutPolicy(TensorDictModuleBase):
    def __init__(self, policy) -> None:
        super().__init__()
        object.__setattr__(self, "_policy", policy)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        policy = self._policy
        actor_td = tensordict.copy()
        with VecNorm.freeze():
            policy.vecnorm(actor_td)
        policy.actor(actor_td)
        for key in (ACTION_KEY, "loc", "scale", f"{ACTION_KEY}_log_prob"):
            if key in actor_td.keys(True, True):
                tensordict.set(key, actor_td[key])
        return tensordict
