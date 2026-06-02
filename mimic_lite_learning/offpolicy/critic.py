from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase

from active_adaptation.learning.modules.common import MLP


CRITIC_INPUT_KEY = "_critic_input"
Q_LOGITS_KEY = "_q_logits"


def _column_like(
    value: torch.Tensor | float,
    reference: torch.Tensor,
) -> torch.Tensor:
    if torch.is_tensor(value):
        value = value.to(device=reference.device, dtype=reference.dtype)
        if value.ndim == 0:
            value = value.expand_as(reference)
    else:
        value = torch.full_like(reference, float(value))
    return value.reshape(-1, 1)


def project_distributional_q(
    q_logits: torch.Tensor,
    rewards: torch.Tensor,
    bootstrap: torch.Tensor,
    discount: torch.Tensor | float,
    q_support: torch.Tensor,
) -> torch.Tensor:
    q_support = q_support.to(device=q_logits.device, dtype=q_logits.dtype)
    num_atoms = q_support.shape[0]
    v_min = q_support[0]
    v_max = q_support[-1]
    delta_z = (v_max - v_min) / (num_atoms - 1)
    batch_size = rewards.shape[0]

    rewards = rewards.to(device=q_logits.device, dtype=q_logits.dtype).reshape(-1)
    bootstrap = _column_like(bootstrap, rewards)
    discount = _column_like(discount, rewards)
    target_z = rewards.unsqueeze(1) + bootstrap * discount * q_support
    target_z = target_z.clamp(v_min.item(), v_max.item())
    b = (target_z - v_min) / delta_z
    lower = torch.floor(b).long()
    upper = torch.ceil(b).long()

    is_integer = upper == lower
    lower_mask = torch.logical_and((lower > 0), is_integer)
    upper_mask = torch.logical_and((lower == 0), is_integer)
    lower = torch.where(lower_mask, lower - 1, lower)
    upper = torch.where(upper_mask, upper + 1, upper)

    offset = (
        torch.arange(batch_size, device=q_logits.device)
        .mul(num_atoms)
        .unsqueeze(1)
        .expand(batch_size, num_atoms)
        .long()
    )
    max_index = batch_size * num_atoms - 1
    lower_indices = torch.clamp((lower + offset).reshape(-1), 0, max_index)
    upper_indices = torch.clamp((upper + offset).reshape(-1), 0, max_index)
    lower_weight = upper.to(dtype=q_logits.dtype) - b
    upper_weight = b - lower.to(dtype=q_logits.dtype)

    projections = []
    for next_logits in q_logits.unbind(dim=1):
        next_dist = F.softmax(next_logits, dim=-1)
        proj_dist = torch.zeros_like(next_dist)
        flat_proj = proj_dist.reshape(-1)
        flat_proj.index_add_(
            0,
            lower_indices,
            (next_dist * lower_weight).reshape(-1),
        )
        flat_proj.index_add_(
            0,
            upper_indices,
            (next_dist * upper_weight).reshape(-1),
        )
        projections.append(proj_dist)
    return torch.stack(projections, dim=1)


def distributional_q_value(
    probs: torch.Tensor,
    q_support: torch.Tensor,
) -> torch.Tensor:
    q_support = q_support.to(device=probs.device, dtype=probs.dtype)
    return torch.sum(probs * q_support, dim=-1)


class DistributionalCritic(TensorDictModuleBase):
    def __init__(
        self,
        *,
        input_dim: int,
        num_atoms: int = 101,
        hidden_dim: int = 768,
        use_layer_norm: bool = True,
        num_q_networks: int = 2,
    ) -> None:
        super().__init__()
        self.in_keys = [CRITIC_INPUT_KEY]
        self.out_keys = [Q_LOGITS_KEY]
        self.num_atoms = num_atoms

        hidden_dims = [hidden_dim, hidden_dim // 2, hidden_dim // 4]
        layer_norm = "pre" if use_layer_norm else None
        self.qnets = nn.ModuleList(
            [
                nn.Sequential(
                    MLP([input_dim, *hidden_dims], nn.SiLU, layer_norm=layer_norm),
                    nn.Linear(hidden_dims[-1], num_atoms),
                )
                for _ in range(num_q_networks)
            ]
        )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        critic_input = tensordict[CRITIC_INPUT_KEY]
        outputs = [qnet(critic_input) for qnet in self.qnets]
        tensordict.set(Q_LOGITS_KEY, torch.stack(outputs, dim=1))
        return tensordict
