import torch
import torch.distributed as dist
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from typing import Sequence

import active_adaptation as aa
from active_adaptation.learning.modules.vecnorm import VecNorm
from active_adaptation.learning.ppo.common import ACTION_KEY


REF_JPOS_KEY = "ref_joint_pos_"
PRIV_TEACHER_KEY = "priv_teacher"
PRIV_STUDENT_KEY = "priv_student"
CMD_SHORT_KEY = "command_short"


class NullVecNorm(VecNorm):
    """Identity VecNorm that keeps the module/state interface intact."""

    def forward(self, input_vector: torch.Tensor):
        return input_vector

    def _update(self, input_vector: torch.Tensor):
        raise RuntimeError("NullVecNorm does not support updating statistics.")

    def _compute(self):
        raise RuntimeError("NullVecNorm does not compute normalization.")

    def synchronize(self, mode: str = "broadcast"):
        del mode
        return None


class EmpiricalNormalizer(nn.Module):
    """Running mean/std normalizer for off-policy actor and critic inputs."""

    def __init__(
        self,
        shape: int | Sequence[int],
        device: torch.device | str,
        eps: float = 1e-2,
        until: int | None = None,
    ) -> None:
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.eps = float(eps)
        self.until = until
        self.register_buffer("_mean", torch.zeros(tuple(shape), device=device).unsqueeze(0))
        self.register_buffer("_var", torch.ones(tuple(shape), device=device).unsqueeze(0))
        self.register_buffer("_std", torch.ones(tuple(shape), device=device).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long, device=device))

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        *,
        center: bool = True,
        update: bool = True,
    ) -> torch.Tensor:
        if x.shape[1:] != self._mean.shape[1:]:
            raise ValueError(
                f"Expected input of shape (*,{self._mean.shape[1:]}), got {tuple(x.shape)}"
            )
        if self.training and update:
            self.update(x)
        if center:
            return (x - self._mean) / (self._std + self.eps)
        return x / (self._std + self.eps)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if self.until is not None and int(self.count.item()) >= self.until:
            return

        if aa.is_distributed() and dist.is_available() and dist.is_initialized():
            local_batch = x.shape[0]
            global_batch = dist.get_world_size() * local_batch

            x_shifted = x - self._mean
            local_sum_shifted = torch.sum(x_shifted, dim=0, keepdim=True)
            local_sum_sq_shifted = torch.sum(x_shifted.pow(2), dim=0, keepdim=True)

            stats = torch.cat([local_sum_shifted, local_sum_sq_shifted], dim=0)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            global_sum_shifted, global_sum_sq_shifted = stats

            batch_mean_shifted = global_sum_shifted / global_batch
            batch_var = global_sum_sq_shifted / global_batch - batch_mean_shifted.pow(2)
            batch_mean = batch_mean_shifted + self._mean
        else:
            global_batch = x.shape[0]
            batch_mean = torch.mean(x, dim=0, keepdim=True)
            batch_var = torch.var(x, dim=0, keepdim=True, unbiased=False)

        new_count = self.count + global_batch
        delta = batch_mean - self._mean
        self._mean.copy_(self._mean + delta * (global_batch / new_count))

        delta2 = batch_mean - self._mean
        m_a = self._var * self.count
        m_b = batch_var * global_batch
        m2 = m_a + m_b + delta2.pow(2) * (self.count * global_batch / new_count)
        self._var.copy_(m2 / new_count)
        self._std.copy_(self._var.sqrt())
        self.count.copy_(new_count)


def check_vecnorm_divergence(vecnorm: VecNorm):
    world_size = aa.get_world_size()

    loc, scale = vecnorm._compute()
    gather_loc = [torch.empty_like(loc) for _ in range(world_size)]
    gather_scale = [torch.empty_like(scale) for _ in range(world_size)]
    dist.all_gather(gather_loc, loc)
    dist.all_gather(gather_scale, scale)

    loc_diffs = []
    scale_diffs = []
    for i in range(world_size):
        loc_diff = torch.abs(gather_loc[i] - loc).sum().item()
        scale_diff = torch.abs(gather_scale[i] - scale).sum().item()
        loc_diffs.append(loc_diff)
        scale_diffs.append(scale_diff)
    return loc_diffs, scale_diffs


class MeanAction(TensorDictModuleBase):
    in_keys = ["loc"]
    out_keys = [ACTION_KEY]

    def forward(self, td):
        td[ACTION_KEY] = td["loc"]
        return td


class ObsOODDetector(TensorDictModuleBase):
    def __init__(self, in_keys, sigma: float = 5.0):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = [("next", f"{k}_ood_ratio") for k in in_keys] + [
            ("next", k) for k in in_keys
        ]
        self.sigma = sigma

    def forward(self, tensordict: TensorDict):
        for in_key in self.in_keys:
            obs = tensordict.get(in_key, None)
            if obs is not None:
                ood_ratio = (obs.abs() > self.sigma).float().mean(dim=-1, keepdim=True)
                tensordict.set(("next", f"{in_key}_ood_ratio"), ood_ratio)
                tensordict.set(("next", in_key), obs)
        return tensordict


class ActorROA(nn.Module):
    def __init__(
        self,
        action_dim: int,
        init_noise_scale: float = 1.0,
        load_noise_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.actor_mean = nn.LazyLinear(action_dim)
        self.actor_std = nn.Parameter(torch.ones(action_dim) * init_noise_scale)
        self.scale_mapping = nn.Identity()
        self.load_noise_scale = load_noise_scale

    def forward(self, features: torch.Tensor):
        loc = self.actor_mean(features)
        scale = torch.ones_like(loc) * self.actor_std
        scale = self.scale_mapping(scale)
        return loc, scale

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if self.load_noise_scale is not None:
            self.actor_std.data.fill_(self.load_noise_scale)
