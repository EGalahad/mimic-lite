from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping, Tuple, Union
import os
import warnings

import torch
import torch.distributions as D
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils._pytree as pytree
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
    set_composite_lp_aggregate,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torchrl.data import Composite as CompositeSpec, TensorSpec
from torchrl.envs.transforms import TensorDictPrimer
from torchrl.modules import ProbabilisticActor

import active_adaptation as aa
from active_adaptation.learning.modules.distributions import IndependentNormal
from active_adaptation.learning.modules.vecnorm import VecNorm
from active_adaptation.learning.ppo.common import (
    ACTION_KEY,
    CMD_KEY,
    DONE_KEY,
    OBS_KEY,
    OBS_PRIV_KEY,
    REWARD_KEY,
    TERM_KEY,
    CatTensors,
    GAE,
    make_batch,
    make_mlp,
)
from active_adaptation.learning.ppo.ppo_base import PPOBase
from active_adaptation.learning.utils.opt import MuonAdamWWrapper
from active_adaptation.learning.utils.valuenorm import ValueNorm1, ValueNormFake
from active_adaptation.utils.profiling import ScopedTimer

from .common import (
    ActorROA,
    MeanAction,
    NullVecNorm,
    ObsOODDetector,
    check_vecnorm_divergence,
)


torch.set_float32_matmul_precision("high")

PROFILE_SYNC_TIMERS = os.environ.get("AA_PROFILE_SYNC_TIMERS", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


class ResidualActorFeatureGate(nn.Module):
    def __init__(self, alpha_init: float) -> None:
        super().__init__()
        self.alpha_raw = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_raw)

    def forward(
        self,
        base_feature: torch.Tensor,
        residual_feature: torch.Tensor,
    ) -> torch.Tensor:
        alpha = self.alpha.to(dtype=base_feature.dtype)
        return base_feature + alpha * residual_feature


class RolloutActorAutocast(nn.Module):
    def __init__(
        self,
        actor: ProbabilisticActor,
        *,
        dist_cls: type[D.Distribution],
        dist_keys: tuple[str, ...],
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.dist_cls = dist_cls
        self.dist_keys = dist_keys
        self.dtype = dtype
        self.in_keys = actor.in_keys
        self.out_keys = [f"{ACTION_KEY}_log_prob", ACTION_KEY] + list(dist_keys)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        actor = self.actor.module if isinstance(self.actor, DDP) else self.actor
        actor_module = (
            actor.module[0] if isinstance(actor.module, nn.ModuleList) else actor.module
        )
        device_type = next(actor_module.parameters()).device.type

        with torch.no_grad(), torch.autocast(
            device_type=device_type,
            dtype=self.dtype,
            enabled=device_type == "cuda",
        ):
            actor_module(tensordict)

        for key in self.dist_keys:
            tensordict.set(key, tensordict[key].float())

        dist_now = self.dist_cls(**tensordict.select(*self.dist_keys))
        action = dist_now.sample()
        with set_composite_lp_aggregate(True):
            log_prob = dist_now.log_prob(action)
        tensordict.set(ACTION_KEY, action.float())
        tensordict.set(f"{ACTION_KEY}_log_prob", log_prob.float())
        return tensordict


@dataclass
class PPOConfig:
    _target_: str = f"{__package__}.ppo.PPOPolicy"
    name: str = "mimic_lite_ppo"
    train_every: int = 32
    ppo_epochs: int = 3
    num_minibatches: int = 8
    clip_param: float = 0.2
    gamma: float = 0.99
    lmbda: float = 0.95

    lr: float = 3e-4
    desired_kl: float | None = 0.01
    opt: str = "muon"
    compile: bool = False

    entropy_coef_start: float = 0.004
    entropy_coef_end: float = 0.004
    entropy_decay_start: int = 0
    entropy_decay_end: int = 0
    init_noise_scale: float = 1.0
    load_noise_scale: float | None = None

    clip_neg_reward: bool = False
    normalize_before_sum: bool = False

    layer_norm: Union[str, None] = "before"
    value_norm: bool = False

    actor_hidden_dims: Tuple[int, ...] = (512, 512, 512)
    res_actor_hidden_dims: Tuple[int, ...] = ()
    actor_residual_alpha_init: float = -4.0
    critic_hidden_dims: Tuple[int, ...] = (1024, 512, 256)
    max_grad_norm: float = 1.0
    in_keys: Tuple[str, ...] = (CMD_KEY, OBS_KEY, OBS_PRIV_KEY)


    vecnorm: bool = True
    freeze_vecnorm: bool = False

    grad_sync_mode: str | None = "manual"
    manual_construct_dist_now: bool = True
    value_chunk_size: int | None = 65536
    rollout_amp_dtype: str | None = None
    
    def __post_init__(self):
        if isinstance(self.opt, str):
            self.opt = self.opt.lower()

        if self.opt not in {"adam", "adamw", "muon"}:
            raise ValueError(
                "opt must be one of {'adam', 'adamw', 'muon'}, " f"got {self.opt!r}"
            )

        if isinstance(self.grad_sync_mode, str):
            self.grad_sync_mode = self.grad_sync_mode.lower()
            if self.grad_sync_mode in {"none", "null"}:
                self.grad_sync_mode = None

        if self.grad_sync_mode not in {"ddp", "manual", None}:
            raise ValueError(
                "grad_sync_mode must be one of {'ddp', 'manual', None}, "
                f"got {self.grad_sync_mode!r}"
            )

        if not self.actor_hidden_dims:
            raise ValueError(
                "actor_hidden_dims must be non-empty."
            )

        if self.value_chunk_size is not None:
            self.value_chunk_size = int(self.value_chunk_size)
            if self.value_chunk_size <= 0:
                self.value_chunk_size = None

        if isinstance(self.rollout_amp_dtype, str):
            self.rollout_amp_dtype = self.rollout_amp_dtype.lower()
            if self.rollout_amp_dtype in {"none", "null", "false", "0"}:
                self.rollout_amp_dtype = None

        if self.rollout_amp_dtype not in {
            "bf16",
            "bfloat16",
            "fp16",
            "float16",
            None,
        }:
            raise ValueError(
                "rollout_amp_dtype must be one of {'bf16', 'bfloat16', "
                "'fp16', 'float16', None}, "
                f"got {self.rollout_amp_dtype!r}"
            )


cs = ConfigStore.instance()
cs.store("mimic_lite_ppo", node=PPOConfig, group="algo")


class PPOPolicy(PPOBase):
    def __init__(
        self,
        cfg: PPOConfig,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device,
        env,
    ):
        super().__init__()
        self.cfg = PPOConfig(**cfg)
        self.device = device
        self.observation_spec = observation_spec

        self.desired_kl = self.cfg.desired_kl
        self.clip_param = self.cfg.clip_param
        self.gae = GAE(gamma=self.cfg.gamma, lmbda=self.cfg.lmbda)

        self.reward_groups = []
        for group_name, group_cfg in env.cfg.reward.items():
            if group_cfg.get("_enabled_", True):
                self.reward_groups.append(group_name)
        num_reward_groups = len(self.reward_groups)
        self.reward_scales = torch.ones(num_reward_groups, device=self.device)
        self.reward_scales /= self.reward_scales.sum()
        value_norm_cls = ValueNorm1 if self.cfg.value_norm else ValueNormFake
        self.value_norm = value_norm_cls(input_shape=num_reward_groups).to(self.device)

        object.__setattr__(self, "env", env)

        self.action_dim = env.action_manager.action_dim
        self.joint_names = env.action_manager.joint_names

        self._build_vecnorm_modules(observation_spec)

        observation_keys = set(observation_spec.keys(True, True))
        missing_keys = sorted(
            set(self.cfg.in_keys).difference(observation_keys)
        )
        if missing_keys:
            raise KeyError(f"Missing required observation keys: {missing_keys}")

        actor_in_keys = [OBS_KEY, CMD_KEY]
        critic_in_keys = [OBS_PRIV_KEY, OBS_KEY, CMD_KEY]

        self.actor = self._build_actor(actor_in_keys)
        self.critic = Seq(
            CatTensors(critic_in_keys, "_critic_input", del_keys=False, sort=False),
            Mod(
                nn.Sequential(
                    make_mlp(
                        list(self.cfg.critic_hidden_dims),
                        norm=self.cfg.layer_norm,
                    ),
                    nn.LazyLinear(num_reward_groups),
                ),
                ["_critic_input"],
                ["state_value"],
            ),
            selected_out_keys=["state_value"],
        ).to(self.device)

        fake_input = observation_spec.zero()
        with VecNorm.freeze():
            self.vecnorm(fake_input)
        self.actor(fake_input)
        self.critic(fake_input)

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.0)

        self.apply(init_)

        if aa.is_distributed():
            self.world_size = aa.get_world_size()
            if self.cfg.grad_sync_mode == "ddp":
                self._wrap_ddp(local_rank=aa.get_local_rank())
            self._broadcast_parameters()

        self.lr_policy = self.cfg.lr
        self.opt_policy = self._make_optimizer([self.actor], lr=self.lr_policy)
        self.opt_critic = self._make_optimizer([self.critic], lr=self.cfg.lr)

        self.update_ppo = self._update_ppo
        if self.cfg.compile and not aa.is_distributed():
            self.update_ppo = torch.compile(self._update_ppo)

        self.num_updates = 0

    def _make_optimizer(
        self, modules: list[nn.Module], *, lr: float
    ) -> torch.optim.Optimizer:
        if self.cfg.opt == "muon":
            return MuonAdamWWrapper(modules, lr=lr, weight_decay=0.01)
        elif self.cfg.opt == "adam":
            return torch.optim.Adam(
                [param for module in modules for param in module.parameters()], lr=lr
            )
        elif self.cfg.opt == "adamw":
            return torch.optim.AdamW(
                [param for module in modules for param in module.parameters()],
                lr=lr,
                weight_decay=0.01,
            )

    def _build_actor(self, in_keys: list[str]) -> ProbabilisticActor:
        actor_modules = [CatTensors(in_keys, "_actor_input", del_keys=False, sort=False)]

        if self.cfg.res_actor_hidden_dims:
            actor_feature_dim = self.cfg.actor_hidden_dims[-1]
            self.actor_residual_gate = ResidualActorFeatureGate(
                self.cfg.actor_residual_alpha_init
            )
            actor_modules.extend(
                [
                    Mod(
                        make_mlp(
                            list(self.cfg.actor_hidden_dims),
                            norm=self.cfg.layer_norm,
                        ),
                        ["_actor_input"],
                        ["_actor_feature_base"],
                    ),
                    Mod(
                        nn.Sequential(
                            make_mlp(
                                list(self.cfg.res_actor_hidden_dims),
                                norm=self.cfg.layer_norm,
                            ),
                            nn.LazyLinear(actor_feature_dim),
                        ),
                        ["_actor_input"],
                        ["_actor_feature_residual"],
                    ),
                    Mod(
                        self.actor_residual_gate,
                        ["_actor_feature_base", "_actor_feature_residual"],
                        ["_actor_feature"],
                    ),
                ]
            )
        else:
            self.actor_residual_gate = None
            actor_modules.append(
                Mod(
                    make_mlp(
                        list(self.cfg.actor_hidden_dims),
                        norm=self.cfg.layer_norm,
                    ),
                    ["_actor_input"],
                    ["_actor_feature"],
                ),
            )

        actor_modules.append(
            Mod(
                ActorROA(
                    self.action_dim,
                    init_noise_scale=self.cfg.init_noise_scale,
                    load_noise_scale=self.cfg.load_noise_scale,
                ),
                ["_actor_feature"],
                ["loc", "scale"],
            ),
        )
        self.dist_cls = IndependentNormal
        self.dist_keys = ["loc", "scale"]
        actor = ProbabilisticActor(
            module=Seq(*actor_modules),
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True,
        ).to(self.device)
        actor.actor_residual_gate = self.actor_residual_gate
        return actor

    def _build_vecnorm_modules(self, observation_spec: CompositeSpec):
        modules = []
        self.vecnorms: Mapping[str, VecNorm] = nn.ModuleDict()
        vecnorm_cls = NullVecNorm if self.cfg.vecnorm is None else VecNorm

        for key in self.cfg.in_keys:
            if key not in observation_spec.keys(True, True):
                continue
            shape = observation_spec[key].shape[-1:]
            vecnorm = vecnorm_cls(input_shape=shape, stats_shape=shape, decay=0.9999)
            self.vecnorms[key] = vecnorm
            modules.append(Mod(vecnorm, [key], [key]))

        self.vecnorm = Seq(*modules).to(self.device)

    @VecNorm.freeze()
    def compute_value(self, tensordict):
        self.vecnorm(tensordict)
        return self.critic(tensordict)

    @torch.no_grad()
    def _critic_values_chunked(self, tensordict: TensorDict) -> torch.Tensor:
        tensordict_flat = tensordict.view(-1)
        numel = tensordict_flat.numel()
        chunk_size = self.cfg.value_chunk_size

        if chunk_size is None or numel <= chunk_size:
            values = self.critic(tensordict_flat)["state_value"]
            return values.view(*tensordict.batch_size, *values.shape[1:])

        values_flat = None
        for start in range(0, numel, chunk_size):
            end = min(start + chunk_size, numel)
            chunk = tensordict_flat[start:end]
            chunk_values = self.critic(chunk)["state_value"]
            if values_flat is None:
                values_flat = chunk_values.new_empty(
                    (numel, *chunk_values.shape[1:])
                )
            values_flat[start:end].copy_(chunk_values)

        assert values_flat is not None
        return values_flat.view(*tensordict.batch_size, *values_flat.shape[1:])

    @VecNorm.freeze()
    @torch.no_grad()
    def compute_rollout_values(self, tensordict: TensorDict, carry: TensorDict):
        values = self._critic_values_chunked(tensordict)
        last_value = self.compute_value(carry)["state_value"]

        next_values = torch.empty_like(values)
        next_values[:, :-1].copy_(values[:, 1:])
        next_values[:, -1].copy_(last_value)

        tensordict.set("state_value", values)
        tensordict.set(
            ("next", "state_value"),
            torch.where(tensordict["next", "done"], values, next_values),
        )
        return tensordict

    def _wrap_ddp(self, local_rank: int):
        ddp_kwargs = dict(
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
            find_unused_parameters=False,
        )

        class DDPWithAttr(DDP):
            def __getattr__(self, name: str):
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    if self.module is not None and hasattr(self.module, name):
                        return getattr(self.module, name)
                    raise

        def wrap_td_module(module):
            return DDPWithAttr(module, **ddp_kwargs)

        self.actor = wrap_td_module(self.actor)
        self.critic = wrap_td_module(self.critic)

    @torch.no_grad()
    def _broadcast_parameters(self):
        for module in (self.actor, self.critic):
            for param in module.parameters():
                dist.broadcast(param, src=0)

    @torch.no_grad()
    def _all_reduce_grads(self, *modules):
        for module in modules:
            for param in module.parameters():
                if param.grad is None:
                    continue
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)

    def get_next_saved_keys(self):
        return ()

    def make_tensordict_primer(self):
        return TensorDictPrimer({}, reset_key="done", expand_specs=False)

    def _get_current_iter(self) -> int:
        return int(getattr(self.env, "current_iter", 0))

    def get_rollout_policy(self, mode: str = "train", critic: bool = False):
        if mode == "deploy":
            in_keys = set(self.actor.in_keys)
            in_keys = [key for key in in_keys if key in self.vecnorms]
            vecnorm = Seq(
                *(Mod(self.vecnorms[key], [key], [key]) for key in in_keys)
            ).to(self.device)
            ood_detector = ObsOODDetector(list(in_keys), sigma=5.0)
            modules = [vecnorm, ood_detector, self.actor]
            modules[-1] = modules[-1].module[0]
            modules.append(MeanAction())
            out_keys = [ACTION_KEY]
        else:
            actor = self.actor
            if self.cfg.rollout_amp_dtype is not None:
                dtype = (
                    torch.bfloat16
                    if self.cfg.rollout_amp_dtype in {"bf16", "bfloat16"}
                    else torch.float16
                )
                actor = RolloutActorAutocast(
                    self.actor,
                    dist_cls=self.dist_cls,
                    dist_keys=tuple(self.dist_keys),
                    dtype=dtype,
                )
            modules = [self.vecnorm, actor]
            out_keys = [f"{ACTION_KEY}_log_prob", ACTION_KEY] + self.dist_keys

        rollout_policy = Seq(*modules, selected_out_keys=out_keys)
        if self.cfg.freeze_vecnorm:
            rollout_policy.forward = VecNorm.freeze()(rollout_policy.forward)
        if self.cfg.compile and not aa.is_distributed() and mode != "deploy":
            rollout_policy = torch.compile(rollout_policy)
        return rollout_policy

    @VecNorm.freeze()
    def train_op(self, tensordict: TensorDict):
        with ScopedTimer("training.exclude_stats", sync=False):
            tensordict = tensordict.exclude("stats")

        with ScopedTimer("training.policy", sync=False):
            info = self.train_policy(tensordict.copy())

        self.num_updates += 1

        if aa.is_distributed():
            with ScopedTimer("training.distributed_sync", sync=PROFILE_SYNC_TIMERS):
                for module in [self.value_norm]:
                    for param in module.parameters():
                        dist.all_reduce(param, op=dist.ReduceOp.AVG)
                    for buffer in module.buffers():
                        dist.all_reduce(buffer, op=dist.ReduceOp.AVG)

                if self.cfg.vecnorm is not None:
                    for name, vecnorm in self.vecnorms.items():
                        loc_diffs, scale_diffs = check_vecnorm_divergence(vecnorm)
                        if aa.is_main_process():
                            info[f"vecnorm/{name}/loc_diff_max"] = max(loc_diffs)
                            info[f"vecnorm/{name}/scale_diff_max"] = max(scale_diffs)
                            info[f"vecnorm/{name}/loc_diff_mean"] = sum(
                                loc_diffs
                            ) / len(loc_diffs)
                            info[f"vecnorm/{name}/scale_diff_mean"] = sum(
                                scale_diffs
                            ) / len(scale_diffs)
                        vecnorm.synchronize(mode="broadcast")

        with ScopedTimer("training.post_metrics", sync=False):
            action_std = self._get_actor_std(self.actor)
            if action_std is not None:
                for joint_name, std in zip(self.joint_names, action_std):
                    info[f"actor_std/{joint_name}"] = std
                info["actor_std/mean"] = action_std.mean()

            residual_alpha = self._get_actor_residual_alpha(self.actor)
            if residual_alpha is not None:
                info["actor/alpha"] = residual_alpha

        return info

    def _get_actor_std(self, actor_module):
        module = actor_module.module if isinstance(actor_module, DDP) else actor_module
        for _, param in module.named_parameters():
            if param.ndim == 1 and param.shape[0] == self.action_dim:
                return param.detach()
        return None

    def _get_actor_residual_alpha(self, actor_module):
        module = actor_module.module if isinstance(actor_module, DDP) else actor_module
        gate = getattr(module, "actor_residual_gate", None)
        if gate is None:
            return None
        return gate.alpha.detach().item()

    def train_policy(self, tensordict: TensorDict):
        infos = []
        with ScopedTimer("training.policy.compute_advantage", sync=False):
            self._compute_advantage(
                tensordict, self.critic, "adv", "ret", update_value_norm=True
            )

        with ScopedTimer("training.policy.entropy_schedule", sync=False):
            current_iter = self._get_current_iter()
            if current_iter <= self.cfg.entropy_decay_start:
                self.entropy_coef = self.cfg.entropy_coef_start
            elif current_iter >= self.cfg.entropy_decay_end:
                self.entropy_coef = self.cfg.entropy_coef_end
            elif self.cfg.entropy_decay_end > self.cfg.entropy_decay_start:
                progress = float(
                    (current_iter - self.cfg.entropy_decay_start)
                    / (self.cfg.entropy_decay_end - self.cfg.entropy_decay_start)
                )
                self.entropy_coef = (
                    self.cfg.entropy_coef_start
                    + (self.cfg.entropy_coef_end - self.cfg.entropy_coef_start)
                    * progress
                )
            else:
                self.entropy_coef = self.cfg.entropy_coef_end

        with ScopedTimer("training.policy.minibatch_loop", sync=False):
            for _ in range(self.cfg.ppo_epochs):
                for minibatch in make_batch(tensordict, self.cfg.num_minibatches):
                    with ScopedTimer("training.policy.update_ppo", sync=False):
                        info = self.update_ppo(minibatch)
                    infos.append(info)

                    with ScopedTimer("training.policy.kl_lr_adjust", sync=False):
                        if self.desired_kl is not None:
                            kl = info["actor/kl"]
                            if aa.is_distributed():
                                dist.all_reduce(kl, op=dist.ReduceOp.AVG)
                            if kl > self.desired_kl * 2.0:
                                self.lr_policy = max(1e-5, self.lr_policy / 1.5)
                            elif kl < self.desired_kl / 2.0 and kl > 0.0:
                                self.lr_policy = min(1e-2, self.lr_policy * 1.5)

                        for param_group in self.opt_policy.param_groups:
                            param_group["lr"] = self.lr_policy

        with ScopedTimer("training.policy.aggregate_infos", sync=False):
            infos = pytree.tree_map(lambda *xs: sum(xs).item() / len(xs), *infos)
            infos["actor/lr"] = self.lr_policy
            infos["actor/entropy_coef"] = self.entropy_coef

            ret = tensordict["ret"]
            ret_mean = ret.mean(dim=(0, 1))
            ret_std = ret.std(dim=(0, 1))
            for i, group_name in enumerate(self.reward_groups):
                infos[f"critic/{group_name}.ret_mean"] = ret_mean[i].item()
                infos[f"critic/{group_name}.ret_std"] = ret_std[i].item()
                infos[f"critic/{group_name}.neg_rew_ratio"] = (
                    (tensordict[REWARD_KEY][:, :, i] <= 0.0).float().mean().item()
                )
        return dict(sorted(infos.items()))

    @torch.no_grad()
    def _compute_advantage(
        self,
        tensordict: TensorDict,
        critic: Mod,
        adv_key: str = "adv",
        ret_key: str = "ret",
        update_value_norm: bool = True,
    ):
        keys = tensordict.keys(True, True)
        if not ("state_value" in keys and ("next", "state_value") in keys):
            with ScopedTimer(
                "training.policy.adv.critic_forward", sync=PROFILE_SYNC_TIMERS
            ):
                if critic is self.critic:
                    tensordict.set(
                        "state_value", self._critic_values_chunked(tensordict)
                    )
                    tensordict.set(
                        ("next", "state_value"),
                        self._critic_values_chunked(tensordict["next"]),
                    )
                else:
                    with tensordict.view(-1) as tensordict_flat:
                        critic(tensordict_flat)
                        critic(tensordict_flat["next"])

        values = tensordict["state_value"]
        next_values = tensordict["next", "state_value"]

        rewards = tensordict[REWARD_KEY]
        if self.cfg.clip_neg_reward:
            rewards = rewards.clamp_min(0.0)
        discount = tensordict["next", "discount"]
        terms = tensordict[TERM_KEY]
        dones = tensordict[DONE_KEY]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        with ScopedTimer("training.policy.adv.gae", sync=PROFILE_SYNC_TIMERS):
            adv, ret = self.gae(rewards, terms, dones, values, next_values, discount)

        def _global_mean_std(x, mask):
            if aa.is_distributed():
                local_count = mask.sum()
                local_sum = (x * mask.unsqueeze(-1)).sum(dim=(0, 1))
                local_sum_sq = (x * x * mask.unsqueeze(-1)).sum(dim=(0, 1))
                expand_count = local_count.float().expand_as(local_sum)

                stats = torch.stack([local_sum, local_sum_sq, expand_count])
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                global_sum, global_sum_sq, global_count = stats
                global_count.clamp_min_(1)

                mean = global_sum / global_count
                var = (global_sum_sq / global_count) - (mean * mean)
                std = var.clamp_min(0.0).sqrt()
            else:
                local_count = mask.sum()
                local_sum = (x * mask.unsqueeze(-1)).sum(dim=(0, 1))
                local_sum_sq = (x * x * mask.unsqueeze(-1)).sum(dim=(0, 1))
                count = local_count.float().expand_as(local_sum).clamp_min_(1)

                mean = local_sum / count
                var = (local_sum_sq / count) - (mean * mean)
                std = var.clamp_min(0.0).sqrt()
            return mean, std

        mask = ~tensordict["is_init"].squeeze(-1)

        with ScopedTimer("training.policy.adv.normalize", sync=PROFILE_SYNC_TIMERS):
            if self.cfg.normalize_before_sum:
                mean, std = _global_mean_std(adv, mask)
                adv_norm = (adv - mean) / (std + 1e-5)
                adv_norm *= self.reward_scales
                adv_final = adv_norm.sum(dim=2, keepdim=True)
            else:
                adv *= self.reward_scales
                adv_sum = adv.sum(dim=2, keepdim=True)
                mean, std = _global_mean_std(adv_sum, mask)
                adv_final = (adv_sum - mean) / (std + 1e-5)

        with ScopedTimer("training.policy.adv.value_norm", sync=PROFILE_SYNC_TIMERS):
            if update_value_norm:
                self.value_norm.update(ret)
            ret = self.value_norm.normalize(ret)

        tensordict.set(adv_key, adv_final)
        tensordict.set(ret_key, ret)
        tensordict["adv_before_norm"] = adv
        return tensordict

    def _update_ppo(self, tensordict: TensorDict):
        dist_kwargs_old = tensordict.select(*self.dist_keys)

        [tensordict.pop(key) for key in self.dist_keys]
        action_old = tensordict.pop(ACTION_KEY)
        logp_old = tensordict.pop(f"{ACTION_KEY}_log_prob")
        with ScopedTimer("training.policy.ppo.actor_dist", sync=PROFILE_SYNC_TIMERS):
            if self.cfg.manual_construct_dist_now:
                actor_base = (
                    self.actor.module if isinstance(self.actor, DDP) else self.actor
                )
                actor_base(tensordict)
                dist_now = self.dist_cls(
                    loc=tensordict["loc"], scale=tensordict["scale"]
                )
            else:
                dist_now: D.Independent = self.actor.get_dist(tensordict)

            with set_composite_lp_aggregate(True):
                log_probs = dist_now.log_prob(action_old)
            entropy = dist_now.entropy().mean()

        valid = ~tensordict["is_init"].squeeze(-1)

        with ScopedTimer("training.policy.ppo.policy_loss", sync=PROFILE_SYNC_TIMERS):
            adv = tensordict["adv"]
            log_ratio = (log_probs - logp_old).unsqueeze(-1)
            ratio = torch.exp(log_ratio)
            surr1 = adv * ratio
            surr2 = adv * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
            policy_loss = -(torch.min(surr1, surr2)[valid]).mean()
            entropy_loss = -self.entropy_coef * entropy

        with ScopedTimer("training.policy.ppo.critic", sync=PROFILE_SYNC_TIMERS):
            b_returns = tensordict["ret"]
            values = self.critic(tensordict)["state_value"]
            value_loss = F.mse_loss(b_returns, values, reduction="none")
            value_loss = value_loss[valid].mean(dim=0)

        loss = policy_loss + entropy_loss + value_loss.mean()

        with ScopedTimer("training.policy.ppo.backward", sync=PROFILE_SYNC_TIMERS):
            self.opt_policy.zero_grad()
            self.opt_critic.zero_grad()
            loss.backward()
        if aa.is_distributed() and self.cfg.grad_sync_mode == "manual":
            with ScopedTimer("training.policy.ppo.grad_sync", sync=PROFILE_SYNC_TIMERS):
                self._all_reduce_grads(self.actor, self.critic)
        with ScopedTimer("training.policy.ppo.clip_grad", sync=PROFILE_SYNC_TIMERS):
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.cfg.max_grad_norm
            )
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.cfg.max_grad_norm
            )
        with ScopedTimer(
            "training.policy.ppo.optimizer_step", sync=PROFILE_SYNC_TIMERS
        ):
            self.opt_policy.step()
            self.opt_critic.step()

        with ScopedTimer("training.policy.ppo.metrics", sync=PROFILE_SYNC_TIMERS):
            with torch.no_grad():
                explained_var = 1 - value_loss / b_returns[valid].var(
                    dim=0, unbiased=False
                )
                clipfrac = ((ratio - 1.0).abs() > self.clip_param).float().mean()
                dist_old = self.dist_cls(**dist_kwargs_old)
                kl = D.kl_divergence(dist_old, dist_now).mean()

        info = {
            "actor/policy_loss": policy_loss.detach(),
            "actor/clamp_ratio": clipfrac.detach(),
            "actor/entropy": entropy.detach(),
            "actor/mean_std": tensordict["scale"].detach().mean(),
            "actor/kl": kl.detach(),
            "opt/grad_norm.actor": actor_grad_norm.detach(),
            "opt/grad_norm.critic": critic_grad_norm.detach(),
        }

        for i, group_name in enumerate(self.reward_groups):
            info[f"critic/{group_name}.explained_var"] = explained_var[i]
            info[f"critic/{group_name}.value_loss"] = value_loss[i].detach()

        return info

    def state_dict(self):
        state_dict = OrderedDict()
        for name, module in self.named_children():
            if isinstance(module, DDP):
                module = module.module
            state_dict[name] = module.state_dict()
        state_dict["last_iter"] = self._get_current_iter()
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        succeed_keys = []
        failed_keys = []
        for name, module in self.named_children():
            _state_dict = state_dict.get(name, {})
            try:
                if isinstance(module, DDP):
                    module.module.load_state_dict(_state_dict, strict=strict)
                else:
                    module.load_state_dict(_state_dict, strict=strict)
                succeed_keys.append(name)
            except Exception as exc:
                warnings.warn(f"Failed to load state dict for {name}: {str(exc)}")
                failed_keys.append(name)
        print(f"Successfully loaded {succeed_keys}.")

        start_iter = state_dict.get("last_iter", 0)
        if hasattr(self.env, "set_progress"):
            self.env.set_progress(start_iter)

        return failed_keys
