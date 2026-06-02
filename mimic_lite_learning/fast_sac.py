from __future__ import annotations

from collections import OrderedDict, deque
from contextlib import nullcontext
from copy import deepcopy
import math
import warnings
from dataclasses import dataclass, field
from typing import Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from tensordict import TensorDictBase
from tensordict.nn import (
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
)
from torchrl.data import Composite as CompositeSpec, LazyTensorStorage, TensorDictReplayBuffer, TensorSpec
from torchrl.data.replay_buffers.samplers import SliceSampler
from torchrl.envs.transforms import TensorDictPrimer
from torchrl.modules import ProbabilisticActor
from torchrl.objectives import hold_out_net

import active_adaptation as aa
from active_adaptation.learning.modules.distributions import TanhNormalWithEntropy
from active_adaptation.learning.modules.common import MLP
from active_adaptation.learning.modules.vecnorm import VecNorm
from active_adaptation.learning.ppo.common import (
    ACTION_KEY,
    CMD_KEY,
    DONE_KEY,
    GAE,
    TERM_KEY,
    OBS_KEY,
    OBS_PRIV_KEY,
    REWARD_KEY,
    CatTensors,
)
from active_adaptation.learning.ppo.ppo_base import PPOBase

from .common import NullVecNorm
from .action_bounds import (
    coerce_action_bounds_config,
    default_action_bounds,
    resolve_fast_sac_action_bounds,
)
from .offpolicy.actor import (
    ACTOR_INPUT_KEY,
    RolloutPolicy,
    TanhActor,
    WarmupUniformRolloutPolicy,
)
from .offpolicy.buffer import (
    BOOTSTRAP_KEY,
    ENV_ID_KEY,
    N_STEP_BOOTSTRAP_KEY,
    N_STEP_DISCOUNT_KEY,
    N_STEP_REWARD_KEY,
    CudaPrefetchNStepReplayBuffer,
)
from .offpolicy.critic import (
    CRITIC_INPUT_KEY,
    Q_LOGITS_KEY,
    DistributionalCritic,
    distributional_q_value,
    project_distributional_q,
)

CRITIC_OBS_KEY = "_critic_obs"
VALUE_PROBE_KEY = "_value_probe"


def _masked_mean(value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return value.mean()

    expanded_mask = mask
    while expanded_mask.ndim < value.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)
    expanded_mask = expanded_mask.expand_as(value)
    denom = expanded_mask.sum().clamp_min(1)
    return (value * expanded_mask).sum() / denom


def _masked_flat_values(value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return value.reshape(-1)

    expanded_mask = mask
    while expanded_mask.ndim < value.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)
    expanded_mask = expanded_mask.expand_as(value)
    return value[expanded_mask]


def _prefix_stats(prefix: str, value: torch.Tensor, mask: torch.Tensor | None) -> dict[str, torch.Tensor]:
    flat = _masked_flat_values(value.detach(), mask)
    if flat.numel() == 0:
        zero = torch.zeros((), device=value.device, dtype=value.dtype)
        return {
            f"{prefix}_mean": zero,
            f"{prefix}_q01": zero,
            f"{prefix}_q05": zero,
            f"{prefix}_q95": zero,
            f"{prefix}_q99": zero,
        }

    quantiles = torch.quantile(
        flat.float(),
        torch.tensor([0.01, 0.05, 0.95, 0.99], device=flat.device),
    )
    return {
        f"{prefix}_mean": flat.float().mean(),
        f"{prefix}_q01": quantiles[0],
        f"{prefix}_q05": quantiles[1],
        f"{prefix}_q95": quantiles[2],
        f"{prefix}_q99": quantiles[3],
    }


def _valid_fraction(mask: torch.Tensor | None, *, device: torch.device) -> torch.Tensor:
    if mask is None:
        return torch.ones((), device=device)
    return mask.float().mean()


def _build_mlp(
    input_dim: int | None,
    hidden_dims: list[int],
    *,
    use_layer_norm: bool = True,
) -> nn.Sequential:
    layer_norm = "pre" if use_layer_norm else None
    if input_dim is None:
        return nn.Sequential(
            nn.LazyLinear(hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            MLP(hidden_dims, nn.SiLU, layer_norm=layer_norm),
        )
    return MLP([input_dim, *hidden_dims], nn.SiLU, layer_norm=layer_norm)


@dataclass
class FastSACConfig:
    _target_: str = f"{__package__}.fast_sac.FastSAC"

    name: str = "fast_sac"
    train_every: int = 2
    # Effective replay capacity = buffer_size * num_envs.
    buffer_size: int = 256
    replay_batch_size: int = 4096
    # Effective transition warmup = warm_up_steps * num_envs.
    warm_up_steps: int = 8
    utd_ratio: int = 4
    policy_frequency: int = 2

    n_step: int = 1
    custom_replay_buffer: bool = True
    custom_replay_prefetch: int = 2

    gamma: float = 0.97
    tau: float = 0.125
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    alpha_init: float = 4e-3
    target_entropy_ratio: float | None = None # 1.0
    weight_decay: float = 1e-3

    actor_hidden_dim: int = 512
    critic_hidden_dim: int = 768
    action_space_mode: str = "holosoma"
    holosoma_action_scale: float = 0.25
    holosoma_use_actor_boundary: bool = True
    action_bounds: dict[str, list[float]] = field(
        default_factory=default_action_bounds
    )
    action_min: float | None = None
    action_max: float | None = None
    v_step: float = 0.05
    v_min: float = -5.0
    v_max: float = 15.0
    actor_q_reduce: str = "min"
    critic_q_reduce: str = "min"
    actor_update_scope: str = "first"
    n_step_entropy_mode: str = "bootstrap"
    log_std_max: float = 0.0
    log_std_min: float = -4.0
    use_layer_norm: bool = True
    max_grad_norm: float = 1.0

    vecnorm: bool = True
    freeze_vecnorm: bool = False
    enable_value_probe: bool = True
    value_probe_hidden_dim: int | None = None
    value_probe_lr: float | None = None
    value_probe_update_every: int = 32
    value_probe_trace_steps: int = 32
    value_probe_inner: int = 2
    gae_lambda: float = 0.95
    action_bound_epsilon_ratio: float = 0.02
    checkpoint_path: Union[str, None] = None
    in_keys: Tuple[str, ...] = (OBS_KEY, CMD_KEY, OBS_PRIV_KEY)
    grad_sync_mode: str | None = "manual"

    def __post_init__(self) -> None:
        self.train_every = int(self.train_every)
        if self.train_every < 1:
            raise ValueError(f"train_every must be >= 1, got {self.train_every}.")
        self.utd_ratio = int(self.utd_ratio)
        if self.utd_ratio < 1:
            raise ValueError(f"utd_ratio must be >= 1, got {self.utd_ratio}.")
        self.policy_frequency = int(self.policy_frequency)
        if self.policy_frequency < 1:
            raise ValueError(
                f"policy_frequency must be >= 1, got {self.policy_frequency}."
            )
        self.n_step = int(self.n_step)
        if self.n_step < 1:
            raise ValueError(f"n_step must be >= 1, got {self.n_step}.")
        self.custom_replay_prefetch = int(self.custom_replay_prefetch)
        if self.custom_replay_prefetch < 0:
            raise ValueError(
                "custom_replay_prefetch must be >= 0, "
                f"got {self.custom_replay_prefetch}."
            )

        self.actor_q_reduce = str(self.actor_q_reduce).lower()
        if self.actor_q_reduce not in {"min", "mean", "q0", "q1"}:
            raise ValueError(
                "actor_q_reduce must be one of {'min', 'mean', 'q0', 'q1'}, "
                f"got {self.actor_q_reduce!r}"
            )
        self.critic_q_reduce = str(self.critic_q_reduce).lower()
        if self.critic_q_reduce not in {"min", "mean", "each"}:
            raise ValueError(
                "critic_q_reduce must be one of {'min', 'mean', 'each'}, "
                f"got {self.critic_q_reduce!r}"
            )
        self.actor_update_scope = str(self.actor_update_scope).lower()
        if self.actor_update_scope not in {"first", "all"}:
            raise ValueError(
                "actor_update_scope must be one of {'first', 'all'}, "
                f"got {self.actor_update_scope!r}"
            )
        self.n_step_entropy_mode = str(self.n_step_entropy_mode).lower()
        if self.n_step_entropy_mode not in {"bootstrap", "all"}:
            raise ValueError(
                "n_step_entropy_mode must be one of {'bootstrap', 'all'}, "
                f"got {self.n_step_entropy_mode!r}"
            )
        self.value_probe_update_every = int(self.value_probe_update_every)
        if self.value_probe_update_every < 1:
            raise ValueError(
                "value_probe_update_every must be >= 1, "
                f"got {self.value_probe_update_every}."
            )
        self.value_probe_trace_steps = int(self.value_probe_trace_steps)
        if self.value_probe_trace_steps < 2:
            raise ValueError(
                "value_probe_trace_steps must be >= 2, "
                f"got {self.value_probe_trace_steps}."
            )
        self.value_probe_inner = int(self.value_probe_inner)
        if self.value_probe_inner < 1:
            raise ValueError(
                f"value_probe_inner must be >= 1, got {self.value_probe_inner}."
            )
        if self.value_probe_hidden_dim is None:
            self.value_probe_hidden_dim = self.critic_hidden_dim
        if self.value_probe_lr is None:
            self.value_probe_lr = self.critic_lr

        if isinstance(self.grad_sync_mode, str):
            self.grad_sync_mode = self.grad_sync_mode.lower()
            if self.grad_sync_mode in {"none", "null"}:
                self.grad_sync_mode = None

        if self.grad_sync_mode not in {"manual", None, "ddp"}:
            raise ValueError(
                "grad_sync_mode must be one of {'manual', None, 'ddp'}, "
                f"got {self.grad_sync_mode!r}"
            )
        self.action_space_mode = str(self.action_space_mode).lower()
        if self.action_space_mode not in {"manual", "holosoma"}:
            raise ValueError(
                "action_space_mode must be one of {'manual', 'holosoma'}, "
                f"got {self.action_space_mode!r}"
            )
        self.action_bounds = coerce_action_bounds_config(
            self.action_bounds,
            action_min=self.action_min,
            action_max=self.action_max,
        )


cs = ConfigStore.instance()
cs.store("fast_sac", node=FastSACConfig(), group="algo")


class FastSAC(PPOBase):
    def __init__(
        self,
        cfg: FastSACConfig,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device,
        env,
    ) -> None:
        super().__init__()
        self.cfg = FastSACConfig(**cfg)
        self.requires_rollout_value = False
        if aa.is_distributed() and self.cfg.grad_sync_mode == "ddp":
            raise NotImplementedError("FastSAC only supports manual gradient sync.")

        self.device = device
        self.observation_spec = observation_spec
        object.__setattr__(self, "env", env)

        observation_keys = set(observation_spec.keys(True, True))
        missing_keys = sorted({OBS_KEY, CMD_KEY, OBS_PRIV_KEY}.difference(observation_keys))
        if missing_keys:
            raise KeyError(f"Missing required observation keys: {missing_keys}")

        self.num_envs = int(getattr(env, "num_envs", observation_spec.shape[0]))
        self.action_dim = int(env.action_manager.action_dim)
        self.joint_names = env.action_manager.joint_names
        self.gradient_step = 0

        self.actor_obs_keys: Tuple[str, ...] = (OBS_KEY, CMD_KEY)
        self.critic_obs_keys: Tuple[str, ...] = (OBS_KEY, CMD_KEY, OBS_PRIV_KEY)
        self.actor_input_dim = sum(
            int(observation_spec[key].shape[-1]) for key in self.actor_obs_keys
        )
        self.critic_obs_dim = sum(
            int(observation_spec[key].shape[-1]) for key in self.critic_obs_keys
        )
        self.critic_input_dim = self.critic_obs_dim + self.action_dim
        vecnorm_cls = VecNorm if self.cfg.vecnorm else NullVecNorm
        self.vecnorms = nn.ModuleDict()
        vecnorm_modules = []
        for key in self.cfg.in_keys:
            shape = observation_spec[key].shape[-1:]
            vecnorm = vecnorm_cls(input_shape=shape, stats_shape=shape, decay=0.99999)
            self.vecnorms[key] = vecnorm
            vecnorm_modules.append(Mod(vecnorm, [key], [key]))
        self.vecnorm = Seq(*vecnorm_modules).to(self.device)

        action_min, action_max = resolve_fast_sac_action_bounds(
            self.cfg,
            self.env,
            self.joint_names,
            self.action_dim,
            self.device,
        )
        self.register_buffer("action_min", action_min.clone())
        self.register_buffer("action_max", action_max.clone())
        self.action_min: torch.Tensor
        self.action_max: torch.Tensor

        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=Seq(
                CatTensors(
                    self.actor_obs_keys,
                    ACTOR_INPUT_KEY,
                    del_keys=False,
                    sort=False,
                ),
                Mod(
                    TanhActor(
                        self.actor_input_dim,
                        self.action_dim,
                        hidden_dim=self.cfg.actor_hidden_dim,
                        log_std_max=self.cfg.log_std_max,
                        log_std_min=self.cfg.log_std_min,
                        use_layer_norm=self.cfg.use_layer_norm,
                    ),
                    [ACTOR_INPUT_KEY],
                    ["loc", "scale"],
                ),
            ),
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=TanhNormalWithEntropy,
            distribution_kwargs={
                "low": self.action_min,
                "high": self.action_max,
                "event_dims": 1,
            },
            return_log_prob=True,
        ).to(self.device)

        num_atoms = int((self.cfg.v_max - self.cfg.v_min) / self.cfg.v_step) + 1
        self.qnet = Seq(
            CatTensors(
                (*self.critic_obs_keys, ACTION_KEY),
                CRITIC_INPUT_KEY,
                del_keys=False,
                sort=False,
            ),
            DistributionalCritic(
                input_dim=self.critic_input_dim,
                num_atoms=num_atoms,
                hidden_dim=self.cfg.critic_hidden_dim,
                use_layer_norm=self.cfg.use_layer_norm,
            ),
        ).to(self.device)
        self.register_buffer(
            "q_support",
            torch.linspace(
                self.cfg.v_min,
                self.cfg.v_max,
                num_atoms,
                device=self.device,
            ),
        )
        self.q_support: torch.Tensor

        fake_input = observation_spec.zero()
        fake_critic_input = fake_input.copy()
        fake_critic_input.set(
            ACTION_KEY,
            torch.zeros(
                (*fake_input.batch_size, self.action_dim),
                device=self.device,
            ),
        )
        with torch.no_grad():
            with VecNorm.freeze():
                self.vecnorm(fake_input)
            self.actor.get_dist(fake_input)
            with VecNorm.freeze():
                self.vecnorm(fake_critic_input)
            self.qnet(fake_critic_input)

        self.qnet_target = deepcopy(self.qnet).to(self.device)
        self.qnet_target.requires_grad_(False)
        fused = str(self.device).startswith("cuda")
        self.gae = GAE(self.cfg.gamma, self.cfg.gae_lambda).to(self.device)
        self.enable_value_probe = bool(self.cfg.enable_value_probe)
        self.value_probe = None
        self.value_optimizer = None
        self.value_trace: deque[TensorDictBase] = deque(
            maxlen=self.cfg.value_probe_trace_steps
        )
        if self.enable_value_probe:
            value_probe_hidden_dim = int(self.cfg.value_probe_hidden_dim)
            value_probe_hidden_dims = [
                value_probe_hidden_dim,
                value_probe_hidden_dim // 2,
                value_probe_hidden_dim // 4,
            ]
            layer_norm = "pre" if self.cfg.use_layer_norm else None
            self.value_probe = Seq(
                CatTensors(
                    self.critic_obs_keys,
                    CRITIC_OBS_KEY,
                    del_keys=False,
                    sort=False,
                ),
                Mod(
                    nn.Sequential(
                        MLP(
                            [self.critic_obs_dim, *value_probe_hidden_dims],
                            nn.SiLU,
                            layer_norm=layer_norm,
                        ),
                        nn.Linear(value_probe_hidden_dims[-1], 1),
                    ),
                    [CRITIC_OBS_KEY],
                    [VALUE_PROBE_KEY],
                ),
            ).to(self.device)
            with torch.no_grad():
                self.value_probe(fake_input.copy())
            self.value_optimizer = torch.optim.AdamW(
                self.value_probe.parameters(),
                lr=float(self.cfg.value_probe_lr),
                weight_decay=self.cfg.weight_decay,
                fused=fused,
                betas=(0.9, 0.95),
            )

        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(self.cfg.alpha_init), device=self.device)
        )
        self.fixed_alpha = self.cfg.target_entropy_ratio is None
        self.target_entropy = (
            None
            if self.fixed_alpha
            else -float(self.action_dim) * float(self.cfg.target_entropy_ratio)
        )
        self.log_alpha.requires_grad_(not self.fixed_alpha)

        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=self.cfg.actor_lr,
            weight_decay=self.cfg.weight_decay,
            fused=fused,
            betas=(0.9, 0.95),
        )
        self.q_optimizer = torch.optim.AdamW(
            self.qnet.parameters(),
            lr=self.cfg.critic_lr,
            weight_decay=self.cfg.weight_decay,
            fused=fused,
            betas=(0.9, 0.95),
        )
        self.alpha_optimizer = None
        if not self.fixed_alpha:
            self.alpha_optimizer = torch.optim.AdamW(
                [self.log_alpha],
                lr=self.cfg.alpha_lr,
                weight_decay=0.0,
                fused=fused,
                betas=(0.9, 0.95),
            )

        self.replay_buffer_capacity_per_env = self.cfg.buffer_size
        self.replay_buffer_capacity = self.replay_buffer_capacity_per_env * self.num_envs
        warmup_transitions = self.cfg.warm_up_steps * self.num_envs
        self.warmup_transition_threshold = min(
            warmup_transitions,
            self.replay_buffer_capacity,
        )
        self.min_replay_sample_transitions = max(
            self.warmup_transition_threshold,
            self.cfg.n_step * self.num_envs,
        )
        self.use_slice_replay = self.cfg.n_step > 1
        self.use_custom_replay_buffer = bool(self.cfg.custom_replay_buffer)
        pin_replay_memory = str(self.device).startswith("cuda")
        if self.use_custom_replay_buffer:
            self.replay_buffer = CudaPrefetchNStepReplayBuffer(
                capacity_per_env=self.replay_buffer_capacity_per_env,
                num_envs=self.num_envs,
                n_step=self.cfg.n_step,
                batch_size=self.cfg.replay_batch_size,
                gamma=self.cfg.gamma,
                device=self.device,
                prefetch=self.cfg.custom_replay_prefetch,
                compact=(
                    self.cfg.actor_update_scope == "first"
                    and self.cfg.n_step_entropy_mode == "bootstrap"
                ),
            )
            self.replay_samples_on_device = True
        elif self.use_slice_replay:
            self.replay_buffer = TensorDictReplayBuffer(
                storage=LazyTensorStorage(max_size=self.replay_buffer_capacity, ndim=2),
                sampler=SliceSampler(
                    slice_len=self.cfg.n_step,
                    traj_key=ENV_ID_KEY,
                    strict_length=True,
                ),
                batch_size=self.cfg.replay_batch_size * self.cfg.n_step,
                dim_extend=0,
                pin_memory=pin_replay_memory,
                prefetch=2,
            )
            self.replay_samples_on_device = False
        else:
            self.replay_buffer = TensorDictReplayBuffer(
                storage=LazyTensorStorage(max_size=self.replay_buffer_capacity),
                batch_size=self.cfg.replay_batch_size,
                pin_memory=pin_replay_memory,
                prefetch=2,
            )
            self.replay_samples_on_device = False

        if aa.is_distributed():
            self.world_size = aa.get_world_size()
            self._broadcast_parameters()
        else:
            self.world_size = 1

    def _broadcast_parameters(self) -> None:
        with torch.no_grad():
            dist.broadcast(self.log_alpha.data, src=0)
            dist.broadcast(self.q_support, src=0)
            for module in (
                self.vecnorm,
                self.actor,
                self.qnet,
                self.qnet_target,
                self.value_probe,
            ):
                if module is None:
                    continue
                for param in module.parameters():
                    dist.broadcast(param, src=0)
                for buf in module.buffers():
                    dist.broadcast(buf, src=0)

    @torch.no_grad()
    def _all_reduce_grads(self, *modules: nn.Module) -> None:
        for module in modules:
            for param in module.parameters():
                if param.grad is None:
                    continue
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)

    @torch.no_grad()
    def _all_reduce_param_grad(self, param: nn.Parameter) -> None:
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)

    def _sample_actor(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        dist = self.actor.get_dist(tensordict)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        tensordict.set(ACTION_KEY, action)
        tensordict.set(f"{ACTION_KEY}_log_prob", log_prob)
        return tensordict

    def _reduce_actor_q_values(self, q_values: torch.Tensor) -> torch.Tensor:
        if self.cfg.actor_q_reduce == "min":
            return q_values.min(dim=1).values
        if self.cfg.actor_q_reduce == "mean":
            return q_values.mean(dim=1)
        if self.cfg.actor_q_reduce == "q0":
            return q_values[:, 0]
        if q_values.shape[1] < 2:
            raise ValueError(
                "actor_q_reduce='q1' requires at least two Q heads."
            )
        return q_values[:, 1]

    def _reduce_target_distributions(
        self,
        target_distributions: torch.Tensor,
        target_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.critic_q_reduce == "each":
            return target_distributions, target_values

        if self.cfg.critic_q_reduce == "min":
            selected_idx = target_values.argmin(dim=1)
            selected = target_distributions[
                torch.arange(target_distributions.shape[0], device=target_distributions.device),
                selected_idx,
            ]
        else:
            selected = target_distributions.mean(dim=1)

        shared_distributions = selected.unsqueeze(1).expand_as(target_distributions)
        shared_values = distributional_q_value(shared_distributions, self.q_support)
        return shared_distributions, shared_values

    def _critic_stats_values(self, q_values: torch.Tensor) -> torch.Tensor:
        if self.cfg.critic_q_reduce == "each":
            return q_values
        if self.cfg.critic_q_reduce == "min":
            return q_values.min(dim=1).values
        return q_values.mean(dim=1)

    def get_next_saved_keys(self) -> tuple[str, ...]:
        return (OBS_KEY, CMD_KEY, OBS_PRIV_KEY)

    def make_tensordict_primer(self):
        return TensorDictPrimer({}, reset_key="done", expand_specs=False)

    def on_stage_start(self, stage: str) -> None:
        del stage
        return None

    def _append_rollout_to_replay(self, tensordict: TensorDictBase) -> None:
        def append_transition(transition: TensorDictBase) -> None:
            if self.enable_value_probe:
                value_keys: list[Union[str, tuple[str, str]]] = [
                    OBS_KEY,
                    CMD_KEY,
                    OBS_PRIV_KEY,
                    REWARD_KEY,
                    DONE_KEY,
                    TERM_KEY,
                    ("next", OBS_KEY),
                    ("next", CMD_KEY),
                    ("next", OBS_PRIV_KEY),
                    "is_init",
                ]
                self.value_trace.append(
                    transition.select(*value_keys, strict=False).detach().cpu()
                )

            replay_keys: list[Union[str, tuple[str, str]]] = [
                OBS_KEY,
                CMD_KEY,
                OBS_PRIV_KEY,
                ACTION_KEY,
                "loc",
                DONE_KEY,
                ("next", OBS_KEY),
                ("next", CMD_KEY),
                ("next", OBS_PRIV_KEY),
                ("next", "done"),
                ("next", "terminated"),
                ("next", "truncated"),
                ("next", "discount"),
                REWARD_KEY,
                "is_init",
            ]
            replay_td = transition.select(*replay_keys, strict=False)
            if "loc" not in replay_td.keys(True, True):
                replay_td.set("loc", torch.zeros_like(replay_td[ACTION_KEY]))
            next_td = transition["next"]
            for key in (OBS_KEY, CMD_KEY, OBS_PRIV_KEY):
                replay_td.set((BOOTSTRAP_KEY, key), next_td[key])
            replay_td.set(
                ENV_ID_KEY,
                torch.arange(self.num_envs, device=replay_td.device),
            )

            if self.use_custom_replay_buffer:
                self.replay_buffer.extend(replay_td.cpu())
                return
            if self.use_slice_replay:
                replay_td = replay_td.unsqueeze(0)
            else:
                replay_td = replay_td.reshape(-1)
            self.replay_buffer.extend(replay_td.cpu())

        if tensordict.batch_dims < 2:
            append_transition(tensordict)
            return

        batch_size = tuple(tensordict.batch_size)
        if int(batch_size[0]) == self.num_envs:
            for step_idx in range(int(batch_size[1])):
                append_transition(tensordict[:, step_idx])
            return

        if int(batch_size[1]) == self.num_envs:
            for step_idx in range(int(batch_size[0])):
                append_transition(tensordict[step_idx])
            return

        raise ValueError(
            "Expected rollout data with batch size [num_envs, T], [T, num_envs], "
            f"or [num_envs], got {batch_size} for num_envs={self.num_envs}."
        )

    def _compute_n_step_target_inputs(
        self,
        tensordict: TensorDictBase,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, n_step = tensordict.batch_size
        rewards = tensordict[REWARD_KEY]
        if rewards.shape[-1] != 1:
            rewards = rewards.sum(-1, keepdim=True)
        rewards = rewards.squeeze(-1)
        dones = tensordict[DONE_KEY].bool().squeeze(-1)
        terminated = tensordict[TERM_KEY].bool().squeeze(-1)

        if N_STEP_REWARD_KEY in tensordict.keys(True, True):
            adjusted_rewards = tensordict[N_STEP_REWARD_KEY].reshape(-1)
            bootstrap = tensordict[N_STEP_BOOTSTRAP_KEY].reshape(-1)
            bootstrap_discount = tensordict[N_STEP_DISCOUNT_KEY].reshape(-1)
            bootstrap_td = tensordict[BOOTSTRAP_KEY].reshape(-1).copy()
            self._sample_actor(bootstrap_td)
            next_log_probs = bootstrap_td[f"{ACTION_KEY}_log_prob"]
            adjusted_rewards = adjusted_rewards + torch.where(
                bootstrap.bool(),
                -bootstrap_discount * self.log_alpha.exp().detach() * next_log_probs,
                torch.zeros_like(adjusted_rewards),
            )
            self.qnet_target(bootstrap_td)
            return (
                adjusted_rewards,
                bootstrap,
                bootstrap_discount,
                bootstrap_td[Q_LOGITS_KEY],
                next_log_probs.reshape(-1),
            )

        bootstrap_td_flat = tensordict[BOOTSTRAP_KEY].reshape(-1)
        next_log_probs = None
        if self.cfg.n_step_entropy_mode == "all":
            bootstrap_td_all = bootstrap_td_flat.copy()
            self._sample_actor(bootstrap_td_all)
            next_log_probs = bootstrap_td_all[f"{ACTION_KEY}_log_prob"].reshape(
                batch_size,
                n_step,
            )

        adjusted_rewards = torch.zeros(batch_size, device=self.device, dtype=rewards.dtype)
        bootstrap = torch.zeros(batch_size, device=self.device, dtype=rewards.dtype)
        bootstrap_discount = torch.zeros(batch_size, device=self.device, dtype=rewards.dtype)
        bootstrap_idx = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        discount = torch.ones(batch_size, device=self.device, dtype=rewards.dtype)
        active = torch.ones(batch_size, device=self.device, dtype=torch.bool)
        alpha = self.log_alpha.exp().detach()

        for step_idx in range(n_step):
            step_active = active
            if not step_active.any():
                break

            adjusted_rewards = adjusted_rewards + torch.where(
                step_active,
                discount * rewards[:, step_idx],
                torch.zeros_like(adjusted_rewards),
            )

            step_terminated = step_active & terminated[:, step_idx]
            can_bootstrap = step_active & ~terminated[:, step_idx]
            next_discount = discount * self.cfg.gamma
            if next_log_probs is not None:
                adjusted_rewards = adjusted_rewards + torch.where(
                    can_bootstrap,
                    -next_discount * alpha * next_log_probs[:, step_idx],
                    torch.zeros_like(adjusted_rewards),
                )

            boundary = can_bootstrap & dones[:, step_idx]
            if boundary.any():
                bootstrap_idx = torch.where(
                    boundary,
                    torch.full_like(bootstrap_idx, step_idx),
                    bootstrap_idx,
                )
                bootstrap_discount = torch.where(boundary, next_discount, bootstrap_discount)
                bootstrap = torch.where(boundary, torch.ones_like(bootstrap), bootstrap)

            active = can_bootstrap & ~dones[:, step_idx]
            discount = torch.where(active, next_discount, discount)

            if step_terminated.any():
                active = active & ~step_terminated

        if active.any():
            bootstrap_idx = torch.where(
                active,
                torch.full_like(bootstrap_idx, n_step - 1),
                bootstrap_idx,
            )
            bootstrap_discount = torch.where(active, discount, bootstrap_discount)
            bootstrap = torch.where(active, torch.ones_like(bootstrap), bootstrap)

        batch_idx = torch.arange(batch_size, device=self.device)
        bootstrap_flat_idx = batch_idx * n_step + bootstrap_idx
        bootstrap_td = bootstrap_td_flat[bootstrap_flat_idx].copy()
        if next_log_probs is None:
            self._sample_actor(bootstrap_td)
            bootstrap_log_probs = bootstrap_td[f"{ACTION_KEY}_log_prob"]
            adjusted_rewards = adjusted_rewards + torch.where(
                bootstrap.bool(),
                -bootstrap_discount * alpha * bootstrap_log_probs,
                torch.zeros_like(adjusted_rewards),
            )
            next_log_probs = bootstrap_log_probs
        self.qnet_target(bootstrap_td)
        bootstrap_q_logits = bootstrap_td[Q_LOGITS_KEY]
        return (
            adjusted_rewards,
            bootstrap,
            bootstrap_discount,
            bootstrap_q_logits,
            next_log_probs.reshape(-1),
        )

    def train_critic(
        self,
        tensordict: TensorDictBase,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        critic_td = tensordict[:, 0].copy()
        rewards = tensordict[REWARD_KEY]
        if rewards.shape[-1] != 1:
            rewards = rewards.sum(-1, keepdim=True)
        rewards = rewards.squeeze(-1)

        with torch.no_grad():
            adjusted_rewards, bootstrap, bootstrap_discount, bootstrap_q_logits, next_log_probs = (
                self._compute_n_step_target_inputs(tensordict)
            )
            target_distributions = project_distributional_q(
                bootstrap_q_logits,
                adjusted_rewards,
                bootstrap,
                bootstrap_discount,
                self.q_support,
            )
            target_values = distributional_q_value(target_distributions, self.q_support)
            target_distributions, target_values = self._reduce_target_distributions(
                target_distributions,
                target_values,
            )

        self.qnet(critic_td)
        q_outputs = critic_td[Q_LOGITS_KEY]
        critic_log_probs = F.log_softmax(q_outputs, dim=-1).clamp(min=-30.0)
        critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
        q_loss = _masked_mean(critic_losses, mask)

        first_rewards = rewards[:, 0]
        q_values = distributional_q_value(
            F.softmax(q_outputs.detach(), dim=-1),
            self.q_support,
        )
        q_stats_values = self._critic_stats_values(q_values)
        target_stats_values = self._critic_stats_values(target_values)
        info = {
            "critic/neg_rew_ratio": _masked_mean(
                (first_rewards.detach() <= 0.0).float(),
                mask,
            ).detach(),
            "critic/q_value": _masked_mean(q_values.mean(dim=1).detach(), mask).detach(),
            "critic/q_max": q_values.detach().max(),
            "critic/q_std": q_values.detach().std(dim=1).mean(),
            "critic/valid_fraction": _valid_fraction(mask, device=self.device),
            "critic/target_q_max": target_values.detach().max(),
            "critic/target_q_min": target_values.detach().min(),
            "critic/target_vmax_frac": (
                target_values.detach() >= (self.cfg.v_max - 1e-4)
            ).float().mean(),
            "critic/target_vmin_frac": (
                target_values.detach() <= (self.cfg.v_min + 1e-4)
            ).float().mean(),
        }
        info.update(_prefix_stats("critic/q", q_stats_values, mask))
        info.pop("critic/q_mean", None)
        info.update(_prefix_stats("critic/target_q", target_stats_values, mask))
        info["critic/q_loss"] = q_loss.detach()

        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        if aa.is_distributed() and self.cfg.grad_sync_mode == "manual":
            self._all_reduce_grads(self.qnet)
        if self.cfg.max_grad_norm > 0:
            q_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.qnet.parameters(),
                self.cfg.max_grad_norm,
            )
        else:
            q_grad_norm = torch.zeros((), device=self.device)
        self.q_optimizer.step()
        with torch.no_grad():
            for target_param, param in zip(self.qnet_target.parameters(), self.qnet.parameters()):
                target_param.data.mul_(1.0 - self.cfg.tau).add_(param.data, alpha=self.cfg.tau)
        info["critic/grad_norm"] = q_grad_norm.detach()
        info["_next_log_probs"] = next_log_probs.detach()

        return info

    def train_actor(
        self,
        tensordict: TensorDictBase,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        actor_td = tensordict.copy()
        replay_loc = tensordict.get("loc", None)
        actor_td = self._sample_actor(actor_td)
        with hold_out_net(self.qnet):
            self.qnet(actor_td)
        q_values = distributional_q_value(
            F.softmax(actor_td[Q_LOGITS_KEY], dim=-1),
            self.q_support,
        )
        q_value = self._reduce_actor_q_values(q_values)
        log_probs = actor_td[f"{ACTION_KEY}_log_prob"]
        action = actor_td[ACTION_KEY]
        action_span = (self.action_max - self.action_min).clamp_min(1.0e-6)
        edge_margin = action_span * float(self.cfg.action_bound_epsilon_ratio)
        actor_loss = _masked_mean(
            self.log_alpha.exp().detach() * log_probs - q_value,
            mask,
        )
        actor_entropy = _masked_mean((-log_probs).detach(), mask)
        loc = actor_td["loc"]
        if replay_loc is None:
            replay_loc = torch.zeros_like(loc)
        action_std = actor_td["scale"].mean(dim=-1)
        action_center = 0.5 * (self.action_max + self.action_min)
        action_half_span = 0.5 * action_span
        tanh_grad = 1.0 - (
            (action.detach() - action_center) / action_half_span
        ).clamp(-1.0, 1.0).square()
        mean_action = action_center + action_half_span * torch.tanh(loc.detach())
        mean_saturation = (
            ((mean_action - self.action_min) <= edge_margin)
            | ((self.action_max - mean_action) <= edge_margin)
        ).float()
        info = {
            "actor/loss": actor_loss.detach(),
            "actor/entropy": actor_entropy.detach(),
            "actor/action_std": _masked_mean(action_std.detach(), mask),
            "actor/mean_change": _masked_mean(
                (loc.detach() - replay_loc.detach()).abs().mean(dim=-1),
                mask,
            ),
            "actor/valid_fraction": _valid_fraction(mask, device=self.device),
            "actor/action_saturation": (
                ((action - self.action_min) <= edge_margin)
                | ((self.action_max - action) <= edge_margin)
            ).float().mean(),
            "actor/mean_saturation": mean_saturation.mean(),
            "actor/max_saturation": mean_saturation.mean(dim=0).max(),
            "actor/tanh_grad": tanh_grad.mean(),
            "actor/tanh_grad_min": tanh_grad.min(),
            "policy/action_clamp_frac": (
                (
                    action.clamp(
                        self.action_min + 1.0e-6,
                        self.action_max - 1.0e-6,
                    )
                    - action
                ).abs()
                > 1.0e-7
            ).float().mean(),
        }
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if aa.is_distributed() and self.cfg.grad_sync_mode == "manual":
            self._all_reduce_grads(self.actor)
        if self.cfg.max_grad_norm > 0:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                self.cfg.max_grad_norm,
            )
        else:
            actor_grad_norm = torch.zeros((), device=self.device)
        self.actor_optimizer.step()

        info["actor/grad_norm"] = actor_grad_norm.detach()
        return info

    def train_v(self) -> dict[str, torch.Tensor]:
        if (
            not self.enable_value_probe
            or self.value_probe is None
            or self.value_optimizer is None
            or len(self.value_trace) < self.cfg.value_probe_trace_steps
        ):
            return {}

        batch = torch.stack(list(self.value_trace), dim=0).to(self.device)
        with VecNorm.freeze():
            self.vecnorm(batch)
            self.vecnorm(batch["next"])
        self.value_probe(batch)
        self.value_probe(batch["next"])
        values_tn = batch[VALUE_PROBE_KEY]
        next_values_tn = batch["next", VALUE_PROBE_KEY]

        rewards = batch[REWARD_KEY]
        if rewards.shape[-1] != 1:
            rewards = rewards.sum(-1, keepdim=True)
        rewards_nt = rewards.squeeze(-1).transpose(0, 1).unsqueeze(-1)
        terms_nt = batch[TERM_KEY].transpose(0, 1).float()
        dones_nt = batch[DONE_KEY].transpose(0, 1).float()
        values_nt = values_tn.transpose(0, 1)
        next_values_nt = next_values_tn.transpose(0, 1)
        _, returns_nt = self.gae(
            rewards_nt,
            terms_nt,
            dones_nt,
            values_nt,
            next_values_nt,
        )
        valid_mask = (~batch["is_init"].transpose(0, 1).bool()).squeeze(-1)
        value_errors = (values_nt - returns_nt).square().squeeze(-1)
        value_loss = _masked_mean(value_errors, valid_mask)

        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        if aa.is_distributed() and self.cfg.grad_sync_mode == "manual":
            self._all_reduce_grads(self.value_probe)
        if self.cfg.max_grad_norm > 0:
            value_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.value_probe.parameters(),
                self.cfg.max_grad_norm,
            )
        else:
            value_grad_norm = torch.zeros((), device=self.device)
        self.value_optimizer.step()

        with torch.no_grad():
            probe_td = batch.reshape(-1).copy()
            self.value_probe(probe_td)
            probe_td = self._sample_actor(probe_td)
            self.qnet(probe_td)
            q_pi = self._reduce_actor_q_values(
                distributional_q_value(
                    F.softmax(probe_td[Q_LOGITS_KEY], dim=-1),
                    self.q_support,
                )
            )
            value_pred = probe_td[VALUE_PROBE_KEY].squeeze(-1)
            flat_mask = ~probe_td["is_init"].bool().squeeze(-1)
            return {
                "critic/v_loss": value_loss.detach(),
                "value/grad_norm": value_grad_norm.detach(),
                "critic/v_value": _masked_mean(value_pred.detach(), flat_mask),
                "value/return_mean": _masked_mean(returns_nt.detach().squeeze(-1), valid_mask),
                "value/q_pi_mean": _masked_mean(q_pi.detach(), flat_mask),
                "value/q_pi_gap": _masked_mean((q_pi - value_pred).detach(), flat_mask),
            }

    def train_op(self, tensordict: TensorDictBase) -> dict[str, float]:
        self._append_rollout_to_replay(tensordict.exclude("stats"))
        self.num_updates += 1
        info: dict[str, float | torch.Tensor] = {
            "rb_size": float(len(self.replay_buffer)),
            "actor/alpha": self.log_alpha.exp().item(),
        }
        if len(self.replay_buffer) < self.min_replay_sample_transitions:
            if aa.is_distributed() and self.cfg.vecnorm:
                for vecnorm in self.vecnorms.values():
                    vecnorm.synchronize(mode="broadcast")
            return dict(sorted(info.items()))

        def sample_batch(*, normalize_bootstrap: bool) -> tuple[
            TensorDictBase,
            torch.Tensor,
            TensorDictBase,
            torch.Tensor,
        ]:
            batch = self.replay_buffer.sample()
            if self.use_custom_replay_buffer:
                pass
            elif self.use_slice_replay:
                batch = batch.reshape(
                    self.cfg.replay_batch_size,
                    self.cfg.n_step,
                )
            else:
                batch = batch.unsqueeze(1)
            if not self.replay_samples_on_device:
                batch = batch.to(self.device)

            batch = batch.copy()
            freeze_vecnorm = VecNorm.freeze() if self.cfg.freeze_vecnorm else nullcontext()
            with freeze_vecnorm:
                self.vecnorm(batch)
                if normalize_bootstrap:
                    self.vecnorm(batch[BOOTSTRAP_KEY])

            valid = ~batch["is_init"].squeeze(-1)
            critic_mask = valid[:, 0]
            if self.cfg.actor_update_scope == "first":
                actor_td = batch[:, 0]
                actor_mask = critic_mask
            else:
                actor_td = batch.reshape(-1)
                actor_mask = valid.reshape(-1)
            return batch, critic_mask, actor_td, actor_mask

        for _ in range(self.cfg.train_every * self.cfg.utd_ratio):
            batch, critic_mask, actor_td, actor_mask = sample_batch(
                normalize_bootstrap=True
            )
            critic_info = self.train_critic(batch, critic_mask)
            next_log_probs = critic_info.pop("_next_log_probs")
            info.update(critic_info)

            if self.gradient_step % self.cfg.policy_frequency == 0:
                info.update(self.train_actor(actor_td, actor_mask))

            if self.fixed_alpha:
                alpha_loss = torch.zeros((), device=self.device)
            else:
                assert self.alpha_optimizer is not None
                assert self.target_entropy is not None
                alpha_loss = -(
                    self.log_alpha.exp()
                    * (next_log_probs.detach() + self.target_entropy)
                ).mean()
                self.alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                if aa.is_distributed() and self.cfg.grad_sync_mode == "manual":
                    self._all_reduce_param_grad(self.log_alpha)
                self.alpha_optimizer.step()
                alpha_loss = alpha_loss.detach()

            info["alpha/loss"] = alpha_loss.detach()
            info["actor/alpha"] = self.log_alpha.exp().detach()
            if (
                self.enable_value_probe
                and self.gradient_step % self.cfg.value_probe_update_every == 0
            ):
                for _ in range(self.cfg.value_probe_inner):
                    info.update(self.train_v())
            self.gradient_step += 1

        info["rb_size"] = float(len(self.replay_buffer))
        info["gradient_step"] = float(self.gradient_step)
        if aa.is_distributed() and self.cfg.vecnorm:
            for vecnorm in self.vecnorms.values():
                vecnorm.synchronize(mode="broadcast")
        for key, value in list(info.items()):
            if key.startswith("_"):
                del info[key]
            elif torch.is_tensor(value):
                info[key] = value.detach().float().mean().item()
            else:
                info[key] = float(value)
        return dict(sorted(info.items()))

    def get_rollout_policy(self, mode: str = "train", critic: bool = False):
        del critic
        rollout_policy = RolloutPolicy(self)
        if mode == "train":
            return WarmupUniformRolloutPolicy(self, rollout_policy)
        return rollout_policy

    def compute_value(self, tensordict: TensorDictBase) -> TensorDictBase:
        work_td = tensordict.copy()
        with torch.no_grad():
            with VecNorm.freeze():
                self.vecnorm(work_td)
            work_td = self._sample_actor(work_td)
            self.qnet(work_td)
            q_probs = F.softmax(work_td[Q_LOGITS_KEY], dim=-1)
            q_values = distributional_q_value(q_probs, self.q_support)
            q_value = self._reduce_actor_q_values(q_values).unsqueeze(-1)
        tensordict.set("state_value", q_value)
        return tensordict

    def state_dict(self):
        state_dict = OrderedDict()
        for name, module in self.named_children():
            state_dict[name] = module.state_dict()
        state_dict["gradient_step"] = self.gradient_step
        state_dict["num_updates"] = self.num_updates
        state_dict["last_iter"] = int(getattr(self.env, "current_iter", 0))
        state_dict["log_alpha"] = self.log_alpha.detach().clone()
        state_dict["q_support"] = self.q_support.detach().clone()
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        succeed_keys = []
        failed_keys = []
        for name, module in self.named_children():
            module_state = state_dict.get(name, {})
            try:
                module.load_state_dict(module_state, strict=strict)
                succeed_keys.append(name)
            except Exception as exc:
                warnings.warn(f"Failed to load state dict for {name}: {str(exc)}")
                failed_keys.append(name)
        print(f"Successfully loaded {succeed_keys}.")

        if "log_alpha" in state_dict:
            self.log_alpha.data.copy_(state_dict["log_alpha"].to(self.log_alpha.device))
        if "q_support" in state_dict:
            self.q_support.copy_(state_dict["q_support"].to(self.q_support.device))
        self.gradient_step = int(state_dict.get("gradient_step", 0))
        self.num_updates = int(state_dict.get("num_updates", 0))
        start_iter = int(state_dict.get("last_iter", 0))
        if hasattr(self.env, "set_progress"):
            self.env.set_progress(start_iter)

        return failed_keys
