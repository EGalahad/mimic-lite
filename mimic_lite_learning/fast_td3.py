from __future__ import annotations

from copy import deepcopy
import warnings
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Mapping, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from tensordict import TensorDictBase
from tensordict.nn import (
    TensorDictModule as Mod,
    TensorDictModuleBase,
    TensorDictSequential as Seq,
)
from torchrl.data import (
    Composite as CompositeSpec,
    LazyTensorStorage,
    TensorDictReplayBuffer,
    TensorSpec,
)
from torchrl.envs.transforms import TensorDictPrimer

import active_adaptation as aa
from active_adaptation.learning.modules.vecnorm import VecNorm
from active_adaptation.learning.ppo.common import (
    ACTION_KEY,
    CMD_KEY,
    DONE_KEY,
    OBS_KEY,
    OBS_PRIV_KEY,
    REWARD_KEY,
    CatTensors,
)
from active_adaptation.learning.ppo.ppo_base import PPOBase

from .action_bounds import (
    coerce_action_bounds_config,
    default_action_bounds,
    resolve_action_bounds,
)
from .common import NullVecNorm
from .fast_sac import (
    ACTOR_INPUT_KEY,
    BOOTSTRAP_KEY,
    CRITIC_INPUT_KEY,
    DistributionalCritic,
    Q_LOGITS_KEY,
    _build_mlp,
    _masked_mean,
    distributional_q_value,
    project_distributional_q,
)


Q_VALUES_KEY = "_q_values"


class ScalarDoubleCritic(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = 512,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        hidden_dims = [hidden_dim, hidden_dim // 2, hidden_dim // 4]
        self.q1_net = _build_mlp(None, hidden_dims, use_layer_norm=use_layer_norm)
        self.q1_out = nn.Linear(hidden_dims[-1], 1)
        self.q2_net = _build_mlp(None, hidden_dims, use_layer_norm=use_layer_norm)
        self.q2_out = nn.Linear(hidden_dims[-1], 1)

    def forward(self, critic_input: torch.Tensor) -> torch.Tensor:
        q1 = self.q1_out(self.q1_net(critic_input)).squeeze(-1)
        q2 = self.q2_out(self.q2_net(critic_input)).squeeze(-1)
        return torch.stack([q1, q2], dim=1)


class ScalarDoubleCriticTD(TensorDictModuleBase):
    def __init__(
        self,
        *,
        hidden_dim: int = 512,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.in_keys = [OBS_KEY, CMD_KEY, OBS_PRIV_KEY, ACTION_KEY]
        self.out_keys = [Q_VALUES_KEY]
        self.cat_tensors = CatTensors(
            self.in_keys,
            "_critic_input",
            del_keys=False,
            sort=False,
        )
        self.model = ScalarDoubleCritic(
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        self.cat_tensors(tensordict)
        tensordict.set(self.out_keys[0], self.model(tensordict["_critic_input"]))
        return tensordict


class FastTD3ActorCore(nn.Module):
    action_min: torch.Tensor
    action_max: torch.Tensor
    action_center: torch.Tensor
    action_scale: torch.Tensor

    def __init__(
        self,
        action_dim: int,
        *,
        hidden_dim: int = 256,
        init_scale: float = 0.01,
        use_layer_norm: bool = True,
        action_min: torch.Tensor,
        action_max: torch.Tensor,
    ) -> None:
        super().__init__()
        hidden_dims = [hidden_dim, hidden_dim // 2, hidden_dim // 4]
        self.net = _build_mlp(None, hidden_dims, use_layer_norm=use_layer_norm)
        self.fc_action = nn.Linear(hidden_dims[-1], action_dim)
        nn.init.normal_(self.fc_action.weight, 0.0, init_scale)
        nn.init.constant_(self.fc_action.bias, 0.0)

        action_min = action_min.to(dtype=torch.float32)
        action_max = action_max.to(dtype=torch.float32)
        self.register_buffer("action_min", action_min)
        self.register_buffer("action_max", action_max)
        self.register_buffer("action_center", (action_max + action_min) * 0.5)
        self.register_buffer("action_scale", (action_max - action_min) * 0.5)

    def forward(self, actor_input: torch.Tensor) -> torch.Tensor:
        hidden = self.net(actor_input)
        action = torch.tanh(self.fc_action(hidden))
        return self.action_center + self.action_scale * action


class TD3ExplorationNoise(nn.Module):
    action_min: torch.Tensor
    action_max: torch.Tensor
    log_std_min: torch.Tensor
    log_std_max: torch.Tensor
    noise_scales: torch.Tensor

    def __init__(
        self,
        *,
        action_dim: int,
        log_std_min: float = -5.0,
        log_std_max: float = 0.0,
        action_min: torch.Tensor,
        action_max: torch.Tensor,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.register_buffer("action_min", action_min.to(dtype=torch.float32))
        self.register_buffer("action_max", action_max.to(dtype=torch.float32))
        self.register_buffer(
            "log_std_min",
            torch.tensor(log_std_min, dtype=torch.float32),
        )
        self.register_buffer(
            "log_std_max",
            torch.tensor(log_std_max, dtype=torch.float32),
        )
        self.register_buffer("noise_scales", torch.empty(0, 1))

    def _sample_scales(self, batch_size: int, device: torch.device) -> torch.Tensor:
        std_min = self.log_std_min.exp()
        std_max = self.log_std_max.exp()
        return torch.rand(batch_size, 1, device=device) * (std_max - std_min) + std_min

    def _ensure_noise_scales(self, batch_size: int, device: torch.device) -> None:
        if (
            self.noise_scales.shape != (batch_size, 1)
            or self.noise_scales.device != device
        ):
            self.noise_scales = self._sample_scales(batch_size, device)

    def forward(
        self,
        action: torch.Tensor,
        is_init: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        batch_size = action.shape[0]
        self._ensure_noise_scales(batch_size, action.device)

        if is_init is not None:
            reset_mask = is_init.reshape(batch_size, -1).any(dim=-1, keepdim=True)
            if reset_mask.any():
                new_scales = self._sample_scales(batch_size, action.device)
                self.noise_scales = torch.where(
                    reset_mask,
                    new_scales,
                    self.noise_scales,
                )

        noise = torch.randn_like(action) * self.noise_scales
        noisy_action = torch.maximum(
            torch.minimum(action + noise, self.action_max),
            self.action_min,
        )
        return (noisy_action,)


@dataclass
class FastTD3Config:
    _target_: str = f"{__package__}.fast_td3.FastTD3"

    name: str = "fast_td3"
    collect_steps: int = 1
    buffer_size: int = 1024
    replay_batch_size: int = 32768
    warm_up_steps: int = 128
    updates_per_step: int = 4
    policy_frequency: int = 2

    gamma: float = 0.97
    tau: float = 0.1
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    weight_decay: float = 1e-3

    actor_hidden_dim: int = 512
    critic_hidden_dim: int = 1024
    init_scale: float = 0.01
    action_bounds: dict[str, list[float]] = field(
        default_factory=default_action_bounds
    )
    action_min: float | None = None
    action_max: float | None = None
    num_atoms: int = 101
    v_min: float = -100.0
    v_max: float = 400.0
    critic_type: str = "distributional"
    log_std_max: float = 0.0
    log_std_min: float = -1.0
    policy_noise: float = 0.001
    noise_clip: float = 0.5
    use_cdq: bool = True
    use_layer_norm: bool = False
    max_grad_norm: float = 1.0

    vecnorm: bool = True
    freeze_vecnorm: bool = False
    checkpoint_path: Union[str, None] = None
    in_keys: Tuple[str, ...] = (OBS_KEY, CMD_KEY, OBS_PRIV_KEY)
    grad_sync_mode: str | None = "manual"

    def __post_init__(self) -> None:
        if isinstance(self.grad_sync_mode, str):
            self.grad_sync_mode = self.grad_sync_mode.lower()
            if self.grad_sync_mode in {"none", "null"}:
                self.grad_sync_mode = None

        if self.grad_sync_mode not in {"manual", None, "ddp"}:
            raise ValueError(
                "grad_sync_mode must be one of {'manual', None, 'ddp'}, "
                f"got {self.grad_sync_mode!r}"
            )
        self.critic_type = str(self.critic_type).lower()
        if self.critic_type not in {"distributional", "scalar"}:
            raise ValueError(
                "critic_type must be one of {'distributional', 'scalar'}, "
                f"got {self.critic_type!r}"
            )
        self.action_bounds = coerce_action_bounds_config(
            self.action_bounds,
            action_min=self.action_min,
            action_max=self.action_max,
        )


cs = ConfigStore.instance()
cs.store("fast_td3", node=FastTD3Config(), group="algo")


class FastTD3(PPOBase):
    def __init__(
        self,
        cfg: FastTD3Config,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device,
        env,
    ) -> None:
        super().__init__()
        self.cfg = FastTD3Config(**cfg)
        if aa.is_distributed() and self.cfg.grad_sync_mode == "ddp":
            raise NotImplementedError("FastTD3 only supports manual gradient sync.")

        self.device = device
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        object.__setattr__(self, "env", env)
        del reward_spec

        observation_keys = set(observation_spec.keys(True, True))
        missing_keys = sorted({OBS_KEY, CMD_KEY, OBS_PRIV_KEY}.difference(observation_keys))
        if missing_keys:
            raise KeyError(f"Missing required observation keys: {missing_keys}")

        self.num_envs = int(getattr(env, "num_envs", observation_spec.shape[0]))
        self.action_dim = int(env.action_manager.action_dim)
        self.joint_names = env.action_manager.joint_names
        self.gradient_step = 0

        self._build_vecnorm_modules(observation_spec)

        action_min, action_max = resolve_action_bounds(
            self.cfg.action_bounds,
            self.joint_names,
            self.device,
        )
        self.register_buffer("action_min", action_min.clone())
        self.register_buffer("action_max", action_max.clone())

        self.actor = Seq(
            CatTensors(
                [OBS_KEY, CMD_KEY],
                ACTOR_INPUT_KEY,
                del_keys=False,
                sort=False,
            ),
            Mod(
                FastTD3ActorCore(
                    self.action_dim,
                    hidden_dim=self.cfg.actor_hidden_dim,
                    init_scale=self.cfg.init_scale,
                    use_layer_norm=self.cfg.use_layer_norm,
                    action_min=action_min,
                    action_max=action_max,
                ),
                [ACTOR_INPUT_KEY],
                [ACTION_KEY],
            ),
            selected_out_keys=[ACTION_KEY],
        ).to(self.device)
        if self.cfg.critic_type == "distributional":
            self.qnet = Seq(
                CatTensors(
                    [OBS_KEY, CMD_KEY, OBS_PRIV_KEY, ACTION_KEY],
                    CRITIC_INPUT_KEY,
                    del_keys=False,
                    sort=False,
                ),
                DistributionalCritic(
                    num_atoms=self.cfg.num_atoms,
                    hidden_dim=self.cfg.critic_hidden_dim,
                    use_layer_norm=self.cfg.use_layer_norm,
                ),
            ).to(self.device)
            self.register_buffer(
                "q_support",
                torch.linspace(
                    self.cfg.v_min,
                    self.cfg.v_max,
                    self.cfg.num_atoms,
                    device=self.device,
                ),
            )
            self._critic_out_key = Q_LOGITS_KEY
        else:
            self.qnet = ScalarDoubleCriticTD(
                hidden_dim=self.cfg.critic_hidden_dim,
                use_layer_norm=self.cfg.use_layer_norm,
            ).to(self.device)
            self._critic_out_key = Q_VALUES_KEY

        fake_input = observation_spec.zero()
        fake_critic_input = fake_input.copy()
        fake_critic_input.set(
            ACTION_KEY,
            torch.zeros(
                (*fake_input.batch_size, self.action_dim),
                device=self.device,
            ),
        )
        with VecNorm.freeze():
            self.vecnorm(fake_input)
            self.actor(fake_input.copy())
            self.qnet(fake_critic_input)

        self.actor_target = deepcopy(self.actor).to(self.device)
        self.actor_target.requires_grad_(False)
        self.qnet_target = deepcopy(self.qnet).to(self.device)
        self.qnet_target.requires_grad_(False)

        self.exploration = Mod(
            TD3ExplorationNoise(
                action_dim=self.action_dim,
                log_std_min=self.cfg.log_std_min,
                log_std_max=self.cfg.log_std_max,
                action_min=self.action_min,
                action_max=self.action_max,
            ),
            [ACTION_KEY, "is_init"],
            [ACTION_KEY],
        ).to(self.device)

        fused = str(self.device).startswith("cuda")
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

        self.replay_buffer_capacity = self.cfg.buffer_size * self.cfg.collect_steps * self.num_envs
        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=self.replay_buffer_capacity),
            batch_size=self.cfg.replay_batch_size,
            prefetch=2,
        )

        if aa.is_distributed():
            self.world_size = aa.get_world_size()
            self._broadcast_parameters()
        else:
            self.world_size = 1

    def _build_vecnorm_modules(self, observation_spec: CompositeSpec) -> None:
        modules = []
        self.vecnorms: Mapping[str, VecNorm] = nn.ModuleDict()
        vecnorm_cls = VecNorm if self.cfg.vecnorm else NullVecNorm
        for key in (OBS_KEY, CMD_KEY, OBS_PRIV_KEY):
            if key not in observation_spec.keys(True, True):
                continue
            shape = observation_spec[key].shape[-1:]
            vecnorm = vecnorm_cls(input_shape=shape, stats_shape=shape, decay=0.9999)
            self.vecnorms[key] = vecnorm
            modules.append(Mod(vecnorm, [key], [key]))
        self.vecnorm = Seq(*modules).to(self.device)

    def _broadcast_parameters(self) -> None:
        with torch.no_grad():
            for module in (
                self.vecnorm,
                self.actor,
                self.actor_target,
                self.qnet,
                self.qnet_target,
                self.exploration,
            ):
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

    def _sync_vecnorms(self) -> None:
        if not aa.is_distributed() or not self.cfg.vecnorm:
            return
        for vecnorm in self.vecnorms.values():
            vecnorm.synchronize(mode="broadcast")

    def _run_frozen_vecnorm(self, tensordict: TensorDictBase) -> TensorDictBase:
        if not self.cfg.vecnorm:
            return tensordict
        with VecNorm.freeze():
            self.vecnorm(tensordict)
            bootstrap_td = tensordict.get(BOOTSTRAP_KEY, None)
            if bootstrap_td is not None:
                self.vecnorm(bootstrap_td)
        return tensordict

    @torch.no_grad()
    def _update_vecnorm_from_batch(self, tensordict: TensorDictBase) -> None:
        if not self.cfg.vecnorm:
            return
        bootstrap_td = tensordict.get(BOOTSTRAP_KEY, None)
        for key, vecnorm in self.vecnorms.items():
            values = [tensordict[key].reshape(-1, tensordict[key].shape[-1])]
            if bootstrap_td is not None:
                values.append(
                    bootstrap_td[key].reshape(-1, bootstrap_td[key].shape[-1])
                )
            vecnorm._update(torch.cat(values, dim=0))

    def _reduce_q_values(self, q_values: torch.Tensor) -> torch.Tensor:
        if self.cfg.use_cdq:
            return torch.minimum(q_values[:, 0], q_values[:, 1])
        return q_values.mean(dim=1)

    def _reward_total(self, tensordict: TensorDictBase) -> torch.Tensor:
        reward = tensordict[REWARD_KEY]
        if reward.shape[-1] != 1:
            reward = reward.sum(-1, keepdim=True)
        return reward.squeeze(-1)

    def _discount(self, tensordict: TensorDictBase) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = tensordict["next"].get("terminated", None)
        if terminated is None:
            terminated = tensordict[DONE_KEY]
        terminated = terminated.float().squeeze(-1)

        discount = tensordict["next"].get("discount", None)
        if discount is None:
            bootstrap = 1.0 - terminated
            discount = torch.full_like(bootstrap, self.cfg.gamma)
            return bootstrap, discount
        bootstrap = (1.0 - terminated) * discount.float().squeeze(-1)
        return bootstrap, torch.full_like(bootstrap, self.cfg.gamma)

    def _collect_replay_data(self, tensordict: TensorDictBase) -> TensorDictBase:
        keys: list[Union[str, tuple[str, str]]] = [
            OBS_KEY,
            CMD_KEY,
            OBS_PRIV_KEY,
            ACTION_KEY,
            DONE_KEY,
            ("next", "done"),
            ("next", "terminated"),
            ("next", "truncated"),
            ("next", "discount"),
            REWARD_KEY,
        ]
        if "is_init" in tensordict.keys(True, True):
            keys.append("is_init")
        replay_td = tensordict.select(*keys, strict=False)
        next_td = tensordict["next"]
        for key in (OBS_KEY, CMD_KEY, OBS_PRIV_KEY):
            replay_td.set((BOOTSTRAP_KEY, key), next_td[key])
        return replay_td

    def observe(self, tensordict: TensorDictBase) -> None:
        self.replay_buffer.extend(self._collect_replay_data(tensordict).reshape(-1).cpu())

    def _soft_update_target(self) -> None:
        with torch.no_grad():
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.mul_(1.0 - self.cfg.tau).add_(param.data, alpha=self.cfg.tau)
            for target_param, param in zip(self.qnet_target.parameters(), self.qnet.parameters()):
                target_param.data.mul_(1.0 - self.cfg.tau).add_(param.data, alpha=self.cfg.tau)

    def _update_critic(
        self,
        tensordict: TensorDictBase,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rewards = self._reward_total(tensordict)
        bootstrap, discount = self._discount(tensordict)

        with torch.no_grad():
            next_td = tensordict[BOOTSTRAP_KEY].copy()
            self.actor_target(next_td)
            next_actions = next_td[ACTION_KEY]
            target_noise = torch.randn_like(next_actions) * self.cfg.policy_noise
            target_noise = target_noise.clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
            next_td.set(
                ACTION_KEY,
                torch.maximum(
                    torch.minimum(next_actions + target_noise, self.action_max),
                    self.action_min,
                ),
            )
            self.qnet_target(next_td)
            if self.cfg.critic_type == "distributional":
                target_distributions = project_distributional_q(
                    next_td[Q_LOGITS_KEY],
                    rewards,
                    bootstrap,
                    discount,
                    self.q_support,
                )
                target_values = distributional_q_value(target_distributions, self.q_support)
                if self.cfg.use_cdq:
                    min_distribution = torch.where(
                        (target_values[:, 0] < target_values[:, 1]).unsqueeze(-1),
                        target_distributions[:, 0],
                        target_distributions[:, 1],
                    )
                    target_distributions = torch.stack(
                        [min_distribution, min_distribution],
                        dim=1,
                    )
                target_summary = self._reduce_q_values(target_values).detach()
            else:
                target_qs = next_td[self._critic_out_key]
                target_summary = (
                    rewards + bootstrap * discount * self._reduce_q_values(target_qs)
                )
                target_distributions = None

        critic_td = tensordict.copy()
        self.qnet(critic_td)
        critic_output = critic_td[self._critic_out_key]
        if self.cfg.critic_type == "distributional":
            critic_log_probs = F.log_softmax(critic_output, dim=-1).clamp(min=-30.0)
            critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
            critic_loss = sum(
                _masked_mean(critic_losses[:, i], mask)
                for i in range(critic_losses.shape[1])
            )
        else:
            critic_losses = (critic_output - target_summary.unsqueeze(-1)).square()
            critic_loss = sum(
                _masked_mean(critic_losses[:, i], mask)
                for i in range(critic_losses.shape[1])
            )

        self.q_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        if aa.is_distributed() and self.cfg.grad_sync_mode == "manual":
            self._all_reduce_grads(self.qnet)
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.qnet.parameters(),
            self.cfg.max_grad_norm,
        )
        self.q_optimizer.step()
        return (
            critic_loss.detach(),
            critic_grad_norm.detach(),
            target_summary.mean().detach(),
            target_summary.detach(),
        )

    def _update_actor(
        self,
        tensordict: TensorDictBase,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_td = tensordict.copy()
        self.actor(actor_td)
        self.qnet(actor_td)
        actor_output = actor_td[self._critic_out_key]
        if self.cfg.critic_type == "distributional":
            actor_values = self._reduce_q_values(
                distributional_q_value(F.softmax(actor_output, dim=-1), self.q_support)
            )
        else:
            actor_values = self._reduce_q_values(actor_output)
        actor_loss = _masked_mean(-actor_values, mask)

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if aa.is_distributed() and self.cfg.grad_sync_mode == "manual":
            self._all_reduce_grads(self.actor)
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.cfg.max_grad_norm,
        )
        self.actor_optimizer.step()
        self._soft_update_target()
        return (
            actor_loss.detach(),
            actor_grad_norm.detach(),
            actor_values.mean().detach(),
            actor_td[ACTION_KEY].abs().mean().detach(),
        )

    def _update_step(self, tensordict: TensorDictBase) -> dict[str, torch.Tensor]:
        tensordict = tensordict.copy()
        self._update_vecnorm_from_batch(tensordict)
        self._run_frozen_vecnorm(tensordict)
        mask = None
        if "is_init" in tensordict.keys(True, True):
            valid = ~tensordict["is_init"].squeeze(-1)
            mask = valid if valid.any() else None

        q_loss, q_grad_norm, target_q_mean, target_summary = self._update_critic(
            tensordict,
            mask,
        )

        zero = torch.zeros((), device=self.device)
        actor_updated = self.gradient_step % self.cfg.policy_frequency == 0
        if actor_updated:
            actor_loss, actor_grad_norm, actor_q_mean, action_abs = self._update_actor(
                tensordict,
                mask,
            )
        else:
            actor_loss = zero
            actor_q_mean = zero
            action_abs = zero
            actor_grad_norm = zero

        self.gradient_step += 1

        return {
            "critic/loss": q_loss,
            "critic/grad_norm": q_grad_norm,
            "critic/target_q_mean": target_q_mean,
            "critic/q_min": target_summary.min().detach(),
            "critic/q_max": target_summary.max().detach(),
            "actor/loss": actor_loss,
            "actor/q_mean": actor_q_mean,
            "actor/action_abs": action_abs,
            "actor/grad_norm": actor_grad_norm,
            "actor/updated": torch.tensor(float(actor_updated), device=self.device),
        }

    def update(self) -> dict[str, float]:
        info: dict[str, float] = {"rb_size": float(len(self.replay_buffer))}
        warmup_transitions = self.cfg.warm_up_steps * self.cfg.collect_steps * self.num_envs
        if len(self.replay_buffer) < min(
            warmup_transitions,
            self.replay_buffer.storage.max_size,
        ):
            self._sync_vecnorms()
            self.num_updates += 1
            return info

        metric_lists: dict[str, list[torch.Tensor]] = defaultdict(list)
        for _ in range(self.cfg.updates_per_step):
            batch = self.replay_buffer.sample().to(self.device)
            step_metrics = self._update_step(batch)
            for key, value in step_metrics.items():
                metric_lists[key].append(value.detach())

        self._sync_vecnorms()
        for key, values in metric_lists.items():
            info[key] = torch.stack(values).float().mean().item()
        info["rb_size"] = float(len(self.replay_buffer))
        info["gradient_step"] = float(self.gradient_step)
        self.num_updates += 1
        return info

    def _get_current_iter(self) -> int:
        return int(getattr(self.env, "current_iter", 0))

    def get_next_saved_keys(self) -> tuple[str, ...]:
        return (OBS_KEY, CMD_KEY, OBS_PRIV_KEY)

    def make_tensordict_primer(self):
        return TensorDictPrimer({}, reset_key="done", expand_specs=False)

    def on_stage_start(self, stage: str) -> None:
        del stage
        return None

    def get_rollout_policy(self, mode: str = "train", critic: bool = False):
        del critic
        modules = [self.vecnorm, self.actor]
        if mode == "train":
            modules.append(self.exploration)
        rollout_policy = Seq(*modules, selected_out_keys=[ACTION_KEY])
        rollout_policy.forward = VecNorm.freeze()(rollout_policy.forward)
        return rollout_policy

    def train_op(self, tensordict: TensorDictBase) -> dict[str, float]:
        self.observe(tensordict.exclude("stats"))
        return self.update()

    def compute_value(self, tensordict: TensorDictBase) -> TensorDictBase:
        work_td = tensordict.copy()
        with torch.no_grad():
            self._run_frozen_vecnorm(work_td)
            self.actor(work_td)
            self.qnet(work_td)
            critic_output = work_td[self._critic_out_key]
            if self.cfg.critic_type == "distributional":
                q_values = distributional_q_value(F.softmax(critic_output, dim=-1), self.q_support)
            else:
                q_values = critic_output
            q_value = self._reduce_q_values(q_values).unsqueeze(-1)
        tensordict.set("state_value", q_value)
        return tensordict

    def state_dict(self):
        state_dict = OrderedDict()
        for name, module in self.named_children():
            state_dict[name] = module.state_dict()
        state_dict["gradient_step"] = self.gradient_step
        state_dict["num_updates"] = self.num_updates
        state_dict["last_iter"] = self._get_current_iter()
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

        self.gradient_step = int(state_dict.get("gradient_step", 0))
        self.num_updates = int(state_dict.get("num_updates", 0))
        start_iter = int(state_dict.get("last_iter", 0))
        if hasattr(self.env, "set_progress"):
            self.env.set_progress(start_iter)

        return failed_keys
