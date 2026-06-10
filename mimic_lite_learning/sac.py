import copy
import math
import einops
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModule as Mod,
    TensorDictModuleBase,
    TensorDictSequential as Seq,
)
from torch.nn.parallel import DistributedDataParallel as DDP

from torchrl.data import Composite, TensorSpec
from torchrl.objectives import hold_out_net

import active_adaptation as aa
from active_adaptation.learning.modules import (
    AmpSafeRMSNorm,
    ConditionalBlock,
    IndependentNormal,
    MLP,
    ResidualMLP,
    SimbaMLP,
    VecNorm,
)
from active_adaptation.learning.ppo.common import (
    ACTION_KEY,
    CMD_KEY,
    DONE_KEY,
    GAE,
    OBS_KEY,
    OBS_PRIV_KEY,
    REWARD_KEY,
    TERM_KEY,
    CatTensors,
    soft_copy_,
)

from .sac_offpolicy.buffer import ReplayBuffer
from .offpolicy.critic import (
    distributional_q_value,
    project_distributional_q,
)
from .sac_offpolicy.reward_normalization import RewardNormalizer
from active_adaptation.learning.utils.opt import MuonAdamWWrapper
from active_adaptation.learning.utils.dormancy import DormancyTracker
from .common import MeanAction, ObsOODDetector

cs = ConfigStore.instance()


clip_grad_norm_ = nn.utils.clip_grad_norm_
ACTOR_INPUT_KEY = "_actor_input"
CRITIC_INPUT_KEY = "_critic_input"
VALUE_INPUT_KEY = "_value_input"
Q_OUTPUT_KEY = "_q_output"


def gaussian_target_entropy(act_dim: int, sigma: float) -> float:
    """Differential entropy of independent \\mathcal N(0, \\sigma^2) in \\mathbb R^d (FlashSAC-style).

    H = (d/2) * log(2 * pi * e * sigma^2). Used as SAC log-alpha target when ``target_entropy_sigma`` is set.
    """
    if sigma <= 0:
        raise ValueError("target_entropy_sigma must be positive for principled entropy.")
    return 0.5 * float(act_dim) * math.log(2.0 * math.pi * math.e * sigma * sigma)


def _init_sac_linear(m: nn.Module, gain: float = 1.0):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        nn.init.zeros_(m.bias)


@dataclass
class SACConfig:
    _target_: str = "mimic_lite_learning.sac.SAC"
    name: str = "mimic_lite_sac"
    train_every: int = 4
    buffer_size: int = 2000
    buffer_device: str = "cpu"
    warm_up_steps: int = 200
    lr: float = 5e-4
    # If True, actor/Q use :class:`~active_adaptation.learning.utils.opt.MuonAdamWWrapper`.
    muon: bool = False
    weight_decay: float = 0.02
    # TD learning
    n_steps: int = 3
    gamma: float = 0.99
    utd_ratio: int = 4
    policy_frequency: int = 4
    # architecture
    actor_init: str = "zeros"
    actor_layer_norm: Any = "pre"
    actor_hidden_dims: Tuple[int, ...] = (384, 384, 384)
    critic_hidden_dims: Tuple[int, ...] = (512, 512, 512)
    critic_layer_norm: Any = "pre"
    distributional: bool = True
    # batch sizes
    critic_batch_size: int = 2048
    actor_batch_size: int = 2048
    # target smoothing: this should help Q(s_t, a_t) to generalize locally around a_t
    target_action_noise: float = 0.01
    # AR(1) Gaussian exploration noise on rollout only: eps_t = rho * eps_{t-1} + sqrt(1-rho^2) * N(0,I).
    use_correlated: bool = True
    # Penalize large actor means. Defaults preserve 0.01 * ((loc / 2.5) ** 6).
    actor_loc_reg_weight: float = 0.01
    actor_loc_reg_scale: float = 8.0
    actor_loc_reg_power: float = 6.0
    # sac specific
    entropy_bonus: float = 1.0
    # If set: H_target = (d/2)*log(2*pi*e*sigma^2) for N(0,sigma^2)^d (FlashSAC).
    # If None: use -dim(A) (common SAC heuristic).
    target_entropy_sigma: float | None = None
    target_entropy_sigma_start: float | None = 0.4
    target_entropy_sigma_end: float | None = 0.25
    target_entropy_decay_start: int = 3000
    target_entropy_decay_end: int = 4000

    tau_actor: float = 0.1 # a relatively large value for faster convergence
    tau_Q: float = 0.02  # a relatively large value for faster convergence
    lr_alpha: float = 5e-4
    max_grad_norm: float = 1.0
    v_update_every: int = 32
    v_trace_steps: int = 32  # on-policy GAE horizon from replay ring (like blade_runner last())
    v_inner: int = 2
    gae_lambda: float = 0.95

    debug: bool = False
    clip_neg_reward: bool = True
    vecnorm: bool = True
    grad_sync_mode: str | None = "ddp"
    # FP16 AMP (CUDA only); GradScaler for critic, V head, standalone train_v, and actor (alpha stays fp32).
    use_amp: bool = True
    # FlashSAC-style: scale learning rewards by running discounted-return stats (buffer stores raw).
    normalize_reward: bool = False
    normalized_G_max: float = 5.0
    reward_norm_epsilon: float = 1e-8

    in_keys: Tuple[str, ...] = (CMD_KEY, OBS_KEY, OBS_PRIV_KEY)

    def __post_init__(self):
        self.utd_ratio = int(self.utd_ratio)
        if self.utd_ratio < 1:
            raise ValueError(f"utd_ratio must be >= 1, got {self.utd_ratio}.")
        self.policy_frequency = int(self.policy_frequency)
        if self.policy_frequency < 1:
            raise ValueError(
                f"policy_frequency must be >= 1, got {self.policy_frequency}."
            )
        self.actor_hidden_dims = tuple(int(x) for x in self.actor_hidden_dims)
        self.critic_hidden_dims = tuple(int(x) for x in self.critic_hidden_dims)
        if not self.actor_hidden_dims:
            raise ValueError("actor_hidden_dims must be non-empty.")
        if not self.critic_hidden_dims:
            raise ValueError("critic_hidden_dims must be non-empty.")
        if self.actor_loc_reg_weight < 0:
            raise ValueError(
                f"actor_loc_reg_weight must be >= 0, got {self.actor_loc_reg_weight}."
            )
        if self.actor_loc_reg_scale <= 0:
            raise ValueError(
                f"actor_loc_reg_scale must be > 0, got {self.actor_loc_reg_scale}."
            )
        if self.actor_loc_reg_power <= 0:
            raise ValueError(
                f"actor_loc_reg_power must be > 0, got {self.actor_loc_reg_power}."
            )
        if self.target_entropy_decay_end < self.target_entropy_decay_start:
            raise ValueError(
                "target_entropy_decay_end must be >= target_entropy_decay_start."
            )


def _same_width_residual_stack(
    input_dim: int,
    hidden_dims: Tuple[int, ...],
    output_dim: int,
    *,
    norm_cls: type[nn.Module],
    activation: type[nn.Module],
    output_non_muon: bool = True,
) -> nn.Sequential:
    width = hidden_dims[0]
    if any(dim != width for dim in hidden_dims):
        raise ValueError(
            "SAC residual trunks require all hidden dims to match; "
            f"got {hidden_dims}."
        )
    layers: list[nn.Module] = [nn.Linear(input_dim, width)]
    layers.extend(
        ConditionalBlock(hidden_dim=width)
        for _ in range(max(0, len(hidden_dims) - 1))
    )
    layers.append(norm_cls(width))
    out_layer = nn.Linear(width, output_dim)
    if output_non_muon:
        out_layer.weight._non_muon = True
    layers.append(out_layer)
    return nn.Sequential(*layers)


cs.store(name="mimic_lite_sac", node=SACConfig, group="algo")


def _normalize_grad_sync_mode(mode: str | None) -> str | None:
    if isinstance(mode, str):
        mode = mode.lower()
        if mode in {"none", "null"}:
            mode = None
    if mode not in {"manual", "ddp", None}:
        raise ValueError(
            "grad_sync_mode must be one of {'manual', 'ddp', None}, "
            f"got {mode!r}"
        )
    return mode


class DDPWithAttr(DDP):
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            module = self.__dict__.get("_modules", {}).get("module")
            if module is not None and hasattr(module, name):
                return getattr(module, name)
            raise


class TwinQNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Tuple[int, ...] = (512, 512, 512),
        activation: type[nn.Module] = nn.SiLU,
        layer_norm: Literal["pre", "post", None] = "pre"
    ):
        super().__init__()
        critic_input_dim = obs_dim + act_dim
        self.critic_1 = _same_width_residual_stack(
            critic_input_dim,
            hidden_dims,
            1,
            norm_cls=AmpSafeRMSNorm,
            activation=activation,
        )
        self.critic_2 = _same_width_residual_stack(
            critic_input_dim,
            hidden_dims,
            1,
            norm_cls=AmpSafeRMSNorm,
            activation=activation,
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        self.critic_1.apply(_init_sac_linear)
        self.critic_2.apply(_init_sac_linear)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat([obs, act], dim=-1)
        q1 = self.critic_1(x)
        q2 = self.critic_2(x)
        return torch.cat([q1, q2], dim=-1)
    
class TwinDistributionalQNetwork(nn.Module):
    """Twin C51-style critics: logits per atom, shared discrete support (see td3dist / FastSAC)."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_atoms: int,
        hidden_dims: Tuple[int, ...] = (512, 512, 512),
        activation: str| type[nn.Module] = nn.SiLU,
        simba_mlp: bool = False,
    ):
        super().__init__()
        if num_atoms < 3:
            raise ValueError("num_atoms must be > 2 for distributional Q.")
        if isinstance(activation, str):
            activation = getattr(nn, activation)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_atoms = num_atoms

        critic_input_dim = obs_dim + act_dim
    
        def make_critic():
            hidden_dim = hidden_dims[0]
            if any(dim != hidden_dim for dim in hidden_dims):
                raise ValueError(
                    "SAC distributional critic requires all hidden dims to match; "
                    f"got {hidden_dims}."
                )
            if simba_mlp:
                in_layer = nn.Linear(critic_input_dim, hidden_dim)
                in_layer.weight._non_muon = True
                out_layer = nn.Linear(hidden_dim, num_atoms)
                out_layer.weight._non_muon = True
                return nn.Sequential(
                    in_layer,
                    SimbaMLP(hidden_dim, max(1, len(hidden_dims) - 1), activation),
                    nn.LayerNorm(hidden_dim),
                    out_layer,
                )
            else:
                return _same_width_residual_stack(
                    critic_input_dim,
                    hidden_dims,
                    num_atoms,
                    norm_cls=AmpSafeRMSNorm,
                    activation=activation,
                )

        self.critic_1 = make_critic()
        self.critic_2 = make_critic()

        self.reset_parameters()

    def reset_parameters(self):
        self.critic_1.apply(_init_sac_linear)
        self.critic_2.apply(_init_sac_linear)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        z1 = self.critic_1(x)
        z2 = self.critic_2(x)
        return torch.cat([z1, z2], dim=-1)

class _SACDormancyScope(nn.Module):
    """Modules exercised during SAC rollout + learner forwards (:class:`DormancyTracker` hooks)."""

    def __init__(
        self,
        actor: nn.Module,
        q_online: nn.Module,
    ):
        super().__init__()
        self.actor = actor
        self.Q = q_online


class TanhNormalActor(nn.Module):
    """Policy trunk + Gaussian + tanh squash (same layout as blade_runner SAC)."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        layer_norm: str = None,
        std_max: float = 1.0,
        std_min: float = 0.001,
        action_init: Literal["zeros", "orthogonal"] = "zeros",
        hidden_dims: Tuple[int, ...] = (384, 384, 384),
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        width = hidden_dims[0]
        if any(dim != width for dim in hidden_dims):
            raise ValueError(
                "SAC actor requires all hidden dims to match; "
                f"got {hidden_dims}."
            )
        trunk_layers: list[nn.Module] = [nn.Linear(obs_dim, width)]
        trunk_layers.extend(
            ConditionalBlock(hidden_dim=width, condition_dim=0)
            for _ in range(max(0, len(hidden_dims) - 1))
        )
        trunk_layers.append(nn.LayerNorm(width))
        self.trunk = nn.Sequential(*trunk_layers)
        self.action = nn.Linear(width, act_dim * 2)
        self.action.weight._non_muon = True
        self.trunk.apply(_init_sac_linear)
        
        if action_init == "orthogonal":
            self.action.apply(lambda m: _init_sac_linear(m, gain=0.01))
        elif action_init == "zeros":
            # zero-init following FastSAC
            nn.init.constant_(self.action.weight, 0.0) # zero-init the weight
            nn.init.constant_(self.action.bias, 0.0) # zero-init the bias
        else:
            raise ValueError(f"Invalid action_init: {action_init}")
        
        if not std_max > 0.0:
            raise ValueError("std_max must be positive")
        self.log_std_max = math.log(std_max)
        self.log_std_min = math.log(std_min)

    def forward(self, obs: torch.Tensor):
        feat = self.trunk(obs)
        mean, raw = self.action(feat).chunk(2, dim=-1)
        # log_std = self.log_std_max - F.softplus(raw)
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (1 + torch.tanh(raw))
        return mean, torch.exp(log_std)


class SAC(TensorDictModuleBase):
    def __init__(
        self,
        cfg: SACConfig,
        observation_spec: Composite,
        action_spec: Composite,
        reward_spec: TensorSpec,
        device,
        env=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.requires_rollout_value = False
        self.device = device
        self.actor_hidden_dims = tuple(int(x) for x in self.cfg.actor_hidden_dims)
        self.critic_hidden_dims = tuple(int(x) for x in self.cfg.critic_hidden_dims)
        if not self.actor_hidden_dims:
            raise ValueError("actor_hidden_dims must be non-empty.")
        if not self.critic_hidden_dims:
            raise ValueError("critic_hidden_dims must be non-empty.")
        self.policy_frequency = int(getattr(self.cfg, "policy_frequency", 4))
        if self.policy_frequency < 1:
            raise ValueError(
                f"policy_frequency must be >= 1, got {self.policy_frequency}."
            )
        if self.cfg.actor_loc_reg_weight < 0:
            raise ValueError(
                f"actor_loc_reg_weight must be >= 0, got {self.cfg.actor_loc_reg_weight}."
            )
        if self.cfg.actor_loc_reg_scale <= 0:
            raise ValueError(
                f"actor_loc_reg_scale must be > 0, got {self.cfg.actor_loc_reg_scale}."
            )
        if self.cfg.actor_loc_reg_power <= 0:
            raise ValueError(
                f"actor_loc_reg_power must be > 0, got {self.cfg.actor_loc_reg_power}."
            )
        if self.cfg.target_entropy_decay_end < self.cfg.target_entropy_decay_start:
            raise ValueError(
                "target_entropy_decay_end must be >= target_entropy_decay_start."
            )
        self.grad_sync_mode = _normalize_grad_sync_mode(
            getattr(self.cfg, "grad_sync_mode", "manual")
        )
        self.world_size = aa.get_world_size()
        self._distributed = aa.is_distributed()
        if self._distributed and not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("Distributed training is enabled but torch.distributed is not initialized.")
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.reward_spec = reward_spec
        self.env = env

        required_obs_keys = (CMD_KEY, OBS_KEY, OBS_PRIV_KEY)
        observation_keys = set(observation_spec.keys(True, True))
        missing_keys = sorted(set(required_obs_keys).difference(observation_keys))
        if missing_keys:
            raise KeyError(f"SAC requires observation keys {required_obs_keys}, missing {missing_keys}.")

        self.actor_obs_keys: Tuple[str, ...] = (CMD_KEY, OBS_KEY)
        self.critic_obs_keys: Tuple[str, ...] = (CMD_KEY, OBS_KEY, OBS_PRIV_KEY)
        self.actor_obs_dim = sum(
            int(observation_spec[key].shape[-1]) for key in self.actor_obs_keys
        )
        self.critic_obs_dim = sum(
            int(observation_spec[key].shape[-1]) for key in self.critic_obs_keys
        )
        act_dim = action_spec.shape[-1]

        vecnorm_modules = []
        self.vecnorms = nn.ModuleDict()
        for key in self.critic_obs_keys:
            shape = observation_spec[key].shape[-1:]
            vecnorm = (
                VecNorm(input_shape=shape, stats_shape=shape, decay=1.0)
                if self.cfg.vecnorm
                else nn.Identity()
            )
            self.vecnorms[key] = vecnorm
            vecnorm_modules.append(Mod(vecnorm, [key], [key]))
        self.vecnorm = Seq(*vecnorm_modules).to(device)

        if self.cfg.vecnorm:
            self.vecnorm_obs = self.vecnorms[OBS_KEY]
        else:
            self.vecnorm_obs = nn.Identity()

        if self.cfg.distributional:
            if self.cfg.normalize_reward:
                v_min = -0.5 # we will not have negative values, but it is a good idea to have a small margin
                v_max = float(self.cfg.normalized_G_max)
                num_atoms = 101
            else:
                v_min, v_max = -1.0, 9.0
                num_atoms = int((v_max - v_min) / 0.05) + 1
            self.register_buffer(
                "q_support",
                torch.linspace(v_min, v_max, num_atoms, device=device),
            )
            self.q_support: torch.Tensor
            q_net = TwinDistributionalQNetwork(
                self.critic_obs_dim,
                act_dim,
                num_atoms=num_atoms,
                hidden_dims=self.critic_hidden_dims,
                simba_mlp=False
            )
            self.V = None  # unused; keeps optim / checkpoint layout stable
            self.V_quantile = 0.7
        else:
            q_net = TwinQNetwork(
                self.critic_obs_dim,
                act_dim,
                hidden_dims=self.critic_hidden_dims,
                layer_norm=self.cfg.critic_layer_norm,
            )
            v_net = nn.Sequential(
                MLP([self.critic_obs_dim, *self.critic_hidden_dims], nn.SiLU),
                nn.Linear(self.critic_hidden_dims[-1], 1),
            )
            v_net.apply(_init_sac_linear)
            self.V = Seq(
                CatTensors(
                    self.critic_obs_keys,
                    VALUE_INPUT_KEY,
                    del_keys=False,
                    sort=False,
                ),
                Mod(v_net, [VALUE_INPUT_KEY], ["value"]),
            ).to(device)
            self.V_quantile = 0.7

        self.gae = GAE(self.cfg.gamma, self.cfg.gae_lambda).to(device)
        actor_net = TanhNormalActor(
            self.actor_obs_dim,
            act_dim,
            layer_norm=self.cfg.actor_layer_norm,
            std_max=1.0,
            std_min=0.001,
            action_init=self.cfg.actor_init,
            hidden_dims=self.actor_hidden_dims,
        )
        self.actor = Seq(
            CatTensors(
                self.actor_obs_keys,
                ACTOR_INPUT_KEY,
                del_keys=False,
                sort=False,
            ),
            Mod(actor_net, [ACTOR_INPUT_KEY], ["loc", "scale"]),
        ).to(device)
        self.Q = Seq(
            CatTensors(
                self.critic_obs_keys,
                CRITIC_INPUT_KEY,
                del_keys=False,
                sort=False,
            ),
            Mod(q_net, [CRITIC_INPUT_KEY, ACTION_KEY], [Q_OUTPUT_KEY]),
        ).to(device)

        self.Q_target = copy.deepcopy(self.Q).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.Q_target.requires_grad_(False)
        self.actor_target.requires_grad_(False)

        self.act_dim = act_dim
        self.target_entropy_sigma: float | None = None
        self.target_entropy = 0.0
        self._set_target_entropy_sigma(self._scheduled_target_entropy_sigma(0))
        self.log_alpha = nn.Parameter(torch.tensor(math.log(0.004), device=device))
        if self._distributed:
            if self.grad_sync_mode == "ddp":
                self._wrap_ddp(local_rank=aa.get_local_rank())
            self._broadcast_parameters()

        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=self.cfg.lr_alpha)
        if self.cfg.muon:
            self.opt_actor = MuonAdamWWrapper(
                [self.actor],
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
            )
            self.opt_Q = MuonAdamWWrapper(
                [self.Q],
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
            )
        else:
            self.opt_actor = torch.optim.AdamW(self.actor.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
            self.opt_Q = torch.optim.AdamW(self.Q.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        
        if self.V is not None:
            self.opt_V = torch.optim.Adam(self.V.parameters(), lr=self.cfg.lr)

        self.global_step = 0
        self.gradient_step = 0
        self.schedule_iter = 0

        if env is None:
            raise ValueError("SAC requires env for ReplayBuffer layout (fake_tensordict).")
        if cfg.buffer_device == "cuda":
            cfg.buffer_device = device
        next_obs_keys = tuple(("next", key) for key in self.critic_obs_keys)
        fake_rb = (
            env.fake_tensordict()
            .exclude(("next", "stats"), *next_obs_keys, "collector")
            .detach()
        ).to(cfg.buffer_device)
        fake_rb[REWARD_KEY] = fake_rb[REWARD_KEY].sum(-1, keepdim=True)
        fake_rb["loc"] = torch.zeros(fake_rb.shape[0], self.act_dim, device=cfg.buffer_device)
        self.rb = ReplayBuffer(
            self.cfg.buffer_size,
            fake_rb,
            gamma=self.cfg.gamma,
            obs_keys=self.critic_obs_keys,
        )
        self.reward_normalizer: RewardNormalizer | None = None
        if self.cfg.normalize_reward:
            self.reward_normalizer = RewardNormalizer(
                gamma=float(self.cfg.gamma),
                G_max=float(self.cfg.normalized_G_max),
                load_rms=False,
                device=self.device if isinstance(self.device, torch.device) else torch.device(self.device),
                epsilon=float(self.cfg.reward_norm_epsilon),
            )

        scope = _SACDormancyScope(
            self.actor,
            self.Q,
        )
        self._dormancy_tracker = DormancyTracker(scope)

        _dev = torch.device(device) if not isinstance(device, torch.device) else device
        self._amp_device_type = _dev.type
        self._amp_enabled = bool(self.cfg.use_amp and _dev.type == "cuda")
        self.grad_scaler = GradScaler(self._amp_device_type, enabled=self._amp_enabled)

    def _unwrap_module(self, module: nn.Module) -> nn.Module:
        return module.module if isinstance(module, DDP) else module

    def _wrap_ddp(self, local_rank: int) -> None:
        device = torch.device(self.device) if not isinstance(self.device, torch.device) else self.device
        ddp_kwargs: dict[str, Any] = {
            "broadcast_buffers": True,
            "find_unused_parameters": False,
        }
        if device.type == "cuda":
            ddp_kwargs.update(device_ids=[local_rank], output_device=local_rank)

        self.actor = DDPWithAttr(self.actor, **ddp_kwargs)
        self.Q = DDPWithAttr(self.Q, **ddp_kwargs)
        if self.V is not None:
            self.V = DDPWithAttr(self.V, **ddp_kwargs)

    def _q_values(self, q_output: torch.Tensor) -> torch.Tensor:
        if not self.cfg.distributional:
            return q_output
        q_logits = q_output.reshape(*q_output.shape[:-1], 2, -1)
        return distributional_q_value(F.softmax(q_logits, dim=-1), self.q_support)

    def _q_loss(
        self,
        q_output: torch.Tensor,
        q_target: torch.Tensor,
    ) -> torch.Tensor:
        if not self.cfg.distributional:
            return (q_output - q_target).square().sum(dim=-1).mean()

        q1, q2 = q_output.chunk(2, dim=-1)
        log_p1 = F.log_softmax(q1, dim=-1).clamp(min=-30.0)
        log_p2 = F.log_softmax(q2, dim=-1).clamp(min=-30.0)
        return -(
            (q_target * log_p1).sum(-1) + (q_target * log_p2).sum(-1)
        ).mean()

    @torch.no_grad()
    def _broadcast_parameters(self) -> None:
        if not self._distributed:
            return
        dist.broadcast(self.log_alpha.data, src=0)
        modules = [
            self.vecnorm,
            self.actor,
            self.actor_target,
            self.Q,
            self.Q_target,
        ]
        if self.V is not None:
            modules.append(self.V)

        for module in modules:
            for param in module.parameters():
                dist.broadcast(param.data, src=0)
            for buffer in module.buffers():
                dist.broadcast(buffer.data, src=0)

    @torch.no_grad()
    def _broadcast_buffers(self, *modules: nn.Module) -> None:
        if not self._distributed:
            return
        for module in modules:
            for buffer in module.buffers():
                dist.broadcast(buffer.data, src=0)

    @torch.no_grad()
    def _all_reduce_grads(self, *modules: nn.Module) -> None:
        if not self._distributed:
            return
        for module in modules:
            for param in module.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)

    @torch.no_grad()
    def _all_reduce_param_grad(self, param: nn.Parameter) -> None:
        if self._distributed and param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)

    def _sync_vecnorms(self) -> None:
        if not self._distributed or not self.cfg.vecnorm:
            return
        for vecnorm in self.vecnorms.values():
            if hasattr(vecnorm, "synchronize"):
                vecnorm.synchronize(mode="broadcast")
            else:
                self._broadcast_buffers(vecnorm)

    def _scheduled_target_entropy_sigma(self, iteration: int) -> float | None:
        start = self.cfg.target_entropy_sigma_start
        end = self.cfg.target_entropy_sigma_end
        base = self.cfg.target_entropy_sigma
        if start is None and end is None:
            return base
        if start is None:
            start = base if base is not None else end
        if end is None:
            end = base if base is not None else start
        assert start is not None and end is not None
        decay_start = int(self.cfg.target_entropy_decay_start)
        decay_end = int(self.cfg.target_entropy_decay_end)
        if iteration <= decay_start:
            return float(start)
        if iteration >= decay_end:
            return float(end)
        if decay_end <= decay_start:
            return float(end)
        progress = (iteration - decay_start) / (decay_end - decay_start)
        return float(start + (end - start) * progress)

    def _set_target_entropy_sigma(self, sigma: float | None) -> None:
        self.target_entropy_sigma = None if sigma is None else float(sigma)
        if self.target_entropy_sigma is None:
            # Preserve the pre-schedule SAC behavior in this file: alpha targets
            # zero entropy unless a sigma or sigma schedule is explicitly set.
            self.target_entropy = 0.0
        else:
            self.target_entropy = gaussian_target_entropy(
                self.act_dim, self.target_entropy_sigma
            )

    def step_schedule(self, progress: float):
        self._set_target_entropy_sigma(
            self._scheduled_target_entropy_sigma(self.schedule_iter)
        )
        self.schedule_iter += 1

    def _autocast(self):
        return autocast(
            device_type=self._amp_device_type,
            dtype=torch.float16,
            enabled=self._amp_enabled,
        )

    def _flush_dormancy(self, infos: dict) -> None:
        dormancy = self._dormancy_tracker.compute_dormancy(0.02)
        for module_name, value in dormancy.items():
            infos[f"dormancy/{module_name}"] = value
        self._dormancy_tracker.reset()

    def make_tensordict_primer(self):
        """Register correlated-noise state **before** constructing :class:`SAC` so replay ``fake_tensordict`` matches rollouts."""
        from torchrl.envs import TensorDictPrimer
        from torchrl.data import UnboundedContinuous, BoundedContinuous, Composite

        shape = tuple(self.action_spec.shape)
        dev = torch.device(self.device)
        spec = {
            "prev_noise": UnboundedContinuous(shape, device=dev),
            "rho": BoundedContinuous(low=0.0, high=1.0, shape=[shape[0], 1], device=dev)
        }
        return TensorDictPrimer(
            Composite(spec, shape=[shape[0]], device=dev),
            random=self.cfg.use_correlated,
            reset_key="done",
            expand_specs=False,
        )

    def get_next_saved_keys(self) -> tuple[str, ...]:
        return ()

    def get_rollout_policy(self, mode: str = "train", critic: bool = False):
        """Train: optional AR(1) pre-tanh rollout noise; eval/deploy: deterministic squash of the Gaussian mean."""
        if mode == "deploy":
            in_keys = [key for key in self.actor_obs_keys if key in self.vecnorms]
            vecnorm = Seq(
                *(Mod(self.vecnorms[key], [key], [key]) for key in in_keys)
            ).to(self.device)
            rollout_policy = Seq(
                vecnorm,
                ObsOODDetector(in_keys, sigma=5.0),
                self._unwrap_module(self.actor),
                MeanAction(),
                selected_out_keys=[ACTION_KEY],
            )
            return rollout_policy

        def policy(tensordict: TensorDict):
            work_td = tensordict.copy()
            with VecNorm.freeze():
                self.vecnorm(work_td)
            self.actor(work_td)
            loc, scale = work_td["loc"], work_td["scale"]
            dist = IndependentNormal(loc, scale)

            if self.cfg.use_correlated:
                prev_noise = tensordict["prev_noise"]
                rho = tensordict["rho"]
                noise = (
                    rho * prev_noise 
                    + torch.sqrt((1.0 - rho.square())) * torch.randn_like(loc)
                )
                sample = loc + noise * scale
                tensordict["next", "prev_noise"] = noise
            else:
                sample = dist.sample()

            tensordict[ACTION_KEY] = sample # + 0.04 * torch.randn_like(sample)
            tensordict["loc"] = loc
            return tensordict

        return policy

    def on_stage_start(self, stage: str):
        self.enable_actor = True

    def train_op(self, tensordict: TensorDict):
        self.global_step += self.cfg.train_every

        next_obs_keys = tuple(("next", key) for key in self.critic_obs_keys)
        td = tensordict.exclude(("next", "stats"), *next_obs_keys, "collector")
        reward = td[REWARD_KEY]
        # KEEP THIS FOR DEBUGGING
        if self.cfg.debug:
            # debug: constant reward scaled by effective horizon
            # the value should converge to 1.0 in this case
            # multi-step return should significantly speed up convergence
            reward = torch.ones_like(reward) * (1.0 - self.cfg.gamma)
            neg_rew_ratio = 0.0
        else:
            reward = reward.sum(-1, keepdim=True)
            neg_rew_ratio = (reward <= 0.).float().mean().item()
            if self.cfg.clip_neg_reward:
                reward = reward.clamp_min(0.)
        td[REWARD_KEY] = reward

        bs = td.batch_size
        # StackingCollector stacks steps on batch dim 1: [num_envs, horizon, …].
        if len(bs) >= 2:
            for ti in range(int(bs[1])):
                sub = td[:, ti]
                if self.reward_normalizer is not None:
                    self.reward_normalizer.update_reward_stats(
                        reward=sub[REWARD_KEY],
                        terminated=sub[TERM_KEY],
                        truncated=sub["next", "truncated"],
                    )
                self.rb.push(sub)
        else:
            if self.reward_normalizer is not None:
                self.reward_normalizer.update_reward_stats(
                    reward=td[REWARD_KEY],
                    terminated=td[TERM_KEY],
                    truncated=td["next", "truncated"],
                )
            self.rb.push(td)

        infos: dict = {"rb_size": len(self.rb), "critic/neg_rew_ratio": neg_rew_ratio}
        if self.global_step < self.cfg.warm_up_steps:
            self._flush_dormancy(infos)
            return infos

        with self._dormancy_tracker.track():
            iters = self.cfg.train_every * self.cfg.utd_ratio
            critic_batch = self.rb.sample(
                batch_size=self.cfg.critic_batch_size * iters,
                steps=self.cfg.n_steps,
            ).to(self.device)
            self.vecnorm(critic_batch)
            self.vecnorm(critic_batch["next"])
            actor_update_count = 0
            if self.enable_actor:
                actor_update_count = sum(
                    1
                    for j in range(iters)
                    if (self.gradient_step + j) % self.policy_frequency == 0
                )
            actor_batch = None
            if actor_update_count:
                actor_batch = self.rb.sample(
                    batch_size=self.cfg.actor_batch_size * actor_update_count,
                    steps=1,
                ).to(self.device)
                self.vecnorm(actor_batch)
                self.vecnorm(actor_batch["next"])
            actor_update_idx = 0
            critic_info = {}
            actor_info = {}
            for i in range(iters):
                # batch, last_indices = self.rb.sample_sequential(
                #     batch_size=self.cfg.critic_batch_size,
                #     steps=self.cfg.n_steps,
                #     last_indices=last_indices,
                #     sequential_prob=0.6,
                #     sequential_offset=-1,
                # )
                s = i * self.cfg.critic_batch_size
                e = s + self.cfg.critic_batch_size
                batch = critic_batch[s:e]
                critic_info = self.train_critic(
                    batch, diagnostics=(i == iters - 1)
                )

                if (
                    self.enable_actor
                    and actor_batch is not None
                    and self.gradient_step % self.policy_frequency == 0
                ):
                    s = actor_update_idx * self.cfg.actor_batch_size
                    e = s + self.cfg.actor_batch_size
                    actor_info = self.train_actor(
                        batch=actor_batch[s:e],
                        diagnostics=(actor_update_idx == actor_update_count - 1),
                    )
                    actor_update_idx += 1
                self.gradient_step += 1
            infos.update(critic_info)
            infos.update(actor_info)

        # if self.global_step % self.cfg.v_update_every == 0:
        #     for _ in range(self.cfg.v_inner):
        #         infos.update(self.train_v())

        self._sync_vecnorms()
        self._flush_dormancy(infos)
        return dict(sorted(infos.items()))

    def train_critic(self, batch: TensorDict, diagnostics: bool = False):
        self.Q.train()
        reward = batch[REWARD_KEY]
        if self.reward_normalizer is not None:
            reward = self.reward_normalizer.normalize_rewards(reward)

        batch = batch.copy()
        next_batch = batch["next"].copy()
        discount = batch["next", "discount"]

        with self._autocast():
            with torch.no_grad():
                # actions are sampled with uncorrelated noise
                self.actor_target(next_batch)
                loc, scale = next_batch["loc"], next_batch["scale"]
                dist = IndependentNormal(loc, scale)
                next_action = dist.sample()

                next_log_prob = dist.log_prob(next_action)
                target_action = next_action + torch.randn_like(next_action) * self.cfg.target_action_noise
                next_batch[ACTION_KEY] = target_action
                alpha = self.log_alpha.exp()
                lp = next_log_prob
                if lp.dim() == 1:
                    lp = lp.unsqueeze(-1)
                if lp.shape != reward.shape:
                    lp = lp.reshape_as(reward)

                if self.cfg.distributional:
                    # Fold soft Bellman entropy into rewards, then categorical projection (FastSAC-style).
                    adjusted_reward = reward + discount * self.cfg.entropy_bonus * (-alpha * lp)
                    next_logits = self.Q_target(next_batch)[Q_OUTPUT_KEY]
                    target_distributions = project_distributional_q(
                        next_logits.reshape(next_logits.shape[0], 2, -1),
                        adjusted_reward,
                        torch.ones_like(discount),
                        discount,
                        self.q_support,
                    )
                    target_values = distributional_q_value(
                        target_distributions,
                        self.q_support,
                    )
                    selected = target_values.argmin(dim=1)
                    q_target = target_distributions[
                        torch.arange(
                            target_distributions.shape[0],
                            device=target_distributions.device,
                        ),
                        selected,
                    ]
                else:
                    entropy_bonus = -alpha * lp
                    if entropy_bonus.shape != reward.shape:
                        entropy_bonus = entropy_bonus.reshape_as(reward)
                    target_qs = self.Q_target(next_batch)[Q_OUTPUT_KEY]
                    target_q = target_qs.mean(dim=-1, keepdim=True)
                    q_target = reward + discount * (
                        target_q + self.cfg.entropy_bonus * entropy_bonus
                    )

            qs: torch.Tensor = self.Q(batch)[Q_OUTPUT_KEY]
            q_loss = self._q_loss(qs, q_target)

        self.opt_Q.zero_grad(set_to_none=True)
        if self._amp_enabled:
            self.grad_scaler.scale(q_loss).backward()
            if self.grad_sync_mode == "manual":
                # Match DDP+AMP ordering: reduce scaled grads before GradScaler checks them.
                self._all_reduce_grads(self.Q)
            # Must unscale before clip / grad norm: clip_grad_norm_ and the logged norm are only
            # meaningful on the physical (unscaled) gradients; grad_scaler.step still runs Inf/NaN checks.
            self.grad_scaler.unscale_(self.opt_Q)
            critic_grad_norm = clip_grad_norm_(
                self.Q.parameters(), max_norm=self.cfg.max_grad_norm
            )
            self.grad_scaler.step(self.opt_Q)
            self.grad_scaler.update()
        else:
            q_loss.backward()
            if self.grad_sync_mode == "manual":
                self._all_reduce_grads(self.Q)
            critic_grad_norm = clip_grad_norm_(self.Q.parameters(), max_norm=self.cfg.max_grad_norm)
            self.opt_Q.step()

        soft_copy_(self.Q, self.Q_target, tau=self.cfg.tau_Q)

        infos: dict = {"critic/q_loss": q_loss.item()}
        if diagnostics:
            with torch.no_grad():
                q_h = self._q_values(qs.detach())
            q_val_mean = q_h.mean().item()
            q_val_max = q_h.max().item()
            q_val_std = q_h.std(dim=-1).mean().item()
            infos.update(
                {
                    "critic/q_value": q_val_mean,
                    "critic/q_max": q_val_max,
                    "critic/q_std": q_val_std,
                    "critic/grad_norm": critic_grad_norm.item(),
                }
            )

        # Optional: use expectile regression to estimate the value
        if self.V is not None:
            with self._autocast():
                v_pred = self.V(batch)["value"]
                q_pred = qs.detach().mean(dim=-1, keepdim=True)
                assert q_pred.shape == v_pred.shape
                v_err = q_pred - v_pred
                vf_sign = (v_err < 0).float()
                vf_weight = (1 - vf_sign) * self.V_quantile + vf_sign * (
                    1 - self.V_quantile
                )
                vf_loss = (vf_weight * (v_err**2)).mean()

            self.opt_V.zero_grad(set_to_none=True)
            if self._amp_enabled:
                self.grad_scaler.scale(vf_loss).backward()
                if self.grad_sync_mode == "manual":
                    self._all_reduce_grads(self.V)
                self.grad_scaler.unscale_(self.opt_V)
                self.grad_scaler.step(self.opt_V)
                self.grad_scaler.update()
            else:
                vf_loss.backward()
                if self.grad_sync_mode == "manual":
                    self._all_reduce_grads(self.V)
                self.opt_V.step()

            if diagnostics:
                infos.update(
                    {
                        "critic/v_loss": vf_loss.item(),
                        "critic/v_value": v_pred.mean().item(),
                        "critic/v_err": v_err.mean().item(),
                    }
                )
        return infos

    def train_actor(
        self,
        batch: TensorDict,
        diagnostics: bool = False,
    ):
        batch = batch.copy()

        with hold_out_net(self.Q), self._autocast():
            self.actor(batch)
            loc, scale = batch["loc"], batch["scale"]
            dist = IndependentNormal(loc, scale)
            action_update = dist.rsample((4,))  # [4, N, D]
            entropy_est = -dist.log_prob(action_update).mean(dim=0)
            action_update_nk = einops.rearrange(action_update, "k n d -> n k d")
            actor_q_td = TensorDict(
                {
                    key: batch[key].unsqueeze(1).expand(
                        -1,
                        action_update_nk.shape[1],
                        -1,
                    )
                    for key in self.critic_obs_keys
                }
                | {ACTION_KEY: action_update_nk},
                batch_size=action_update_nk.shape[:2],
                device=batch.device,
            )
            self.Q(actor_q_td)
            q = self._q_values(actor_q_td[Q_OUTPUT_KEY]).mean(dim=-1)
            policy_term = -q.mean(dim=1)

        alpha = self.log_alpha.exp()
        loc_reg = (
            self.cfg.actor_loc_reg_weight
            * (loc.abs() / self.cfg.actor_loc_reg_scale)
            .pow(self.cfg.actor_loc_reg_power)
            .sum(-1)
            .reshape_as(policy_term)
        )
        actor_loss = (
            policy_term
            + alpha.detach() * (-entropy_est.reshape_as(policy_term))
            + loc_reg
        ).mean()

        q_action_grad_norm: torch.Tensor | None = None
        if diagnostics:
            (grad_q_wrt_a,) = torch.autograd.grad(
                q.sum(),
                action_update,
                retain_graph=True,
                create_graph=False,
            )
            q_action_grad_norm = grad_q_wrt_a.norm(dim=-1).mean()

        self.opt_alpha.zero_grad(set_to_none=True)
        alpha_loss = -(alpha * (-entropy_est.detach() + self.target_entropy)).mean()
        alpha_loss.backward()
        if self.grad_sync_mode in {"manual", "ddp"}:
            self._all_reduce_param_grad(self.log_alpha)
        self.opt_alpha.step()

        self.opt_actor.zero_grad(set_to_none=True)
        if self._amp_enabled:
            self.grad_scaler.scale(actor_loss).backward()
            if self.grad_sync_mode == "manual":
                self._all_reduce_grads(self.actor)
            self.grad_scaler.unscale_(self.opt_actor)
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), max_norm=self.cfg.max_grad_norm
            )
            self.grad_scaler.step(self.opt_actor)
            self.grad_scaler.update()
        else:
            actor_loss.backward()
            if self.grad_sync_mode == "manual":
                self._all_reduce_grads(self.actor)
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), max_norm=self.cfg.max_grad_norm
            )
            self.opt_actor.step()
        soft_copy_(self.actor, self.actor_target, tau=self.cfg.tau_actor)

        if not diagnostics:
            return 

        assert q_action_grad_norm is not None
        infos = {
            "actor/loss": actor_loss.item(),
            "actor/grad_norm": actor_grad_norm.item(),
            "actor/alpha": alpha.detach().item(),
            "actor/entropy": entropy_est.mean().item(),
            "actor/target_entropy": float(self.target_entropy),
            "actor/q_std": q.std(dim=1).mean().item(),
            "actor/q_action_grad_norm": q_action_grad_norm.item(),
            "actor/mean_loc": loc.abs().mean().item(),
            "actor/mean_scale": scale.mean().item(),
            "actor/loc_reg": loc_reg.mean().item(),
        }
        if "loc" in batch.keys():
            mean_change = (
                (dist.loc[: batch.shape[0]].detach() - batch["loc"]).abs().mean()
            )
            infos["actor/mean_change"] = mean_change.item()
        if self.target_entropy_sigma is not None:
            infos["actor/target_entropy_sigma"] = self.target_entropy_sigma

        return infos

    def train_v(self):
        """On-policy-style V update: last `v_trace_steps` ring-buffer rows + GAE (ppo.common layout [N, T, …])."""
        if self.V is None:
            return {}
        if len(self.rb) <= self.cfg.v_trace_steps:
            return {}
        trace = self.rb.last(steps=self.cfg.v_trace_steps + 1).to(self.device)
        batch = trace[:-1].copy()

        reward = batch[REWARD_KEY]
        if self.reward_normalizer is not None:
            reward = self.reward_normalizer.normalize_rewards(reward)

        # Ring buffer layout: [T, N, …]. GAE expects [N, T, …].
        next_batch = trace[1:].copy()
        done_tn = batch[DONE_KEY].bool()
        for key in self.critic_obs_keys:
            next_batch[key] = torch.where(done_tn, batch[key], next_batch[key])

        with VecNorm.freeze():
            self.vecnorm(batch)
            self.vecnorm(next_batch)
        with self._autocast():
            self.V(batch)
            self.V(next_batch)
            vals_tn = batch["value"]
            next_vals_tn = next_batch["value"]

        r_nt = reward.transpose(0, 1)
        term_nt = batch[TERM_KEY].transpose(0, 1).float()
        done_nt = batch[DONE_KEY].transpose(0, 1).float()
        val_nt = vals_tn.transpose(0, 1).float()
        next_val_nt = next_vals_tn.transpose(0, 1).float()

        with torch.no_grad():
            _, ret = self.gae(r_nt, term_nt, done_nt, val_nt, next_val_nt)

        pred_nt = vals_tn.transpose(0, 1)
        with self._autocast():
            v_loss = F.mse_loss(pred_nt, ret)

        self.opt_V.zero_grad(set_to_none=True)
        if self._amp_enabled:
            self.grad_scaler.scale(v_loss).backward()
            if self.grad_sync_mode == "manual":
                self._all_reduce_grads(self.V)
            self.grad_scaler.unscale_(self.opt_V)
            self.grad_scaler.step(self.opt_V)
            self.grad_scaler.update()
        else:
            v_loss.backward()
            if self.grad_sync_mode == "manual":
                self._all_reduce_grads(self.V)
            self.opt_V.step()

        return {
            "critic/v_loss": v_loss.item(),
            "critic/v_value": pred_nt.mean().item(),
        }

    def state_dict(self):
        state_dict = OrderedDict()
        Q = self._unwrap_module(self.Q)
        actor = self._unwrap_module(self.actor)
        state_dict["Q"] = Q.state_dict()
        # state_dict["V"] = self.V.state_dict()
        state_dict["actor"] = actor.state_dict()
        # do not store opt states as they make the ckpt very large
        # state_dict["opt_actor"] = self.opt_actor.state_dict()
        # state_dict["opt_Q"] = self.opt_Q.state_dict()
        # state_dict["opt_V"] = self.opt_V.state_dict()
        state_dict["opt_alpha"] = self.opt_alpha.state_dict()
        state_dict["log_alpha"] = self.log_alpha.detach()
        state_dict["vecnorm"] = self.vecnorm.state_dict()
        if hasattr(self, "q_support"):
            state_dict["q_support"] = self.q_support.detach()
        return state_dict

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        Q = self._unwrap_module(self.Q)
        actor = self._unwrap_module(self.actor)
        actor_state = {
            key: value
            for key, value in state_dict["actor"].items()
            if not key.endswith("upscale")
        }
        Q.load_state_dict(state_dict["Q"], strict=strict)
        # self.V.load_state_dict(state_dict["V"], strict=strict)
        actor.load_state_dict(actor_state, strict=strict)
        # reuse the same state dict for target networks
        self.Q_target.load_state_dict(state_dict["Q"], strict=strict)
        self.actor_target.load_state_dict(actor_state, strict=strict)
        # do not store opt states as they make the ckpt very large
        # self.opt_actor.load_state_dict(state_dict["opt_actor"])
        # self.opt_Q.load_state_dict(state_dict["opt_Q"])
        # self.opt_V.load_state_dict(state_dict["opt_V"])
        self.opt_alpha.load_state_dict(state_dict["opt_alpha"])
        self.log_alpha.data = state_dict["log_alpha"].to(self.device)
        self.vecnorm.load_state_dict(state_dict["vecnorm"], strict=strict)
        if "q_support" in state_dict and hasattr(self, "q_support"):
            self.q_support.copy_(state_dict["q_support"].to(self.q_support.device))
