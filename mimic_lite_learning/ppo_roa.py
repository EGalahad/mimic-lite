import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Tuple, Union, Mapping
import os

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
from active_adaptation.learning.utils.valuenorm import ValueNorm1, ValueNormFake
from active_adaptation.learning.ppo.ppo_base import PPOBase
from active_adaptation.utils.profiling import ScopedTimer
from .common import (
    ActorROA,
    CMD_SHORT_KEY,
    MeanAction,
    NullVecNorm,
    ObsOODDetector,
    PRIV_STUDENT_KEY,
    PRIV_TEACHER_KEY,
    REF_JPOS_KEY,
    check_vecnorm_divergence,
)


torch.set_float32_matmul_precision("high")

PROFILE_SYNC_TIMERS = os.environ.get("AA_PROFILE_SYNC_TIMERS", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


@dataclass
class PPOConfig:
    _target_: str = f"{__package__}.ppo_roa.PPOROA"
    name: str = "ppo_roa"
    train_every: int = 32
    ppo_epochs: int = 3
    num_minibatches: int = 8
    clip_param: float = 0.2
    gamma: float = 0.99
    lmbda: float = 0.95

    residual_action: bool = False
    teacher_use_priv: bool = False

    lr: float = 3e-4
    desired_kl: float | None = 0.01
    reg_coef: float = 0.2

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

    latent_dim: int = 512
    encoder_teacher_dims: Tuple[int, ...] = (512,)
    encoder_student_dims: Tuple[int, ...] = (512, 512)
    actor_hidden_dims: Tuple[int, ...] = (512, 512, 512)
    critic_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    max_grad_norm: float = 1.0

    phase: str = "train"
    vecnorm: bool = True
    freeze_vecnorm: bool = False
    checkpoint_path: Union[str, None] = None
    in_keys: Tuple[str, ...] = (CMD_KEY, CMD_SHORT_KEY, OBS_KEY, OBS_PRIV_KEY)

    # TODO: ddp sync mode has some bugs
    grad_sync_mode: str | None = "manual" # Options: "ddp", "manual", None
    manual_construct_dist_now: bool = True

    def __post_init__(self):
        if isinstance(self.grad_sync_mode, str):
            self.grad_sync_mode = self.grad_sync_mode.lower()
            if self.grad_sync_mode in {"none", "null"}:
                self.grad_sync_mode = None

        if self.grad_sync_mode not in {"ddp", "manual", None}:
            raise ValueError(
                "grad_sync_mode must be one of {'ddp', 'manual', None}, "
                f"got {self.grad_sync_mode!r}"
            )


cs = ConfigStore.instance()
cs.store("ppo_roa_train", node=PPOConfig(phase="train"), group="algo")
cs.store("ppo_roa_adapt", node=PPOConfig(phase="adapt"), group="algo")
cs.store("ppo_roa_finetune", node=PPOConfig(phase="finetune"), group="algo")


class PPOROA(PPOBase):
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
        assert self.cfg.phase in ["train", "adapt", "finetune"]

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

        encoder_student_in_keys = [OBS_KEY, CMD_SHORT_KEY]
        critic_in_keys = [OBS_PRIV_KEY, OBS_KEY, CMD_KEY]

        latent_dim = self.cfg.latent_dim
        encoder_teacher_in_keys = (
            [OBS_PRIV_KEY] if self.cfg.teacher_use_priv else [OBS_KEY, CMD_KEY]
        )
        self.encoder_teacher = Seq(
            CatTensors(
                encoder_teacher_in_keys,
                "_encoder_teacher_inp",
                del_keys=False,
                sort=False,
            ),
            Mod(
                nn.Sequential(
                    make_mlp(
                        list(self.cfg.encoder_teacher_dims),
                        norm=self.cfg.layer_norm,
                    ),
                    nn.LazyLinear(latent_dim),
                ),
                "_encoder_teacher_inp",
                PRIV_TEACHER_KEY,
            ),
            selected_out_keys=[PRIV_TEACHER_KEY],
        ).to(self.device)

        self.encoder_student = Seq(
            CatTensors(
                encoder_student_in_keys,
                "_encoder_student_inp",
                del_keys=False,
                sort=False,
            ),
            Mod(
                nn.Sequential(
                    make_mlp(
                        list(self.cfg.encoder_student_dims),
                        norm=self.cfg.layer_norm,
                    ),
                    nn.LazyLinear(latent_dim),
                ),
                "_encoder_student_inp",
                PRIV_STUDENT_KEY,
            ),
            selected_out_keys=[PRIV_STUDENT_KEY],
        ).to(self.device)

        if self.cfg.residual_action:
            assert REF_JPOS_KEY in observation_spec.keys(
                True, True
            ), f"Residual action requires {REF_JPOS_KEY} observation"

            class RefJointPos(nn.Module):
                def forward(self, ref_jpos, action):
                    return (ref_jpos + action,)

            residual_module = Mod(RefJointPos(), [REF_JPOS_KEY, "loc"], ["loc"])
        else:
            residual_module = None

        def build_actor(
            in_keys: List[str], residual: Mod | None = None
        ) -> ProbabilisticActor:
            actor_modules = [
                CatTensors(in_keys, "_actor_inp", del_keys=False, sort=False),
                Mod(
                    make_mlp(
                        list(self.cfg.actor_hidden_dims), norm=self.cfg.layer_norm
                    ),
                    ["_actor_inp"],
                    ["_actor_feature"],
                ),
                Mod(
                    ActorROA(
                        self.action_dim,
                        init_noise_scale=self.cfg.init_noise_scale,
                        load_noise_scale=self.cfg.load_noise_scale,
                    ),
                    ["_actor_feature"],
                    ["loc", "scale"],
                ),
            ]
            if residual is not None:
                actor_modules.append(residual)
            return ProbabilisticActor(
                module=Seq(*actor_modules),
                in_keys=["loc", "scale"],
                out_keys=[ACTION_KEY],
                distribution_class=IndependentNormal,
                return_log_prob=True,
            ).to(self.device)

        self.dist_cls = IndependentNormal
        self.dist_keys = ["loc", "scale"]

        self.actor_teacher = build_actor(
            [OBS_KEY, PRIV_TEACHER_KEY],
            residual=residual_module,
        )
        self.actor_student = build_actor([OBS_KEY, PRIV_STUDENT_KEY])

        self.critic = Seq(
            CatTensors(critic_in_keys, "_critic_input", del_keys=False),
            Mod(
                nn.Sequential(
                    make_mlp(
                        list(self.cfg.critic_hidden_dims), norm=self.cfg.layer_norm
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
        self.encoder_teacher(fake_input)
        self.actor_teacher(fake_input)
        self.encoder_student(fake_input)
        self.actor_student(fake_input)
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

        if self.cfg.phase in {"train", "finetune"}:
            # PPO related optimizers
            policy_modules: List[nn.Module] = []
            if self.cfg.phase == "train":
                policy_modules += [self.actor_teacher, self.encoder_teacher]
            else:
                policy_modules += [self.actor_student, self.encoder_student]

            self.lr_policy = self.cfg.lr
            policy_params = [
                {"params": module.parameters()} for module in policy_modules
            ]
            self.opt_policy = torch.optim.Adam(policy_params, lr=self.lr_policy)
            self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr)

        if self.cfg.phase in {"train", "adapt"}:
            adapt_modules: List[nn.Module] = [self.encoder_student]
            if self.cfg.residual_action:
                adapt_modules.append(self.actor_student)
            adapt_parameters = [
                {"params": module.parameters()} for module in adapt_modules
            ]
            self.opt_adapt = torch.optim.Adam(adapt_parameters, lr=self.cfg.lr)

        self.num_updates = 0

    def _build_vecnorm_modules(self, observation_spec: CompositeSpec):
        modules = []
        self.vecnorms: Mapping[str, VecNorm] = nn.ModuleDict()
        vecnorm_cls = NullVecNorm if self.cfg.vecnorm is None else VecNorm

        keys_to_norm = [CMD_KEY, CMD_SHORT_KEY, OBS_KEY, OBS_PRIV_KEY]
        for key in keys_to_norm:
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

        self.actor_teacher = wrap_td_module(self.actor_teacher)
        self.actor_student = wrap_td_module(self.actor_student)
        self.encoder_teacher = wrap_td_module(self.encoder_teacher)
        self.encoder_student = wrap_td_module(self.encoder_student)
        self.critic = wrap_td_module(self.critic)

    @torch.no_grad()
    def _broadcast_parameters(self):
        for module in (
            self.actor_teacher,
            self.actor_student,
            self.encoder_teacher,
            self.encoder_student,
            self.critic,
        ):
            for param in module.parameters():
                dist.broadcast(param, src=0)

    @torch.no_grad()
    def _all_reduce_grads(self, *modules):
        for module in modules:
            if module is None:
                continue
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
            vecnorms = []
            in_keys = set(self.encoder_student.in_keys).union(
                set(self.actor_student.in_keys)
            )
            in_keys = [key for key in in_keys if key in self.vecnorms]
            for key in in_keys:
                vecnorms.append(Mod(self.vecnorms[key], [key], [key]))
            vecnorm = Seq(*vecnorms).to(self.device)
            ood_detector = ObsOODDetector(list(in_keys), sigma=5.0)
            modules = [vecnorm, ood_detector]
        else:
            modules = [self.vecnorm]

        if self.cfg.phase == "train":
            modules += [self.encoder_teacher, self.actor_teacher]
        elif self.cfg.phase == "adapt":
            modules += [self.encoder_student, self.actor_student]
        elif self.cfg.phase == "finetune":
            modules += [self.encoder_student, self.actor_student]

        if mode == "deploy":
            modules[-1] = modules[-1].module[0]
            modules.append(MeanAction())
            out_keys = [ACTION_KEY]
        else:
            out_keys = [f"{ACTION_KEY}_log_prob", ACTION_KEY] + self.dist_keys

        rollout_policy = Seq(*modules, selected_out_keys=out_keys)
        if self.cfg.freeze_vecnorm:
            rollout_policy.forward = VecNorm.freeze()(rollout_policy.forward)
        return rollout_policy

    @VecNorm.freeze()
    def train_op(self, tensordict: TensorDict):
        with ScopedTimer("training.exclude_stats", sync=False):
            tensordict = tensordict.exclude("stats")

        info = {}
        if self.cfg.phase == "train":
            with ScopedTimer("training.train_policy", sync=False):
                info.update(self.train_policy(tensordict.copy()))
            with ScopedTimer("training.train_adapt", sync=False):
                info.update(self.train_adapt(tensordict.copy()))
        elif self.cfg.phase == "adapt":
            with ScopedTimer("training.train_adapt", sync=False):
                info.update(self.train_adapt(tensordict.copy()))
        elif self.cfg.phase == "finetune":
            with ScopedTimer("training.train_policy", sync=False):
                info.update(self.train_policy(tensordict.copy()))

        self.num_updates += 1

        if aa.is_distributed():
            with ScopedTimer("training.distributed_sync", sync=PROFILE_SYNC_TIMERS):
                for m in [self.value_norm]:
                    for p in m.parameters():
                        dist.all_reduce(p, op=dist.ReduceOp.AVG)
                    for b in m.buffers():
                        dist.all_reduce(b, op=dist.ReduceOp.AVG)

                if self.cfg.vecnorm is not None:
                    for name, vecnorm in self.vecnorms.items():
                        loc_diffs, scale_diffs = check_vecnorm_divergence(vecnorm)
                        if aa.is_main_process():
                            info[f"vecnorm/{name}/loc_diff_max"] = max(loc_diffs)
                            info[f"vecnorm/{name}/scale_diff_max"] = max(scale_diffs)
                            info[f"vecnorm/{name}/loc_diff_mean"] = sum(loc_diffs) / len(
                                loc_diffs
                            )
                            info[f"vecnorm/{name}/scale_diff_mean"] = sum(
                                scale_diffs
                            ) / len(scale_diffs)
                        vecnorm.synchronize(mode="broadcast")

        with ScopedTimer("training.post_metrics", sync=False):
            action_std = self._get_actor_std(
                self.actor_teacher if self.cfg.phase == "train" else self.actor_student
            )
            if action_std is not None:
                for joint_name, std in zip(self.joint_names, action_std):
                    info[f"actor_std/{joint_name}"] = std
                info["actor_std/mean"] = action_std.mean()

            if PRIV_TEACHER_KEY in tensordict.keys():
                info["adapt/priv_feature_norm"] = (
                    tensordict[PRIV_TEACHER_KEY].norm(p=2, dim=-1).mean().detach()
                )
            if PRIV_STUDENT_KEY in tensordict.keys():
                info["adapt/priv_pred_norm"] = (
                    tensordict[PRIV_STUDENT_KEY].norm(p=2, dim=-1).mean().detach()
                )

        return info

    def _get_actor_std(self, actor_module):
        module = actor_module.module if isinstance(actor_module, DDP) else actor_module
        for _, p in module.named_parameters():
            if p.ndim == 1 and p.shape[0] == self.action_dim:
                return p.detach()
        return None

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
                    + (self.cfg.entropy_coef_end - self.cfg.entropy_coef_start) * progress
                )
            else:
                self.entropy_coef = self.cfg.entropy_coef_end

        with ScopedTimer("training.policy.minibatch_loop", sync=False):
            for _ in range(self.cfg.ppo_epochs):
                for minibatch in make_batch(tensordict, self.cfg.num_minibatches):
                    with ScopedTimer("training.policy.update_ppo", sync=False):
                        info = self._update_ppo(minibatch)
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

    def train_adapt(self, tensordict: TensorDict):
        infos = []

        with ScopedTimer("training.adapt.teacher_encoder", sync=PROFILE_SYNC_TIMERS):
            with torch.no_grad():
                self.encoder_teacher(tensordict)

        with ScopedTimer("training.adapt.minibatch_loop", sync=False):
            for _ in range(2):
                for minibatch in make_batch(
                    tensordict, self.cfg.num_minibatches, self.cfg.train_every
                ):
                    info = {}
                    valid = ~minibatch["is_init"].squeeze(-1)

                    with ScopedTimer(
                        "training.adapt.student_forward_loss",
                        sync=PROFILE_SYNC_TIMERS,
                    ):
                        self.encoder_student(minibatch)
                        priv_loss = F.mse_loss(
                            minibatch[PRIV_STUDENT_KEY],
                            minibatch[PRIV_TEACHER_KEY],
                            reduction="none",
                        )
                        priv_loss = priv_loss[valid].mean()

                    if self.cfg.residual_action:
                        with ScopedTimer(
                            "training.adapt.actor_dist_loss",
                            sync=PROFILE_SYNC_TIMERS,
                        ):
                            with torch.no_grad():
                                dist_teacher = self.actor_teacher.get_dist(minibatch)

                            minibatch[PRIV_STUDENT_KEY] = minibatch[
                                PRIV_STUDENT_KEY
                            ].detach()
                            dist_student = self.actor_student.get_dist(minibatch)
                            actor_loss = F.mse_loss(
                                dist_teacher.mean, dist_student.mean, reduction="none"
                            )
                            actor_loss = actor_loss[valid].mean()
                    else:
                        actor_loss = torch.zeros((), device=self.device)

                    info["adapt/priv_loss"] = priv_loss.detach()
                    info["adapt/actor_loss"] = actor_loss.detach()

                    with ScopedTimer(
                        "training.adapt.backward", sync=PROFILE_SYNC_TIMERS
                    ):
                        self.opt_adapt.zero_grad()
                        (priv_loss + actor_loss).backward()
                    if aa.is_distributed() and self.cfg.grad_sync_mode == "manual":
                        with ScopedTimer(
                            "training.adapt.grad_sync", sync=PROFILE_SYNC_TIMERS
                        ):
                            self._all_reduce_grads(
                                self.encoder_student, self.actor_student
                            )
                    with ScopedTimer(
                        "training.adapt.optimizer_step", sync=PROFILE_SYNC_TIMERS
                    ):
                        self.opt_adapt.step()

                    infos.append(TensorDict(info, []))

        with ScopedTimer("training.adapt.aggregate_infos", sync=False):
            return {k: v.mean().item() for k, v in sorted(torch.stack(infos).items())}

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

        with ScopedTimer("training.policy.ppo.encode", sync=PROFILE_SYNC_TIMERS):
            if self.cfg.phase == "train":
                self.encoder_teacher(tensordict)
                actor = self.actor_teacher
            elif self.cfg.phase == "finetune":
                self.encoder_student(tensordict)
                actor = self.actor_student
            else:
                raise ValueError(f"Invalid phase: {self.cfg.phase}")

        [tensordict.pop(key) for key in self.dist_keys]
        action_old = tensordict.pop(ACTION_KEY)
        logp_old = tensordict.pop(f"{ACTION_KEY}_log_prob")
        with ScopedTimer(
            "training.policy.ppo.actor_dist", sync=PROFILE_SYNC_TIMERS
        ):
            if self.cfg.manual_construct_dist_now:
                actor_base = actor.module if isinstance(actor, DDP) else actor
                actor_base(tensordict)
                dist_now = self.dist_cls(
                    loc=tensordict["loc"], scale=tensordict["scale"]
                )
            else:
                dist_now: D.Independent = actor.get_dist(tensordict)

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

        with ScopedTimer("training.policy.ppo.reg_loss", sync=PROFILE_SYNC_TIMERS):
            if self.cfg.phase == "train":
                if PRIV_STUDENT_KEY not in tensordict.keys():
                    with torch.no_grad():
                        self.encoder_student(tensordict)
                reg_loss = F.mse_loss(
                    tensordict[PRIV_STUDENT_KEY],
                    tensordict[PRIV_TEACHER_KEY],
                    reduction="none",
                )
                reg_loss = torch.mean(reg_loss[valid])
            else:
                reg_loss = torch.zeros((), device=self.device)

        loss = (
            policy_loss
            + entropy_loss
            + value_loss.mean()
            + self.cfg.reg_coef * reg_loss
        )

        with ScopedTimer("training.policy.ppo.backward", sync=PROFILE_SYNC_TIMERS):
            self.opt_policy.zero_grad()
            self.opt_critic.zero_grad()
            loss.backward()
        if aa.is_distributed() and self.cfg.grad_sync_mode == "manual":
            with ScopedTimer(
                "training.policy.ppo.grad_sync", sync=PROFILE_SYNC_TIMERS
            ):
                self._all_reduce_grads(
                    self.actor_teacher,
                    self.actor_student,
                    self.encoder_teacher,
                    self.encoder_student,
                    self.critic,
                )
        with ScopedTimer("training.policy.ppo.clip_grad", sync=PROFILE_SYNC_TIMERS):
            actor_grad_norm = nn.utils.clip_grad_norm_(
                actor.parameters(), self.cfg.max_grad_norm
            )
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.cfg.max_grad_norm
            )
            if self.cfg.phase == "train":
                priv_grad_norm = nn.utils.clip_grad_norm_(
                    self.encoder_teacher.parameters(), self.cfg.max_grad_norm
                )
            else:
                priv_grad_norm = torch.zeros(1, device=self.device)
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
            "actor/reg_loss": reg_loss.detach(),
            "actor/clamp_ratio": clipfrac.detach(),
            "actor/entropy": entropy.detach(),
            "actor/mean_std": tensordict["scale"].detach().mean(),
            "actor/kl": kl.detach(),
        }

        for i, group_name in enumerate(self.reward_groups):
            info[f"critic/{group_name}.explained_var"] = explained_var[i]
            info[f"critic/{group_name}.value_loss"] = value_loss[i].detach()

        info["opt/grad_norm.critic"] = critic_grad_norm.detach()
        info["opt/grad_norm.encoder_priv"] = priv_grad_norm.detach()
        info["opt/grad_norm.actor"] = actor_grad_norm.detach()
        return info

    def state_dict(self):
        if self.cfg.phase == "train":
            if not self.cfg.residual_action:
                for src, dst in zip(
                    self.actor_teacher.parameters(), self.actor_student.parameters()
                ):
                    dst.data.copy_(src.data)
            else:
                # copy actor_std
                for name, param in self.actor_teacher.named_parameters():
                    if "actor_std" in name:
                        for (
                            adapt_name,
                            adapt_param,
                        ) in self.actor_student.named_parameters():
                            if "actor_std" in adapt_name:
                                adapt_param.data.copy_(param.data)

        state_dict = OrderedDict()
        for name, module in self.named_children():
            if isinstance(module, DDP):
                module = module.module
            state_dict[name] = module.state_dict()
        state_dict["last_phase"] = self.cfg.phase
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
            except Exception as e:
                warnings.warn(f"Failed to load state dict for {name}: {str(e)}")
                failed_keys.append(name)
        print(f"Successfully loaded {succeed_keys}.")

        start_iter = state_dict.get("last_iter", 0)
        if self.cfg.phase != state_dict.get("last_phase", self.cfg.phase):
            start_iter = 0
        if hasattr(self.env, "set_progress"):
            self.env.set_progress(start_iter)

        return failed_keys
