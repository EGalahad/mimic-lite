from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from torchrl.data import Composite, TensorSpec

from active_adaptation.learning.offpolicy.base import OffPolicyPolicyBase
from active_adaptation.learning.offpolicy.common import (
    OffPolicyBaseConfig,
    detach_item,
    global_grad_norm,
    hard_copy_,
    maybe_all_reduce_grads,
    maybe_all_reduce_tensor_grads,
    maybe_autocast,
    soft_copy_,
)
from active_adaptation.learning.offpolicy.networks import (
    DistributionalCritic,
    TanhGaussianActor,
    cross_entropy_distribution_loss,
)


@dataclass
class FastSACConfig(OffPolicyBaseConfig):
    _target_: str = "hdmi_learning.lqz_fast_sac.fast_sac.FastSACPolicy"
    name: str = "lqz-fast_sac"

    # Anchored to Holosoma G1 29-DoF WBT FastSAC preset.
    num_learning_iterations: int = 400_000
    buffer_size: int = 512
    batch_size: int = 8192
    learning_starts: int = 10
    num_updates: int = 4
    gamma: float = 0.95
    tau: float = 0.05
    policy_frequency: int = 2
    num_steps: int = 1

    actor_hidden_dim: int = 512
    critic_hidden_dim: int = 768
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    use_layer_norm: bool = True

    num_atoms: int = 501
    v_min: float = -20.0
    v_max: float = 20.0
    num_q_networks: int = 2

    log_std_min: float = -5.0
    log_std_max: float = 0.5
    init_log_std: float = 0.0
    alpha_init: float = 0.001
    target_entropy_ratio: float = 0.25
    use_autotune: bool = True
    actor_q_reduction: str = "mean"  # Holosoma FastSAC uses mean over Q heads for actor update.
    init_scale: float = 0.01
    holosoma_action_scale: float = 0.25


cs = ConfigStore.instance()
cs.store("lqz-fast_sac", node=FastSACConfig(vecnorm="train", symmetry_enabled=False), group="algo")


class FastSACPolicy(OffPolicyPolicyBase):
    def __init__(
        self,
        cfg: FastSACConfig,
        observation_spec: Composite,
        action_spec: Composite,
        reward_spec: TensorSpec,
        device: str = "cuda:0",
        env=None,
    ) -> None:
        super().__init__(cfg, observation_spec, action_spec, reward_spec, device=device, env=env)
        self.cfg: FastSACConfig

        action_scale = self._configure_holosoma_action_scale()
        self.actor = TanhGaussianActor(
            self.actor_obs_dim,
            self.action_dim,
            hidden_dim=cfg.actor_hidden_dim,
            layer_norm=cfg.use_layer_norm,
            activation="silu",
            log_std_min=cfg.log_std_min,
            log_std_max=cfg.log_std_max,
            init_scale=cfg.init_scale,
            init_log_std=cfg.init_log_std,
            action_scale=action_scale,
        ).to(self.device)
        self.critic = DistributionalCritic(
            self.critic_obs_dim,
            self.action_dim,
            num_atoms=cfg.num_atoms,
            v_min=cfg.v_min,
            v_max=cfg.v_max,
            hidden_dim=cfg.critic_hidden_dim,
            num_q_networks=cfg.num_q_networks,
            layer_norm=cfg.use_layer_norm,
            activation="silu",
        ).to(self.device)
        self.critic_target = DistributionalCritic(
            self.critic_obs_dim,
            self.action_dim,
            num_atoms=cfg.num_atoms,
            v_min=cfg.v_min,
            v_max=cfg.v_max,
            hidden_dim=cfg.critic_hidden_dim,
            num_q_networks=cfg.num_q_networks,
            layer_norm=cfg.use_layer_norm,
            activation="silu",
        ).to(self.device)
        hard_copy_(self.critic, self.critic_target)

        self.log_alpha = nn.Parameter(torch.tensor([math.log(cfg.alpha_init)], device=self.device, dtype=torch.float32))
        self.target_entropy = -float(self.action_dim) * float(cfg.target_entropy_ratio)

        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=cfg.actor_learning_rate, betas=tuple(cfg.adam_betas), weight_decay=cfg.weight_decay
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=cfg.critic_learning_rate, betas=tuple(cfg.adam_betas), weight_decay=cfg.weight_decay
        )
        self.alpha_optimizer = torch.optim.AdamW([self.log_alpha], lr=cfg.alpha_learning_rate, betas=tuple(cfg.adam_betas))

    def _configure_holosoma_action_scale(self) -> torch.Tensor:
        manager = getattr(self.env, "action_manager", None)
        if manager is None:
            return torch.ones(self.action_dim, device=self.device, dtype=torch.float32)

        env_action_scale = self._compute_holosoma_env_action_scale(manager)
        if env_action_scale.numel() != self.action_dim:
            raise ValueError(
                f"FastSAC env action scale has {env_action_scale.numel()} entries, "
                f"expected action_dim={self.action_dim}."
            )
        manager.action_scaling = env_action_scale.to(manager.device)

        action_scale = self._compute_action_boundary_from_limits(manager, env_action_scale)
        print(
            "[Info] FastSAC Holosoma action scaling: "
            f"env_scale_min={env_action_scale.min().item():.4f}, "
            f"env_scale_max={env_action_scale.max().item():.4f}, "
            f"actor_boundary_min={action_scale.min().item():.4f}, "
            f"actor_boundary_max={action_scale.max().item():.4f}, "
            f"boundary_logsum={action_scale.log().sum().item():.4f}"
        )
        return action_scale

    def _compute_holosoma_env_action_scale(self, manager) -> torch.Tensor:
        asset = manager.asset
        actuator_names = list(asset.actuator_names)
        ctrl_ids = torch.as_tensor(asset.indexing.ctrl_ids, device=self.device, dtype=torch.long)
        if len(actuator_names) != int(ctrl_ids.numel()):
            raise RuntimeError(
                f"Expected one actuator name per control id, got {len(actuator_names)} names "
                f"and {int(ctrl_ids.numel())} control ids."
            )
        name_to_ctrl_id = {name: ctrl_ids[i] for i, name in enumerate(actuator_names)}
        missing = [name for name in manager.joint_names if name not in name_to_ctrl_id]
        if missing:
            raise RuntimeError(f"Cannot compute Holosoma action scale; missing actuators for joints: {missing}")

        selected_ctrl_ids = torch.stack([name_to_ctrl_id[name] for name in manager.joint_names])
        force_range = manager.env.sim.get_default_field("actuator_forcerange").to(self.device)
        gainprm = manager.env.sim.get_default_field("actuator_gainprm").to(self.device)
        effort_limit = force_range[selected_ctrl_ids].abs().max(dim=-1).values
        stiffness = gainprm[selected_ctrl_ids, 0].abs().clamp_min(1.0e-6)
        return (float(self.cfg.holosoma_action_scale) * effort_limit / stiffness).clamp_min(1.0e-6)

    def _compute_action_boundary_from_limits(self, manager, env_action_scale: torch.Tensor) -> torch.Tensor:
        if not hasattr(manager.asset.data, "joint_pos_limits"):
            raise RuntimeError("FastSAC Holosoma action boundary requires asset.data.joint_pos_limits.")

        limits = manager.asset.data.joint_pos_limits[0, manager.joint_ids].to(self.device)
        default_pos = manager.default_joint_pos[0, manager.joint_ids].to(self.device)
        lower = limits[..., 0]
        upper = limits[..., 1]
        max_offset = torch.maximum((default_pos - lower).abs(), (upper - default_pos).abs())
        return (max_offset / env_action_scale.abs().clamp_min(1.0e-6)).clamp_min(1.0e-6)

    def _clamp_actions_to_policy_bounds(self, actions: torch.Tensor) -> torch.Tensor:
        scale = self.actor.action_scale.to(actions.device).view(1, -1)
        bias = self.actor.action_bias.to(actions.device).view(1, -1)
        return torch.minimum(torch.maximum(actions, bias - scale), bias + scale)

    def modules_to_broadcast(self) -> list[nn.Module]:
        return [self.actor, self.critic, self.critic_target]

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, td, deterministic: bool = False) -> torch.Tensor:
        obs = self.normalize_actor_obs(self.actor_obs(td).to(self.device), update=False)
        action, _, mean_action, _ = self.actor.sample(obs, deterministic=deterministic, with_logprob=False)
        return mean_action if deterministic else action

    def _critic_update(self, batch) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        critic_obs = batch["critic_observations"]
        actions = self._clamp_actions_to_policy_bounds(batch["actions"])
        rewards = batch["next", "rewards"]
        next_obs = batch["next", "observations"]
        next_critic_obs = batch["next", "critic_observations"]
        bootstrap = batch["next", "bootstrap"]
        effective_n = batch["next", "effective_n_steps"]
        dones = batch["next", "dones"]
        truncated = batch["next", "truncated"]
        discount = torch.full_like(rewards, cfg.gamma).pow(effective_n)
        valid = self.valid_mask(batch)

        with torch.no_grad(), maybe_autocast(self.device, cfg.amp, cfg.amp_dtype):
            next_actions, next_logp, _, _ = self.actor.sample(next_obs, deterministic=False, with_logprob=True)
            assert next_logp is not None
            entropy_adjusted_reward = rewards - discount * bootstrap * self.alpha.detach() * next_logp
            target_logits = self.critic_target(next_critic_obs, next_actions)
            target_probs = self.critic_target.project_distribution(
                target_logits, entropy_adjusted_reward, bootstrap, discount
            )
            target_values = self.critic_target.probs_to_values(target_probs)
            target_edge_mass = (target_probs[..., 0] + target_probs[..., -1]).mean()

        with maybe_autocast(self.device, cfg.amp, cfg.amp_dtype):
            logits = self.critic(critic_obs, actions)
            critic_loss = cross_entropy_distribution_loss(logits, target_probs, valid)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        maybe_all_reduce_grads(self.critic)
        critic_grad_norm = global_grad_norm(self.critic.parameters()).to(self.device)
        if cfg.max_grad_norm and cfg.max_grad_norm > 0:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
        self.critic_optimizer.step()

        alpha_info = self._alpha_update(next_logp, valid)

        with torch.no_grad():
            q_values = self.critic.logits_to_values(logits.detach())
        info = {
            "critic_loss": critic_loss.detach(),
            "critic_grad_norm": critic_grad_norm.detach(),
            "q_mean": q_values.mean(),
            "q_min": q_values.min(),
            "q_max": q_values.max(),
            "target_q_min": target_values.min(),
            "target_q_max": target_values.max(),
            "target_edge_mass": target_edge_mass,
            "next_logp": next_logp.detach().mean(),
            "bootstrap_fraction": bootstrap.detach().mean(),
            "effective_n_steps": effective_n.detach().mean(),
            "done_fraction": dones.float().mean(),
            "truncated_fraction": truncated.float().mean(),
            "valid_fraction": valid.detach().mean(),
        }
        info.update(alpha_info)
        return info

    def _alpha_update(self, logp: torch.Tensor, valid: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        alpha_loss = torch.tensor(0.0, device=self.device)
        logp_flat = logp.detach().view(-1)
        if valid is None:
            weight = torch.ones_like(logp_flat)
        else:
            weight = valid.float().view(-1)
        denom = weight.sum().clamp_min(1.0)
        if cfg.use_autotune:
            alpha_terms = -self.alpha * (logp_flat + self.target_entropy)
            alpha_loss = (alpha_terms * weight).sum() / denom
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            maybe_all_reduce_tensor_grads([self.log_alpha])
            self.alpha_optimizer.step()

        return {
            "alpha_loss": alpha_loss.detach(),
            "alpha": self.alpha.detach(),
            "alpha_entropy": ((-logp_flat) * weight).sum() / denom,
        }

    def _actor_update(self, batch) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        obs = batch["observations"]
        critic_obs = batch["critic_observations"]
        valid = self.valid_mask(batch)
        weight = valid.float().view(-1)
        denom = weight.sum().clamp_min(1.0)
        critic_requires_grad = [p.requires_grad for p in self.critic.parameters()]
        for p in self.critic.parameters():
            p.requires_grad_(False)
        try:
            with maybe_autocast(self.device, cfg.amp, cfg.amp_dtype):
                actions, logp, _, log_std = self.actor.sample(obs, deterministic=False, with_logprob=True)
                assert logp is not None
                logits = self.critic(critic_obs, actions)
                q_values = self.critic.logits_to_values(logits)
                if cfg.actor_q_reduction == "min":
                    q_for_actor = q_values.min(dim=0).values
                elif cfg.actor_q_reduction == "mean":
                    q_for_actor = q_values.mean(dim=0)
                else:
                    raise ValueError(f"Unsupported actor_q_reduction={cfg.actor_q_reduction}")
                actor_terms = self.alpha.detach() * logp.view(-1) - q_for_actor.view(-1)
                actor_loss = (actor_terms * weight).sum() / denom
        finally:
            for p, requires_grad in zip(self.critic.parameters(), critic_requires_grad):
                p.requires_grad_(requires_grad)

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        maybe_all_reduce_grads(self.actor)
        actor_grad_norm = global_grad_norm(self.actor.parameters()).to(self.device)
        if cfg.max_grad_norm and cfg.max_grad_norm > 0:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
        self.actor_optimizer.step()

        return {
            "actor_loss": actor_loss.detach(),
            "actor_grad_norm": actor_grad_norm.detach(),
            "actor_q_mean": (q_for_actor.detach().view(-1) * weight).sum() / denom,
            "entropy": ((-logp.detach().view(-1)) * weight).sum() / denom,
            "log_std_mean": log_std.detach().mean(),
            "actor_valid_fraction": valid.detach().mean(),
        }

    def _actor_alpha_update(self, batch) -> dict[str, torch.Tensor]:
        """Compatibility path for subclasses that still call the old combined update."""
        info = self._actor_update(batch)
        with torch.no_grad(), maybe_autocast(self.device, self.cfg.amp, self.cfg.amp_dtype):
            _, logp, _, _ = self.actor.sample(batch["observations"], deterministic=False, with_logprob=True)
            assert logp is not None
        info.update(self._alpha_update(logp, self.valid_mask(batch)))
        return info

    def train_op(self, data=None, vecnorm=None) -> dict[str, float]:
        self.add_rollout_to_replay(data)
        self.train_step += 1
        cfg = self.cfg
        if not self.replay.can_sample(cfg.batch_size, cfg.learning_starts):
            return {
                "train/replay_steps": float(len(self.replay)),
                "train/replay_transitions": float(self.replay.num_transitions),
                "train/updates": 0.0,
                "train/alpha": detach_item(self.alpha),
            }

        metrics: dict[str, list[float]] = {}
        actor_updates = 0
        for _ in range(cfg.num_updates):
            batch = self.prepare_batch(self.replay.sample(cfg.batch_size), update_normalizers=True)
            critic_info = self._critic_update(batch)
            self.update_step += 1
            for k, v in critic_info.items():
                metrics.setdefault(k, []).append(detach_item(v))

            if self.update_step % cfg.policy_frequency == 0:
                actor_info = self._actor_update(batch)
                actor_updates += 1
                for k, v in actor_info.items():
                    metrics.setdefault(k, []).append(detach_item(v))
            soft_copy_(self.critic, self.critic_target, cfg.tau)

        out = {f"train/{k}": float(sum(v) / max(1, len(v))) for k, v in metrics.items()}
        out.update(
            {
                "train/replay_steps": float(len(self.replay)),
                "train/replay_transitions": float(self.replay.num_transitions),
                "train/updates": float(cfg.num_updates),
                "train/actor_updates": float(actor_updates),
                "train/target_entropy": float(self.target_entropy),
            }
        )
        return out

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        sd = self.checkpoint_state()
        sd["actor"] = self.actor.state_dict()
        sd["critic"] = self.critic.state_dict()
        sd["critic_target"] = self.critic_target.state_dict()
        sd["log_alpha"] = self.log_alpha.detach().cpu()
        sd["actor_optimizer"] = self._opt_state(self.actor_optimizer)
        sd["critic_optimizer"] = self._opt_state(self.critic_optimizer)
        sd["alpha_optimizer"] = self._opt_state(self.alpha_optimizer)
        return sd

    def load_state_dict(self, state_dict: OrderedDict, strict: bool = True):  # type: ignore[override]
        self.load_common_state(state_dict)
        self.actor.load_state_dict(state_dict["actor"], strict=strict)
        self.critic.load_state_dict(state_dict["critic"], strict=strict)
        self.critic_target.load_state_dict(state_dict.get("critic_target", state_dict["critic"]), strict=strict)
        if "log_alpha" in state_dict:
            self.log_alpha.data.copy_(state_dict["log_alpha"].to(self.device))
        if "actor_optimizer" in state_dict:
            self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        if "critic_optimizer" in state_dict:
            self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        if "alpha_optimizer" in state_dict:
            self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])
        return nn.modules.module._IncompatibleKeys([], [])


__all__ = ["FastSACConfig", "FastSACPolicy"]
