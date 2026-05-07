from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from torchrl.data import Composite, TensorSpec

from hdmi_learning.lqz_fast_sac.fast_sac import FastSACConfig, FastSACPolicy
from active_adaptation.learning.offpolicy.base import RewardRunningNormalizer
from active_adaptation.learning.offpolicy.common import (
    ACTION_KEY,
    DONE_KEY,
    REWARD_KEY,
    TERM_KEY,
    OffPolicyBaseConfig,
    detach_item,
    get_truncated,
    maybe_all_reduce_grads,
    maybe_autocast,
    soft_copy_,
)
from active_adaptation.learning.offpolicy.networks import cross_entropy_distribution_loss


@dataclass
class FlashSACConfig(FastSACConfig):
    _target_: str = "hdmi_learning.lqz_flash_sac.flash_sac.FlashSACPolicy"
    name: str = "lqz-flash_sac"

    # FlashSAC-style scaling: fewer gradient updates, larger batches/models,
    # n-step targets, reward normalization and bounded update dynamics.
    num_learning_iterations: int = 250_000
    buffer_size: int = 1024
    batch_size: int = 65_536
    learning_starts: int = 10
    num_updates: int = 1
    num_steps: int = 3
    gamma: float = 0.99
    tau: float = 0.03
    policy_frequency: int = 2

    actor_hidden_dim: int = 1024
    critic_hidden_dim: int = 1024
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    use_layer_norm: bool = True

    num_atoms: int = 501
    v_min: float = -5.0
    v_max: float = 5.0
    num_q_networks: int = 2

    alpha_init: float = 0.01
    target_entropy_ratio: float = 0.5
    actor_q_reduction: str = "min"

    reward_normalization: bool = True
    reward_norm_decay: float = 0.99
    reward_norm_clip: float = 10.0
    weight_clip: float = 10.0
    weight_row_norm: bool = True
    critic_target_reduction: str = "min"


cs = ConfigStore.instance()
cs.store("lqz-flash_sac", node=FlashSACConfig(vecnorm="train", symmetry_enabled=False), group="algo")


class FlashSACPolicy(FastSACPolicy):
    """FlashSAC-inspired high-dimensional SAC implementation.

    This keeps the same TensorDict/replay interface as FastSAC but changes the
    training recipe to the FlashSAC design axis: high-throughput batches, fewer
    updates, n-step returns, and explicit critic-dynamics stabilizers.
    """

    cfg: FlashSACConfig

    def __init__(
        self,
        cfg: FlashSACConfig,
        observation_spec: Composite,
        action_spec: Composite,
        reward_spec: TensorSpec,
        device: str = "cuda:0",
        env=None,
    ) -> None:
        super().__init__(cfg, observation_spec, action_spec, reward_spec, device=device, env=env)
        self.reward_normalizer = RewardRunningNormalizer(
            self.device,
            decay=cfg.reward_norm_decay,
            clip=cfg.reward_norm_clip,
        )

    @torch.no_grad()
    def add_to_replay(self, td):
        done = td.get(DONE_KEY).bool()
        terminated = td.get(TERM_KEY).bool()
        truncated = td.get(("next", "truncated"), get_truncated(done, terminated)).bool()
        reward = self.scalar_reward(td)
        if self.cfg.reward_normalization:
            self.reward_normalizer.update(reward)
        self.replay.add(
            observations=self.replay_observations(td),
            actions=td.get(ACTION_KEY),
            rewards=reward,
            next_observations=self.next_replay_observations(td),
            dones=done,
            terminated=terminated,
            truncated=truncated,
            is_init=self.transition_is_init(td, done),
        )

    def _apply_weight_bounds(self) -> None:
        clip = float(self.cfg.weight_clip)
        with torch.no_grad():
            for module in (self.actor, self.critic):
                if bool(self.cfg.weight_row_norm):
                    for submodule in module.modules():
                        if isinstance(submodule, nn.Linear):
                            submodule.weight.copy_(F.normalize(submodule.weight, dim=-1, eps=1e-8))
                if clip > 0:
                    for p in module.parameters():
                        p.clamp_(-clip, clip)

    def _critic_update(self, batch) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        critic_obs = batch["critic_observations"]
        actions = self._clamp_actions_to_policy_bounds(batch["actions"])
        rewards = batch["next", "rewards"]
        if cfg.reward_normalization:
            rewards = self.reward_normalizer.normalize(rewards)
        next_obs = batch["next", "observations"]
        next_critic_obs = batch["next", "critic_observations"]
        bootstrap = batch["next", "bootstrap"]
        effective_n = batch["next", "effective_n_steps"]
        valid = self.valid_mask(batch)
        discount = torch.full_like(rewards, cfg.gamma).pow(effective_n)

        with torch.no_grad(), maybe_autocast(self.device, cfg.amp, cfg.amp_dtype):
            next_actions, next_logp, _, _ = self.actor.sample(next_obs, deterministic=False, with_logprob=True)
            assert next_logp is not None
            entropy_adjusted_reward = rewards - discount * bootstrap * self.alpha.detach() * next_logp
            target_logits = self.critic_target(next_critic_obs, next_actions)
            selected_logits = self.critic_target.select_target_distribution(target_logits, reduction=cfg.critic_target_reduction)
            target_probs = self.critic_target.project_distribution(
                selected_logits, entropy_adjusted_reward, bootstrap, discount
            )

        with maybe_autocast(self.device, cfg.amp, cfg.amp_dtype):
            logits = self.critic(critic_obs, actions)
            critic_loss = cross_entropy_distribution_loss(logits, target_probs, valid)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        maybe_all_reduce_grads(self.critic)
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
        self.critic_optimizer.step()
        self._apply_weight_bounds()

        with torch.no_grad():
            q_values = self.critic.logits_to_values(logits.detach())
        return {
            "critic_loss": critic_loss.detach(),
            "critic_grad_norm": critic_grad_norm.detach(),
            "q_mean": q_values.mean(),
            "q_min": q_values.min(),
            "q_max": q_values.max(),
            "next_logp": next_logp.detach().mean(),
            "valid_fraction": valid.detach().mean(),
            "reward_norm_mean": self.reward_normalizer.mean.detach(),
            "reward_norm_std": self.reward_normalizer.var.sqrt().detach(),
        }

    def _actor_alpha_update(self, batch) -> dict[str, torch.Tensor]:
        info = super()._actor_alpha_update(batch)
        self._apply_weight_bounds()
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
                "train/reward_norm_std": detach_item(self.reward_normalizer.var.sqrt()),
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
                actor_info = self._actor_alpha_update(batch)
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
        sd = super().state_dict(*args, **kwargs)
        sd["reward_normalizer"] = self.reward_normalizer.state_dict()
        return sd

    def load_state_dict(self, state_dict: OrderedDict, strict: bool = True):  # type: ignore[override]
        result = super().load_state_dict(state_dict, strict=strict)
        if "reward_normalizer" in state_dict:
            self.reward_normalizer.load_state_dict(state_dict["reward_normalizer"], strict=strict)
        return result


__all__ = ["FlashSACConfig", "FlashSACPolicy"]
