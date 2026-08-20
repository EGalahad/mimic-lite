from __future__ import annotations

from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from tensordict import TensorDict

from active_adaptation.learning.ppo.common import REWARD_KEY
from mimic_lite_learning.ppo_roa import PPOConfig, PPOROA


class PPOROAAlignmentTest(unittest.TestCase):
    def test_teacher_recipe_defaults_are_aligned(self) -> None:
        cfg = PPOConfig()

        self.assertTrue(cfg.teacher_use_priv)
        self.assertEqual(cfg.opt, "muon")
        self.assertEqual(cfg.policy_lr, 3e-4)
        self.assertEqual(cfg.critic_lr, 3e-4)
        self.assertNotIn("lr", cfg.__dataclass_fields__)
        self.assertTrue(cfg.tail_kl_lr_control)
        self.assertEqual(cfg.desired_kl_end, 0.005)
        self.assertEqual(cfg.kl_lr_high_threshold_ratio, 2.0)
        self.assertFalse(cfg.epoch_kl_early_stop)
        self.assertFalse(cfg.finetune_freeze_encoder)
        self.assertFalse(cfg.finetune_clip_encoder_grads)
        self.assertEqual((cfg.ppo_epochs, cfg.num_minibatches), (5, 8))
        self.assertEqual(cfg.latent_dim, 256)
        self.assertEqual(cfg.encoder_teacher_dims, (512,))
        self.assertEqual(cfg.encoder_student_dims, (512, 512))
        self.assertEqual(cfg.actor_hidden_dims, (1024, 512, 512))
        self.assertEqual(cfg.critic_hidden_dims, (1024, 512, 512))

    def test_bf16_ddp_recipe_is_supported(self) -> None:
        cfg = PPOConfig(grad_sync_mode="ddp", train_amp_dtype="bf16")

        self.assertEqual(cfg.grad_sync_mode, "ddp")
        self.assertEqual(cfg.train_amp_dtype, "bf16")

        with self.assertRaisesRegex(ValueError, "train_amp_dtype"):
            PPOConfig(train_amp_dtype="fp8")

    def test_regularization_warms_up_between_500_and_1000(self) -> None:
        schedule = PPOROA._linear_schedule

        self.assertEqual(schedule(0, 0.0, 0.2, 500, 1000), 0.0)
        self.assertEqual(schedule(500, 0.0, 0.2, 500, 1000), 0.0)
        self.assertAlmostEqual(schedule(750, 0.0, 0.2, 500, 1000), 0.1)
        self.assertEqual(schedule(1000, 0.0, 0.2, 500, 1000), 0.2)
        self.assertEqual(schedule(4000, 0.0, 0.2, 500, 1000), 0.2)

    def test_kl_update_changes_all_policy_optimizer_groups_once(self) -> None:
        policy = object.__new__(PPOROA)
        policy.desired_kl = 0.01
        policy.cfg = SimpleNamespace(tail_kl_lr_control=True)
        policy.lr_policy = 3e-4
        policy.opt_policy = SimpleNamespace(
            param_groups=[{"lr": 3e-4}, {"lr": 3e-4}]
        )

        policy._update_policy_lr(0.01, 0.03, 2.0)

        self.assertAlmostEqual(policy.lr_policy, 2.5e-4)
        self.assertEqual(
            [group["lr"] for group in policy.opt_policy.param_groups],
            [policy.lr_policy, policy.lr_policy],
        )

        policy.lr_policy = 3e-4
        policy._update_policy_lr(0.004, 0.004, 2.0)

        self.assertAlmostEqual(policy.lr_policy, 3.6e-4)
        self.assertEqual(
            [group["lr"] for group in policy.opt_policy.param_groups],
            [policy.lr_policy, policy.lr_policy],
        )

        policy.lr_policy = 3e-4
        policy._update_policy_lr(0.004, 0.03, 2.0)
        self.assertAlmostEqual(policy.lr_policy, 2.5e-4)

        policy.lr_policy = 5.1e-7
        policy._update_policy_lr(0.01, 0.03, 2.0)
        self.assertAlmostEqual(policy.lr_policy, 4.25e-7)
        self.assertEqual(
            [group["lr"] for group in policy.opt_policy.param_groups],
            [policy.lr_policy, policy.lr_policy],
        )

    def test_distributed_update_metrics_use_one_collective(self) -> None:
        policy = object.__new__(PPOROA)
        nn.Module.__init__(policy)
        policy.cfg = SimpleNamespace(
            entropy_coef_start=0.01,
            entropy_coef_end=0.002,
            entropy_decay_start=0.75,
            entropy_decay_end=1.0,
            reg_coef=0.2,
            reg_warmup_start=500,
            reg_warmup_end=1000,
            desired_kl=0.01,
            desired_kl_end=0.005,
            kl_lr_high_threshold_ratio=2.0,
            ppo_epochs=1,
            num_minibatches=1,
            epoch_kl_early_stop=False,
            tail_kl_lr_control=True,
        )
        policy.device = torch.device("cpu")
        policy.entropy_decay_start = 3000
        policy.entropy_decay_end = 4000
        policy.critic = nn.Identity()
        policy.desired_kl = 0.01
        policy.lr_policy = 3e-4
        policy.opt_policy = SimpleNamespace(param_groups=[{"lr": 3e-4}])
        policy.reward_groups = ["tracking"]
        policy._get_current_iter = MethodType(lambda self: 3500, policy)
        policy._compute_advantage = MethodType(
            lambda self, *args, **kwargs: None,
            policy,
        )
        policy._update_ppo = MethodType(
            lambda self, minibatch: {
                "actor/kl": torch.tensor(0.01),
                "actor/clamp_ratio": torch.tensor(0.2),
            },
            policy,
        )
        batch = TensorDict(
            {
                "ret": torch.ones(2, 1, 1),
                REWARD_KEY: torch.ones(2, 1, 1),
            },
            batch_size=[2, 1],
        )

        with (
            patch("mimic_lite_learning.ppo_roa.aa.is_distributed", return_value=True),
            patch("mimic_lite_learning.ppo_roa.dist.all_reduce") as all_reduce,
            patch(
                "mimic_lite_learning.ppo_roa.make_batch",
                side_effect=lambda value, _: [value],
            ),
        ):
            policy.train_policy(batch)

        all_reduce.assert_called_once()
        self.assertEqual(tuple(all_reduce.call_args.args[0].shape), (2, 1))
        self.assertAlmostEqual(policy.entropy_coef, 0.006)
        self.assertAlmostEqual(policy.desired_kl, 0.0075)

if __name__ == "__main__":
    unittest.main()
