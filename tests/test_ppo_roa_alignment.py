from __future__ import annotations

from types import SimpleNamespace
import unittest

from mimic_lite_learning.ppo_roa import PPOConfig, PPOROA


class PPOROAAlignmentTest(unittest.TestCase):
    def test_teacher_recipe_defaults_are_aligned(self) -> None:
        cfg = PPOConfig()

        self.assertTrue(cfg.teacher_use_priv)
        self.assertEqual(cfg.opt, "muon")
        self.assertEqual(cfg.policy_lr, 3e-4)
        self.assertEqual(cfg.critic_lr, 3e-4)
        self.assertNotIn("lr", cfg.__dataclass_fields__)
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
        policy.lr_policy = 3e-4
        policy.opt_policy = SimpleNamespace(
            param_groups=[{"lr": 3e-4}, {"lr": 3e-4}]
        )

        policy._update_policy_lr(0.03)

        self.assertAlmostEqual(policy.lr_policy, 2.5e-4)
        self.assertEqual(
            [group["lr"] for group in policy.opt_policy.param_groups],
            [policy.lr_policy, policy.lr_policy],
        )

        policy.lr_policy = 3e-4
        policy._update_policy_lr(0.004)

        self.assertAlmostEqual(policy.lr_policy, 3.6e-4)
        self.assertEqual(
            [group["lr"] for group in policy.opt_policy.param_groups],
            [policy.lr_policy, policy.lr_policy],
        )

if __name__ == "__main__":
    unittest.main()
