from __future__ import annotations

from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from tensordict import TensorDict

from active_adaptation.learning.ppo.common import REWARD_KEY
from mimic_lite_learning.common import check_vecnorm_divergence
from mimic_lite_learning.ppo import PPOConfig, PPOPolicy


class PPOKLControlTest(unittest.TestCase):
    @staticmethod
    def _policy(current_iter: int) -> PPOPolicy:
        policy = object.__new__(PPOPolicy)
        nn.Module.__init__(policy)
        policy.cfg = SimpleNamespace(
            entropy_decay_start=500,
            entropy_decay_end=3500,
            entropy_coef_start=0.008,
            entropy_coef_end=0.002,
            desired_kl=0.01,
            desired_kl_end=None,
            ppo_epochs=1,
            num_minibatches=1,
            tail_kl_lr_control=False,
            kl_lr_high_threshold_ratio=2.0,
            epoch_kl_early_stop=False,
        )
        policy.device = torch.device("cpu")
        policy.critic = nn.Identity()
        policy.desired_kl = 0.01
        policy.lr_policy = 5.8142009840344475e-5
        policy.opt_policy = SimpleNamespace(
            param_groups=[{"lr": policy.lr_policy}]
        )
        policy.reward_groups = ["tracking"]
        policy._get_current_iter = MethodType(
            lambda self: current_iter,
            policy,
        )
        policy._compute_advantage = MethodType(
            lambda self, *args, **kwargs: None,
            policy,
        )
        policy.update_ppo = MethodType(
            lambda self, minibatch: {
                "actor/kl": torch.tensor(0.01),
                "actor/clamp_ratio": torch.tensor(0.2),
            },
            policy,
        )
        return policy

    @staticmethod
    def _batch() -> TensorDict:
        return TensorDict(
            {
                "ret": torch.ones(2, 1, 1),
                REWARD_KEY: torch.ones(2, 1, 1),
            },
            batch_size=[2, 1],
        )

    def test_defaults_enable_tail_control_without_cap_or_early_stop(self) -> None:
        cfg = PPOConfig()
        self.assertTrue(cfg.tail_kl_lr_control)
        self.assertIsNone(cfg.desired_kl_end)
        self.assertEqual(cfg.kl_lr_high_threshold_ratio, 2.0)
        self.assertFalse(cfg.epoch_kl_early_stop)

    def test_kl_lr_high_threshold_ratio_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            PPOConfig(kl_lr_high_threshold_ratio=0)

    def test_desired_kl_end_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            PPOConfig(desired_kl_end=0)

    def test_kl_high_threshold_ratio_stays_fixed(self) -> None:
        with patch(
            "mimic_lite_learning.ppo.make_batch",
            side_effect=lambda batch, _: [batch],
        ):
            before = self._policy(500).train_policy(self._batch())
            middle = self._policy(2000).train_policy(self._batch())
            after = self._policy(3500).train_policy(self._batch())

        self.assertAlmostEqual(before["actor/kl_high_threshold_ratio"], 2.0)
        self.assertAlmostEqual(middle["actor/kl_high_threshold_ratio"], 2.0)
        self.assertAlmostEqual(after["actor/kl_high_threshold_ratio"], 2.0)

    def test_desired_kl_decays_with_entropy_schedule(self) -> None:
        with patch(
            "mimic_lite_learning.ppo.make_batch",
            side_effect=lambda batch, _: [batch],
        ):
            before_policy = self._policy(500)
            before_policy.cfg.desired_kl_end = 0.003
            before = before_policy.train_policy(self._batch())
            middle_policy = self._policy(2000)
            middle_policy.cfg.desired_kl_end = 0.003
            middle = middle_policy.train_policy(self._batch())
            after_policy = self._policy(3500)
            after_policy.cfg.desired_kl_end = 0.003
            after = after_policy.train_policy(self._batch())

        self.assertAlmostEqual(before["actor/desired_kl"], 0.01)
        self.assertAlmostEqual(middle["actor/desired_kl"], 0.0065)
        self.assertAlmostEqual(after["actor/desired_kl"], 0.003)

    def test_update_level_kl_diagnostics_do_not_change_mean_controller(self) -> None:
        policy = self._policy(100)
        updates = iter(
            [
                {
                    "actor/kl": torch.tensor(0.004),
                    "actor/clamp_ratio": torch.tensor(0.1),
                },
                {
                    "actor/kl": torch.tensor(0.016),
                    "actor/clamp_ratio": torch.tensor(0.3),
                },
            ]
        )
        policy.cfg.ppo_epochs = 1
        policy.cfg.num_minibatches = 2
        policy.update_ppo = MethodType(
            lambda self, minibatch: next(updates),
            policy,
        )

        with patch(
            "mimic_lite_learning.ppo.make_batch",
            side_effect=lambda batch, _: [batch, batch],
        ):
            info = policy.train_policy(self._batch())

        self.assertAlmostEqual(info["actor/kl"], 0.01)
        self.assertAlmostEqual(info["actor/kl_update_max"], 0.016)
        self.assertAlmostEqual(info["actor/kl_update_last"], 0.016)
        self.assertAlmostEqual(info["actor/clamp_ratio_update_max"], 0.3)
        self.assertAlmostEqual(policy.lr_policy, 5.8142009840344475e-5)

    def test_tail_controller_reduces_lr_before_mean_crosses_threshold(self) -> None:
        policy = self._policy(100)
        policy.cfg.tail_kl_lr_control = True
        updates = iter(
            [
                {
                    "actor/kl": torch.tensor(0.010),
                    "actor/clamp_ratio": torch.tensor(0.1),
                },
                {
                    "actor/kl": torch.tensor(0.026),
                    "actor/clamp_ratio": torch.tensor(0.3),
                },
            ]
        )
        policy.cfg.num_minibatches = 2
        policy.update_ppo = MethodType(
            lambda self, minibatch: next(updates),
            policy,
        )

        with patch(
            "mimic_lite_learning.ppo.make_batch",
            side_effect=lambda batch, _: [batch, batch],
        ):
            info = policy.train_policy(self._batch())

        self.assertLess(info["actor/kl"], 0.02)
        self.assertGreater(info["actor/kl_update_p90"], 0.02)
        self.assertAlmostEqual(
            policy.lr_policy,
            5.8142009840344475e-5 / 1.2,
        )

    def test_distributed_update_metrics_use_one_collective(self) -> None:
        policy = self._policy(100)

        with (
            patch("mimic_lite_learning.ppo.aa.is_distributed", return_value=True),
            patch("mimic_lite_learning.ppo.dist.all_reduce") as all_reduce,
            patch(
                "mimic_lite_learning.ppo.make_batch",
                side_effect=lambda batch, _: [batch],
            ),
        ):
            policy.train_policy(self._batch())

        all_reduce.assert_called_once()
        self.assertEqual(tuple(all_reduce.call_args.args[0].shape), (2, 1))

    def test_vecnorm_divergence_gathers_loc_and_scale_together(self) -> None:
        vecnorm = SimpleNamespace(
            _compute=lambda: (torch.zeros(3), torch.ones(3))
        )

        def all_gather(outputs, value):
            outputs[0].copy_(value)
            outputs[1].copy_(value)
            outputs[1][0].add_(1.0)
            outputs[1][1].add_(2.0)

        with (
            patch("mimic_lite_learning.common.aa.get_world_size", return_value=2),
            patch(
                "mimic_lite_learning.common.dist.all_gather",
                side_effect=all_gather,
            ) as gather,
        ):
            loc_diffs, scale_diffs = check_vecnorm_divergence(vecnorm)

        gather.assert_called_once()
        self.assertEqual(loc_diffs, [0.0, 3.0])
        self.assertEqual(scale_diffs, [0.0, 6.0])

    def test_tail_controller_supports_tighter_high_threshold(self) -> None:
        policy = self._policy(100)
        policy.cfg.tail_kl_lr_control = True
        policy.cfg.kl_lr_high_threshold_ratio = 1.5
        updates = iter(
            [
                {
                    "actor/kl": torch.tensor(0.010),
                    "actor/clamp_ratio": torch.tensor(0.1),
                },
                {
                    "actor/kl": torch.tensor(0.017),
                    "actor/clamp_ratio": torch.tensor(0.2),
                },
            ]
        )
        policy.cfg.num_minibatches = 2
        policy.update_ppo = MethodType(
            lambda self, minibatch: next(updates),
            policy,
        )

        with patch(
            "mimic_lite_learning.ppo.make_batch",
            side_effect=lambda batch, _: [batch, batch],
        ):
            info = policy.train_policy(self._batch())

        self.assertLess(info["actor/kl"], 0.02)
        self.assertGreater(info["actor/kl_update_p90"], 0.015)
        self.assertLess(info["actor/kl_update_p90"], 0.02)
        self.assertAlmostEqual(
            policy.lr_policy,
            5.8142009840344475e-5 / 1.2,
        )

    def test_tail_controller_lr_floor_is_5e_7(self) -> None:
        policy = self._policy(100)
        policy.cfg.tail_kl_lr_control = True
        policy.lr_policy = 5.1e-7
        policy.opt_policy.param_groups[0]["lr"] = policy.lr_policy
        policy.update_ppo = MethodType(
            lambda self, minibatch: {
                "actor/kl": torch.tensor(0.03),
                "actor/clamp_ratio": torch.tensor(0.3),
            },
            policy,
        )

        with patch(
            "mimic_lite_learning.ppo.make_batch",
            side_effect=lambda batch, _: [batch],
        ):
            policy.train_policy(self._batch())

        self.assertEqual(policy.lr_policy, 5e-7)
        self.assertEqual(policy.opt_policy.param_groups[0]["lr"], 5e-7)

    def test_tail_controller_requires_low_tail_before_increasing_lr(self) -> None:
        policy = self._policy(100)
        policy.cfg.tail_kl_lr_control = True
        updates = iter(
            [
                {
                    "actor/kl": torch.tensor(0.0),
                    "actor/clamp_ratio": torch.tensor(0.1),
                },
                {
                    "actor/kl": torch.tensor(0.0),
                    "actor/clamp_ratio": torch.tensor(0.1),
                },
                {
                    "actor/kl": torch.tensor(0.0),
                    "actor/clamp_ratio": torch.tensor(0.1),
                },
                {
                    "actor/kl": torch.tensor(0.0),
                    "actor/clamp_ratio": torch.tensor(0.1),
                },
                {
                    "actor/kl": torch.tensor(0.0),
                    "actor/clamp_ratio": torch.tensor(0.1),
                },
                {
                    "actor/kl": torch.tensor(0.0),
                    "actor/clamp_ratio": torch.tensor(0.1),
                },
                {
                    "actor/kl": torch.tensor(0.011),
                    "actor/clamp_ratio": torch.tensor(0.2),
                },
                {
                    "actor/kl": torch.tensor(0.011),
                    "actor/clamp_ratio": torch.tensor(0.2),
                },
            ]
        )
        policy.cfg.num_minibatches = 8
        policy.update_ppo = MethodType(
            lambda self, minibatch: next(updates),
            policy,
        )

        with patch(
            "mimic_lite_learning.ppo.make_batch",
            side_effect=lambda batch, _: [batch] * 8,
        ):
            info = policy.train_policy(self._batch())

        self.assertLess(info["actor/kl"], 0.005)
        self.assertGreater(info["actor/kl_update_p90"], 0.005)
        self.assertAlmostEqual(policy.lr_policy, 5.8142009840344475e-5)

    def test_epoch_tail_kl_stops_remaining_epochs(self) -> None:
        policy = self._policy(100)
        policy.cfg.epoch_kl_early_stop = True
        policy.cfg.kl_lr_high_threshold_ratio = 1.5
        policy.cfg.ppo_epochs = 5
        policy.cfg.num_minibatches = 2
        updates = iter(
            [
                {
                    "actor/kl": torch.tensor(0.016),
                    "actor/clamp_ratio": torch.tensor(0.2),
                },
                {
                    "actor/kl": torch.tensor(0.017),
                    "actor/clamp_ratio": torch.tensor(0.3),
                },
            ]
        )
        policy.update_ppo = MethodType(
            lambda self, minibatch: next(updates),
            policy,
        )

        with patch(
            "mimic_lite_learning.ppo.make_batch",
            side_effect=lambda batch, _: [batch, batch],
        ):
            info = policy.train_policy(self._batch())

        self.assertEqual(info["actor/epochs_completed"], 1)
        self.assertEqual(info["actor/kl_early_stop"], 1.0)
        self.assertGreater(info["actor/kl_epoch_last_p90"], 0.015)
        self.assertLess(info["actor/kl_epoch_last_p90"], 0.02)


if __name__ == "__main__":
    unittest.main()
