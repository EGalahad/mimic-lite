from types import SimpleNamespace
import unittest

from mimic_lite_learning.ppo import PPOConfig
from mimic_lite_learning.ppo_roa import PPOConfig as PPOROAConfig
from mimic_lite_learning.sac import SAC, SACConfig


class FractionalDecayScheduleTest(unittest.TestCase):
    def test_ppo_rejects_legacy_iteration_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "decay fractions"):
            PPOConfig(entropy_decay_start=3000, entropy_decay_end=4000)
        with self.assertRaisesRegex(ValueError, "decay fractions"):
            PPOROAConfig(entropy_decay_start=3000, entropy_decay_end=4000)

    def test_sac_target_entropy_uses_fractional_bounds(self) -> None:
        policy = object.__new__(SAC)
        policy.total_iters = 5000
        policy.cfg = SimpleNamespace(
            target_entropy_sigma=0.4,
            target_entropy_sigma_start=0.4,
            target_entropy_sigma_end=0.25,
            target_entropy_decay_start=0.6,
            target_entropy_decay_end=0.8,
        )

        self.assertAlmostEqual(policy._scheduled_target_entropy_sigma(3000), 0.4)
        self.assertAlmostEqual(policy._scheduled_target_entropy_sigma(3500), 0.325)
        self.assertAlmostEqual(policy._scheduled_target_entropy_sigma(4000), 0.25)

    def test_sac_rejects_legacy_iteration_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "decay fractions"):
            SACConfig(
                target_entropy_decay_start=3000,
                target_entropy_decay_end=4000,
            )


if __name__ == "__main__":
    unittest.main()
