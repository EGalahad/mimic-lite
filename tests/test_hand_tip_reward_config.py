"""CPU-only check that hand-tip rewards do not change observation/termination bodies."""
from pathlib import Path
import re
import unittest

from hydra import compose, initialize_config_dir


class HandTipRewardConfigTest(unittest.TestCase):
    def test_reward_landmarks_are_separate_from_observation_and_termination(self):
        cfg_dir = Path(__file__).resolve().parents[1] / "cfg"
        original = ["pelvis", "torso_link"] + [
            f"{side}_{part}_link"
            for part in ("hip_yaw", "knee", "toe", "shoulder_yaw", "elbow", "wrist_yaw")
            for side in ("left", "right")
        ]
        tips = ["left_palm_link", "right_palm_link"]
        available = original + tips

        def select(patterns):
            return [name for name in available if any(re.fullmatch(p, name) for p in patterns)]

        with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
            for extra in ([], ["+task/patches=teacher_future_t16"],
                          ["task/observation/priv=full"]):
                with self.subTest(overrides=extra):
                    task = compose(overrides=["+task=tracking-base", *extra]).task
                    self.assertEqual(select(task.shared.tracking_body_names), original)
                    self.assertEqual(select(task.command.tracking_body_names), available)
                    expected = [n for n in original if "wrist_yaw" not in n] + tips
                    self.assertEqual(select(task.shared.reward_body_names), expected)
                    for name in ("body_pos", "body_ori", "body_linvel", "body_angvel"):
                        self.assertEqual(select(task.reward.tracking[name].body_names), expected)
                    for name in ("body_pos", "body_ori"):
                        self.assertEqual(select(task.reward.tracking_metrics[name].body_names), expected)
                    for name, term in task.observation.priv.items():
                        if name.startswith("diff_body_") or name == "body_spatial_error_local":
                            self.assertEqual(select(term.body_names), original)
                    self.assertFalse(set(select(task.command.obs_body_names)) & set(tips))
                    for term in task.termination.values():
                        names = term.get("body_names", [])
                        if not isinstance(names, str):
                            self.assertFalse(set(select(names)) & set(tips))
                    self.assertTrue(task.termination.root_pos_error.is_timeout)


if __name__ == "__main__":
    unittest.main()
