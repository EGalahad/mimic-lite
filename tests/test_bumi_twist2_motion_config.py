from __future__ import annotations

from pathlib import Path
import unittest

from omegaconf import OmegaConf


class BumiTwist2MotionConfigTest(unittest.TestCase):
    def test_training_dataset_contract(self) -> None:
        config_root = (
            Path(__file__).resolve().parents[1]
            / "cfg"
            / "task"
            / "motion"
            / "bumi"
        )
        twist2_cfg = OmegaConf.load(config_root / "twist2_robot_retarget.yaml")
        amass_gmr_cfg = OmegaConf.load(config_root / "amass_gmr.yaml")

        self.assertEqual(twist2_cfg.windowed_next_window_device, "current")
        self.assertIs(twist2_cfg.windowed_pin_window_load, True)
        self.assertEqual(
            twist2_cfg.windowed_next_window_device,
            amass_gmr_cfg.windowed_next_window_device,
        )
        self.assertEqual(
            twist2_cfg.windowed_pin_window_load,
            amass_gmr_cfg.windowed_pin_window_load,
        )
        self.assertEqual(
            OmegaConf.to_container(twist2_cfg.motion_cfgs, resolve=True),
            {
                "bumi_twist2_robot_retarget": {
                    "path": (
                        ".cache/mimic-lite/motions/bumi/"
                        "twist2_robot_retarget"
                    ),
                    "weight": 1.0,
                    "full_motion": False,
                    "shard": False,
                }
            },
        )


if __name__ == "__main__":
    unittest.main()
