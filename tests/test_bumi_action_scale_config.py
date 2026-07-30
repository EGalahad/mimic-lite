from __future__ import annotations

from pathlib import Path
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


class BumiActionScaleConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config_root = str(Path(__file__).resolve().parents[1] / "cfg")

    def compose_task(self, task: str, *overrides: str) -> DictConfig:
        with initialize_config_dir(
            version_base=None,
            config_dir=self.config_root,
        ):
            return compose(
                config_name=f"task/{task}",
                overrides=list(overrides),
            )

    def test_twist2_profile_resolves_jointwise_upper_body_scales(self) -> None:
        cfg = self.compose_task("tracking-base-bumi-twist2")
        OmegaConf.resolve(cfg)

        self.assertEqual(cfg.task.arm_pitch_action_scale, 0.88)
        self.assertEqual(cfg.task.arm_roll_action_scale, 0.46)
        self.assertEqual(cfg.task.arm_yaw_action_scale, 0.53)
        self.assertEqual(cfg.task.elbow_pitch_action_scale, 0.64)
        self.assertEqual(
            OmegaConf.to_container(
                cfg.task.input.action.action_scaling,
                resolve=True,
            ),
            {
                ".*_leg_pitch_joint": 0.25,
                "waist_yaw_joint": 0.12735849,
                ".*_leg_roll_joint": 0.1125,
                ".*_arm_pitch_joint": 0.88,
                ".*_leg_yaw_joint": 0.1125,
                ".*_arm_roll_joint": 0.46,
                ".*_knee_pitch_joint": 0.33333333,
                ".*_arm_yaw_joint": 0.53,
                ".*_ankle_pitch_joint": 0.325,
                ".*_elbow_pitch_joint": 0.64,
                ".*_ankle_roll_joint": 0.325,
            },
        )

    def test_base_profile_keeps_uniform_override_compatibility(self) -> None:
        default_cfg = self.compose_task("tracking-base-bumi")
        OmegaConf.resolve(default_cfg)
        default_scaling = default_cfg.task.input.action.action_scaling
        for joint_pattern in (
            ".*_arm_pitch_joint",
            ".*_arm_roll_joint",
            ".*_arm_yaw_joint",
            ".*_elbow_pitch_joint",
        ):
            self.assertEqual(default_scaling[joint_pattern], 0.11458333)

        cfg = self.compose_task(
            "tracking-base-bumi",
            "task.arm_action_scale=0.45833332",
        )
        OmegaConf.resolve(cfg)

        action_scaling = cfg.task.input.action.action_scaling
        self.assertEqual(action_scaling[".*_arm_pitch_joint"], 0.45833332)
        self.assertEqual(action_scaling[".*_arm_roll_joint"], 0.45833332)
        self.assertEqual(action_scaling[".*_arm_yaw_joint"], 0.45833332)
        self.assertEqual(action_scaling[".*_elbow_pitch_joint"], 0.45833332)


if __name__ == "__main__":
    unittest.main()
