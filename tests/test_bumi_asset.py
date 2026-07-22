from __future__ import annotations

from pathlib import Path
import re
from typing import Any, cast
import unittest

import active_adaptation as aa
from omegaconf import OmegaConf


try:
    aa.get_backend()
except RuntimeError:
    aa.set_backend("mjlab")

from active_adaptation.registry import Registry  # noqa: E402
from mimic_lite.assets.bumi import (  # noqa: E402
    BUMI_ACTION_SCALE,
    BUMI_ACTUATOR_SPECS,
    BUMI_BODY_NAMES,
    BUMI_DELAY_MAX_LAG,
    BUMI_DELAY_MIN_LAG,
    BUMI_JOINT_SYMMETRY,
    BUMI_MOTION_BODY_NAMES,
    BUMI_MOTION_JOINT_NAMES,
    BUMI_POLICY_JOINT_NAMES,
    BUMI_SPATIAL_SYMMETRY,
    BUMI_XML,
    get_bumi_spec,
    make_cfg,
)


class BumiAssetTest(unittest.TestCase):
    def test_task_applies_delay_once_and_uses_bumi_capacity(self) -> None:
        task_path = (
            Path(__file__).resolve().parents[1]
            / "cfg"
            / "task"
            / "tracking-base-bumi.yaml"
        )
        task_cfg = OmegaConf.load(task_path)
        action_cfg = task_cfg.input.action
        self.assertEqual(action_cfg.min_delay, 0)
        self.assertEqual(action_cfg.max_delay, 0)
        self.assertEqual(action_cfg.alpha, 1.0)
        self.assertEqual(action_cfg.expected_action_dim, 21)
        self.assertEqual(set(action_cfg.action_scaling), set(BUMI_ACTION_SCALE))
        for expression, scale in BUMI_ACTION_SCALE.items():
            self.assertAlmostEqual(action_cfg.action_scaling[expression], scale)

        self.assertEqual(task_cfg.robot.name, "bumi")
        self.assertEqual(task_cfg.sim.mjlab.nconmax, 128)
        self.assertEqual(task_cfg.sim.mjlab.njmax, 640)
        self.assertEqual(task_cfg.sim.mjlab.contact_sensor_maxmatch, 256)
        self.assertEqual(task_cfg.sim.mjlab.ccd_iterations, 50)

    def test_names_and_action_scale_are_complete(self) -> None:
        self.assertEqual(len(BUMI_POLICY_JOINT_NAMES), 21)
        self.assertEqual(len(set(BUMI_POLICY_JOINT_NAMES)), 21)
        self.assertEqual(len(BUMI_BODY_NAMES), 22)
        self.assertEqual(len(set(BUMI_BODY_NAMES)), 22)
        self.assertEqual(set(BUMI_MOTION_JOINT_NAMES), set(BUMI_POLICY_JOINT_NAMES))
        self.assertEqual(BUMI_MOTION_BODY_NAMES[0], "motion_root")
        self.assertEqual(BUMI_MOTION_BODY_NAMES[1:], BUMI_BODY_NAMES)

        matched = []
        for joint_name in BUMI_POLICY_JOINT_NAMES:
            matching_specs = [
                spec
                for spec in BUMI_ACTUATOR_SPECS
                if any(
                    re.fullmatch(expr, joint_name) for expr in spec.target_names_expr
                )
            ]
            self.assertEqual(len(matching_specs), 1, joint_name)
            matched.append(joint_name)
        self.assertEqual(len(matched), 21)
        self.assertEqual(
            set(BUMI_ACTION_SCALE),
            {spec.target_names_expr[0] for spec in BUMI_ACTUATOR_SPECS},
        )

    def test_symmetry_mappings_are_involutions(self) -> None:
        self.assertEqual(set(BUMI_JOINT_SYMMETRY), set(BUMI_POLICY_JOINT_NAMES))
        for joint_name, (sign, mirror_name) in BUMI_JOINT_SYMMETRY.items():
            mirror_sign, original_name = BUMI_JOINT_SYMMETRY[mirror_name]
            self.assertEqual(original_name, joint_name)
            self.assertEqual(sign * mirror_sign, 1)

        self.assertEqual(set(BUMI_SPATIAL_SYMMETRY), set(BUMI_BODY_NAMES))
        for body_name, mirror_name in BUMI_SPATIAL_SYMMETRY.items():
            self.assertEqual(BUMI_SPATIAL_SYMMETRY[mirror_name], body_name)

    def test_registry_and_backend_contract(self) -> None:
        self.assertIn("bumi", Registry.instance().list_all("asset"))
        with self.assertRaisesRegex(NotImplementedError, "MJLab-only"):
            make_cfg("isaaclab")

    @unittest.skipUnless(BUMI_XML.is_file(), "Bumi cache has not been prepared")
    def test_mjlab_config_compiles_with_actuator_delay(self) -> None:
        import mujoco

        cfg, sensors = make_cfg("mjlab")
        self.assertIsNotNone(cfg.articulation)
        actuators = cast(Any, cfg.articulation).actuators
        self.assertEqual(len(actuators), len(BUMI_ACTUATOR_SPECS))
        for actuator, spec in zip(actuators, BUMI_ACTUATOR_SPECS, strict=True):
            self.assertEqual(actuator.target_names_expr, spec.target_names_expr)
            self.assertEqual(actuator.stiffness, spec.stiffness)
            self.assertEqual(actuator.damping, spec.damping)
            self.assertEqual(actuator.effort_limit, spec.effort_limit)
            self.assertEqual(actuator.armature, spec.armature)
            self.assertEqual(actuator.frictionloss, spec.frictionloss)
            self.assertEqual(actuator.delay_min_lag, BUMI_DELAY_MIN_LAG)
            self.assertEqual(actuator.delay_max_lag, BUMI_DELAY_MAX_LAG)
        self.assertEqual(
            [sensor.name for sensor in sensors], ["contact_forces", "self_collision"]
        )

        model = get_bumi_spec().compile()
        joint_type = getattr(mujoco, "mjtJoint")
        self.assertEqual(int((model.jnt_type == joint_type.mjJNT_HINGE).sum()), 21)
        self.assertEqual(int((model.jnt_type == joint_type.mjJNT_FREE).sum()), 1)
        self.assertEqual(model.nbody - 1, 22)
        hinge_names = [
            model.joint(index).name
            for index in range(model.njnt)
            if model.jnt_type[index] == joint_type.mjJNT_HINGE
        ]
        body_names = [model.body(index).name for index in range(1, model.nbody)]
        self.assertEqual(hinge_names, BUMI_POLICY_JOINT_NAMES)
        self.assertEqual(body_names, BUMI_BODY_NAMES)


if __name__ == "__main__":
    unittest.main()
