from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from mimic_lite_conversion.amass_gmr import sha256_file


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "report_bumi_retarget_quality.py"
)
SPEC = importlib.util.spec_from_file_location("report_bumi_retarget_quality", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BumiRetargetQualityTest(unittest.TestCase):
    def _write_record(
        self,
        root: Path,
        *,
        joint_velocity: float,
        joint_index: int = 0,
        geometry_pass: bool = True,
    ) -> None:
        tracker = root / "tracker_50hz" / "clip.npz"
        tracker.parent.mkdir(parents=True, exist_ok=True)
        body_quat = np.zeros((3, 23, 4), dtype=np.float32)
        body_quat[..., 0] = 1.0
        joint_vel = np.zeros((3, 21), dtype=np.float32)
        joint_vel[:, joint_index] = joint_velocity
        np.savez_compressed(
            tracker,
            joint_pos=np.zeros((3, 21), dtype=np.float32),
            joint_vel=joint_vel,
            body_pos_w=np.zeros((3, 23, 3), dtype=np.float32),
            body_quat_w=body_quat,
            body_lin_vel_w=np.zeros((3, 23, 3), dtype=np.float32),
            body_ang_vel_w=np.zeros((3, 23, 3), dtype=np.float32),
        )
        metadata = root / "metadata" / "clip.json"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(
            json.dumps(
                {
                    "tracker_sha256": sha256_file(tracker),
                    "result": {
                        "clip_id": "clip",
                        "source_relative_path": "D/S/M.npz",
                        "source_fps": 120.0,
                        "gate_position_pass": geometry_pass,
                        "gate_foot_position_pass": geometry_pass,
                        "gate_orientation_pass": geometry_pass,
                    },
                }
            )
        )
        reports = root / "reports"
        reports.mkdir(exist_ok=True)
        (reports / "rejected.jsonl").write_text("")

    def test_production_gate_uses_joint_family_velocity_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_record(root, joint_velocity=12.0)
            report = MODULE.build_report(root)
            self.assertTrue(report["pipeline_integrity_ready"])
            self.assertTrue(report["production_ready"])
            self.assertEqual(report["max_joint_velocity_ratio"], 1.0)
            self.assertEqual(report["automatic_training_ready"], 1)

            self._write_record(root, joint_velocity=13.0)
            report = MODULE.build_report(root)
            self.assertFalse(report["pipeline_integrity_ready"])
            self.assertFalse(report["production_ready"])
            self.assertGreater(report["max_joint_velocity_ratio"], 1.05)
            self.assertEqual(report["integrity_failures"], 1)

    def test_soft_review_sets_are_separate_from_pipeline_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # Motion joint index 5 is an arm with a 50 rad/s physical limit,
            # but 25 rad/s exceeds the conservative v1 training gate.
            self._write_record(root, joint_velocity=25.0, joint_index=5)
            report = MODULE.build_report(root)
            self.assertTrue(report["pipeline_integrity_ready"])
            self.assertFalse(report["production_ready"])
            self.assertEqual(report["dynamics_review"], 1)

            self._write_record(root, joint_velocity=0.0, geometry_pass=False)
            report = MODULE.build_report(root)
            self.assertTrue(report["pipeline_integrity_ready"])
            self.assertEqual(report["geometry_review"], 1)
            self.assertEqual(report["automatic_training_ready"], 0)

    def test_current_batch_report_hides_stale_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_record(root, joint_velocity=0.0)
            (root / "reports" / "converted.jsonl").write_text("")
            report = MODULE.build_report(root)
            self.assertEqual(report["succeeded"], 0)
            self.assertEqual(report["stale_metadata_clip_ids"], ["clip"])


if __name__ == "__main__":
    unittest.main()
