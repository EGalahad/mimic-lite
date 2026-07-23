from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import active_adaptation as aa
import numpy as np


try:
    aa.get_backend()
except RuntimeError:
    aa.set_backend("mjlab")

from mimic_lite.assets.bumi import (  # noqa: E402
    BUMI_MOTION_BODY_NAMES,
    BUMI_MOTION_JOINT_NAMES,
)

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "prepare_bumi_motion_dataset.py"
)
SPEC = importlib.util.spec_from_file_location(
    "prepare_bumi_motion_dataset", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_motion(path: Path, *, fps: int = 50, frames: int = 3) -> None:
    body_quat = np.zeros((frames, 23, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    np.savez_compressed(
        path,
        fps=np.array([fps], dtype=np.int32),
        joint_pos=np.zeros((frames, 21), dtype=np.float32),
        joint_vel=np.zeros((frames, 21), dtype=np.float32),
        body_pos_w=np.zeros((frames, 23, 3), dtype=np.float32),
        body_quat_w=body_quat,
        body_lin_vel_w=np.zeros((frames, 23, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((frames, 23, 3), dtype=np.float32),
    )


class BumiMotionDatasetTest(unittest.TestCase):
    def test_metadata_names_match_asset_adapter(self) -> None:
        self.assertEqual(MODULE.BUMI_MOTION_JOINT_NAMES, BUMI_MOTION_JOINT_NAMES)
        self.assertEqual(MODULE.BUMI_MOTION_BODY_NAMES, BUMI_MOTION_BODY_NAMES)

    def test_symlink_and_copy_layout(self) -> None:
        for link_mode in ("symlink", "copy"):
            with (
                self.subTest(link_mode=link_mode),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                root = Path(temp_dir)
                source = root / "source"
                source.mkdir()
                _write_motion(source / "walk.npz")
                output = root / "output"

                result = MODULE.prepare_bumi_motion_dataset(
                    source,
                    output,
                    link_mode=link_mode,
                    report_json=root / "report.json",
                )

                motion_path = output / "walk" / "motion.npz"
                self.assertEqual(result["motions"], 1)
                self.assertEqual(result["frames"], 3)
                self.assertEqual(motion_path.is_symlink(), link_mode == "symlink")
                meta = json.loads((output / "walk" / "meta.json").read_text())
                self.assertEqual(meta["fps"], 50)
                self.assertEqual(len(meta["joint_names"]), 21)
                self.assertEqual(len(meta["body_names"]), 23)
                report = json.loads((root / "report.json").read_text())
                self.assertEqual(report["max_motion_root_pos_abs"], 0.0)
                self.assertEqual(report["max_motion_root_quat_error"], 0.0)

    def test_missing_field_and_bad_shapes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing = root / "missing.npz"
            np.savez_compressed(missing, fps=np.array([50]))
            with self.assertRaisesRegex(ValueError, "missing fields"):
                MODULE.validate_motion(missing)

            bad_fps = root / "bad_fps.npz"
            _write_motion(bad_fps, fps=60)
            with self.assertRaisesRegex(ValueError, "fps=50"):
                MODULE.validate_motion(bad_fps)

            empty = root / "empty.npz"
            _write_motion(empty, frames=0)
            with self.assertRaisesRegex(ValueError, "at least one frame"):
                MODULE.validate_motion(empty)

    def test_non_finite_and_bad_quaternion_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "bad.npz"
            _write_motion(path)
            with np.load(path) as payload:
                arrays = {key: payload[key] for key in payload.files}
            arrays["joint_pos"] = arrays["joint_pos"].copy()
            arrays["joint_pos"][0, 0] = np.nan
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                MODULE.validate_motion(path)

            _write_motion(path)
            with np.load(path) as payload:
                arrays = {key: payload[key] for key in payload.files}
            arrays["body_quat_w"] = arrays["body_quat_w"].copy()
            arrays["body_quat_w"][0, 0] = 0.0
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "quat"):
                MODULE.validate_motion(path)

    def test_joint_positions_use_per_joint_mjcf_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "position_limits.npz"
            _write_motion(path)
            with np.load(path) as payload:
                arrays = {key: payload[key] for key in payload.files}
            arrays["joint_pos"] = arrays["joint_pos"].copy()
            # l_arm_pitch allows [-3.14, 1.57], so a valid value below the old
            # blanket -3 rad limit must pass.
            arrays["joint_pos"][0, 5] = -3.1
            np.savez_compressed(path, **arrays)
            MODULE.validate_motion(path)

            arrays["joint_pos"][0, 5] = -3.15
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "l_arm_pitch_joint"):
                MODULE.validate_motion(path)

    def test_different_output_requires_force(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            source.mkdir()
            _write_motion(source / "walk.npz")
            output = root / "output"
            MODULE.prepare_bumi_motion_dataset(source, output, link_mode="copy")
            (output / "walk" / "meta.json").write_text("{}\n")

            with self.assertRaisesRegex(FileExistsError, "--force"):
                MODULE.prepare_bumi_motion_dataset(source, output, link_mode="copy")

            result = MODULE.prepare_bumi_motion_dataset(
                source, output, link_mode="copy", force=True
            )
            self.assertEqual(result["status"], "replaced")

    def test_quality_report_selects_only_automatic_training_ready_clips(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            conversion = root / "conversion"
            source = conversion / "tracker_50hz"
            source.mkdir(parents=True)
            _write_motion(source / "accepted.npz")
            _write_motion(source / "review.npz")
            quality_report = conversion / "reports" / "quality_summary.json"
            quality_report.parent.mkdir()
            quality_report.write_text(
                json.dumps(
                    {
                        "conversion_root": str(conversion.resolve()),
                        "pipeline_integrity_ready": True,
                        "automatic_training_ready_clip_ids": ["accepted"],
                    }
                )
            )

            output = root / "output"
            result = MODULE.prepare_bumi_motion_dataset(
                source,
                output,
                quality_report=quality_report,
            )

            self.assertEqual(result["motions"], 1)
            self.assertEqual(result["selection"], "automatic_training_ready_clip_ids")
            self.assertTrue((output / "accepted" / "motion.npz").is_symlink())
            self.assertFalse((output / "review").exists())

            quality_report.write_text(
                json.dumps(
                    {
                        "conversion_root": str(conversion.resolve()),
                        "pipeline_integrity_ready": False,
                        "automatic_training_ready_clip_ids": ["accepted"],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "pipeline_integrity_ready"):
                MODULE.prepare_bumi_motion_dataset(
                    source,
                    root / "blocked-output",
                    quality_report=quality_report,
                )


if __name__ == "__main__":
    unittest.main()
