from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import mujoco
import numpy as np

from mimic_lite_conversion.bumi import (
    BUMI_MOTION_BODY_NAMES,
    BUMI_MOTION_JOINT_NAMES,
)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "view_bumi_retarget_viser.py"
SPEC = importlib.util.spec_from_file_location("view_bumi_retarget_viser", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_motion(path: Path, *, frames: int = 3) -> None:
    body_pos = np.zeros(
        (frames, len(BUMI_MOTION_BODY_NAMES), 3),
        dtype=np.float32,
    )
    body_pos[:, 1, 2] = 0.5
    body_quat = np.zeros(
        (frames, len(BUMI_MOTION_BODY_NAMES), 4),
        dtype=np.float32,
    )
    body_quat[..., 0] = 1.0
    np.savez_compressed(
        path,
        fps=np.asarray([50], dtype=np.int32),
        joint_pos=np.zeros(
            (frames, len(BUMI_MOTION_JOINT_NAMES)),
            dtype=np.float32,
        ),
        joint_vel=np.zeros(
            (frames, len(BUMI_MOTION_JOINT_NAMES)),
            dtype=np.float32,
        ),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
    )


def _minimal_bumi_joint_model() -> mujoco.MjModel:
    nested_bodies = "".join(
        (
            f'<body name="link_{index}">'
            f'<joint name="{name}" type="hinge" axis="0 0 1"/>'
            '<geom type="sphere" size="0.01"/>'
        )
        for index, name in enumerate(BUMI_MOTION_JOINT_NAMES)
    )
    nested_bodies += "</body>" * len(BUMI_MOTION_JOINT_NAMES)
    return mujoco.MjModel.from_xml_string(
        f"""
        <mujoco>
          <worldbody>
            <body name="base_link">
              <freejoint/>
              <geom type="sphere" size="0.05"/>
              {nested_bodies}
            </body>
          </worldbody>
        </mujoco>
        """
    )


class BumiRetargetViserTest(unittest.TestCase):
    def test_catalog_is_sorted_and_quality_filterable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_motion(root / "b.npz")
            _write_motion(root / "a.npz")
            report = root / "quality.json"
            report.write_text(
                json.dumps(
                    {
                        "automatic_training_ready_clip_ids": ["a"],
                        "geometry_review_clip_ids": ["b"],
                    }
                ),
                encoding="utf-8",
            )

            all_entries = MODULE.build_catalog(
                root,
                quality_report=report,
                quality_group="all",
                recursive=False,
            )
            automatic = MODULE.build_catalog(
                root,
                quality_report=report,
                quality_group="automatic",
                recursive=False,
            )

            self.assertEqual([entry.clip_id for entry in all_entries], ["a", "b"])
            self.assertEqual([entry.clip_id for entry in automatic], ["a"])
            self.assertEqual(automatic[0].quality_group, "automatic")

    def test_load_clip_uses_tracker_joint_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "clip.npz"
            _write_motion(path, frames=4)
            entry = MODULE.ClipEntry(
                path=path,
                label=path.name,
                clip_id=path.stem,
                quality_group="unclassified",
            )

            clip = MODULE.load_tracking_clip(entry)

            self.assertEqual(clip.frame_count, 4)
            self.assertEqual(clip.fps, 50.0)
            self.assertEqual(clip.joint_names, BUMI_MOTION_JOINT_NAMES)

    def test_apply_frame_maps_root_and_named_joints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "clip.npz"
            _write_motion(path, frames=2)
            entry = MODULE.ClipEntry(
                path=path,
                label=path.name,
                clip_id=path.stem,
                quality_group="unclassified",
            )
            clip = MODULE.load_tracking_clip(entry)
            clip.body_pos_w[1, 1] = np.asarray([1.0, 2.0, 0.8])
            clip.joint_pos[1] = np.linspace(
                -0.2,
                0.2,
                len(BUMI_MOTION_JOINT_NAMES),
                dtype=np.float32,
            )
            model = _minimal_bumi_joint_model()
            data = mujoco.MjData(model)
            qpos, dof = MODULE.resolve_joint_addresses(model, clip.joint_names)

            MODULE.apply_clip_frame(
                model,
                data,
                clip,
                1,
                joint_qpos_addresses=qpos,
                joint_dof_addresses=dof,
            )

            np.testing.assert_allclose(data.qpos[:3], [1.0, 2.0, 0.8])
            np.testing.assert_allclose(data.qpos[3:7], [1.0, 0.0, 0.0, 0.0])
            for index, address in enumerate(qpos):
                self.assertAlmostEqual(
                    float(data.qpos[address]),
                    float(clip.joint_pos[1, index]),
                )

    def test_recursive_staging_clip_id_uses_parent_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clip_dir = root / "clip_id"
            clip_dir.mkdir()
            _write_motion(clip_dir / "motion.npz")

            entries = MODULE.build_catalog(
                root,
                quality_report=None,
                quality_group="all",
                recursive=True,
            )

            self.assertEqual(entries[0].clip_id, "clip_id")
            self.assertEqual(entries[0].label, "clip_id/motion.npz")


if __name__ == "__main__":
    unittest.main()
