from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import mujoco
import numpy as np
from mjhub import resolve_asset_reference


H2_XML = resolve_asset_reference(
    "hf://elijahgalahad/h2_model@beb532e8717b99816b93baace0c649a599538715/h2.xml"
)


def _converter_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "convert_h2_dataset.py"
    spec = importlib.util.spec_from_file_location("convert_h2_dataset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class H2ConvertDatasetTest(unittest.TestCase):
    def test_visual_meshes_and_collision_primitives_are_separate(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(H2_XML))
        collision_geom_ids = [
            geom_id
            for geom_id in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").endswith(
                "_collision"
            )
        ]
        visual_geom_ids = [
            geom_id
            for geom_id in range(model.ngeom)
            if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
        ]
        self.assertEqual(len(collision_geom_ids), 33)
        self.assertEqual(len(visual_geom_ids), 32)
        self.assertTrue(np.all(model.geom_group[collision_geom_ids] == 3))
        self.assertTrue(np.all(model.geom_group[visual_geom_ids] == 2))
        self.assertTrue(np.all(model.geom_contype[visual_geom_ids] == 0))
        self.assertTrue(np.all(model.geom_conaffinity[visual_geom_ids] == 0))
        self.assertTrue(np.all(model.geom_rgba[collision_geom_ids, 3] > 0.0))
        for side in ("left", "right"):
            ankle_pitch_body = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_pitch_link"
            )
            foot_geom_ids = [
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot{i}_collision"
                )
                for i in range(1, 8)
            ]
            self.assertTrue(np.all(model.geom_bodyid[foot_geom_ids] == ankle_pitch_body))

    def test_converts_sonic_joblib_to_portable_any4hdmi(self) -> None:
        converter = _converter_module()
        model = mujoco.MjModel.from_xml_path(str(H2_XML))
        frames = 3
        root_pos = np.zeros((frames, 3), dtype=np.float32)
        root_pos[:, 2] = 1.2
        root_rot = np.tile([0.0, 0.0, 0.0, 2.0], (frames, 1)).astype(np.float32)
        dof = np.linspace(0.0, 0.3, frames * 31, dtype=np.float32).reshape(frames, 31)
        pose_aa = np.zeros((frames, model.nbody - 1, 3), dtype=np.float32)
        pose_aa[:, 1:] = dof[:, :, None] * model.jnt_axis[None, 1:]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "motion.pkl"
            output = root / "dataset"
            joblib.dump(
                {
                    "motion": {
                        "root_trans_offset": root_pos,
                        "root_rot": root_rot,
                        "dof": dof,
                        "pose_aa": pose_aa,
                        "fps": 30,
                    }
                },
                source,
            )
            summary = converter.convert_h2_dataset(
                source_root=source,
                out_dir=output,
                mjcf_path=H2_XML,
                target_fps=50,
            )

            with np.load(output / "motions" / "motion.npz", allow_pickle=False) as archive:
                qpos = archive["qpos"]
            self.assertEqual(summary["total_frames"], 4)
            self.assertEqual(qpos.shape, (4, 38))
            np.testing.assert_allclose(qpos[:, 3:7], [[1.0, 0.0, 0.0, 0.0]] * 4)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["timestep"], 1.0 / 50.0)
            self.assertEqual(manifest["qpos_dim"], 38)
            self.assertEqual(manifest["mjcf"], "mjcf/h2.xml")
            self.assertEqual(manifest["source"]["source_revision"], converter.SOURCE_REVISION)


if __name__ == "__main__":
    unittest.main()
