from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from mimic_lite_conversion.bumi import (
    BUMI_BODY_NAMES,
    BUMI_MOTION_JOINT_NAMES,
    export_bumi_tracking_npz,
    nominal_bumi_qpos,
    resample_bumi_qpos,
)


PREPARE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_bumi_motion_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepare_bumi_motion_dataset_export_test", PREPARE_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PREPARE_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREPARE_MODULE
SPEC.loader.exec_module(PREPARE_MODULE)


class BumiAmassExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_path = (
            Path(__file__).resolve().parents[3]
            / ".cache"
            / "aa-robot-models"
            / "bumi"
            / "bumi.xml"
        )
        if not cls.model_path.is_file():
            raise unittest.SkipTest(f"Bumi model cache not found: {cls.model_path}")
        cls.model = mujoco.MjModel.from_xml_path(str(cls.model_path))

    def _native_motion(self) -> tuple[np.ndarray, np.ndarray]:
        timestamps = np.arange(13, dtype=np.float64) / 120.0
        qpos = np.repeat(nominal_bumi_qpos(self.model)[None, :], len(timestamps), axis=0)
        qpos[:, 0] = timestamps
        yaw = 0.5 * timestamps
        quat_xyzw = R.from_rotvec(
            np.stack((np.zeros_like(yaw), np.zeros_like(yaw), yaw), axis=-1)
        ).as_quat()
        qpos[:, 3:7] = quat_xyzw[:, [3, 0, 1, 2]]
        qpos[1::2, 3:7] *= -1.0
        waist_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "waist_yaw_joint",
        )
        qpos[:, self.model.jnt_qposadr[waist_id]] = 0.1 * timestamps
        return timestamps, qpos

    def test_resample_uses_source_duration_and_single_50hz_grid(self) -> None:
        timestamps, qpos = self._native_motion()
        target_times, target_qpos = resample_bumi_qpos(
            self.model,
            timestamps,
            qpos,
            target_fps=50.0,
        )
        np.testing.assert_allclose(target_times, np.arange(6) / 50.0)
        np.testing.assert_allclose(target_qpos[:, 0], target_times, atol=1.0e-10)
        yaw = R.from_quat(target_qpos[:, [4, 5, 6, 3]]).as_euler("xyz")[:, 2]
        np.testing.assert_allclose(yaw, 0.5 * target_times, atol=1.0e-8)
        np.testing.assert_allclose(np.linalg.norm(target_qpos[:, 3:7], axis=-1), 1.0)

    def test_export_matches_tracker_schema_and_recomputed_fk(self) -> None:
        timestamps, qpos = self._native_motion()
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "motion.npz"
            stats = export_bumi_tracking_npz(
                output,
                model_path=self.model_path,
                source_timestamps_s=timestamps,
                source_qpos=qpos,
                target_fps=50.0,
            )
            PREPARE_MODULE.validate_motion(output)
            with np.load(output, allow_pickle=False) as payload:
                arrays = {name: np.asarray(payload[name]) for name in payload.files}

        self.assertEqual(stats["frames"], 6)
        self.assertEqual(arrays["joint_pos"].shape, (6, 21))
        self.assertEqual(arrays["body_pos_w"].shape, (6, 23, 3))
        np.testing.assert_array_equal(arrays["body_pos_w"][:, 0], 0.0)
        np.testing.assert_array_equal(
            arrays["body_quat_w"][:, 0],
            np.broadcast_to(np.asarray([1.0, 0.0, 0.0, 0.0]), (6, 4)),
        )
        np.testing.assert_array_equal(arrays["body_lin_vel_w"][:, 0], 0.0)
        np.testing.assert_array_equal(arrays["body_ang_vel_w"][:, 0], 0.0)

        data = mujoco.MjData(self.model)
        for frame_idx in range(arrays["joint_pos"].shape[0]):
            reconstructed = self.model.qpos0.copy()
            reconstructed[:3] = arrays["body_pos_w"][frame_idx, 1]
            reconstructed[3:7] = arrays["body_quat_w"][frame_idx, 1]
            for motion_idx, joint_name in enumerate(BUMI_MOTION_JOINT_NAMES):
                joint_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    joint_name,
                )
                reconstructed[self.model.jnt_qposadr[joint_id]] = arrays["joint_pos"][
                    frame_idx,
                    motion_idx,
                ]
            data.qpos[:] = reconstructed
            mujoco.mj_forward(self.model, data)
            expected_pos = np.stack(
                [
                    data.xpos[
                        mujoco.mj_name2id(
                            self.model,
                            mujoco.mjtObj.mjOBJ_BODY,
                            body_name,
                        )
                    ]
                    for body_name in BUMI_BODY_NAMES
                ]
            )
            np.testing.assert_allclose(
                arrays["body_pos_w"][frame_idx, 1:],
                expected_pos,
                atol=1.0e-6,
            )

        np.testing.assert_allclose(
            arrays["body_lin_vel_w"][:, 1, 0],
            1.0,
            # mj_differentiatePos/objectVelocity introduces about 2e-5 of
            # floating-point error when translation and yaw change together.
            atol=3.0e-5,
        )


if __name__ == "__main__":
    unittest.main()
