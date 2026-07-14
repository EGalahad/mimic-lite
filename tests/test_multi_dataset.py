from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import torch
import active_adaptation as aa
from any4hdmi import FullMotionDataset, MotionData

aa.set_backend("mjlab")
from mimic_lite.tasks import multi_dataset as multi_dataset_module  # noqa: E402
from mimic_lite.tasks.multi_dataset import (  # noqa: E402
    MotionDatasetConfig,
    WeightedMultiMotionDataset,
    load_motion_dataset_collection,
    normalize_motion_cfgs,
)
from mimic_lite.tasks.motion import create_dataset_from_path  # noqa: E402


def _full_dataset(offset: float) -> FullMotionDataset:
    scalar = torch.tensor([offset, offset + 1]).reshape(2, 1, 1)
    vec3 = scalar.expand(-1, 1, 3).clone()
    quat = torch.zeros((2, 1, 4))
    quat[..., 0] = 1
    return FullMotionDataset(
        body_names=["pelvis"],
        joint_names=["joint"],
        motion_paths=[],
        starts=[0],
        ends=[2],
        data=MotionData(
            motion_id=torch.tensor([0, 0]),
            step=torch.tensor([0, 1]),
            body_pos_w=vec3,
            body_lin_vel_w=torch.zeros_like(vec3),
            body_quat_w=quat,
            body_ang_vel_w=torch.zeros_like(vec3),
            joint_pos=scalar,
            joint_vel=torch.zeros_like(scalar),
            batch_size=(2,),
            device=torch.device("cpu"),
        ),
        num_envs=2,
        output_float_dtype=torch.float32,
    )


class MultiDatasetTest(unittest.TestCase):
    def test_shard_defaults_false_and_can_be_enabled(self) -> None:
        configs = normalize_motion_cfgs(
            {
                "lafan": {"path": "lafan", "weight": 1, "full_motion": True},
                "sonic": {
                    "path": "sonic",
                    "weight": 2,
                    "full_motion": False,
                    "shard": True,
                },
            }
        )
        self.assertFalse(configs[0].shard)
        self.assertTrue(configs[1].shard)

    def test_weight_rejects_legacy_numeric_strings(self) -> None:
        with self.assertRaisesRegex(TypeError, "weight must be numeric"):
            normalize_motion_cfgs(
                {
                    "sonic": {
                        "path": "sonic",
                        "weight": "1.0",
                        "full_motion": False,
                    }
                }
            )

    def test_collection_passes_independent_runtime_and_partition_flags(self) -> None:
        child = _full_dataset(1)
        create_dataset = Mock(return_value=child)
        result = load_motion_dataset_collection(
            [MotionDatasetConfig("sonic", "sonic", 1, False, True)],
            create_dataset_fn=create_dataset,
            target_fps=50,
            num_envs=2,
        )
        self.assertIs(result, child)
        self.assertFalse(create_dataset.call_args.kwargs["full_motion"])
        self.assertTrue(create_dataset.call_args.kwargs["shard"])

    def test_mimic_lite_passes_torchrun_context_explicitly(self) -> None:
        child = _full_dataset(1)
        with (
            patch.dict("os.environ", {"RANK": "3", "WORLD_SIZE": "8"}),
            patch(
                "mimic_lite.tasks.motion.load_any4hdmi_dataset",
                return_value=child,
            ) as load,
        ):
            result = create_dataset_from_path(
                "sonic", full_motion=False, shard=True
            )
        self.assertIs(result, child)
        self.assertEqual(load.call_args.kwargs["rank"], 3)
        self.assertEqual(load.call_args.kwargs["world_size"], 8)

    def test_weighted_sampling_policy_stays_in_mimic_lite(self) -> None:
        dataset = WeightedMultiMotionDataset(
            motion_cfgs=[
                MotionDatasetConfig("first", "first", 1, True),
                MotionDatasetConfig("second", "second", 1, True),
            ],
            datasets=[_full_dataset(1), _full_dataset(3)],
            num_envs=2,
        ).to("cpu")
        with patch(
            "mimic_lite.tasks.multi_dataset.torch.multinomial",
            return_value=torch.tensor([0, 1]),
        ):
            sampled = dataset.sample_motion(
                torch.tensor([0, 1]),
                terminated_t=torch.zeros(2, dtype=torch.long),
                rewind_mask=torch.zeros(2, dtype=torch.bool),
                rewind_steps=torch.zeros(2, dtype=torch.long),
            )
        torch.testing.assert_close(sampled.motion_id, torch.tensor([0, 1]))

    def test_compatibility_index_was_removed(self) -> None:
        self.assertFalse(
            hasattr(multi_dataset_module, "DeferredCombinedDatasetIndex")
        )


if __name__ == "__main__":
    unittest.main()
