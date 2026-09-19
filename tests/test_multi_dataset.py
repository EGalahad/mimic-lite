from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from any4hdmi import FullMotionDataset, MotionData

import active_adaptation as aa

aa.set_backend("mjlab")
from mimic_lite.tasks import multi_dataset as multi_dataset_module
from mimic_lite.tasks.motion import create_dataset_from_path
from mimic_lite.tasks.multi_dataset import (
    MotionDatasetConfig,
    MultiMotionDataset,
    load_motion_dataset_collection,
    normalize_motion_cfgs,
)


def _full_dataset(offset: float) -> FullMotionDataset:
    scalar = torch.tensor([offset, offset + 1]).reshape(2, 1, 1)
    vec3 = scalar.expand(-1, 1, 3).clone()
    quat = torch.zeros((2, 1, 4))
    quat[..., 0] = 1
    return FullMotionDataset(
        body_names=["pelvis"],
        joint_names=["joint"],
        motion_paths=[Path(f"{offset}.npz")],
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


def _windowed_stub(offset: float = 10.0):
    source = _full_dataset(offset)
    stub = SimpleNamespace(
        body_names=["pelvis"],
        joint_names=["joint"],
        motion_paths=[Path("windowed.npz")],
        starts=torch.tensor([0]),
        ends=torch.tensor([2]),
        lengths=torch.tensor([2]),
        num_motions=1,
        num_steps=2,
        sample_id_span=4,
        device=torch.device("cpu"),
    )
    stub.to = Mock(return_value=stub)
    stub.get_slice = Mock(
        side_effect=lambda motion_ids, starts, steps: source.get_slice(
            torch.zeros_like(motion_ids), starts, steps
        )
    )
    return stub


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
        self.assertIsInstance(result, MultiMotionDataset)
        self.assertEqual(result.datasets, [child])
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
        dataset = MultiMotionDataset(
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

    def test_fixed_environment_binding_overrides_weights_and_survives_rewind(self) -> None:
        dataset = MultiMotionDataset(
            motion_cfgs=[
                MotionDatasetConfig("first", "first", 100, True),
                MotionDatasetConfig("second", "second", 1, True),
            ],
            datasets=[_full_dataset(1), _full_dataset(3)],
            num_envs=2,
        ).to("cpu")
        dataset.bind_env_dataset_ids(torch.tensor([1, 0]))
        first = dataset.sample_motion(
            torch.tensor([0, 1]),
            terminated_t=torch.zeros(2, dtype=torch.long),
            rewind_mask=torch.zeros(2, dtype=torch.bool),
            rewind_steps=torch.zeros(2, dtype=torch.long),
        )
        rewound = dataset.sample_motion(
            torch.tensor([0, 1]),
            terminated_t=torch.ones(2, dtype=torch.long),
            rewind_mask=torch.ones(2, dtype=torch.bool),
            rewind_steps=torch.ones(2, dtype=torch.long),
        )
        torch.testing.assert_close(first.motion_id, torch.tensor([1, 0]))
        torch.testing.assert_close(rewound.motion_id, torch.tensor([1, 0]))
        torch.testing.assert_close(dataset.env_dataset_ids, torch.tensor([1, 0]))

    def test_binding_is_validated_and_cannot_change_after_sampling(self) -> None:
        dataset = MultiMotionDataset(
            motion_cfgs=[MotionDatasetConfig("only", "only", 1, True)],
            datasets=[_full_dataset(1)],
            num_envs=2,
        ).to("cpu")
        with self.assertRaisesRegex(ValueError, "shape"):
            dataset.bind_env_dataset_ids(torch.tensor([0]))
        with self.assertRaisesRegex(ValueError, "out of range"):
            dataset.bind_env_dataset_ids(torch.tensor([0, 1]))
        dataset.sample_motion(
            torch.tensor([0]),
            terminated_t=torch.zeros(1, dtype=torch.long),
            rewind_mask=torch.zeros(1, dtype=torch.bool),
            rewind_steps=torch.zeros(1, dtype=torch.long),
        )
        with self.assertRaisesRegex(RuntimeError, "after sampling"):
            dataset.bind_env_dataset_ids(torch.zeros(2, dtype=torch.long))

    def test_resident_children_share_fp16_store_and_preserve_batch_order(self) -> None:
        dataset = MultiMotionDataset(
            motion_cfgs=[
                MotionDatasetConfig("first", "first", 1, True),
                MotionDatasetConfig("second", "second", 1, True),
            ],
            datasets=[_full_dataset(1), _full_dataset(3)],
            num_envs=2,
        ).to("cpu")
        assert dataset._resident_data is not None
        assert dataset._resident_data.body_pos_w.dtype == torch.float16
        result = dataset.get_slice(
            torch.tensor([1, 0]), torch.tensor([0, 0]), torch.tensor([0, 1])
        )
        torch.testing.assert_close(
            result.body_pos_w[:, :, 0, 0], torch.tensor([[3.0, 4.0], [1.0, 2.0]])
        )

    def test_mixed_resident_windowed_routing_keeps_input_order(self) -> None:
        windowed = _windowed_stub()
        dataset = MultiMotionDataset(
            motion_cfgs=[
                MotionDatasetConfig("before", "before", 1, True),
                MotionDatasetConfig("windowed", "windowed", 1, False),
                MotionDatasetConfig("after", "after", 1, True),
            ],
            datasets=[_full_dataset(1), windowed, _full_dataset(20)],
            num_envs=6,
        ).to("cpu")
        result = dataset.get_slice(
            torch.tensor([5, 4, 0, 1, 5, 0]),
            torch.tensor([0, 0, 1, 1, 1, 0]),
            torch.tensor([0, 1]),
        )
        torch.testing.assert_close(
            result.body_pos_w[:, :, 0, 0],
            torch.tensor(
                [[20, 21], [10, 11], [2, 2], [11, 11], [21, 21], [1, 2]],
                dtype=torch.float32,
            ),
        )

    def test_compatibility_index_was_removed(self) -> None:
        self.assertFalse(
            hasattr(multi_dataset_module, "DeferredCombinedDatasetIndex")
        )


if __name__ == "__main__":
    unittest.main()
