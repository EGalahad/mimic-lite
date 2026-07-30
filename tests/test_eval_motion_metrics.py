from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from omegaconf import OmegaConf
from tensordict import TensorDict
import torch


EVAL_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval.py"
SPEC = importlib.util.spec_from_file_location("mimic_lite_eval", EVAL_PATH)
assert SPEC is not None and SPEC.loader is not None
EVAL_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL_MODULE)


class EvalMotionMetricsTest(unittest.TestCase):
    def test_aggregate_reports_coverage_and_per_motion_stats(self) -> None:
        metrics = EVAL_MODULE._aggregate_motion_progress(
            torch.tensor([2, 0, 2, 1]),
            torch.tensor([0.5, 1.0, 0.9, 0.25]),
            num_motions=4,
        )

        torch.testing.assert_close(metrics["motion_id"], torch.tensor([0, 1, 2]))
        torch.testing.assert_close(metrics["count"], torch.tensor([1, 1, 2]))
        torch.testing.assert_close(
            metrics["progress_mean"], torch.tensor([1.0, 0.25, 0.7])
        )
        torch.testing.assert_close(
            metrics["progress_std"], torch.tensor([0.0, 0.0, 0.2])
        )
        torch.testing.assert_close(
            metrics["progress_min"], torch.tensor([1.0, 0.25, 0.5])
        )

    def test_empty_input_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            EVAL_MODULE._aggregate_motion_progress(
                torch.tensor([], dtype=torch.long),
                torch.tensor([], dtype=torch.float32),
                num_motions=4,
            )

    def test_out_of_range_motion_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be in"):
            EVAL_MODULE._aggregate_motion_progress(
                torch.tensor([4]), torch.tensor([0.5]), num_motions=4
            )

    def test_summary_json_maps_motion_paths_and_names(self) -> None:
        metrics = EVAL_MODULE._aggregate_motion_progress(
            torch.tensor([1, 0, 1]),
            torch.tensor([0.5, 1.0, 0.75]),
            num_motions=2,
        )
        result = TensorDict(
            {
                "motion_metrics": metrics,
                "summary": EVAL_MODULE._scalar_tensordict(
                    {
                        "num_total_motions": 2,
                        "num_unique_motions": 2,
                        "motion_coverage": 1.0,
                    },
                    device=torch.device("cpu"),
                ),
                "reward_stats": TensorDict({}, batch_size=[]),
                "episode_summary": TensorDict({}, batch_size=[]),
            },
            batch_size=[],
        )
        cfg = OmegaConf.create(
            {"checkpoint_path": "/tmp/checkpoint.pt", "eval_output": "eval.pt"}
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "summary.json"
            EVAL_MODULE._write_summary_json(
                output,
                cfg,
                result,
                motion_paths=(
                    "/motions/walk/motion.npz",
                    "/motions/run/motion.npz",
                ),
            )
            payload = json.loads(output.read_text())

        self.assertEqual(payload["motion_metrics"]["per_motion"]["0"]["name"], "walk")
        self.assertEqual(payload["motion_metrics"]["per_motion"]["1"]["name"], "run")
        self.assertEqual(payload["motion_metrics"]["per_motion"]["1"]["count"], 2)


if __name__ == "__main__":
    unittest.main()
