from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mimic_lite_conversion.amass_gmr import (
    AmassToBumiConverter,
    inspect_amass_source,
    load_manifest,
    run_batch,
    select_diverse_subset,
    sha256_file,
    split_by_subject,
    write_json,
    write_manifest,
)


class BumiAmassManifestTest(unittest.TestCase):
    def _write_source(
        self,
        root: Path,
        dataset: str,
        subject: str,
        sequence: str,
        *,
        fps: float,
        frames: int = 5,
    ) -> Path:
        output = root / dataset / subject / f"{sequence}_stageii.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            mocap_frame_rate=np.asarray(fps),
            pose_body=np.zeros((frames, 63), dtype=np.float32),
            root_orient=np.zeros((frames, 3), dtype=np.float32),
            trans=np.zeros((frames, 3), dtype=np.float32),
            betas=np.linspace(-0.1, 0.1, 16, dtype=np.float32),
            gender=np.asarray("neutral"),
        )
        return output

    def _entries(self, root: Path):
        entries = []
        for index in range(8):
            path = self._write_source(
                root,
                f"dataset_{index % 3}",
                f"subject_{index}",
                f"motion_{index}",
                fps=(30.0, 59.94, 120.0)[index % 3],
                frames=5 + index,
            )
            entries.append(inspect_amass_source(path, amass_root=root))
        return entries

    def test_inventory_preserves_native_fps_and_round_trips_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self._write_source(
                root,
                "CMU_SMPLX",
                "78",
                "78_10",
                fps=120.0,
                frames=13,
            )
            entry = inspect_amass_source(path, amass_root=root)
            self.assertEqual(entry.source_fps, 120.0)
            self.assertEqual(entry.frames, 13)
            self.assertAlmostEqual(entry.duration_s, 0.1)
            self.assertEqual(entry.relative_path, "CMU_SMPLX/78/78_10_stageii.npz")
            manifest = root / "manifest.jsonl"
            write_manifest(manifest, [entry])
            self.assertEqual(load_manifest(manifest), [entry])

    def test_subject_split_and_fixture_selection_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            entries = self._entries(Path(temporary))
            train_a, val_a = split_by_subject(entries, validation_fraction=0.25, seed="x")
            train_b, val_b = split_by_subject(entries, validation_fraction=0.25, seed="x")
            self.assertEqual(train_a, train_b)
            self.assertEqual(val_a, val_b)
            train_subjects = {f"{item.dataset}/{item.subject}" for item in train_a}
            val_subjects = {f"{item.dataset}/{item.subject}" for item in val_a}
            self.assertFalse(train_subjects & val_subjects)
            fixture_a = select_diverse_subset(train_a, 3, seed="fixture")
            fixture_b = select_diverse_subset(train_a, 3, seed="fixture")
            self.assertEqual(fixture_a, fixture_b)
            self.assertEqual(len({item.dataset for item in fixture_a}), 3)

    def test_changed_source_hash_is_visible_in_manifest_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self._write_source(root, "D", "S", "M", fps=60.0)
            entry = inspect_amass_source(path, amass_root=root)
            original_hash = entry.source_sha256
            with path.open("ab") as stream:
                stream.write(b"changed")
            self.assertNotEqual(sha256_file(path), original_hash)

    def test_resume_requires_provenance_and_tracker_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tracker = root / "tracker.npz"
            tracker.write_bytes(b"tracker")
            metadata = root / "metadata.json"
            expected = {"source_sha256": "a" * 64, "target_fps": 50.0}
            write_json(
                metadata,
                {
                    "provenance": expected,
                    "tracker_sha256": sha256_file(tracker),
                    "result": {"status": "converted", "tracker_frames": 10},
                },
            )
            converter = object.__new__(AmassToBumiConverter)
            paths = {"metadata": metadata, "tracker": tracker}
            cached = converter._cached_result(paths, expected)
            self.assertIsNotNone(cached)
            self.assertEqual(cached["status"], "cached")
            tracker.write_bytes(b"different")
            self.assertIsNone(converter._cached_result(paths, expected))

    def test_batch_isolates_rejects_and_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = self._entries(root)[:3]

            def convert(entry):
                if entry == entries[1]:
                    raise RuntimeError("synthetic failure")
                return {
                    "status": "converted",
                    "source_relative_path": entry.relative_path,
                    "source_frames": entry.frames,
                    "tracker_frames": entry.frames,
                }

            output = root / "output"
            summary = run_batch(entries, output_root=output, converter=convert)
            self.assertEqual(summary["succeeded"], 2)
            self.assertEqual(summary["rejected"], 1)
            rejected = [
                json.loads(line)
                for line in (output / "reports" / "rejected.jsonl").read_text().splitlines()
            ]
            self.assertEqual(rejected[0]["error_type"], "RuntimeError")
            self.assertIn("synthetic failure", rejected[0]["error"])


if __name__ == "__main__":
    unittest.main()
