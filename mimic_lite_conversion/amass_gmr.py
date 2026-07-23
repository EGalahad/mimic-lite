"""Manifest and batch conversion support for AMASS -> HumanPose24 -> Bumi.

The temporal contract is intentionally strict: AMASS and GMR stay on the
source timeline, then :mod:`mimic_lite_conversion.bumi` performs the sole
resampling operation when it exports the final 50 Hz tracker file.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

import mujoco
import numpy as np

from .bumi import (
    BUMI_JOINT_VELOCITY_LIMITS,
    BUMI_POLICY_JOINT_NAMES,
    BUMI_ROOT_ANGULAR_VELOCITY_LIMIT,
    BUMI_ROOT_LINEAR_VELOCITY_LIMIT,
    TARGET_TRACKER_FPS,
    export_bumi_tracking_npz,
    nominal_bumi_qpos,
    validate_bumi_model,
)


MANIFEST_SCHEMA_VERSION = 1
PIPELINE_VERSION = 2
REQUIRED_AMASS_FIELDS = (
    "pose_body",
    "root_orient",
    "trans",
    "betas",
    "gender",
)


@dataclass(frozen=True)
class AmassManifestEntry:
    source_path: str
    relative_path: str
    dataset: str
    subject: str
    sequence: str
    source_fps: float
    frames: int
    duration_s: float
    gender: str
    betas_sha256: str
    source_sha256: str
    source_size_bytes: int
    source_mtime_ns: int
    schema_version: int = MANIFEST_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AmassManifestEntry":
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown AMASS manifest fields: {sorted(unknown)}")
        entry = cls(**value)
        entry.validate()
        return entry

    def validate(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported manifest schema {self.schema_version}; "
                f"expected {MANIFEST_SCHEMA_VERSION}"
            )
        if not self.relative_path or Path(self.relative_path).is_absolute():
            raise ValueError(f"relative_path must be relative: {self.relative_path!r}")
        if not self.dataset or not self.subject or not self.sequence:
            raise ValueError("dataset, subject and sequence must not be empty")
        if not math.isfinite(self.source_fps) or self.source_fps <= 0.0:
            raise ValueError(f"Invalid source FPS: {self.source_fps}")
        if self.frames <= 0:
            raise ValueError(f"frames must be positive, got {self.frames}")
        expected_duration = (self.frames - 1) / self.source_fps
        if not math.isclose(self.duration_s, expected_duration, abs_tol=1.0e-9):
            raise ValueError(
                f"duration_s={self.duration_s} does not match frames/FPS "
                f"({expected_duration})"
            )
        for label, digest in (
            ("betas_sha256", self.betas_sha256),
            ("source_sha256", self.source_sha256),
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"{label} is not a SHA256 digest: {digest!r}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_files(paths: Sequence[str | Path]) -> str:
    """Hash a labeled group of source files for cache invalidation."""

    digest = hashlib.sha256()
    for path_value in sorted((Path(path).resolve() for path in paths), key=str):
        digest.update(path_value.name.encode())
        digest.update(b"\0")
        digest.update(sha256_file(path_value).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _scalar_string(value: Any) -> str:
    item = np.asarray(value).reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _source_fps(payload: Mapping[str, Any]) -> float:
    for key in ("mocap_frame_rate", "mocap_framerate"):
        if key in payload:
            fps = float(np.asarray(payload[key]).reshape(-1)[0])
            if not math.isfinite(fps) or fps <= 0.0:
                raise ValueError(f"{key} must be finite and positive, got {fps}")
            return fps
    raise ValueError("missing mocap_frame_rate/mocap_framerate")


def inspect_amass_source(
    path: str | Path,
    *,
    amass_root: str | Path,
) -> AmassManifestEntry:
    source = Path(path).expanduser().resolve()
    root = Path(amass_root).expanduser().resolve()
    try:
        relative = source.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"AMASS source is outside root: {source} / {root}") from exc
    if len(relative.parts) < 3:
        raise ValueError(
            "Expected AMASS layout <dataset>/<subject>/<sequence>.npz, got "
            f"{relative}"
        )

    with np.load(source, allow_pickle=True) as payload:
        missing = [field for field in REQUIRED_AMASS_FIELDS if field not in payload]
        if missing:
            raise ValueError(f"missing AMASS fields: {', '.join(missing)}")
        fps = _source_fps(payload)
        pose_body = np.asarray(payload["pose_body"])
        root_orient = np.asarray(payload["root_orient"])
        trans = np.asarray(payload["trans"])
        betas = np.asarray(payload["betas"], dtype=np.float32).reshape(-1)
        gender = _scalar_string(payload["gender"]).strip().lower()

    if pose_body.ndim != 2 or pose_body.shape[1] != 63:
        raise ValueError(f"pose_body must be [T,63], got {pose_body.shape}")
    frames = int(pose_body.shape[0])
    if root_orient.shape != (frames, 3):
        raise ValueError(f"root_orient must be [{frames},3], got {root_orient.shape}")
    if trans.shape != (frames, 3):
        raise ValueError(f"trans must be [{frames},3], got {trans.shape}")
    if frames <= 0 or betas.size == 0:
        raise ValueError("AMASS motion and betas must not be empty")
    for label, array in (
        ("pose_body", pose_body),
        ("root_orient", root_orient),
        ("trans", trans),
        ("betas", betas),
    ):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{label} contains non-finite values")
    if gender not in {"female", "male", "neutral"}:
        raise ValueError(f"Unsupported AMASS gender: {gender!r}")

    stat = source.stat()
    sequence = source.stem.removesuffix("_stageii")
    return AmassManifestEntry(
        source_path=str(source),
        relative_path=relative.as_posix(),
        dataset=relative.parts[0],
        subject=relative.parts[1],
        sequence=sequence,
        source_fps=fps,
        frames=frames,
        duration_s=(frames - 1) / fps,
        gender=gender,
        betas_sha256=_sha256_array(betas),
        source_sha256=sha256_file(source),
        source_size_bytes=int(stat.st_size),
        source_mtime_ns=int(stat.st_mtime_ns),
    )


def inventory_amass(
    amass_root: str | Path,
    *,
    pattern: str = "**/*_stageii.npz",
    limit: int | None = None,
    on_reject: Callable[[Path, Exception], None] | None = None,
) -> list[AmassManifestEntry]:
    root = Path(amass_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"AMASS root not found: {root}")
    paths = sorted(path for path in root.glob(pattern) if path.is_file())
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No AMASS files matched {pattern!r} under {root}")
    entries = []
    for path in paths:
        try:
            entries.append(inspect_amass_source(path, amass_root=root))
        except Exception as exc:
            if on_reject is None:
                raise
            on_reject(path, exc)
    return entries


def _stable_score(value: str, seed: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}\0{value}".encode()).digest()[:8],
        "big",
    )


def split_by_subject(
    entries: Sequence[AmassManifestEntry],
    *,
    validation_fraction: float = 0.1,
    seed: str = "bumi-amass-v1",
) -> tuple[list[AmassManifestEntry], list[AmassManifestEntry]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    groups = sorted({f"{entry.dataset}/{entry.subject}" for entry in entries})
    if len(groups) < 2:
        raise ValueError("At least two dataset/subject groups are required for a split")
    validation_count = max(1, min(len(groups) - 1, round(len(groups) * validation_fraction)))
    ranked_groups = sorted(groups, key=lambda value: (_stable_score(value, seed), value))
    validation_groups = set(ranked_groups[:validation_count])
    train = []
    validation = []
    for entry in entries:
        group = f"{entry.dataset}/{entry.subject}"
        (validation if group in validation_groups else train).append(entry)
    return (
        sorted(train, key=lambda item: item.relative_path),
        sorted(validation, key=lambda item: item.relative_path),
    )


def select_diverse_subset(
    entries: Sequence[AmassManifestEntry],
    count: int,
    *,
    seed: str,
) -> list[AmassManifestEntry]:
    """Select deterministically while cycling datasets and avoiding subjects first."""

    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if count > len(entries):
        raise ValueError(f"Cannot select {count} entries from {len(entries)}")
    per_dataset: dict[str, list[AmassManifestEntry]] = defaultdict(list)
    for entry in entries:
        per_dataset[entry.dataset].append(entry)
    for dataset_entries in per_dataset.values():
        dataset_entries.sort(
            key=lambda item: (_stable_score(item.relative_path, seed), item.relative_path)
        )

    selected: list[AmassManifestEntry] = []
    selected_paths: set[str] = set()
    selected_subjects: set[str] = set()
    datasets = sorted(per_dataset)
    for require_new_subject in (True, False):
        made_progress = True
        while len(selected) < count and made_progress:
            made_progress = False
            for dataset in datasets:
                for entry in per_dataset[dataset]:
                    subject_key = f"{entry.dataset}/{entry.subject}"
                    if entry.relative_path in selected_paths:
                        continue
                    if require_new_subject and subject_key in selected_subjects:
                        continue
                    selected.append(entry)
                    selected_paths.add(entry.relative_path)
                    selected_subjects.add(subject_key)
                    made_progress = True
                    break
                if len(selected) == count:
                    break
    if len(selected) != count:
        raise RuntimeError(f"Subset selection produced {len(selected)} of {count} entries")
    return selected


def _atomic_write_text(path: str | Path, text: str) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{output.name}.",
            dir=output.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path: str | Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: str | Path, values: Iterable[Mapping[str, Any]]) -> None:
    lines = [json.dumps(dict(value), sort_keys=True) for value in values]
    _atomic_write_text(path, "".join(f"{line}\n" for line in lines))


def write_manifest(path: str | Path, entries: Sequence[AmassManifestEntry]) -> None:
    write_jsonl(path, (asdict(entry) for entry in entries))


def load_manifest(path: str | Path) -> list[AmassManifestEntry]:
    manifest = Path(path).expanduser().resolve()
    entries = []
    with manifest.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                entries.append(AmassManifestEntry.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Invalid {manifest}:{line_number}: {exc}") from exc
    if not entries:
        raise ValueError(f"Manifest is empty: {manifest}")
    relative_paths = [entry.relative_path for entry in entries]
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError(f"Manifest contains duplicate relative paths: {manifest}")
    return entries


def inventory_summary(entries: Sequence[AmassManifestEntry]) -> dict[str, Any]:
    fps_counts = Counter(f"{entry.source_fps:g}" for entry in entries)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "motions": len(entries),
        "subjects": len({f"{entry.dataset}/{entry.subject}" for entry in entries}),
        "datasets": dict(sorted(Counter(entry.dataset for entry in entries).items())),
        "source_fps": dict(sorted(fps_counts.items(), key=lambda item: float(item[0]))),
        "frames": sum(entry.frames for entry in entries),
        "duration_s": sum(entry.duration_s for entry in entries),
    }


def _git_commit(path: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _atomic_save_sequence(sequence: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.stem}.",
            suffix=".npz",
            dir=output.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        sequence.save_npz(temporary)
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_save_npz(output: Path, **arrays: np.ndarray) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.stem}.",
            suffix=".npz",
            dir=output.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def clip_id(entry: AmassManifestEntry) -> str:
    readable = "__".join((entry.dataset, entry.subject, entry.sequence))
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", readable)[:120].strip("._")
    return f"{readable}__{entry.source_sha256[:10]}"


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


class AmassToBumiConverter:
    """Stateful, single-worker converter with clip-local GMR resets."""

    def __init__(
        self,
        *,
        gmr_root: str | Path,
        smplx_model_dir: str | Path,
        bumi_mjcf: str | Path,
        output_root: str | Path,
        target_fps: float = TARGET_TRACKER_FPS,
        actual_human_height: float = 1.6,
        resume: bool = False,
    ) -> None:
        self.gmr_root = Path(gmr_root).expanduser().resolve()
        self.smplx_model_dir = Path(smplx_model_dir).expanduser().resolve()
        self.bumi_mjcf = Path(bumi_mjcf).expanduser().resolve()
        self.output_root = Path(output_root).expanduser().resolve()
        self.target_fps = float(target_fps)
        self.actual_human_height = float(actual_human_height)
        self.resume = bool(resume)
        if not self.gmr_root.is_dir():
            raise FileNotFoundError(f"GMR root not found: {self.gmr_root}")
        if not self.smplx_model_dir.is_dir():
            raise FileNotFoundError(f"SMPL-X model directory not found: {self.smplx_model_dir}")
        if not self.bumi_mjcf.is_file():
            raise FileNotFoundError(f"Bumi MJCF not found: {self.bumi_mjcf}")
        if not math.isclose(self.target_fps, TARGET_TRACKER_FPS):
            raise ValueError(
                f"MimicLite Bumi tracker requires {TARGET_TRACKER_FPS:g} Hz, "
                f"got {self.target_fps:g}"
            )
        if not math.isfinite(self.actual_human_height) or self.actual_human_height <= 0:
            raise ValueError(f"Invalid actual_human_height: {self.actual_human_height}")

        gmr_path = str(self.gmr_root)
        if gmr_path not in sys.path:
            sys.path.insert(0, gmr_path)
        from general_motion_retargeting import GeneralMotionRetargeting
        from general_motion_retargeting.human_pose24 import HumanPose24Sequence
        from general_motion_retargeting.utils.smpl import load_smplx_file
        from general_motion_retargeting.utils.smplx_to_human_pose24 import (
            human_pose24_from_smplx_output,
        )

        self._human_pose_type = HumanPose24Sequence
        self._load_smplx_file = load_smplx_file
        self._human_pose_from_smplx = human_pose24_from_smplx_output
        self.model = mujoco.MjModel.from_xml_path(str(self.bumi_mjcf))
        validate_bumi_model(self.model)
        self.initial_qpos = nominal_bumi_qpos(self.model)
        self.gmr_config = (
            self.gmr_root
            / "general_motion_retargeting"
            / "ik_configs"
            / "xrobot_to_bumi.json"
        )
        if not self.gmr_config.is_file():
            raise FileNotFoundError(f"Bumi GMR config not found: {self.gmr_config}")
        self.retargeter = GeneralMotionRetargeting(
            src_human="xrobot",
            tgt_robot="bumi",
            actual_human_height=self.actual_human_height,
            verbose=False,
            robot_xml_path=self.bumi_mjcf,
            ik_config_path=self.gmr_config,
            initial_qpos=self.initial_qpos,
            joint_velocity_limits=BUMI_JOINT_VELOCITY_LIMITS,
            root_linear_velocity_limit=BUMI_ROOT_LINEAR_VELOCITY_LIMIT,
            root_angular_velocity_limit=BUMI_ROOT_ANGULAR_VELOCITY_LIMIT,
        )
        gmr_code_paths = (
            self.gmr_root / "general_motion_retargeting" / "motion_retarget.py",
            self.gmr_root / "general_motion_retargeting" / "human_pose24.py",
            self.gmr_root
            / "general_motion_retargeting"
            / "utils"
            / "smplx_to_human_pose24.py",
        )
        human_pose_code_paths = (
            self.gmr_root / "general_motion_retargeting" / "human_pose24.py",
            self.gmr_root
            / "general_motion_retargeting"
            / "utils"
            / "smplx_to_human_pose24.py",
            self.gmr_root
            / "general_motion_retargeting"
            / "utils"
            / "smpl.py",
        )
        self.human_pose_common_provenance = {
            "human_pose_cache_version": 1,
            "human_pose_code_sha256": sha256_files(human_pose_code_paths),
            "smplx_model_dir": str(self.smplx_model_dir),
        }
        self.common_provenance = {
            "pipeline_version": PIPELINE_VERSION,
            "gmr_commit": _git_commit(self.gmr_root),
            "gmr_code_sha256": sha256_files(gmr_code_paths),
            "gmr_config_sha256": sha256_file(self.gmr_config),
            "bumi_mjcf_sha256": sha256_file(self.bumi_mjcf),
            "smplx_model_dir": str(self.smplx_model_dir),
            "actual_human_height": self.actual_human_height,
            "target_fps": self.target_fps,
            "joint_velocity_limits": BUMI_JOINT_VELOCITY_LIMITS,
            "root_linear_velocity_limit": BUMI_ROOT_LINEAR_VELOCITY_LIMIT,
            "root_angular_velocity_limit": BUMI_ROOT_ANGULAR_VELOCITY_LIMIT,
        }

    def _paths(self, entry: AmassManifestEntry) -> dict[str, Path]:
        identifier = clip_id(entry)
        return {
            "human_pose24": self.output_root / "human_pose24" / f"{identifier}.npz",
            "native_qpos": self.output_root / "native_qpos" / f"{identifier}.npz",
            "tracker": self.output_root / "tracker_50hz" / f"{identifier}.npz",
            "metadata": self.output_root / "metadata" / f"{identifier}.json",
        }

    def _expected_provenance(self, entry: AmassManifestEntry) -> dict[str, Any]:
        return {
            **self.common_provenance,
            "manifest_schema_version": entry.schema_version,
            "source_path": entry.source_path,
            "source_relative_path": entry.relative_path,
            "source_sha256": entry.source_sha256,
            "source_fps": entry.source_fps,
            "source_frames": entry.frames,
            "betas_sha256": entry.betas_sha256,
            "gender": entry.gender,
        }

    def _human_pose_provenance(self, entry: AmassManifestEntry) -> dict[str, Any]:
        return {
            **self.human_pose_common_provenance,
            "source_path": entry.source_path,
            "source_relative_path": entry.relative_path,
            "source_sha256": entry.source_sha256,
            "source_fps": entry.source_fps,
            "source_frames": entry.frames,
            "betas_sha256": entry.betas_sha256,
            "gender": entry.gender,
        }

    def _cached_result(
        self,
        paths: Mapping[str, Path],
        expected_provenance: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        metadata_path = paths["metadata"]
        tracker_path = paths["tracker"]
        if not metadata_path.is_file() or not tracker_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("provenance") != dict(expected_provenance):
                return None
            if metadata.get("tracker_sha256") != sha256_file(tracker_path):
                return None
            result = dict(metadata["result"])
            result["status"] = "cached"
            return result
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            return None

    def _load_or_create_human_pose(
        self,
        entry: AmassManifestEntry,
        output: Path,
        provenance: Mapping[str, Any],
    ) -> Any:
        if self.resume and output.is_file():
            try:
                cached = self._human_pose_type.load_npz(output)
                if cached.metadata.get("pipeline_provenance") == dict(provenance):
                    return cached
            except (KeyError, OSError, ValueError, json.JSONDecodeError):
                pass

        source = Path(entry.source_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"AMASS source not found: {source}")
        actual_digest = sha256_file(source)
        if actual_digest != entry.source_sha256:
            raise ValueError(
                f"AMASS source hash changed for {entry.relative_path}: "
                f"manifest={entry.source_sha256}, actual={actual_digest}"
            )
        smplx_data, body_model, smplx_output, _ = self._load_smplx_file(
            source,
            self.smplx_model_dir,
        )
        sequence = self._human_pose_from_smplx(
            smplx_data,
            body_model,
            smplx_output,
            metadata={
                "source_path": str(source),
                "source_relative_path": entry.relative_path,
                "source_sha256": entry.source_sha256,
                "gender": entry.gender,
                "betas_sha256": entry.betas_sha256,
                "pipeline_provenance": dict(provenance),
            },
        )
        if sequence.num_frames != entry.frames:
            raise ValueError(
                f"HumanPose24 frame mismatch: {sequence.num_frames} / {entry.frames}"
            )
        if not math.isclose(sequence.source_fps, entry.source_fps):
            raise ValueError(
                f"HumanPose24 FPS mismatch: {sequence.source_fps} / {entry.source_fps}"
            )
        _atomic_save_sequence(sequence, output)
        return sequence

    def _qpos_quality(
        self,
        qpos: np.ndarray,
        timestamps_s: np.ndarray,
    ) -> dict[str, float]:
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape != (timestamps_s.shape[0], self.model.nq):
            raise ValueError(f"Unexpected native qpos shape: {qpos.shape}")
        if not np.all(np.isfinite(qpos)):
            raise ValueError("Native GMR qpos contains non-finite values")
        quaternion_norm_error = np.abs(np.linalg.norm(qpos[:, 3:7], axis=-1) - 1.0)
        if float(quaternion_norm_error.max()) > 1.0e-5:
            raise ValueError(
                "Native root quaternion norm error exceeds 1e-5: "
                f"{float(quaternion_norm_error.max())}"
            )

        qpos_addresses = []
        minimum_margin = math.inf
        for joint_name in BUMI_POLICY_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            address = int(self.model.jnt_qposadr[joint_id])
            qpos_addresses.append(address)
            lower, upper = self.model.jnt_range[joint_id]
            minimum = float(qpos[:, address].min())
            maximum = float(qpos[:, address].max())
            if minimum < lower - 1.0e-6 or maximum > upper + 1.0e-6:
                raise ValueError(
                    f"GMR output violates {joint_name} limits [{lower}, {upper}]: "
                    f"[{minimum}, {maximum}]"
                )
            minimum_margin = min(minimum_margin, minimum - lower, upper - maximum)
        if qpos.shape[0] > 1:
            dt = np.diff(timestamps_s)
            joint_speed = np.abs(np.diff(qpos[:, qpos_addresses], axis=0) / dt[:, None])
            max_joint_speed = float(joint_speed.max())
            p95_joint_speed = _percentile(joint_speed, 95.0)
            joint_velocity_limits = np.asarray(
                [BUMI_JOINT_VELOCITY_LIMITS[name] for name in BUMI_POLICY_JOINT_NAMES]
            )
            joint_speed_ratio = joint_speed / joint_velocity_limits[None, :]
            max_joint_speed_ratio = float(joint_speed_ratio.max())
        else:
            max_joint_speed = 0.0
            p95_joint_speed = 0.0
            max_joint_speed_ratio = 0.0
        return {
            "max_root_quaternion_norm_error": float(quaternion_norm_error.max()),
            "minimum_joint_limit_margin": float(minimum_margin),
            "max_native_joint_speed": max_joint_speed,
            "p95_native_joint_speed": p95_joint_speed,
            "max_native_joint_velocity_ratio": max_joint_speed_ratio,
        }

    def convert(self, entry: AmassManifestEntry) -> dict[str, Any]:
        entry.validate()
        paths = self._paths(entry)
        provenance = self._expected_provenance(entry)
        if self.resume:
            cached = self._cached_result(paths, provenance)
            if cached is not None:
                return cached

        sequence = self._load_or_create_human_pose(
            entry,
            paths["human_pose24"],
            self._human_pose_provenance(entry),
        )
        qpos, diagnostics = self.retargeter.retarget_sequence(
            sequence,
            initial_qpos=self.initial_qpos,
        )
        qpos_quality = self._qpos_quality(qpos, sequence.timestamps_s)
        diagnostics_arrays = {
            key: np.asarray([item[key] for item in diagnostics])
            for key in (
                "table1_error",
                "table2_error",
                "table1_iterations",
                "table2_iterations",
                "rate_limited",
                "max_desired_velocity_ratio",
                "max_emitted_velocity_ratio",
            )
        }
        task_error_names = tuple(sorted(diagnostics[0]["task_errors"]))
        task_human_body_names = tuple(
            diagnostics[0]["task_errors"][name]["human_body"]
            for name in task_error_names
        )
        for frame_idx, item in enumerate(diagnostics):
            if tuple(sorted(item["task_errors"])) != task_error_names:
                raise ValueError(f"GMR task diagnostics changed at frame {frame_idx}")
        task_position_error = np.asarray(
            [
                [item["task_errors"][name]["position_error"] for name in task_error_names]
                for item in diagnostics
            ],
            dtype=np.float64,
        )
        task_orientation_error_rad = np.asarray(
            [
                [
                    item["task_errors"][name]["orientation_error_rad"]
                    for name in task_error_names
                ]
                for item in diagnostics
            ],
            dtype=np.float64,
        )
        _atomic_save_npz(
            paths["native_qpos"],
            timestamps_s=sequence.timestamps_s,
            qpos=qpos,
            source_fps=np.asarray([sequence.source_fps], dtype=np.float64),
            task_error_names=np.asarray(task_error_names),
            task_human_body_names=np.asarray(task_human_body_names),
            task_position_error=task_position_error,
            task_orientation_error_rad=task_orientation_error_rad,
            **diagnostics_arrays,
        )
        export_stats = dict(
            export_bumi_tracking_npz(
                paths["tracker"],
                model_path=self.bumi_mjcf,
                source_timestamps_s=sequence.timestamps_s,
                source_qpos=qpos,
                target_fps=self.target_fps,
            )
        )
        # The destination grid is floor(duration * fps) + 1 and may not extend
        # past the source.  Its tail truncation is therefore strictly less
        # than one target interval (not half an interval).
        if abs(float(export_stats["duration_drift_s"])) >= 1.0 / self.target_fps:
            raise ValueError(f"Tracker duration drift is too large: {export_stats}")

        task_name_to_index = {
            name: index for index, name in enumerate(task_error_names)
        }
        foot_task_names = (
            "table2:l_ankle_roll_link",
            "table2:r_ankle_roll_link",
        )
        high_priority_task_names = (
            "table2:base_link",
            *foot_task_names,
        )
        foot_indices = [task_name_to_index[name] for name in foot_task_names]
        high_priority_indices = [
            task_name_to_index[name] for name in high_priority_task_names
        ]
        foot_position_p95 = _percentile(task_position_error[:, foot_indices], 95.0)
        high_priority_position_p95 = _percentile(
            task_position_error[:, high_priority_indices],
            95.0,
        )
        high_priority_orientation_p95_rad = _percentile(
            task_orientation_error_rad[:, high_priority_indices],
            95.0,
        )
        result = {
            "status": "converted",
            "clip_id": clip_id(entry),
            "source_relative_path": entry.relative_path,
            "source_frames": entry.frames,
            "source_fps": entry.source_fps,
            "tracker_frames": int(export_stats["frames"]),
            "tracker_path": str(paths["tracker"]),
            "human_pose24_path": str(paths["human_pose24"]),
            "native_qpos_path": str(paths["native_qpos"]),
            "duration_drift_s": float(export_stats["duration_drift_s"]),
            "gmr_table1_error_median": _percentile(
                diagnostics_arrays["table1_error"], 50.0
            ),
            "gmr_table1_error_p95": _percentile(
                diagnostics_arrays["table1_error"], 95.0
            ),
            "gmr_table2_error_median": _percentile(
                diagnostics_arrays["table2_error"], 50.0
            ),
            "gmr_table2_error_p95": _percentile(
                diagnostics_arrays["table2_error"], 95.0
            ),
            "gmr_rate_limited_fraction": float(
                np.mean(diagnostics_arrays["rate_limited"])
            ),
            "gmr_max_desired_velocity_ratio": float(
                diagnostics_arrays["max_desired_velocity_ratio"].max(initial=0.0)
            ),
            "gmr_max_emitted_velocity_ratio": float(
                diagnostics_arrays["max_emitted_velocity_ratio"].max(initial=0.0)
            ),
            "gmr_task_position_error_p95": _percentile(task_position_error, 95.0),
            "gmr_task_orientation_error_p95_rad": _percentile(
                task_orientation_error_rad,
                95.0,
            ),
            "gmr_high_priority_position_error_p95": high_priority_position_p95,
            "gmr_foot_position_error_p95": foot_position_p95,
            "gmr_high_priority_orientation_error_p95_rad": (
                high_priority_orientation_p95_rad
            ),
            "gate_position_pass": bool(high_priority_position_p95 < 0.10),
            "gate_foot_position_pass": bool(foot_position_p95 < 0.06),
            "gate_orientation_pass": bool(
                high_priority_orientation_p95_rad < np.deg2rad(15.0)
            ),
            **qpos_quality,
        }
        metadata = {
            "provenance": provenance,
            "tracker_sha256": sha256_file(paths["tracker"]),
            "human_pose24_sha256": sha256_file(paths["human_pose24"]),
            "native_qpos_sha256": sha256_file(paths["native_qpos"]),
            "result": result,
        }
        write_json(paths["metadata"], metadata)
        return result


def _finalize_batch(
    entries: Sequence[AmassManifestEntry],
    successes: Sequence[Mapping[str, Any]],
    rejects: Sequence[Mapping[str, Any]],
    *,
    output_root: str | Path,
) -> dict[str, Any]:
    output = Path(output_root).expanduser().resolve()
    reports = output / "reports"
    ordered_successes = sorted(
        (dict(item) for item in successes),
        key=lambda item: str(item.get("source_relative_path", "")),
    )
    ordered_rejects = sorted(
        (dict(item) for item in rejects),
        key=lambda item: str(item.get("source_relative_path", "")),
    )
    write_jsonl(reports / "converted.jsonl", ordered_successes)
    write_jsonl(reports / "rejected.jsonl", ordered_rejects)
    summary = {
        "requested": len(entries),
        "converted": sum(
            item.get("status") == "converted" for item in ordered_successes
        ),
        "cached": sum(item.get("status") == "cached" for item in ordered_successes),
        "succeeded": len(ordered_successes),
        "rejected": len(ordered_rejects),
        "source_frames": sum(
            int(item.get("source_frames", 0)) for item in ordered_successes
        ),
        "tracker_frames": sum(
            int(item.get("tracker_frames", 0)) for item in ordered_successes
        ),
    }
    write_json(reports / "summary.json", summary)
    return summary


def run_batch(
    entries: Sequence[AmassManifestEntry],
    *,
    output_root: str | Path,
    converter: Callable[[AmassManifestEntry], Mapping[str, Any]],
) -> dict[str, Any]:
    """Run every clip sequentially, isolating rejects and writing reports."""

    successes: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    for entry in entries:
        try:
            successes.append(dict(converter(entry)))
        except Exception as exc:
            rejects.append(
                {
                    "source_path": entry.source_path,
                    "source_relative_path": entry.relative_path,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return _finalize_batch(
        entries,
        successes,
        rejects,
        output_root=output_root,
    )


_PROCESS_CONVERTER: AmassToBumiConverter | None = None


def _initialize_converter_worker(
    converter_kwargs: Mapping[str, Any],
    torch_threads: int,
) -> None:
    global _PROCESS_CONVERTER
    import torch

    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(1)
    _PROCESS_CONVERTER = AmassToBumiConverter(**dict(converter_kwargs))


def _convert_in_worker(
    entry: AmassManifestEntry,
) -> tuple[bool, dict[str, Any]]:
    if _PROCESS_CONVERTER is None:
        raise RuntimeError("AMASS converter worker was not initialized")
    try:
        return True, dict(_PROCESS_CONVERTER.convert(entry))
    except Exception as exc:
        return False, {
            "source_path": entry.source_path,
            "source_relative_path": entry.relative_path,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def run_parallel_converter_batch(
    entries: Sequence[AmassManifestEntry],
    *,
    output_root: str | Path,
    converter_kwargs: Mapping[str, Any],
    workers: int,
    torch_threads_per_worker: int = 4,
) -> dict[str, Any]:
    """Convert independent clips in spawned workers and write one report set."""

    if workers <= 1:
        raise ValueError(f"Parallel conversion requires workers > 1, got {workers}")
    if torch_threads_per_worker <= 0:
        raise ValueError(
            "torch_threads_per_worker must be positive, got "
            f"{torch_threads_per_worker}"
        )
    worker_count = min(int(workers), len(entries))
    context = multiprocessing.get_context("spawn")
    successes: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    started_at = time.monotonic()
    last_progress_at = started_at
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=_initialize_converter_worker,
        initargs=(dict(converter_kwargs), int(torch_threads_per_worker)),
    ) as executor:
        futures = [executor.submit(_convert_in_worker, entry) for entry in entries]
        for completed_count, future in enumerate(as_completed(futures), start=1):
            succeeded, result = future.result()
            (successes if succeeded else rejects).append(result)
            now = time.monotonic()
            if (
                completed_count == 1
                or completed_count == len(entries)
                or now - last_progress_at >= 30.0
            ):
                elapsed = max(now - started_at, 1.0e-9)
                rate = completed_count / elapsed
                remaining = (len(entries) - completed_count) / max(rate, 1.0e-9)
                print(
                    "AMASS -> Bumi progress:",
                    f"completed={completed_count}/{len(entries)}",
                    f"succeeded={len(successes)}",
                    f"rejected={len(rejects)}",
                    f"rate={rate:.2f}_clips/s",
                    f"eta={remaining / 60.0:.1f}_min",
                    flush=True,
                )
                last_progress_at = now
    return _finalize_batch(
        entries,
        successes,
        rejects,
        output_root=output_root,
    )
