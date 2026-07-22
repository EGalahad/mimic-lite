#!/usr/bin/env python3
"""Validate Bumi tracking NPZ files and stage an any4hdmi legacy dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Literal
from uuid import uuid4

import numpy as np

from mimic_lite_conversion.bumi import (
    BUMI_V1_TRAINING_MAX_BODY_ANGULAR_VELOCITY_NORM,
    BUMI_V1_TRAINING_MAX_BODY_LINEAR_VELOCITY_NORM,
    BUMI_V1_TRAINING_MAX_JOINT_VELOCITY_ABS,
)

REQUIRED_FIELDS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
QUAT_NORM_ATOL = 1.0e-3

BUMI_MOTION_JOINT_NAMES = [
    "l_leg_pitch_joint",
    "r_leg_pitch_joint",
    "waist_yaw_joint",
    "l_leg_roll_joint",
    "r_leg_roll_joint",
    "l_arm_pitch_joint",
    "r_arm_pitch_joint",
    "l_leg_yaw_joint",
    "r_leg_yaw_joint",
    "l_arm_roll_joint",
    "r_arm_roll_joint",
    "l_knee_pitch_joint",
    "r_knee_pitch_joint",
    "l_arm_yaw_joint",
    "r_arm_yaw_joint",
    "l_ankle_pitch_joint",
    "r_ankle_pitch_joint",
    "l_elbow_pitch_joint",
    "r_elbow_pitch_joint",
    "l_ankle_roll_joint",
    "r_ankle_roll_joint",
]
BUMI_MOTION_BODY_NAMES = [
    "motion_root",
    "base_link",
    "waist_yaw_link",
    "l_arm_pitch_link",
    "l_arm_roll_link",
    "l_arm_yaw_link",
    "l_elbow_pitch_link",
    "r_arm_pitch_link",
    "r_arm_roll_link",
    "r_arm_yaw_link",
    "r_elbow_pitch_link",
    "l_leg_pitch_link",
    "l_leg_roll_link",
    "l_leg_yaw_link",
    "l_knee_pitch_link",
    "l_ankle_pitch_link",
    "l_ankle_roll_link",
    "r_leg_pitch_link",
    "r_leg_roll_link",
    "r_leg_yaw_link",
    "r_knee_pitch_link",
    "r_ankle_pitch_link",
    "r_ankle_roll_link",
]


@dataclass(frozen=True)
class MotionStats:
    frames: int
    max_joint_pos_abs: float
    max_joint_vel_abs: float
    max_body_pos_z: float
    max_body_lin_vel_norm: float
    max_body_ang_vel_norm: float
    max_body_quat_norm_error: float
    max_motion_root_pos_abs: float
    max_motion_root_quat_error: float
    max_motion_root_lin_vel_abs: float
    max_motion_root_ang_vel_abs: float


def validate_motion(path: Path) -> MotionStats:
    with np.load(path, allow_pickle=False) as payload:
        missing = [field for field in REQUIRED_FIELDS if field not in payload]
        if missing:
            raise ValueError(f"{path.name}: missing fields: {', '.join(missing)}")

        fps_values = np.asarray(payload["fps"]).reshape(-1)
        if fps_values.size != 1 or not math.isclose(float(fps_values[0]), 50.0):
            raise ValueError(f"{path.name}: expected fps=50, got {fps_values.tolist()}")

        arrays = {field: np.asarray(payload[field]) for field in REQUIRED_FIELDS[1:]}

    joint_pos = arrays["joint_pos"]
    if joint_pos.ndim != 2:
        raise ValueError(
            f"{path.name}: joint_pos must be rank 2, got {joint_pos.shape}"
        )
    frames = int(joint_pos.shape[0])
    if frames == 0:
        raise ValueError(f"{path.name}: motion must contain at least one frame")
    expected_shapes = {
        "joint_pos": (frames, 21),
        "joint_vel": (frames, 21),
        "body_pos_w": (frames, 23, 3),
        "body_quat_w": (frames, 23, 4),
        "body_lin_vel_w": (frames, 23, 3),
        "body_ang_vel_w": (frames, 23, 3),
    }
    for field, expected_shape in expected_shapes.items():
        if arrays[field].shape != expected_shape:
            raise ValueError(
                f"{path.name}: {field} expected shape {expected_shape}, got {arrays[field].shape}"
            )
        if not np.isfinite(arrays[field]).all():
            raise ValueError(f"{path.name}: {field} contains non-finite values")

    joint_pos_abs = np.abs(arrays["joint_pos"])
    joint_vel_abs = np.abs(arrays["joint_vel"])
    body_pos_z = arrays["body_pos_w"][..., 2]
    body_lin_vel_norm = np.linalg.norm(arrays["body_lin_vel_w"], axis=-1)
    body_ang_vel_norm = np.linalg.norm(arrays["body_ang_vel_w"], axis=-1)
    body_quat_norm_error = np.abs(np.linalg.norm(arrays["body_quat_w"], axis=-1) - 1.0)
    motion_root_quat = arrays["body_quat_w"][:, 0]
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    motion_root_quat_error = np.minimum(
        np.linalg.norm(motion_root_quat - identity, axis=-1),
        np.linalg.norm(motion_root_quat + identity, axis=-1),
    )

    checks = (
        ("joint_pos abs", joint_pos_abs, 3.0),
        (
            "joint_vel abs",
            joint_vel_abs,
            BUMI_V1_TRAINING_MAX_JOINT_VELOCITY_ABS,
        ),
        ("body_pos z", body_pos_z, 2.5),
        (
            "body_lin_vel norm",
            body_lin_vel_norm,
            BUMI_V1_TRAINING_MAX_BODY_LINEAR_VELOCITY_NORM,
        ),
        (
            "body_ang_vel norm",
            body_ang_vel_norm,
            BUMI_V1_TRAINING_MAX_BODY_ANGULAR_VELOCITY_NORM,
        ),
        ("body_quat norm error", body_quat_norm_error, QUAT_NORM_ATOL),
    )
    for label, values, upper_bound in checks:
        maximum = float(values.max(initial=0.0))
        violates_limit = (
            maximum > upper_bound
            if label == "body_quat norm error"
            else maximum >= upper_bound
        )
        if violates_limit:
            raise ValueError(
                f"{path.name}: {label} exceeds limit {upper_bound:g} (max={maximum:g})"
            )

    return MotionStats(
        frames=frames,
        max_joint_pos_abs=float(joint_pos_abs.max(initial=0.0)),
        max_joint_vel_abs=float(joint_vel_abs.max(initial=0.0)),
        max_body_pos_z=float(body_pos_z.max(initial=0.0)),
        max_body_lin_vel_norm=float(body_lin_vel_norm.max(initial=0.0)),
        max_body_ang_vel_norm=float(body_ang_vel_norm.max(initial=0.0)),
        max_body_quat_norm_error=float(body_quat_norm_error.max(initial=0.0)),
        max_motion_root_pos_abs=float(
            np.abs(arrays["body_pos_w"][:, 0]).max(initial=0.0)
        ),
        max_motion_root_quat_error=float(
            motion_root_quat_error.max(initial=0.0)
        ),
        max_motion_root_lin_vel_abs=float(
            np.abs(arrays["body_lin_vel_w"][:, 0]).max(initial=0.0)
        ),
        max_motion_root_ang_vel_abs=float(
            np.abs(arrays["body_ang_vel_w"][:, 0]).max(initial=0.0)
        ),
    )


def _tree_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        if path.is_symlink():
            digest.update(b"symlink\0")
            digest.update(os.readlink(path).encode())
        elif path.is_file():
            digest.update(b"file\0")
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        elif path.is_dir():
            digest.update(b"dir\0")
    return digest.hexdigest()


def _install_staged_tree(stage: Path, output: Path, *, force: bool) -> str:
    if output.exists():
        if _tree_fingerprint(stage) == _tree_fingerprint(output):
            return "unchanged"
        if not force:
            raise FileExistsError(
                f"Output exists with different content: {output}. Pass --force to replace it."
            )
        backup = output.parent / f".{output.name}.backup-{uuid4().hex}"
        output.rename(backup)
        try:
            stage.rename(output)
        except Exception:
            backup.rename(output)
            raise
        shutil.rmtree(backup)
        return "replaced"
    stage.rename(output)
    return "created"


def prepare_bumi_motion_dataset(
    source: Path,
    output: Path,
    *,
    link_mode: Literal["symlink", "copy"] = "symlink",
    force: bool = False,
    report_json: Path | None = None,
    quality_report: Path | None = None,
) -> dict[str, int | float | str]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Bumi motion source directory not found: {source}")
    motion_paths = sorted(source.glob("*.npz"))
    if not motion_paths:
        raise FileNotFoundError(f"No NPZ motions found directly under {source}")
    selection = "all"
    if quality_report is not None:
        quality_path = quality_report.expanduser().resolve()
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        expected_source = (
            Path(quality["conversion_root"]).expanduser().resolve() / "tracker_50hz"
        )
        if expected_source != source:
            raise ValueError(
                "Quality report does not describe the requested source: "
                f"{expected_source} / {source}"
            )
        if quality.get("pipeline_integrity_ready") is not True:
            raise ValueError(
                "Quality report has not passed pipeline_integrity_ready; "
                "resolve rejects, missing clips, stale provenance, or physical-limit "
                "violations before staging"
            )
        selected_ids = quality.get("automatic_training_ready_clip_ids")
        if not isinstance(selected_ids, list) or not all(
            isinstance(value, str) for value in selected_ids
        ):
            raise ValueError(
                "Quality report is missing automatic_training_ready_clip_ids"
            )
        paths_by_id = {path.stem: path for path in motion_paths}
        missing = sorted(set(selected_ids) - set(paths_by_id))
        if missing:
            raise ValueError(
                "Quality report selects missing tracker clips: " + ", ".join(missing)
            )
        motion_paths = [paths_by_id[identifier] for identifier in selected_ids]
        if not motion_paths:
            raise ValueError("Quality report selected no automatic training-ready clips")
        selection = "automatic_training_ready_clip_ids"
    if link_mode not in {"symlink", "copy"}:
        raise ValueError(f"Unsupported link mode: {link_mode}")

    stats = [validate_motion(path) for path in motion_paths]
    meta = {
        "fps": 50,
        "joint_names": BUMI_MOTION_JOINT_NAMES,
        "body_names": BUMI_MOTION_BODY_NAMES,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.stage-", dir=output.parent
    ) as temp_dir:
        stage = Path(temp_dir) / output.name
        stage.mkdir()
        for motion_path in motion_paths:
            motion_dir = stage / motion_path.stem
            motion_dir.mkdir()
            target = motion_dir / "motion.npz"
            if link_mode == "symlink":
                target.symlink_to(motion_path.resolve())
            else:
                shutil.copy2(motion_path, target)
            (motion_dir / "meta.json").write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8"
            )
        status = _install_staged_tree(stage, output, force=force)

    result: dict[str, int | float | str] = {
        "status": status,
        "source": str(source),
        "output": str(output),
        "motions": len(motion_paths),
        "frames": sum(item.frames for item in stats),
        "fps": 50,
        "selection": selection,
        "max_joint_pos_abs": max(item.max_joint_pos_abs for item in stats),
        "max_joint_vel_abs": max(item.max_joint_vel_abs for item in stats),
        "max_body_pos_z": max(item.max_body_pos_z for item in stats),
        "max_body_lin_vel_norm": max(item.max_body_lin_vel_norm for item in stats),
        "max_body_ang_vel_norm": max(item.max_body_ang_vel_norm for item in stats),
        "max_body_quat_norm_error": max(
            item.max_body_quat_norm_error for item in stats
        ),
        "max_motion_root_pos_abs": max(item.max_motion_root_pos_abs for item in stats),
        "max_motion_root_quat_error": max(
            item.max_motion_root_quat_error for item in stats
        ),
        "max_motion_root_lin_vel_abs": max(
            item.max_motion_root_lin_vel_abs for item in stats
        ),
        "max_motion_root_ang_vel_abs": max(
            item.max_motion_root_ang_vel_abs for item in stats
        ),
    }
    if report_json is not None:
        report_output = report_json.expanduser().resolve()
        report_output.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=f".{report_output.name}.",
                dir=report_output.parent,
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(result, stream, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(temporary, report_output)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    print(
        "Prepared Bumi motions:",
        " ".join(f"{key}={value}" for key, value in result.items()),
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / ".cache"
        / "mimic-lite"
        / "motions"
        / "bumi"
        / "omni",
    )
    parser.add_argument("--link-mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-json", type=Path)
    parser.add_argument(
        "--quality-report",
        type=Path,
        help="Stage only automatic_training_ready_clip_ids from a quality report.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prepare_bumi_motion_dataset(
        args.source,
        args.output,
        link_mode=args.link_mode,
        force=args.force,
        report_json=args.report_json,
        quality_report=args.quality_report,
    )


if __name__ == "__main__":
    main()
