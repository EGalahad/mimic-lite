#!/usr/bin/env python3
"""Aggregate conversion provenance, geometry gates and final Bumi NPZ quality."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from mimic_lite_conversion.amass_gmr import (
    clip_id,
    load_manifest,
    sha256_file,
    write_json,
)
from mimic_lite_conversion.bumi import (
    BUMI_JOINT_POSITION_LIMITS,
    BUMI_JOINT_VELOCITY_LIMITS,
    BUMI_MOTION_BODY_NAMES,
    BUMI_MOTION_JOINT_NAMES,
    BUMI_V1_TRAINING_MAX_BODY_ANGULAR_VELOCITY_NORM,
    BUMI_V1_TRAINING_MAX_BODY_LINEAR_VELOCITY_NORM,
    BUMI_V1_TRAINING_MAX_JOINT_VELOCITY_ABS,
)


def _tracker_quality(path: Path) -> dict[str, float]:
    with np.load(path, allow_pickle=False) as payload:
        joint_pos = np.asarray(payload["joint_pos"], dtype=np.float64)
        joint_vel = np.asarray(payload["joint_vel"], dtype=np.float64)
        body_pos = np.asarray(payload["body_pos_w"], dtype=np.float64)
        body_quat = np.asarray(payload["body_quat_w"], dtype=np.float64)
        body_lin_vel = np.asarray(payload["body_lin_vel_w"], dtype=np.float64)
        body_ang_vel = np.asarray(payload["body_ang_vel_w"], dtype=np.float64)
    velocity_limits = np.asarray(
        [BUMI_JOINT_VELOCITY_LIMITS[name] for name in BUMI_MOTION_JOINT_NAMES],
        dtype=np.float64,
    )
    velocity_ratio = np.abs(joint_vel) / velocity_limits[None, :]
    position_lower = np.asarray(
        [BUMI_JOINT_POSITION_LIMITS[name][0] for name in BUMI_MOTION_JOINT_NAMES],
        dtype=np.float64,
    )
    position_upper = np.asarray(
        [BUMI_JOINT_POSITION_LIMITS[name][1] for name in BUMI_MOTION_JOINT_NAMES],
        dtype=np.float64,
    )
    position_violation = np.maximum(
        position_lower[None, :] - joint_pos,
        joint_pos - position_upper[None, :],
    )
    quaternion_norm_error = np.abs(np.linalg.norm(body_quat, axis=-1) - 1.0)
    if body_pos.shape[0] > 1:
        max_body_step_m = float(
            np.linalg.norm(np.diff(body_pos[:, 1:], axis=0), axis=-1).max()
        )
    else:
        max_body_step_m = 0.0

    foot_indices = [
        BUMI_MOTION_BODY_NAMES.index("l_ankle_roll_link"),
        BUMI_MOTION_BODY_NAMES.index("r_ankle_roll_link"),
    ]
    foot_z = body_pos[:, foot_indices, 2]
    foot_horizontal_speed = np.linalg.norm(
        body_lin_vel[:, foot_indices, :2],
        axis=-1,
    )
    low_threshold = np.percentile(foot_z, 10.0, axis=0) + 0.02
    low_mask = foot_z <= low_threshold[None, :]
    low_foot_speed = foot_horizontal_speed[low_mask]
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    motion_root_quat_error = np.minimum(
        np.linalg.norm(body_quat[:, 0] - identity, axis=-1),
        np.linalg.norm(body_quat[:, 0] + identity, axis=-1),
    )
    return {
        "max_joint_position_limit_violation": float(
            position_violation.max(initial=0.0)
        ),
        "max_joint_velocity_abs": float(np.abs(joint_vel).max(initial=0.0)),
        "max_joint_velocity_ratio": float(velocity_ratio.max(initial=0.0)),
        "max_body_linear_velocity_norm": float(
            np.linalg.norm(body_lin_vel, axis=-1).max(initial=0.0)
        ),
        "max_body_angular_velocity_norm": float(
            np.linalg.norm(body_ang_vel, axis=-1).max(initial=0.0)
        ),
        "max_body_quaternion_norm_error": float(
            quaternion_norm_error.max(initial=0.0)
        ),
        "max_body_step_m": max_body_step_m,
        "min_ankle_body_height_m": float(foot_z.min(initial=np.inf)),
        "p95_low_ankle_horizontal_speed_mps": float(
            np.percentile(low_foot_speed, 95.0) if low_foot_speed.size else 0.0
        ),
        "max_motion_root_pos_abs": float(np.abs(body_pos[:, 0]).max(initial=0.0)),
        "max_motion_root_lin_vel_abs": float(
            np.abs(body_lin_vel[:, 0]).max(initial=0.0)
        ),
        "max_motion_root_ang_vel_abs": float(
            np.abs(body_ang_vel[:, 0]).max(initial=0.0)
        ),
        "max_motion_root_quat_error": float(
            motion_root_quat_error.max(initial=0.0)
        ),
    }


def build_report(
    conversion_root: Path,
    *,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    root = conversion_root.expanduser().resolve()
    manifest_entries = load_manifest(manifest_path) if manifest_path is not None else []
    expected_by_id = {clip_id(entry): entry for entry in manifest_entries}
    expected_ids = set(expected_by_id) if manifest_path is not None else None

    rejected_path = root / "reports" / "rejected.jsonl"
    rejects = []
    if rejected_path.is_file():
        rejects = [
            json.loads(line)
            for line in rejected_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    converted_path = root / "reports" / "converted.jsonl"
    current_success_ids: set[str] | None = None
    if converted_path.is_file():
        current_success_ids = {
            str(item["clip_id"])
            for item in (
                json.loads(line)
                for line in converted_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }

    metadata_paths = sorted((root / "metadata").glob("*.json"))
    records = []
    stale_metadata_clip_ids = []
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        result = dict(metadata["result"])
        identifier = str(result["clip_id"])
        if expected_ids is not None and identifier not in expected_ids:
            continue
        if current_success_ids is not None and identifier not in current_success_ids:
            stale_metadata_clip_ids.append(identifier)
            continue
        if expected_ids is not None:
            entry = expected_by_id[identifier]
            provenance = metadata.get("provenance", {})
            if (
                provenance.get("source_relative_path") != entry.relative_path
                or provenance.get("source_sha256") != entry.source_sha256
                or provenance.get("source_frames") != entry.frames
                or not np.isclose(
                    float(provenance.get("source_fps", np.nan)),
                    entry.source_fps,
                )
            ):
                stale_metadata_clip_ids.append(identifier)
                continue
        tracker_path = root / "tracker_50hz" / f"{result['clip_id']}.npz"
        if not tracker_path.is_file():
            raise FileNotFoundError(f"Missing tracker file: {tracker_path}")
        if sha256_file(tracker_path) != metadata["tracker_sha256"]:
            raise ValueError(f"Tracker hash mismatch: {tracker_path}")
        records.append({**result, **_tracker_quality(tracker_path)})
    actual_ids = {str(record["clip_id"]) for record in records}
    missing_ids = sorted(expected_ids - actual_ids) if expected_ids is not None else []

    gate_keys = (
        "gate_position_pass",
        "gate_foot_position_pass",
        "gate_orientation_pass",
        "gate_ground_contact_pass",
    )
    gate_failures = {
        key: [
            record["source_relative_path"]
            for record in records
            if not bool(record.get(key, True))
        ]
        for key in gate_keys
    }
    max_velocity_ratio = max(
        (record["max_joint_velocity_ratio"] for record in records),
        default=0.0,
    )
    max_position_violation = max(
        (record["max_joint_position_limit_violation"] for record in records),
        default=0.0,
    )
    max_native_velocity_ratio = max(
        (
            record.get(
                "max_native_joint_velocity_ratio",
                record["max_joint_velocity_ratio"],
            )
            for record in records
        ),
        default=0.0,
    )
    max_quaternion_error = max(
        (record["max_body_quaternion_norm_error"] for record in records),
        default=0.0,
    )
    motion_root_error = max(
        (
            max(
                record["max_motion_root_pos_abs"],
                record["max_motion_root_quat_error"],
                record["max_motion_root_lin_vel_abs"],
                record["max_motion_root_ang_vel_abs"],
            )
            for record in records
        ),
        default=0.0,
    )
    pipeline_integrity_ready = bool(
        records
        and not rejects
        and not missing_ids
        and not stale_metadata_clip_ids
        and max_position_violation <= 1.0e-5
        and max_velocity_ratio <= 1.05
        and max_native_velocity_ratio <= 1.05
        and max_quaternion_error < 1.0e-5
        and motion_root_error == 0.0
    )

    def geometry_pass(record: dict[str, Any]) -> bool:
        return all(bool(record.get(key, True)) for key in gate_keys)

    def conservative_dynamics_pass(record: dict[str, Any]) -> bool:
        return bool(
            record["max_joint_velocity_abs"]
            < BUMI_V1_TRAINING_MAX_JOINT_VELOCITY_ABS
            and record["max_body_linear_velocity_norm"]
            < BUMI_V1_TRAINING_MAX_BODY_LINEAR_VELOCITY_NORM
            and record["max_body_angular_velocity_norm"]
            < BUMI_V1_TRAINING_MAX_BODY_ANGULAR_VELOCITY_NORM
        )

    def record_integrity_pass(record: dict[str, Any]) -> bool:
        root_error = max(
            record["max_motion_root_pos_abs"],
            record["max_motion_root_quat_error"],
            record["max_motion_root_lin_vel_abs"],
            record["max_motion_root_ang_vel_abs"],
        )
        return bool(
            record["max_joint_position_limit_violation"] <= 1.0e-5
            and record["max_joint_velocity_ratio"] <= 1.05
            and record.get(
                "max_native_joint_velocity_ratio",
                record["max_joint_velocity_ratio"],
            )
            <= 1.05
            and record["max_body_quaternion_norm_error"] < 1.0e-5
            and root_error == 0.0
        )

    automatic_training_ready = sorted(
        str(record["clip_id"])
        for record in records
        if record_integrity_pass(record)
        and geometry_pass(record)
        and conservative_dynamics_pass(record)
    )
    geometry_review = sorted(
        str(record["clip_id"])
        for record in records
        if record_integrity_pass(record)
        and not geometry_pass(record)
        and conservative_dynamics_pass(record)
    )
    dynamics_review = sorted(
        str(record["clip_id"])
        for record in records
        if record_integrity_pass(record) and not conservative_dynamics_pass(record)
    )
    integrity_failures = sorted(
        str(record["clip_id"])
        for record in records
        if not record_integrity_pass(record)
    )
    production_ready = bool(
        pipeline_integrity_ready
        and len(automatic_training_ready) == len(records)
    )
    return {
        "conversion_root": str(root),
        "expected": len(expected_ids) if expected_ids is not None else None,
        "succeeded": len(records),
        "rejected": len(rejects),
        "missing_clip_ids": missing_ids,
        "stale_metadata_clip_ids": sorted(set(stale_metadata_clip_ids)),
        "source_fps": dict(
            sorted(Counter(f"{record['source_fps']:g}" for record in records).items())
        ),
        "gate_failures": gate_failures,
        "max_joint_position_limit_violation": max_position_violation,
        "max_joint_velocity_ratio": max_velocity_ratio,
        "max_native_joint_velocity_ratio": max_native_velocity_ratio,
        "max_joint_velocity_abs": max(
            (record["max_joint_velocity_abs"] for record in records), default=0.0
        ),
        "max_body_linear_velocity_norm": max(
            (record["max_body_linear_velocity_norm"] for record in records),
            default=0.0,
        ),
        "max_body_angular_velocity_norm": max(
            (record["max_body_angular_velocity_norm"] for record in records),
            default=0.0,
        ),
        "max_body_quaternion_norm_error": max_quaternion_error,
        "max_motion_root_contract_error": motion_root_error,
        "max_body_step_m": max(
            (record["max_body_step_m"] for record in records), default=0.0
        ),
        "min_ankle_body_height_m": min(
            (record["min_ankle_body_height_m"] for record in records),
            default=0.0,
        ),
        "max_p95_low_ankle_horizontal_speed_mps": max(
            (
                record["p95_low_ankle_horizontal_speed_mps"]
                for record in records
            ),
            default=0.0,
        ),
        "max_abs_ground_requested_root_z_offset_m": max(
            (
                abs(record.get("ground_requested_root_z_offset_m", 0.0))
                for record in records
            ),
            default=0.0,
        ),
        "max_abs_ground_applied_root_z_offset_m": max(
            (
                abs(record.get("ground_applied_root_z_offset_m", 0.0))
                for record in records
            ),
            default=0.0,
        ),
        "ground_alignment_clipped": sum(
            bool(record.get("ground_alignment_clipped", False))
            for record in records
        ),
        "min_foot_contact_height_after_m": min(
            (
                record.get(
                    "min_foot_contact_height_after_m",
                    record.get("native_min_foot_contact_height_m", 0.0),
                )
                for record in records
            ),
            default=0.0,
        ),
        "max_foot_contact_below_ground_fraction_after": max(
            (
                record.get(
                    "foot_contact_below_ground_fraction_after",
                    record.get(
                        "native_foot_contact_below_ground_fraction",
                        0.0,
                    ),
                )
                for record in records
            ),
            default=0.0,
        ),
        "min_native_contact_foot_height_p05_m": min(
            (
                record.get("native_contact_foot_height_p05_m", 0.0)
                for record in records
            ),
            default=0.0,
        ),
        "min_native_contact_foot_height_median_m": min(
            (
                record.get("native_contact_foot_height_median_m", 0.0)
                for record in records
            ),
            default=0.0,
        ),
        "max_native_contact_foot_height_median_m": max(
            (
                record.get("native_contact_foot_height_median_m", 0.0)
                for record in records
            ),
            default=0.0,
        ),
        "max_native_contact_foot_below_minus_5mm_fraction": max(
            (
                record.get(
                    "native_contact_foot_below_minus_5mm_fraction",
                    0.0,
                )
                for record in records
            ),
            default=0.0,
        ),
        "min_native_double_support_low_foot_height_median_m": min(
            (
                record.get(
                    "native_double_support_low_foot_height_median_m",
                    0.0,
                )
                for record in records
            ),
            default=0.0,
        ),
        "max_native_double_support_high_foot_height_median_m": max(
            (
                record.get(
                    "native_double_support_high_foot_height_median_m",
                    0.0,
                )
                for record in records
            ),
            default=0.0,
        ),
        "max_abs_input_height_offset_m": max(
            (
                abs(record.get("input_height_offset_m", 0.0))
                for record in records
            ),
            default=0.0,
        ),
        "min_tracker_foot_contact_height_m": min(
            (
                record.get("tracker_min_foot_contact_height_m", 0.0)
                for record in records
            ),
            default=0.0,
        ),
        "max_tracker_foot_contact_below_ground_fraction": max(
            (
                record.get(
                    "tracker_foot_contact_below_ground_fraction",
                    0.0,
                )
                for record in records
            ),
            default=0.0,
        ),
        "pipeline_integrity_ready": pipeline_integrity_ready,
        "geometry_all_pass": all(not failures for failures in gate_failures.values()),
        "automatic_training_ready": len(automatic_training_ready),
        "automatic_training_ready_clip_ids": automatic_training_ready,
        "geometry_review": len(geometry_review),
        "geometry_review_clip_ids": geometry_review,
        "dynamics_review": len(dynamics_review),
        "dynamics_review_clip_ids": dynamics_review,
        "integrity_failures": len(integrity_failures),
        "integrity_failure_clip_ids": integrity_failures,
        "production_ready": production_ready,
        "rejected_records": rejects,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-production-ready", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_report(args.conversion_root, manifest_path=args.manifest)
    output = args.output or args.conversion_root / "reports" / "quality_summary.json"
    write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_production_ready and not report["production_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
