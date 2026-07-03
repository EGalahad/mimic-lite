from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

from any4hdmi import BaseDataset as Any4HDMIBaseDataset, MotionData, load_any4hdmi_dataset
MOTION_DATASET_VALIDATE_CHUNK_SIZE = 131072
MOTION_DATASET_QUAT_NORM_ATOL = 1e-3


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def create_dataset_from_path(
    root_path: str | List[str],
    target_fps: int = 50,
    num_envs: int = 1,
    full_motion: bool = True,
    filenames: list[str] | None = None,
    filenames_path: str | None = None,
    body_names: list[str] | None = None,
    joint_names: list[str] | None = None,
    windowed_next_window_device: str | torch.device | None = "current",
    windowed_pin_window_load: bool = True,
) -> Any4HDMIBaseDataset:
    import active_adaptation

    base_dir = Path(active_adaptation.__file__).parent.parent
    dataset = load_any4hdmi_dataset(
        root_path=root_path,
        target_fps=target_fps,
        base_dir=base_dir,
        num_envs=num_envs,
        full_motion=full_motion,
        filenames=filenames,
        filenames_path=filenames_path,
        body_names=body_names,
        joint_names=joint_names,
        windowed_next_window_device=windowed_next_window_device,
        windowed_pin_window_load=windowed_pin_window_load,
    )
    if not isinstance(dataset, Any4HDMIBaseDataset):
        raise TypeError(
            f"Expected any4hdmi BaseDataset-compatible loader, got {type(dataset)!r}"
        )
    return dataset


def _motion_dataset_total_steps(dataset: Any4HDMIBaseDataset) -> int:
    ends = dataset.ends
    if not isinstance(ends, torch.Tensor):
        raise TypeError(f"Expected dataset.ends to be a torch.Tensor, got {type(ends)!r}")
    if ends.numel() == 0:
        return 0
    return int(ends.max().item())


def _motion_dataset_field_chunk(
    dataset: Any4HDMIBaseDataset,
    *,
    field_name: str,
    start: int,
    end: int,
) -> torch.Tensor:
    if hasattr(dataset, "_storage_cpu"):
        storage = getattr(dataset, "_storage_cpu")
        if field_name not in storage:
            raise KeyError(f"Dataset storage does not contain field {field_name!r}")
        field = storage[field_name][start:end]
    elif hasattr(dataset, "data") and hasattr(dataset.data, field_name):
        field = getattr(dataset.data, field_name)[start:end]
    else:
        raise AttributeError(f"Dataset does not expose field {field_name!r} for validation")
    if not isinstance(field, torch.Tensor):
        field = torch.as_tensor(field)
    return field.detach().to(device="cpu")


def _max_finite_or_nan(x: torch.Tensor) -> float:
    finite = x[torch.isfinite(x)]
    if finite.numel() == 0:
        return float("nan")
    return float(finite.max().item())


def _validate_motion_dataset(dataset: Any4HDMIBaseDataset) -> None:
    total_steps = _motion_dataset_total_steps(dataset)
    if total_steps == 0:
        print("Motion dataset validation skipped: dataset is empty")
        return

    chunk_size = _env_int("MIMIC_LITE_MOTION_VALIDATE_CHUNK_SIZE", MOTION_DATASET_VALIDATE_CHUNK_SIZE)
    quat_norm_atol = float(os.environ.get("MIMIC_LITE_MOTION_VALIDATE_QUAT_NORM_ATOL", MOTION_DATASET_QUAT_NORM_ATOL))
    violation_counts = {
        "joint_pos_abs": 0,
        "joint_vel_abs": 0,
        "body_pos_z": 0,
        "body_lin_vel_norm": 0,
        "body_ang_vel_norm": 0,
        "body_quat_norm": 0,
        "any": 0,
    }
    max_stats = {
        "joint_pos_abs": float("-inf"),
        "joint_vel_abs": float("-inf"),
        "body_pos_z": float("-inf"),
        "body_lin_vel_norm": float("-inf"),
        "body_ang_vel_norm": float("-inf"),
        "body_quat_norm_error": float("-inf"),
    }

    chunk_starts = range(0, total_steps, chunk_size)
    for start in tqdm(chunk_starts, total=(total_steps + chunk_size - 1) // chunk_size, desc="Validating motion dataset", unit="chunk"):
        end = min(total_steps, start + chunk_size)
        joint_pos = _motion_dataset_field_chunk(dataset, field_name="joint_pos", start=start, end=end)
        joint_vel = _motion_dataset_field_chunk(dataset, field_name="joint_vel", start=start, end=end)
        body_pos_w = _motion_dataset_field_chunk(dataset, field_name="body_pos_w", start=start, end=end)
        body_lin_vel_w = _motion_dataset_field_chunk(dataset, field_name="body_lin_vel_w", start=start, end=end)
        body_ang_vel_w = _motion_dataset_field_chunk(dataset, field_name="body_ang_vel_w", start=start, end=end)
        body_quat_w = _motion_dataset_field_chunk(dataset, field_name="body_quat_w", start=start, end=end)

        joint_pos_abs = joint_pos.abs()
        joint_vel_abs = joint_vel.abs()
        body_pos_z = body_pos_w[..., 2]
        body_lin_vel_norm = torch.linalg.vector_norm(body_lin_vel_w, dim=-1)
        body_ang_vel_norm = torch.linalg.vector_norm(body_ang_vel_w, dim=-1)
        body_quat_norm = torch.linalg.vector_norm(body_quat_w, dim=-1)
        body_quat_norm_error = (body_quat_norm - 1.0).abs()

        joint_pos_bad = (~torch.isfinite(joint_pos_abs)) | (joint_pos_abs >= 3.0)
        joint_vel_bad = (~torch.isfinite(joint_vel_abs)) | (joint_vel_abs >= 20.0)
        body_pos_z_bad = (~torch.isfinite(body_pos_z)) | (body_pos_z >= 2.5)
        body_lin_vel_bad = (~torch.isfinite(body_lin_vel_norm)) | (body_lin_vel_norm >= 10.0)
        body_ang_vel_bad = (~torch.isfinite(body_ang_vel_norm)) | (body_ang_vel_norm >= 30.0)
        body_quat_bad = (~torch.isfinite(body_quat_norm_error)) | (body_quat_norm_error > quat_norm_atol)

        joint_pos_bad_frames = joint_pos_bad.any(dim=1)
        joint_vel_bad_frames = joint_vel_bad.any(dim=1)
        body_pos_z_bad_frames = body_pos_z_bad.any(dim=1)
        body_lin_vel_bad_frames = body_lin_vel_bad.any(dim=1)
        body_ang_vel_bad_frames = body_ang_vel_bad.any(dim=1)
        body_quat_bad_frames = body_quat_bad.any(dim=1)
        any_bad_frames = (
            joint_pos_bad_frames
            | joint_vel_bad_frames
            | body_pos_z_bad_frames
            | body_lin_vel_bad_frames
            | body_ang_vel_bad_frames
            | body_quat_bad_frames
        )

        violation_counts["joint_pos_abs"] += int(joint_pos_bad_frames.sum().item())
        violation_counts["joint_vel_abs"] += int(joint_vel_bad_frames.sum().item())
        violation_counts["body_pos_z"] += int(body_pos_z_bad_frames.sum().item())
        violation_counts["body_lin_vel_norm"] += int(body_lin_vel_bad_frames.sum().item())
        violation_counts["body_ang_vel_norm"] += int(body_ang_vel_bad_frames.sum().item())
        violation_counts["body_quat_norm"] += int(body_quat_bad_frames.sum().item())
        violation_counts["any"] += int(any_bad_frames.sum().item())

        max_stats["joint_pos_abs"] = max(max_stats["joint_pos_abs"], _max_finite_or_nan(joint_pos_abs))
        max_stats["joint_vel_abs"] = max(max_stats["joint_vel_abs"], _max_finite_or_nan(joint_vel_abs))
        max_stats["body_pos_z"] = max(max_stats["body_pos_z"], _max_finite_or_nan(body_pos_z))
        max_stats["body_lin_vel_norm"] = max(
            max_stats["body_lin_vel_norm"],
            _max_finite_or_nan(body_lin_vel_norm),
        )
        max_stats["body_ang_vel_norm"] = max(
            max_stats["body_ang_vel_norm"],
            _max_finite_or_nan(body_ang_vel_norm),
        )
        max_stats["body_quat_norm_error"] = max(
            max_stats["body_quat_norm_error"],
            _max_finite_or_nan(body_quat_norm_error),
        )

    if violation_counts["any"] > 0:
        raise RuntimeError(
            "Motion dataset validation failed: "
            f"{violation_counts['any']} invalid frames out of {total_steps}. "
            f"joint_pos abs < 2 violated on {violation_counts['joint_pos_abs']} frames (max={max_stats['joint_pos_abs']:.6g}); "
            f"joint_vel abs < 20 violated on {violation_counts['joint_vel_abs']} frames (max={max_stats['joint_vel_abs']:.6g}); "
            f"body_pos[..., 2] < 2.5 violated on {violation_counts['body_pos_z']} frames (max={max_stats['body_pos_z']:.6g}); "
            f"body_lin_vel norm < 10 violated on {violation_counts['body_lin_vel_norm']} frames (max={max_stats['body_lin_vel_norm']:.6g}); "
            f"body_ang_vel norm < 10 violated on {violation_counts['body_ang_vel_norm']} frames (max={max_stats['body_ang_vel_norm']:.6g}); "
            f"body_quat norm ~= 1 violated on {violation_counts['body_quat_norm']} frames "
            f"(max error={max_stats['body_quat_norm_error']:.6g}, atol={quat_norm_atol:.6g})"
        )

    print(
        "Motion dataset validation passed:",
        f"frames={total_steps}",
        f"max_joint_pos_abs={max_stats['joint_pos_abs']:.6g}",
        f"max_joint_vel_abs={max_stats['joint_vel_abs']:.6g}",
        f"max_body_pos_z={max_stats['body_pos_z']:.6g}",
        f"max_body_lin_vel_norm={max_stats['body_lin_vel_norm']:.6g}",
        f"max_body_ang_vel_norm={max_stats['body_ang_vel_norm']:.6g}",
        f"max_body_quat_norm_error={max_stats['body_quat_norm_error']:.6g}",
        f"quat_norm_atol={quat_norm_atol:.6g}",
    )


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a motion dataset and materialize the qpos cache when applicable."
    )
    parser.add_argument(
        "dataset_path",
        nargs="+",
        help="Dataset root, subdirectory, or motion file path to load.",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=50,
        help="Target FPS used for interpolation and qpos cache keys.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    root_path: str | list[str]
    if len(args.dataset_path) == 1:
        root_path = args.dataset_path[0]
    else:
        root_path = args.dataset_path

    dataset = create_dataset_from_path(
        root_path=root_path,
        target_fps=args.target_fps,
    )
    _validate_motion_dataset(dataset)
    print(
        "Loaded motion dataset:",
        f"num_motions={dataset.num_motions}",
        f"num_steps={dataset.num_steps}",
        f"device={dataset.device}",
    )


if __name__ == "__main__":
    main()
