#!/usr/bin/env python3
"""Browse Bumi tracker NPZ motions with a lazy-loading Viser player.

The player accepts either one tracker NPZ or a directory of NPZ files.  It
renders Bumi through the same MuJoCo model used by the exporter, the original
SMPL-X FK motion saved before GMR as HumanPose24, and an optional final-Bumi-FK
debug overlay.  Only the selected clip is held in memory, so the full AMASS
conversion can be browsed without loading every motion at startup.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any

import mujoco
import numpy as np

from mimic_lite_conversion.bumi import (
    BUMI_MOTION_BODY_NAMES,
    BUMI_MOTION_JOINT_NAMES,
    validate_bumi_model,
)


_BASE_BODY_INDEX = BUMI_MOTION_BODY_NAMES.index("base_link")
_POINT_COLOR = np.asarray([40, 180, 255], dtype=np.uint8)
_LINE_COLOR = np.asarray([255, 186, 73], dtype=np.uint8)
_SOURCE_POINT_COLOR = np.asarray([202, 132, 255], dtype=np.uint8)
_SOURCE_LINE_COLOR = np.asarray([92, 230, 154], dtype=np.uint8)
_DEFAULT_OVERLAY_Y_OFFSET = 0.6
_DEFAULT_SOURCE_Y_OFFSET = -1.0
_HUMAN_POSE24_NAMES = (
    "Pelvis",
    "Left_Hip",
    "Right_Hip",
    "Spine1",
    "Left_Knee",
    "Right_Knee",
    "Spine2",
    "Left_Ankle",
    "Right_Ankle",
    "Spine3",
    "Left_Foot",
    "Right_Foot",
    "Neck",
    "Left_Collar",
    "Right_Collar",
    "Head",
    "Left_Shoulder",
    "Right_Shoulder",
    "Left_Elbow",
    "Right_Elbow",
    "Left_Wrist",
    "Right_Wrist",
    "Left_Hand",
    "Right_Hand",
)
_HUMAN_POSE24_PARENTS = (
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
)
_HUMAN_POSE24_LINKS = tuple(
    (parent, child) for child, parent in enumerate(_HUMAN_POSE24_PARENTS) if parent >= 0
)
_HUMAN_POSE24_PELVIS_INDEX = _HUMAN_POSE24_NAMES.index("Pelvis")
_QUALITY_REPORT_FIELDS = {
    "automatic": "automatic_training_ready_clip_ids",
    "geometry_review": "geometry_review_clip_ids",
    "dynamics_review": "dynamics_review_clip_ids",
    "integrity_failure": "integrity_failure_clip_ids",
}
_END_BEHAVIORS = ("Loop current", "Next clip", "Pause")


@dataclass(frozen=True)
class ClipEntry:
    path: Path
    label: str
    clip_id: str
    quality_group: str


@dataclass(frozen=True)
class TrackingClip:
    entry: ClipEntry
    fps: float
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    joint_names: tuple[str, ...]

    @property
    def frame_count(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def duration_s(self) -> float:
        return max(0, self.frame_count - 1) / self.fps


@dataclass(frozen=True)
class SourcePose24:
    path: Path
    body_pos_w: np.ndarray
    timestamps_s: np.ndarray
    source_fps: float

    @property
    def frame_count(self) -> int:
        return int(self.body_pos_w.shape[0])


@dataclass
class PlaybackState:
    clip_index: int
    clip: TrackingClip
    joint_qpos_addresses: tuple[int, ...]
    joint_dof_addresses: tuple[int, ...]
    source_pose: SourcePose24 | None
    frame: int = 0
    paused: bool = False


def active_adaptation_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_motion_path() -> Path:
    configured = os.environ.get("BUMI_MOTION_DIR")
    if configured:
        return Path(configured).expanduser()
    return (
        active_adaptation_root()
        / ".cache"
        / "mimic-lite"
        / "retarget"
        / "bumi"
        / "amass"
        / "train"
        / "tracker_50hz"
    )


def default_model_path() -> Path:
    configured = os.environ.get("BUMI_MJCF")
    if configured:
        return Path(configured).expanduser()
    return active_adaptation_root() / ".cache" / "aa-robot-models" / "bumi" / "bumi.xml"


def discover_npz_files(
    motion_path: Path,
    *,
    recursive: bool,
) -> tuple[Path, ...]:
    path = motion_path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != ".npz":
            raise ValueError(f"Expected a .npz file, got {path}")
        return (path,)
    if not path.is_dir():
        raise FileNotFoundError(f"Motion path does not exist: {path}")

    candidates = path.rglob("*.npz") if recursive else path.glob("*.npz")
    files = tuple(sorted(candidate for candidate in candidates if candidate.is_file()))
    if not files:
        mode = "recursively" if recursive else "at its top level"
        raise FileNotFoundError(
            f"Motion directory contains no .npz files {mode}: {path}"
        )
    return files


def infer_quality_report(
    motion_path: Path,
    explicit_report: Path | None,
) -> Path | None:
    if explicit_report is not None:
        return explicit_report.expanduser().resolve()
    path = motion_path.expanduser().resolve()
    if path.is_dir() and path.name == "tracker_50hz":
        candidate = path.parent / "reports" / "quality_summary.json"
        if candidate.is_file():
            return candidate
    return None


def load_quality_groups(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Quality report does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups: dict[str, str] = {}
    for group, field in _QUALITY_REPORT_FIELDS.items():
        values = payload.get(field, [])
        if not isinstance(values, list):
            raise ValueError(f"{path}: {field} must be a list")
        for raw_clip_id in values:
            clip_id = str(raw_clip_id)
            previous = groups.get(clip_id)
            if previous is not None:
                raise ValueError(
                    f"{path}: clip {clip_id!r} is in both {previous} and {group}"
                )
            groups[clip_id] = group
    return groups


def _clip_id_from_path(path: Path) -> str:
    if path.name == "motion.npz":
        return path.parent.name
    return path.stem


def build_catalog(
    motion_path: Path,
    *,
    quality_report: Path | None,
    quality_group: str,
    recursive: bool,
    max_clips: int | None = None,
) -> tuple[ClipEntry, ...]:
    files = discover_npz_files(motion_path, recursive=recursive)
    root = motion_path.expanduser().resolve()
    quality_groups = (
        load_quality_groups(quality_report) if quality_report is not None else {}
    )
    if quality_group != "all" and quality_report is None:
        raise ValueError("--quality-group requires a quality report")

    entries: list[ClipEntry] = []
    for path in files:
        clip_id = _clip_id_from_path(path)
        group = quality_groups.get(clip_id, "unclassified")
        if quality_group != "all" and group != quality_group:
            continue
        label = path.name if root.is_file() else path.relative_to(root).as_posix()
        entries.append(
            ClipEntry(
                path=path,
                label=label,
                clip_id=clip_id,
                quality_group=group,
            )
        )

    if max_clips is not None:
        if max_clips <= 0:
            raise ValueError(f"--max-clips must be positive, got {max_clips}")
        entries = entries[:max_clips]
    if not entries:
        raise ValueError(
            f"No clips remain after applying quality group {quality_group!r}"
        )

    labels = [entry.label for entry in entries]
    duplicate_labels = sorted(
        label for label, count in Counter(labels).items() if count > 1
    )
    if duplicate_labels:
        raise ValueError(f"Catalog contains duplicate labels: {duplicate_labels[:5]}")
    return tuple(entries)


def load_tracking_clip(entry: ClipEntry) -> TrackingClip:
    with np.load(entry.path, allow_pickle=False) as payload:
        required = (
            "fps",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
        )
        missing = [field for field in required if field not in payload]
        if missing:
            raise ValueError(
                f"{entry.path}: missing required NPZ fields: {', '.join(missing)}"
            )

        fps_values = np.asarray(payload["fps"]).reshape(-1)
        if fps_values.size != 1:
            raise ValueError(
                f"{entry.path}: fps must contain one value, got {fps_values.shape}"
            )
        fps = float(fps_values[0])
        joint_pos = np.asarray(payload["joint_pos"], dtype=np.float32)
        joint_vel = np.asarray(payload["joint_vel"], dtype=np.float32)
        body_pos_w = np.asarray(payload["body_pos_w"], dtype=np.float32)
        body_quat_w = np.asarray(payload["body_quat_w"], dtype=np.float32)
        if "joint_names" in payload:
            joint_names = tuple(
                str(name) for name in np.asarray(payload["joint_names"]).tolist()
            )
        else:
            joint_names = BUMI_MOTION_JOINT_NAMES

    _validate_clip_arrays(
        entry.path,
        fps=fps,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        joint_names=joint_names,
    )
    return TrackingClip(
        entry=entry,
        fps=fps,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        joint_names=joint_names,
    )


def resolve_source_pose_path(
    entry: ClipEntry,
    source_pose_dir: Path | None,
) -> Path | None:
    candidates: list[Path] = []
    if source_pose_dir is not None:
        configured = source_pose_dir.expanduser().resolve()
        candidates.append(
            configured if configured.is_file() else configured / f"{entry.clip_id}.npz"
        )

    # Production conversion layout:
    #   <split>/tracker_50hz/<clip>.npz
    #   <split>/human_pose24/<clip>.npz
    # Check the unresolved and resolved paths so staged tracker symlinks can
    # still find their original source pose.
    for tracker_path in (entry.path, entry.path.resolve()):
        if tracker_path.parent.name == "tracker_50hz":
            candidates.append(
                tracker_path.parent.parent / "human_pose24" / f"{entry.clip_id}.npz"
            )

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


def load_source_pose24(
    entry: ClipEntry,
    source_pose_dir: Path | None,
) -> SourcePose24 | None:
    path = resolve_source_pose_path(entry, source_pose_dir)
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as payload:
        required = ("body_names", "body_pos_w", "timestamps_s", "source_fps")
        missing = [field for field in required if field not in payload]
        if missing:
            raise ValueError(
                f"{path}: missing HumanPose24 fields: {', '.join(missing)}"
            )
        body_names = tuple(str(name) for name in payload["body_names"].tolist())
        body_pos_w = np.asarray(payload["body_pos_w"], dtype=np.float32)
        timestamps_s = np.asarray(payload["timestamps_s"], dtype=np.float64)
        fps_values = np.asarray(payload["source_fps"]).reshape(-1)

    if body_names != _HUMAN_POSE24_NAMES:
        raise ValueError(
            f"{path}: HumanPose24 body order mismatch: "
            f"expected {_HUMAN_POSE24_NAMES}, got {body_names}"
        )
    if body_pos_w.ndim != 3 or body_pos_w.shape[1:] != (
        len(_HUMAN_POSE24_NAMES),
        3,
    ):
        raise ValueError(
            f"{path}: body_pos_w must have shape (frames, 24, 3), "
            f"got {body_pos_w.shape}"
        )
    if body_pos_w.shape[0] == 0:
        raise ValueError(f"{path}: HumanPose24 contains no frames")
    if timestamps_s.shape != (body_pos_w.shape[0],):
        raise ValueError(
            f"{path}: timestamps_s must have shape ({body_pos_w.shape[0]},), "
            f"got {timestamps_s.shape}"
        )
    if fps_values.size != 1:
        raise ValueError(
            f"{path}: source_fps must contain one value, got {fps_values.shape}"
        )
    source_fps = float(fps_values[0])
    if not math.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError(f"{path}: invalid source_fps {source_fps}")
    if not np.isfinite(body_pos_w).all():
        raise ValueError(f"{path}: body_pos_w contains non-finite values")
    if not np.isfinite(timestamps_s).all():
        raise ValueError(f"{path}: timestamps_s contains non-finite values")
    if timestamps_s.shape[0] > 1 and np.any(np.diff(timestamps_s) <= 0.0):
        raise ValueError(f"{path}: timestamps_s must be strictly increasing")
    return SourcePose24(
        path=path,
        body_pos_w=body_pos_w,
        timestamps_s=timestamps_s,
        source_fps=source_fps,
    )


def source_pose_arrays(
    source_pose: SourcePose24,
    time_s: float,
    *,
    recenter_root_xy: bool,
    source_y_offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    timestamp = float(
        np.clip(time_s, source_pose.timestamps_s[0], source_pose.timestamps_s[-1])
    )
    upper = int(np.searchsorted(source_pose.timestamps_s, timestamp, side="right"))
    if upper == 0:
        points = source_pose.body_pos_w[0].copy()
    elif upper >= source_pose.frame_count:
        points = source_pose.body_pos_w[-1].copy()
    else:
        lower = upper - 1
        span = source_pose.timestamps_s[upper] - source_pose.timestamps_s[lower]
        alpha = float((timestamp - source_pose.timestamps_s[lower]) / span)
        points = (
            (1.0 - alpha) * source_pose.body_pos_w[lower]
            + alpha * source_pose.body_pos_w[upper]
        ).astype(np.float32)

    offset = np.zeros(3, dtype=np.float32)
    if recenter_root_xy:
        offset[:2] -= points[_HUMAN_POSE24_PELVIS_INDEX, :2]
    offset[1] += float(source_y_offset)
    points += offset
    segments = np.asarray(
        [[points[parent], points[child]] for parent, child in _HUMAN_POSE24_LINKS],
        dtype=np.float32,
    )
    return points, segments


def _validate_clip_arrays(
    path: Path,
    *,
    fps: float,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    joint_names: tuple[str, ...],
) -> None:
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"{path}: fps must be finite and positive, got {fps}")
    if joint_pos.ndim != 2 or joint_pos.shape[0] == 0:
        raise ValueError(
            f"{path}: joint_pos must have shape (frames, joints), got {joint_pos.shape}"
        )
    frames = int(joint_pos.shape[0])
    expected_shapes = {
        "joint_pos": (frames, len(BUMI_MOTION_JOINT_NAMES)),
        "joint_vel": (frames, len(BUMI_MOTION_JOINT_NAMES)),
        "body_pos_w": (frames, len(BUMI_MOTION_BODY_NAMES), 3),
        "body_quat_w": (frames, len(BUMI_MOTION_BODY_NAMES), 4),
    }
    arrays = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
    }
    for field, expected_shape in expected_shapes.items():
        if arrays[field].shape != expected_shape:
            raise ValueError(
                f"{path}: {field} expected {expected_shape}, got {arrays[field].shape}"
            )
        if not np.isfinite(arrays[field]).all():
            raise ValueError(f"{path}: {field} contains non-finite values")
    if len(joint_names) != joint_pos.shape[1]:
        raise ValueError(
            f"{path}: joint_names has {len(joint_names)} values for "
            f"{joint_pos.shape[1]} joints"
        )
    if set(joint_names) != set(BUMI_MOTION_JOINT_NAMES):
        raise ValueError(f"{path}: joint_names do not match the Bumi tracker contract")

    quaternion_norms = np.linalg.norm(body_quat_w, axis=-1)
    if np.any(quaternion_norms <= 1.0e-8):
        raise ValueError(f"{path}: body_quat_w contains a zero quaternion")
    max_quaternion_error = float(np.max(np.abs(quaternion_norms - 1.0)))
    if max_quaternion_error > 1.0e-3:
        raise ValueError(
            f"{path}: body_quat_w norm error {max_quaternion_error:g} exceeds 1e-3"
        )


def resolve_joint_addresses(
    model: mujoco.MjModel,
    joint_names: tuple[str, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    qpos_addresses: list[int] = []
    dof_addresses: list[int] = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
        if joint_id < 0:
            raise ValueError(f"Bumi model is missing joint {joint_name!r}")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"Bumi joint {joint_name!r} is not a hinge")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        dof_addresses.append(int(model.jnt_dofadr[joint_id]))
    return tuple(qpos_addresses), tuple(dof_addresses)


def apply_clip_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    clip: TrackingClip,
    frame_index: int,
    *,
    joint_qpos_addresses: tuple[int, ...],
    joint_dof_addresses: tuple[int, ...],
    recenter_root_xy: bool,
) -> int:
    frame = int(np.clip(frame_index, 0, clip.frame_count - 1))
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0

    root_quaternion = clip.body_quat_w[frame, _BASE_BODY_INDEX].astype(
        np.float64,
        copy=True,
    )
    root_quaternion /= np.linalg.norm(root_quaternion)
    root_position = clip.body_pos_w[frame, _BASE_BODY_INDEX].astype(
        np.float64,
        copy=True,
    )
    if recenter_root_xy:
        root_position[:2] = 0.0
    data.qpos[:3] = root_position
    data.qpos[3:7] = root_quaternion
    data.qpos[list(joint_qpos_addresses)] = clip.joint_pos[frame]
    data.qvel[list(joint_dof_addresses)] = clip.joint_vel[frame]
    data.time = frame / clip.fps
    mujoco.mj_forward(model, data)
    return frame


def build_body_links(model: mujoco.MjModel) -> tuple[tuple[int, int], ...]:
    if model.nbody != len(BUMI_MOTION_BODY_NAMES):
        raise ValueError(
            f"Bumi model has {model.nbody} bodies including world, "
            f"expected {len(BUMI_MOTION_BODY_NAMES)}"
        )
    links: list[tuple[int, int]] = []
    for body_id in range(1, model.nbody):
        parent_id = int(model.body_parentid[body_id])
        if parent_id > 0:
            links.append((parent_id, body_id))
    return tuple(links)


def body_overlay_arrays(
    clip: TrackingClip,
    body_links: tuple[tuple[int, int], ...],
    frame_index: int,
    *,
    recenter_root_xy: bool,
    overlay_y_offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    frame = int(np.clip(frame_index, 0, clip.frame_count - 1))
    offset = np.zeros(3, dtype=np.float32)
    if recenter_root_xy:
        offset[:2] -= clip.body_pos_w[frame, _BASE_BODY_INDEX, :2]
    offset[1] += float(overlay_y_offset)
    points = clip.body_pos_w[frame, 1:].astype(np.float32, copy=True)
    points += offset
    segments = np.asarray(
        [
            [
                clip.body_pos_w[frame, parent] + offset,
                clip.body_pos_w[frame, child] + offset,
            ]
            for parent, child in body_links
        ],
        dtype=np.float32,
    )
    if not body_links:
        segments = np.zeros((0, 2, 3), dtype=np.float32)
    return points, segments


def line_segment_colors(segment_count: int) -> np.ndarray:
    return np.tile(_LINE_COLOR[None, None, :], (segment_count, 2, 1))


def source_line_segment_colors(segment_count: int) -> np.ndarray:
    return np.tile(_SOURCE_LINE_COLOR[None, None, :], (segment_count, 2, 1))


def print_catalog_summary(entries: tuple[ClipEntry, ...]) -> None:
    groups = Counter(entry.quality_group for entry in entries)
    group_summary = ", ".join(
        f"{name}={count}" for name, count in sorted(groups.items())
    )
    print(f"Discovered {len(entries)} Bumi tracker clip(s): {group_summary}")


def validate_catalog(entries: tuple[ClipEntry, ...]) -> None:
    total_frames = 0
    total_duration_s = 0.0
    errors: list[str] = []
    for index, entry in enumerate(entries, start=1):
        try:
            clip = load_tracking_clip(entry)
        except Exception as exc:  # noqa: BLE001 - collect every invalid clip
            errors.append(f"{entry.path}: {exc}")
            continue
        total_frames += clip.frame_count
        total_duration_s += clip.duration_s
        if index % 100 == 0 or index == len(entries):
            print(f"Validated {index}/{len(entries)} clips")
    if errors:
        for error in errors[:20]:
            print(f"ERROR: {error}")
        if len(errors) > 20:
            print(f"... and {len(errors) - 20} more errors")
        raise RuntimeError(f"{len(errors)} of {len(entries)} clips failed validation")
    print(
        f"Validated {len(entries)} clips: frames={total_frames}, "
        f"duration={total_duration_s / 60.0:.1f} min"
    )


def _clip_markdown(
    state: PlaybackState,
    entries: tuple[ClipEntry, ...],
) -> str:
    clip = state.clip
    return (
        f"### {state.clip_index + 1} / {len(entries)}\n"
        f"`{clip.entry.label}`\n\n"
        f"- quality: **{clip.entry.quality_group}**\n"
        f"- frames: **{clip.frame_count}** @ **{clip.fps:g} Hz**\n"
        f"- duration: **{clip.duration_s:.2f} s**"
    )


def _source_status_text(source_pose: SourcePose24 | None) -> str:
    if source_pose is None:
        return "not found for this clip"
    return f"loaded: {source_pose.frame_count} frames @ {source_pose.source_fps:g} Hz"


def run_viewer(
    entries: tuple[ClipEntry, ...],
    *,
    model_file: Path,
    source_pose_dir: Path | None,
    host: str,
    port: int,
    speed: float,
    start_index: int,
    start_paused: bool,
    show_source: bool,
    show_reference: bool,
    recenter_root_xy: bool,
    source_y_offset: float,
    overlay_y_offset: float,
    run_seconds: float | None,
) -> None:
    try:
        import viser
        from mjviser import ViserMujocoScene
    except ImportError as exc:
        raise ImportError(
            "Viser playback needs mjviser. Run this script with "
            "`uv --project venv/mjlab run --with mjviser==0.0.14 ...`."
        ) from exc

    if speed <= 0.0:
        raise ValueError(f"--speed must be positive, got {speed}")
    if not 0 <= start_index < len(entries):
        raise ValueError(
            f"--start-index must be in [0, {len(entries) - 1}], got {start_index}"
        )
    if run_seconds is not None and run_seconds <= 0.0:
        raise ValueError(f"--run-seconds must be positive, got {run_seconds}")
    if not math.isfinite(source_y_offset):
        raise ValueError(f"--source-y-offset must be finite, got {source_y_offset}")
    if not math.isfinite(overlay_y_offset):
        raise ValueError(f"--overlay-y-offset must be finite, got {overlay_y_offset}")

    model = mujoco.MjModel.from_xml_path(model_file.as_posix())
    validate_bumi_model(model)
    data = mujoco.MjData(model)
    initial_clip = load_tracking_clip(entries[start_index])
    initial_source_pose = load_source_pose24(
        entries[start_index],
        source_pose_dir,
    )
    qpos_addresses, dof_addresses = resolve_joint_addresses(
        model,
        initial_clip.joint_names,
    )
    state = PlaybackState(
        clip_index=start_index,
        clip=initial_clip,
        joint_qpos_addresses=qpos_addresses,
        joint_dof_addresses=dof_addresses,
        source_pose=initial_source_pose,
        paused=start_paused,
    )
    state_lock = threading.RLock()

    server = viser.ViserServer(host=host, port=port, label="Bumi retarget browser")
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/world/grid",
        width=8.0,
        height=8.0,
        cell_size=0.25,
        plane="xy",
    )
    scene = ViserMujocoScene(server, model, num_envs=1)
    # mjviser camera tracking subtracts the tracked body's full XYZ position.
    # That is inappropriate here because it moves the physical ground through
    # the robot.  Root following is instead applied explicitly to XY only.
    scene.camera_tracking_enabled = False
    body_links = build_body_links(model)

    apply_clip_frame(
        model,
        data,
        state.clip,
        0,
        joint_qpos_addresses=state.joint_qpos_addresses,
        joint_dof_addresses=state.joint_dof_addresses,
        recenter_root_xy=recenter_root_xy,
    )
    initial_points, initial_segments = body_overlay_arrays(
        state.clip,
        body_links,
        0,
        recenter_root_xy=recenter_root_xy,
        overlay_y_offset=overlay_y_offset,
    )
    point_handle = server.scene.add_point_cloud(
        "/reference/body_points",
        points=initial_points,
        colors=np.tile(_POINT_COLOR[None, :], (initial_points.shape[0], 1)),
        point_size=0.03,
        point_shape="circle",
        visible=show_reference,
    )
    line_handle = server.scene.add_line_segments(
        "/reference/body_links",
        points=initial_segments,
        colors=line_segment_colors(initial_segments.shape[0]),
        line_width=3.0,
        visible=show_reference,
    )
    root_frame = server.scene.add_frame(
        "/reference/base_link",
        axes_length=0.2,
        axes_radius=0.008,
        visible=show_reference,
    )
    if initial_source_pose is None:
        initial_source_points = np.zeros(
            (len(_HUMAN_POSE24_NAMES), 3),
            dtype=np.float32,
        )
        initial_source_segments = np.zeros(
            (len(_HUMAN_POSE24_LINKS), 2, 3),
            dtype=np.float32,
        )
    else:
        initial_source_points, initial_source_segments = source_pose_arrays(
            initial_source_pose,
            0.0,
            recenter_root_xy=recenter_root_xy,
            source_y_offset=source_y_offset,
        )
    source_point_handle = server.scene.add_point_cloud(
        "/source_smplx/joints",
        points=initial_source_points,
        colors=np.tile(
            _SOURCE_POINT_COLOR[None, :],
            (len(_HUMAN_POSE24_NAMES), 1),
        ),
        point_size=0.035,
        point_shape="circle",
        visible=show_source and initial_source_pose is not None,
    )
    source_line_handle = server.scene.add_line_segments(
        "/source_smplx/bones",
        points=initial_source_segments,
        colors=source_line_segment_colors(len(_HUMAN_POSE24_LINKS)),
        line_width=4.0,
        visible=show_source and initial_source_pose is not None,
    )

    labels = tuple(entry.label for entry in entries)
    index_by_label = {label: index for index, label in enumerate(labels)}

    camera_look_at = np.asarray([0.0, 0.0, 0.5])
    camera_position = camera_look_at + np.asarray([1.17, -2.03, 0.86])

    @server.on_client_connect
    def _set_initial_camera(client: Any) -> None:
        client.camera.position = camera_position
        client.camera.look_at = camera_look_at

    tabs = server.gui.add_tab_group()
    with tabs.add_tab("Playback"):
        server.gui.add_markdown(
            "**Purple/green:** original SMPL-X FK source motion represented "
            "by its 24 selected joints, before GMR/IK. **Blue/orange:** "
            "optional final Bumi FK debug overlay. `Follow root XY` recenters "
            "the SMPL-X pelvis and Bumi `base_link` independently in XY only; "
            "both retain their original Z above the world ground."
        )
        clip_dropdown = server.gui.add_dropdown(
            "Clip",
            options=labels,
            initial_value=labels[start_index],
        )
        previous_button = server.gui.add_button("Previous clip")
        next_button = server.gui.add_button("Next clip")
        pause_button = server.gui.add_button("Pause / Resume")
        back_button = server.gui.add_button("Step backward")
        forward_button = server.gui.add_button("Step forward")
        reset_button = server.gui.add_button("Reset clip")
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=initial_clip.frame_count - 1,
            step=1,
            initial_value=0,
        )
        speed_slider = server.gui.add_slider(
            "Speed",
            min=0.1,
            max=4.0,
            step=0.1,
            initial_value=speed,
        )
        end_dropdown = server.gui.add_dropdown(
            "At clip end",
            options=_END_BEHAVIORS,
            initial_value="Next clip",
        )
        source_checkbox = server.gui.add_checkbox(
            "Show source SMPL-X skeleton",
            initial_value=show_source,
        )
        source_y_slider = server.gui.add_slider(
            "Source SMPL-X Y offset",
            min=-2.5,
            max=2.5,
            step=0.05,
            initial_value=source_y_offset,
        )
        reference_checkbox = server.gui.add_checkbox(
            "Show Bumi FK debug overlay",
            initial_value=show_reference,
        )
        follow_root_xy_checkbox = server.gui.add_checkbox(
            "Follow root XY",
            initial_value=recenter_root_xy,
        )
        overlay_y_slider = server.gui.add_slider(
            "Overlay Y offset",
            min=-1.5,
            max=1.5,
            step=0.05,
            initial_value=overlay_y_offset,
        )
        quality_text = server.gui.add_text(
            "Quality group",
            initial_value=initial_clip.entry.quality_group,
            disabled=True,
        )
        source_status = server.gui.add_text(
            "Source SMPL-X",
            initial_value=_source_status_text(initial_source_pose),
            disabled=True,
        )
        clip_info = server.gui.add_markdown(_clip_markdown(state, entries))
    with tabs.add_tab("Visualization"):
        scene.create_overlay_gui()
    with tabs.add_tab("Groups"):
        scene.create_groups_gui()

    def render_frame(frame_index: int) -> None:
        with state_lock:
            state.frame = apply_clip_frame(
                model,
                data,
                state.clip,
                frame_index,
                joint_qpos_addresses=state.joint_qpos_addresses,
                joint_dof_addresses=state.joint_dof_addresses,
                recenter_root_xy=bool(follow_root_xy_checkbox.value),
            )
            scene.update_from_mjdata(data)
            points, segments = body_overlay_arrays(
                state.clip,
                body_links,
                state.frame,
                recenter_root_xy=bool(follow_root_xy_checkbox.value),
                overlay_y_offset=float(overlay_y_slider.value),
            )
            offset = np.zeros(3, dtype=np.float32)
            if follow_root_xy_checkbox.value:
                offset[:2] -= state.clip.body_pos_w[
                    state.frame,
                    _BASE_BODY_INDEX,
                    :2,
                ]
            offset[1] += float(overlay_y_slider.value)
            if state.source_pose is None:
                source_points = np.zeros(
                    (len(_HUMAN_POSE24_NAMES), 3),
                    dtype=np.float32,
                )
                source_segments = np.zeros(
                    (len(_HUMAN_POSE24_LINKS), 2, 3),
                    dtype=np.float32,
                )
            else:
                source_points, source_segments = source_pose_arrays(
                    state.source_pose,
                    state.frame / state.clip.fps,
                    recenter_root_xy=bool(follow_root_xy_checkbox.value),
                    source_y_offset=float(source_y_slider.value),
                )
            with server.atomic():
                point_handle.points = points
                line_handle.points = segments
                root_frame.position = (
                    state.clip.body_pos_w[state.frame, _BASE_BODY_INDEX] + offset
                )
                root_frame.wxyz = state.clip.body_quat_w[
                    state.frame,
                    _BASE_BODY_INDEX,
                ]
                source_point_handle.points = source_points
                source_line_handle.points = source_segments
                source_visible = (
                    bool(source_checkbox.value) and state.source_pose is not None
                )
                source_point_handle.visible = source_visible
                source_line_handle.visible = source_visible
                if int(frame_slider.value) != state.frame:
                    frame_slider.value = state.frame

    def select_clip(index: int, *, sync_dropdown: bool) -> None:
        with state_lock:
            index %= len(entries)
            clip = load_tracking_clip(entries[index])
            source_pose = load_source_pose24(entries[index], source_pose_dir)
            qpos, dof = resolve_joint_addresses(model, clip.joint_names)
            state.clip_index = index
            state.clip = clip
            state.joint_qpos_addresses = qpos
            state.joint_dof_addresses = dof
            state.source_pose = source_pose
            state.frame = 0
            frame_slider.max = clip.frame_count - 1
            quality_text.value = clip.entry.quality_group
            source_status.value = _source_status_text(source_pose)
            clip_info.content = _clip_markdown(state, entries)
            if sync_dropdown and clip_dropdown.value != entries[index].label:
                clip_dropdown.value = entries[index].label
            print(
                "[view_bumi_retarget_viser] "
                f"selected {index + 1}/{len(entries)} "
                f"group={clip.entry.quality_group} path={clip.entry.path}"
            )
            render_frame(0)

    @clip_dropdown.on_update
    def _on_clip_update(event: Any) -> None:
        del event
        requested_index = index_by_label[str(clip_dropdown.value)]
        with state_lock:
            if requested_index == state.clip_index:
                return
        select_clip(requested_index, sync_dropdown=False)

    @previous_button.on_click
    def _on_previous(event: Any) -> None:
        del event
        select_clip(state.clip_index - 1, sync_dropdown=True)

    @next_button.on_click
    def _on_next(event: Any) -> None:
        del event
        select_clip(state.clip_index + 1, sync_dropdown=True)

    @pause_button.on_click
    def _on_pause(event: Any) -> None:
        del event
        with state_lock:
            state.paused = not state.paused
            status = "paused" if state.paused else "playing"
            print(f"[view_bumi_retarget_viser] {status}")

    @back_button.on_click
    def _on_back(event: Any) -> None:
        del event
        with state_lock:
            state.paused = True
            render_frame(state.frame - 1)

    @forward_button.on_click
    def _on_forward(event: Any) -> None:
        del event
        with state_lock:
            state.paused = True
            render_frame(state.frame + 1)

    @reset_button.on_click
    def _on_reset(event: Any) -> None:
        del event
        with state_lock:
            state.paused = True
            render_frame(0)

    @frame_slider.on_update
    def _on_frame_update(event: Any) -> None:
        del event
        with state_lock:
            requested = int(frame_slider.value)
            if requested != state.frame:
                state.paused = True
                render_frame(requested)

    @reference_checkbox.on_update
    def _on_reference_update(event: Any) -> None:
        del event
        with server.atomic():
            visible = bool(reference_checkbox.value)
            point_handle.visible = visible
            line_handle.visible = visible
            root_frame.visible = visible

    @source_checkbox.on_update
    def _on_source_update(event: Any) -> None:
        del event
        with server.atomic():
            visible = bool(source_checkbox.value) and state.source_pose is not None
            source_point_handle.visible = visible
            source_line_handle.visible = visible

    @source_y_slider.on_update
    def _on_source_offset_update(event: Any) -> None:
        del event
        with state_lock:
            render_frame(state.frame)

    @overlay_y_slider.on_update
    def _on_overlay_offset_update(event: Any) -> None:
        del event
        with state_lock:
            render_frame(state.frame)

    @follow_root_xy_checkbox.on_update
    def _on_follow_root_xy_update(event: Any) -> None:
        del event
        with state_lock:
            render_frame(state.frame)

    actual_port = server.get_port()
    browser_host = "localhost" if host in {"0.0.0.0", "127.0.0.1"} else host
    print(f"[view_bumi_retarget_viser] loaded catalog with {len(entries)} clips")
    print(f"[view_bumi_retarget_viser] model: {model_file}")
    print(f"[view_bumi_retarget_viser] server: http://{browser_host}:{actual_port}")
    print("[view_bumi_retarget_viser] Press Ctrl+C to quit.")
    render_frame(0)

    deadline = time.monotonic() + run_seconds if run_seconds is not None else None
    try:
        while deadline is None or time.monotonic() < deadline:
            tick_start = time.perf_counter()
            with state_lock:
                clip_fps = state.clip.fps
                if not state.paused:
                    next_frame = state.frame + 1
                    if next_frame >= state.clip.frame_count:
                        end_behavior = str(end_dropdown.value)
                        if end_behavior == "Loop current":
                            render_frame(0)
                        elif end_behavior == "Next clip":
                            select_clip(state.clip_index + 1, sync_dropdown=True)
                        else:
                            state.paused = True
                    else:
                        render_frame(next_frame)
                playback_speed = float(speed_slider.value)
            target_dt = (1.0 / clip_fps) / playback_speed
            elapsed = time.perf_counter() - tick_start
            if target_dt > elapsed:
                time.sleep(target_dt - elapsed)
    except KeyboardInterrupt:
        print("\n[view_bumi_retarget_viser] Shutting down.")
    finally:
        server.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browse retargeted Bumi tracker NPZ clips with Viser"
    )
    parser.add_argument(
        "--motion-dir",
        type=Path,
        default=default_motion_path(),
        help=(
            "One tracker NPZ or a directory of NPZ files; defaults to the "
            "production AMASS train tracker_50hz directory"
        ),
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=default_model_path(),
        help="Bumi MJCF used to render FK meshes",
    )
    parser.add_argument(
        "--source-pose-dir",
        type=Path,
        default=None,
        help=(
            "HumanPose24 NPZ directory (or one NPZ); by default infer the "
            "human_pose24 sibling of tracker_50hz"
        ),
    )
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=None,
        help=(
            "quality_summary.json; inferred next to a tracker_50hz directory "
            "when present"
        ),
    )
    parser.add_argument(
        "--quality-group",
        choices=("all", *_QUALITY_REPORT_FIELDS, "unclassified"),
        default="all",
        help="Only browse clips from one quality split",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Find NPZ files recursively, including staged <clip>/motion.npz trees",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Limit the sorted catalog; mainly useful for quick checks",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based initial clip index after filtering",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Viser bind host")
    parser.add_argument("--port", type=int, default=8090, help="Viser port")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed")
    parser.add_argument(
        "--start-paused",
        action="store_true",
        help="Load the first frame without starting playback",
    )
    parser.add_argument(
        "--hide-reference",
        action="store_true",
        help="Initially hide the final Bumi FK debug overlay",
    )
    parser.add_argument(
        "--hide-source",
        action="store_true",
        help="Initially hide the original SMPL-X/HumanPose24 source skeleton",
    )
    parser.add_argument(
        "--recenter-root-xy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Recenter base_link XY for display while retaining its world Z; "
            "disable to show the original world trajectory"
        ),
    )
    parser.add_argument(
        "--source-y-offset",
        type=float,
        default=_DEFAULT_SOURCE_Y_OFFSET,
        help="Initial lateral offset for the source SMPL-X 24-joint skeleton",
    )
    parser.add_argument(
        "--overlay-y-offset",
        type=float,
        default=_DEFAULT_OVERLAY_Y_OFFSET,
        help=(
            "Initial lateral offset for the stored FK skeleton; use 0 for "
            "exact mesh alignment"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate every selected NPZ and exit without importing Viser",
    )
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=None,
        help="Automatically stop the server after this many seconds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motion_path = args.motion_dir.expanduser().resolve()
    quality_report = infer_quality_report(motion_path, args.quality_report)
    entries = build_catalog(
        motion_path,
        quality_report=quality_report,
        quality_group=str(args.quality_group),
        recursive=bool(args.recursive),
        max_clips=args.max_clips,
    )
    print_catalog_summary(entries)
    if quality_report is not None:
        print(f"Quality report: {quality_report}")
    if args.dry_run:
        validate_catalog(entries)
        return

    model_file = args.model_file.expanduser().resolve()
    if not model_file.is_file():
        raise FileNotFoundError(f"Bumi model does not exist: {model_file}")
    run_viewer(
        entries,
        model_file=model_file,
        source_pose_dir=args.source_pose_dir,
        host=str(args.host),
        port=int(args.port),
        speed=float(args.speed),
        start_index=int(args.start_index),
        start_paused=bool(args.start_paused),
        show_source=not bool(args.hide_source),
        show_reference=not bool(args.hide_reference),
        recenter_root_xy=bool(args.recenter_root_xy),
        source_y_offset=float(args.source_y_offset),
        overlay_y_offset=float(args.overlay_y_offset),
        run_seconds=args.run_seconds,
    )


if __name__ == "__main__":
    main()
