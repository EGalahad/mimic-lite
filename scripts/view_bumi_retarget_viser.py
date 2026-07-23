#!/usr/bin/env python3
"""Browse Bumi tracker NPZ motions with a lazy-loading Viser player.

The player accepts either one tracker NPZ or a directory of NPZ files.  It
renders Bumi through the same MuJoCo model used by the exporter and can overlay
the body positions stored in the NPZ.  Only the selected clip is held in
memory, so the full AMASS conversion can be browsed without loading every
motion at startup.
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


@dataclass
class PlaybackState:
    clip_index: int
    clip: TrackingClip
    joint_qpos_addresses: tuple[int, ...]
    joint_dof_addresses: tuple[int, ...]
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
) -> int:
    frame = int(np.clip(frame_index, 0, clip.frame_count - 1))
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0

    root_quaternion = clip.body_quat_w[frame, _BASE_BODY_INDEX].astype(
        np.float64,
        copy=True,
    )
    root_quaternion /= np.linalg.norm(root_quaternion)
    data.qpos[:3] = clip.body_pos_w[frame, _BASE_BODY_INDEX]
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
    follow_root: bool,
) -> tuple[np.ndarray, np.ndarray]:
    frame = int(np.clip(frame_index, 0, clip.frame_count - 1))
    offset = (
        -clip.body_pos_w[frame, _BASE_BODY_INDEX]
        if follow_root
        else np.zeros(3, dtype=np.float32)
    )
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


def run_viewer(
    entries: tuple[ClipEntry, ...],
    *,
    model_file: Path,
    host: str,
    port: int,
    speed: float,
    start_index: int,
    start_paused: bool,
    show_reference: bool,
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

    model = mujoco.MjModel.from_xml_path(model_file.as_posix())
    validate_bumi_model(model)
    data = mujoco.MjData(model)
    initial_clip = load_tracking_clip(entries[start_index])
    qpos_addresses, dof_addresses = resolve_joint_addresses(
        model,
        initial_clip.joint_names,
    )
    state = PlaybackState(
        clip_index=start_index,
        clip=initial_clip,
        joint_qpos_addresses=qpos_addresses,
        joint_dof_addresses=dof_addresses,
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
    body_links = build_body_links(model)

    apply_clip_frame(
        model,
        data,
        state.clip,
        0,
        joint_qpos_addresses=state.joint_qpos_addresses,
        joint_dof_addresses=state.joint_dof_addresses,
    )
    initial_points, initial_segments = body_overlay_arrays(
        state.clip,
        body_links,
        0,
        follow_root=scene.camera_tracking_enabled,
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

    labels = tuple(entry.label for entry in entries)
    index_by_label = {label: index for index, label in enumerate(labels)}
    with server.gui.add_folder("Batch playback"):
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
            initial_value="Loop current",
        )
        reference_checkbox = server.gui.add_checkbox(
            "Show stored body overlay",
            initial_value=show_reference,
        )
        clip_info = server.gui.add_markdown(_clip_markdown(state, entries))
    scene.create_visualization_gui(
        camera_distance=2.5,
        camera_azimuth=120.0,
        camera_elevation=20.0,
    )

    def render_frame(frame_index: int) -> None:
        with state_lock:
            state.frame = apply_clip_frame(
                model,
                data,
                state.clip,
                frame_index,
                joint_qpos_addresses=state.joint_qpos_addresses,
                joint_dof_addresses=state.joint_dof_addresses,
            )
            scene.update_from_mjdata(data)
            points, segments = body_overlay_arrays(
                state.clip,
                body_links,
                state.frame,
                follow_root=scene.camera_tracking_enabled,
            )
            offset = (
                -state.clip.body_pos_w[state.frame, _BASE_BODY_INDEX]
                if scene.camera_tracking_enabled
                else np.zeros(3, dtype=np.float32)
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
                if int(frame_slider.value) != state.frame:
                    frame_slider.value = state.frame

    def select_clip(index: int, *, sync_dropdown: bool) -> None:
        with state_lock:
            index %= len(entries)
            clip = load_tracking_clip(entries[index])
            qpos, dof = resolve_joint_addresses(model, clip.joint_names)
            state.clip_index = index
            state.clip = clip
            state.joint_qpos_addresses = qpos
            state.joint_dof_addresses = dof
            state.frame = 0
            frame_slider.max = clip.frame_count - 1
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
        help="Initially hide stored body points/links over the MuJoCo mesh",
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
        host=str(args.host),
        port=int(args.port),
        speed=float(args.speed),
        start_index=int(args.start_index),
        start_paused=bool(args.start_paused),
        show_reference=not bool(args.hide_reference),
        run_seconds=args.run_seconds,
    )


if __name__ == "__main__":
    main()
