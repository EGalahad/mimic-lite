from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path
from typing import Any

try:
    import joblib
except ImportError:  # pragma: no cover - optional dependency in some envs
    joblib = None

import mujoco
import numpy as np
import torch
from tqdm import tqdm
from mjhub import resolve_asset_reference

from any4hdmi.core.format import load_motion, save_motion, write_manifest
from any4hdmi.dataset.interpolation import interpolate_qpos_torch
from any4hdmi.dataset.loading import resolve_input_paths
from any4hdmi.utils.mjcf import qpos_names_from_model


SOURCE_REPO_ID = "junsooki/h2_retargeted_motions"
SOURCE_REVISION = "295019f02a1ea913d2679602a59783e68a5c40bc"
DEFAULT_SOURCE_ROOT = f"hf://{SOURCE_REPO_ID}@{SOURCE_REVISION}"
DEFAULT_TARGET_FPS = 50.0
DEFAULT_MJCF_PATH = "hf://elijahgalahad/h2_model@beb532e8717b99816b93baace0c649a599538715/h2.xml"

_ROOT_POS_KEYS = ("root_pos", "root_trans_offset")
_DOF_KEYS = ("dof", "dof_pos")


def _resolve_single_path(path_like: str | Path) -> Path:
    return resolve_input_paths(Path.cwd(), path_like)[0]


def _motion_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if all(key in payload for key in ("root_rot", "pose_aa", "fps")) and any(
        key in payload for key in _ROOT_POS_KEYS
    ) and any(key in payload for key in _DOF_KEYS):
        candidates.append(payload)

    for value in payload.values():
        if isinstance(value, dict):
            candidates.append(value)
    return candidates


def _load_motion_dict(path: Path) -> dict[str, Any]:
    payload = None
    if joblib is not None:
        try:
            payload = joblib.load(path)
        except Exception:
            payload = None
    if payload is None:
        with path.open("rb") as f:
            payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a motion dictionary in {path}")

    for motion in _motion_candidates(payload):
        if all(key in motion for key in ("root_rot", "pose_aa", "fps")) and any(
            key in motion for key in _ROOT_POS_KEYS
        ) and any(key in motion for key in _DOF_KEYS):
            return motion

    raise ValueError(
        f"Expected a motion dict with root_pos/root_trans_offset, root_rot, dof, pose_aa and fps in {path}"
    )


def _field(motion: dict[str, Any], keys: tuple[str, ...], *, path: Path) -> np.ndarray:
    for key in keys:
        if key in motion:
            return np.asarray(motion[key])
    raise KeyError(f"Motion in {path} is missing all of: {', '.join(keys)}")


def _normalize_root_rot_xyzw(root_rot: np.ndarray, *, path: Path) -> np.ndarray:
    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        raise ValueError(f"Expected root_rot shape (T, 4), got {root_rot.shape} in {path}")
    norm = np.linalg.norm(root_rot, axis=1, keepdims=True)
    if np.any(norm <= 1e-8):
        raise ValueError(f"root_rot contains a zero quaternion in {path}")
    return (root_rot / norm)[:, [3, 0, 1, 2]]


def _source_motion_paths(source_root: Path) -> list[Path]:
    if source_root.is_file():
        return [source_root]
    paths = sorted({*source_root.rglob("*.pkl"), *source_root.rglob("*.joblib")})
    if not paths:
        raise ValueError(f"No H2 motion pickles found under {source_root}")
    return paths


def _relative_motion_path(path: Path, source_root: Path) -> Path:
    if source_root.is_file():
        return Path(path.name)
    return path.relative_to(source_root)


def _convert_motion(
    path: Path,
    *,
    model: mujoco.MjModel,
    target_fps: float,
) -> tuple[np.ndarray, float]:
    motion = _load_motion_dict(path)

    root_pos = _field(motion, _ROOT_POS_KEYS, path=path)
    root_rot = np.asarray(motion["root_rot"])
    dof = _field(motion, _DOF_KEYS, path=path)
    pose_aa = np.asarray(motion["pose_aa"])
    fps = float(motion["fps"])

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"Expected root_pos shape (T, 3), got {root_pos.shape} in {path}")
    if dof.ndim != 2:
        raise ValueError(f"Expected dof to be rank 2, got {dof.shape} in {path}")
    if pose_aa.ndim != 3 or pose_aa.shape[2] != 3:
        raise ValueError(f"Expected pose_aa shape (T, B, 3), got {pose_aa.shape} in {path}")
    if not (len(root_pos) == len(root_rot) == len(dof) == len(pose_aa)):
        raise ValueError(
            "root_pos, root_rot, dof, and pose_aa must have the same frame count "
            f"in {path}: {len(root_pos)}, {len(root_rot)}, {len(dof)}, {len(pose_aa)}"
        )

    expected_body_count = model.nbody - 1
    if pose_aa.shape != (len(dof), expected_body_count, 3):
        raise ValueError(
            f"Expected pose_aa shape (T, {expected_body_count}, 3), got {pose_aa.shape} in {path}"
        )

    joint_axes = np.asarray(model.jnt_axis[1:], dtype=np.float64)
    expected_pose_aa = np.asarray(dof, dtype=np.float64)[:, :, None] * joint_axes[None, :, :]
    if not np.allclose(pose_aa[:, 1:], expected_pose_aa, atol=1e-5):
        raise ValueError(f"pose_aa and dof do not match the H2 MJCF joint order in {path}")

    qpos = np.zeros((len(dof), model.nq), dtype=np.float32)
    qpos[:, :3] = np.asarray(root_pos, dtype=np.float32)
    qpos[:, 3:7] = _normalize_root_rot_xyzw(root_rot, path=path)
    qpos[:, 7:] = np.asarray(dof, dtype=np.float32)
    if qpos.shape[1] != model.nq or not np.isfinite(qpos).all():
        raise ValueError(f"Invalid qpos produced from {path}: {qpos.shape}")

    if fps != target_fps and len(qpos) > 1:
        qpos_t = interpolate_qpos_torch(
            torch.from_numpy(qpos),
            source_fps=fps,
            target_fps=target_fps,
        )
        qpos = qpos_t.cpu().numpy().astype(np.float32, copy=False)

    return qpos, fps


def convert_h2_dataset(
    *,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    out_dir: str | Path,
    mjcf_path: str | Path = DEFAULT_MJCF_PATH,
    target_fps: float = DEFAULT_TARGET_FPS,
    max_motions: int | None = None,
) -> dict[str, Any]:
    source_root_path = _resolve_single_path(source_root)
    out_dir = Path(out_dir).expanduser().resolve()
    # Preserve the HF snapshot path: resolving its symlink loses sibling meshes.
    mjcf_path = Path(resolve_asset_reference(str(mjcf_path))).expanduser().absolute()

    if (out_dir / "manifest.json").exists():
        raise FileExistsError(f"Completed dataset already exists: {out_dir}")

    motion_paths = _source_motion_paths(source_root_path)
    if max_motions is not None:
        motion_paths = motion_paths[: max(0, max_motions)]
    if not motion_paths:
        raise ValueError("No motions selected for conversion")

    dataset_mjcf = out_dir / "mjcf" / "h2.xml"
    dataset_mjcf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mjcf_path, dataset_mjcf)
    source_meshes = mjcf_path.parent / "meshes"
    if source_meshes.is_dir():
        shutil.copytree(source_meshes, dataset_mjcf.parent / "meshes")
    model = mujoco.MjModel.from_xml_path(str(dataset_mjcf))
    qpos_names = qpos_names_from_model(model)

    source_fps_values: list[float] = []
    total_frames = 0

    for index, path in enumerate(tqdm(motion_paths, desc="Converting H2 motions", unit="clip"), start=1):
        qpos, fps = _convert_motion(path, model=model, target_fps=target_fps)
        source_fps_values.append(float(fps))
        total_frames += int(qpos.shape[0])

        relative = _relative_motion_path(path, source_root_path).with_suffix(".npz")
        destination = out_dir / "motions" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            existing = load_motion(destination)
            if (
                existing.shape != qpos.shape
                or not np.isfinite(existing).all()
                or not np.array_equal(existing, qpos)
            ):
                raise ValueError(f"Invalid existing motion output: {destination}")
        else:
            temporary = destination.with_name(f".{destination.name}.tmp.npz")
            save_motion(temporary, qpos)
            temporary.replace(destination)

        if index == 1 or index % 100 == 0 or index == len(motion_paths):
            tqdm.write(f"Converted {index}/{len(motion_paths)} motions ({total_frames} frames)")

    manifest = write_manifest(
        out_dir,
        dataset_name="h2_retargeted_motions",
        mjcf=dataset_mjcf,
        timestep=1.0 / float(target_fps),
        qpos_names=qpos_names,
        num_motions=len(motion_paths),
        total_hours=total_frames / float(target_fps) / 3600.0,
        source={
            "source_repo": SOURCE_REPO_ID,
            "source_revision": SOURCE_REVISION,
            "source_root": str(source_root_path),
            "source_schema": "joblib dict with root_pos/root_trans_offset, root_rot, dof, pose_aa, fps",
            "source_fps_values": sorted(set(source_fps_values)),
            "root_rot_format": "xyzw",
            "qpos_root_format": "xyz + wxyz",
            "target_fps": float(target_fps),
            "conversion": "joblib -> qpos",
        },
    )

    return {
        "out_dir": str(out_dir),
        "manifest": str(manifest),
        "num_motions": len(motion_paths),
        "total_frames": total_frames,
        "source_fps_values": sorted(set(source_fps_values)),
        "target_fps": float(target_fps),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        default=DEFAULT_SOURCE_ROOT,
        help="Local directory/file path or hf:// dataset reference.",
    )
    parser.add_argument(
        "--out-dir",
        default=Path("data/h2/retargeted"),
        type=Path,
        help="Output any4hdmi dataset root.",
    )
    parser.add_argument(
        "--mjcf-path",
        default=DEFAULT_MJCF_PATH,
        help="Local or hf:// H2 MJCF copied into the output dataset.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=DEFAULT_TARGET_FPS,
        help="Target output FPS. Source clips are resampled to this rate.",
    )
    parser.add_argument(
        "--max-motions",
        type=int,
        default=None,
        help="Optional limit for smoke tests.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    summary = convert_h2_dataset(
        source_root=args.source_root,
        out_dir=args.out_dir,
        mjcf_path=args.mjcf_path,
        target_fps=args.target_fps,
        max_motions=args.max_motions,
    )
    print(
        "H2 dataset prepared: "
        f"motions={summary['num_motions']} "
        f"frames={summary['total_frames']} "
        f"target_fps={summary['target_fps']} "
        f"source_fps_values={summary['source_fps_values']} "
        f"out_dir={summary['out_dir']}"
    )


if __name__ == "__main__":
    main()
