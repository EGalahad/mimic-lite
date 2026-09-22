from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from any4hdmi import BaseDataset, MotionData, MotionSample
from any4hdmi.dataset.full import FullMotionDataset, convert_motion_data

_MOTION_DATA_FIELD_NAMES = (
    "motion_id",
    "step",
    "body_pos_w",
    "body_lin_vel_w",
    "body_quat_w",
    "body_ang_vel_w",
    "joint_pos",
    "joint_vel",
)
_FLOAT_MOTION_DATA_FIELD_NAMES = _MOTION_DATA_FIELD_NAMES[2:]


@dataclass(frozen=True)
class MotionDatasetConfig:
    name: str
    path: str | list[str]
    weight: float
    full_motion: bool
    shard: bool = False
    filenames: list[str] | None = None
    filenames_path: str | None = None


def _normalize_data_path(value: Any) -> str | list[str]:
    if isinstance(value, (str, os.PathLike)):
        return os.fspath(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        paths = []
        for item in value:
            if not isinstance(item, (str, os.PathLike)):
                raise TypeError(
                    "motion_cfgs path sequences must contain only strings or path-like values"
                )
            paths.append(os.fspath(item))
        if not paths:
            raise ValueError("motion_cfgs path sequences must not be empty")
        return paths
    raise TypeError(
        "motion_cfgs entries must provide a string path or a non-empty sequence of paths"
    )


def _normalize_motion_filenames(value: Any, *, name: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(
            f"motion_cfgs[{name!r}] filenames must be a sequence of strings"
        )
    filenames = []
    for item in value:
        if not isinstance(item, (str, os.PathLike)):
            raise TypeError(f"motion_cfgs[{name!r}] filenames entries must be strings")
        filename = os.fspath(item).strip()
        if filename:
            filenames.append(filename)
    if not filenames:
        raise ValueError(f"motion_cfgs[{name!r}] filenames must not be empty")
    return filenames


def _normalize_optional_path(value: Any, *, name: str, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"motion_cfgs[{name!r}] {key} must be a string path")
    path = os.fspath(value).strip()
    if not path:
        raise ValueError(f"motion_cfgs[{name!r}] {key} must not be empty")
    return path


def normalize_motion_cfgs(motion_cfgs: Mapping[str, object]) -> list[MotionDatasetConfig]:
    if not isinstance(motion_cfgs, Mapping):
        raise TypeError("motion_cfgs must be a mapping of dataset names to configs")

    configs = []
    allowed_keys = {
        "path",
        "weight",
        "full_motion",
        "shard",
        "filenames",
        "filenames_path",
    }
    for raw_name, raw_cfg in motion_cfgs.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("motion_cfgs dataset names must not be empty")
        if not isinstance(raw_cfg, Mapping):
            raise TypeError(f"motion_cfgs[{name!r}] must be a mapping")
        missing_keys = [
            key for key in ("path", "weight", "full_motion") if key not in raw_cfg
        ]
        if missing_keys:
            raise ValueError(
                f"motion_cfgs[{name!r}] is missing required keys: {', '.join(missing_keys)}"
            )
        unexpected_keys = sorted(
            str(key) for key in raw_cfg if key not in allowed_keys
        )
        if unexpected_keys:
            raise ValueError(
                f"motion_cfgs[{name!r}] has unexpected keys: {', '.join(unexpected_keys)}"
            )

        raw_weight = raw_cfg["weight"]
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
            raise TypeError(f"motion_cfgs[{name!r}] weight must be numeric")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(
                f"motion_cfgs[{name!r}] weight must be positive and finite"
            )
        full_motion = raw_cfg["full_motion"]
        shard = raw_cfg.get("shard", False)
        if not isinstance(full_motion, bool):
            raise TypeError(f"motion_cfgs[{name!r}] full_motion must be a boolean")
        if not isinstance(shard, bool):
            raise TypeError(f"motion_cfgs[{name!r}] shard must be a boolean")

        filenames = _normalize_motion_filenames(raw_cfg.get("filenames"), name=name)
        filenames_path = _normalize_optional_path(
            raw_cfg.get("filenames_path"), name=name, key="filenames_path"
        )
        if filenames is not None and filenames_path is not None:
            raise ValueError(
                f"motion_cfgs[{name!r}] must provide only one of filenames or filenames_path"
            )
        configs.append(
            MotionDatasetConfig(
                name=name,
                path=_normalize_data_path(raw_cfg["path"]),
                weight=weight,
                full_motion=full_motion,
                shard=shard,
                filenames=filenames,
                filenames_path=filenames_path,
            )
        )
    if not configs:
        raise ValueError("motion_cfgs must contain at least one dataset entry")
    return configs


def motion_cfgs_to_dict(
    motion_cfgs: Sequence[MotionDatasetConfig],
) -> dict[str, dict[str, str | list[str] | float | bool]]:
    serialized = {}
    for cfg in motion_cfgs:
        serialized[cfg.name] = {
            "path": list(cfg.path) if isinstance(cfg.path, list) else cfg.path,
            "weight": float(cfg.weight),
            "full_motion": cfg.full_motion,
            "shard": cfg.shard,
        }
        if cfg.filenames is not None:
            serialized[cfg.name]["filenames"] = list(cfg.filenames)
        if cfg.filenames_path is not None:
            serialized[cfg.name]["filenames_path"] = cfg.filenames_path
    return serialized


def load_motion_dataset_collection(
    motion_cfgs: Sequence[MotionDatasetConfig],
    *,
    create_dataset_fn: Callable[..., BaseDataset],
    target_fps: int,
    num_envs: int,
    body_names: list[str] | None = None,
    joint_names: list[str] | None = None,
    windowed_next_window_device: str | None = "current",
    windowed_pin_window_load: bool = True,
) -> BaseDataset:
    datasets = [
        create_dataset_fn(
            cfg.path,
            target_fps=target_fps,
            num_envs=num_envs,
            full_motion=cfg.full_motion,
            shard=cfg.shard,
            filenames=cfg.filenames,
            filenames_path=cfg.filenames_path,
            body_names=body_names,
            joint_names=joint_names,
            windowed_next_window_device=windowed_next_window_device,
            windowed_pin_window_load=windowed_pin_window_load,
        )
        for cfg in motion_cfgs
    ]
    return MultiMotionDataset(
        motion_cfgs=motion_cfgs,
        datasets=datasets,
        num_envs=num_envs,
    )


class MultiMotionDataset(BaseDataset):
    dataset_kind = "multi"

    def __init__(
        self,
        *,
        motion_cfgs: Sequence[MotionDatasetConfig],
        datasets: Sequence[BaseDataset],
        num_envs: int,
    ) -> None:
        if len(motion_cfgs) != len(datasets):
            raise ValueError("motion_cfgs and datasets must have identical lengths")
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        self.motion_cfgs = list(motion_cfgs)
        self.dataset_names = [cfg.name for cfg in self.motion_cfgs]
        self.num_envs = int(num_envs)
        if not datasets:
            raise ValueError("MultiMotionDataset requires at least one dataset")
        self.datasets = list(datasets)
        self.body_names = list(self.datasets[0].body_names)
        self.joint_names = list(self.datasets[0].joint_names)
        for index, dataset in enumerate(self.datasets[1:], start=1):
            if list(dataset.body_names) != self.body_names:
                raise ValueError(f"datasets[{index}] body_names do not match")
            if list(dataset.joint_names) != self.joint_names:
                raise ValueError(f"datasets[{index}] joint_names do not match")

        self.device = torch.device(self.datasets[0].device)
        route_offsets = []
        route_ends = []
        route_offset = 0
        step_offset = 0
        starts = []
        ends = []
        self.motion_paths = []
        for dataset in self.datasets:
            route_offsets.append(route_offset)
            route_offset += int(dataset.sample_id_span)
            route_ends.append(route_offset)
            starts.append(dataset.starts.to(self.device) + step_offset)
            ends.append(dataset.ends.to(self.device) + step_offset)
            step_offset += int(dataset.num_steps)
            self.motion_paths.extend(dataset.motion_paths)
        self.route_offsets = torch.tensor(
            route_offsets, device=self.device, dtype=torch.long
        )
        self.route_ends = torch.tensor(
            route_ends, device=self.device, dtype=torch.long
        )
        self.starts = torch.cat(starts)
        self.ends = torch.cat(ends)
        self.lengths = self.ends - self.starts
        self._resident_flags = [
            isinstance(dataset, FullMotionDataset) for dataset in self.datasets
        ]
        self._resident_dataset_mask = torch.tensor(
            self._resident_flags, device=self.device, dtype=torch.bool
        )
        self._resident_motion_id_offsets = torch.tensor(
            [getattr(dataset, "motion_id_offset", 0) for dataset in self.datasets],
            device=self.device,
            dtype=torch.long,
        )
        self._resident_data: MotionData | None = None
        self._resident_route_starts: torch.Tensor | None = None
        self._resident_route_ends: torch.Tensor | None = None
        weights = torch.tensor([cfg.weight for cfg in self.motion_cfgs])
        self._dataset_probs = (weights / weights.sum()).to(self.device)
        self._current_env_dataset_ids = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )
        self._fixed_env_dataset_ids: torch.Tensor | None = None
        self._has_sampled = False

    @property
    def num_motions(self) -> int:
        return sum(int(dataset.num_motions) for dataset in self.datasets)

    @property
    def num_steps(self) -> int:
        return sum(int(dataset.num_steps) for dataset in self.datasets)

    @property
    def env_dataset_ids(self) -> torch.Tensor:
        if self._fixed_env_dataset_ids is not None:
            return self._fixed_env_dataset_ids
        return self._current_env_dataset_ids

    def bind_env_dataset_ids(self, dataset_ids: torch.Tensor) -> None:
        if self._has_sampled:
            raise RuntimeError("Cannot bind environment datasets after sampling")
        ids = torch.as_tensor(dataset_ids, device=self.device, dtype=torch.long)
        if ids.shape != (self.num_envs,):
            raise ValueError(
                "dataset_ids shape must be "
                f"({self.num_envs},), got {tuple(ids.shape)}"
            )
        if torch.any((ids < 0) | (ids >= len(self.datasets))):
            raise ValueError("dataset_ids contain out of range values")
        self._fixed_env_dataset_ids = ids.clone()

    def to(self, device: torch.device | str) -> MultiMotionDataset:
        self.device = torch.device(device)
        resident_datasets = [
            dataset
            for dataset, resident in zip(self.datasets, self._resident_flags, strict=True)
            if resident
        ]
        if resident_datasets:
            total_frames = sum(int(dataset.num_steps) for dataset in resident_datasets)
            packed_fields = {}
            for field_name in _MOTION_DATA_FIELD_NAMES:
                sources = [
                    getattr(dataset.data, field_name) for dataset in resident_datasets
                ]
                output = torch.empty(
                    (total_frames, *sources[0].shape[1:]),
                    device=self.device,
                    dtype=(
                        torch.float16
                        if field_name in _FLOAT_MOTION_DATA_FIELD_NAMES
                        else sources[0].dtype
                    ),
                )
                frame_start = 0
                for dataset, source in zip(
                    resident_datasets, sources, strict=True
                ):
                    frame_end = frame_start + int(dataset.num_steps)
                    output[frame_start:frame_end].copy_(source)
                    setattr(dataset.data, field_name, output[frame_start:frame_end])
                    frame_start = frame_end
                packed_fields[field_name] = output
            self._resident_data = MotionData(
                **packed_fields,
                device=self.device,
                batch_size=(total_frames,),
            )

        self.datasets = [dataset.to(self.device) for dataset in self.datasets]
        self.route_offsets = self.route_offsets.to(self.device)
        self.route_ends = self.route_ends.to(self.device)
        self.starts = self.starts.to(self.device)
        self.ends = self.ends.to(self.device)
        self.lengths = self.lengths.to(self.device)
        self._resident_dataset_mask = self._resident_dataset_mask.to(self.device)
        self._resident_motion_id_offsets = self._resident_motion_id_offsets.to(
            self.device
        )
        self._dataset_probs = self._dataset_probs.to(self.device)
        self._current_env_dataset_ids = self._current_env_dataset_ids.to(self.device)
        if self._fixed_env_dataset_ids is not None:
            self._fixed_env_dataset_ids = self._fixed_env_dataset_ids.to(self.device)

        if resident_datasets:
            total_route_span = int(self.route_ends[-1].item())
            self._resident_route_starts = torch.full(
                (total_route_span,), -1, device=self.device, dtype=torch.long
            )
            self._resident_route_ends = torch.full_like(
                self._resident_route_starts, -1
            )
            frame_offset = 0
            for dataset_index, (dataset, resident) in enumerate(
                zip(self.datasets, self._resident_flags, strict=True)
            ):
                if not resident:
                    continue
                route_ids = self.route_offsets[dataset_index] + torch.arange(
                    int(dataset.num_motions), device=self.device
                )
                self._resident_route_starts[route_ids] = dataset.starts + frame_offset
                self._resident_route_ends[route_ids] = dataset.ends + frame_offset
                frame_offset += int(dataset.num_steps)
        return self

    def get_slice(
        self,
        motion_ids: torch.Tensor,
        starts: torch.Tensor,
        steps: torch.Tensor,
    ) -> MotionData:
        motion_ids = motion_ids.to(device=self.device, dtype=torch.long)
        starts = starts.to(device=self.device, dtype=torch.long)
        steps = steps.to(device=self.device, dtype=torch.long)
        if motion_ids.numel() == 0:
            return self.datasets[0].get_slice(motion_ids, starts, steps)

        dataset_ids = torch.bucketize(motion_ids, self.route_ends, right=True)
        if self._resident_data is not None and all(self._resident_flags):
            # All routes have fixed-size GPU storage: dynamic nonzero() routing
            # would synchronize the host on every policy frame.
            assert self._resident_route_starts is not None
            assert self._resident_route_ends is not None
            resident_starts = self._resident_route_starts[motion_ids]
            resident_ends = self._resident_route_ends[motion_ids]
            index = (resident_starts + starts).unsqueeze(1) + steps.unsqueeze(0)
            index.clamp_max_(resident_ends.unsqueeze(1) - 1)
            index.clamp_min_(resident_starts.unsqueeze(1))
            result = convert_motion_data(
                self._resident_data[index], float_dtype=torch.float32
            )
            result.motion_id = (
                result.motion_id
                - self._resident_motion_id_offsets[dataset_ids].unsqueeze(1)
            )
            return result

        parts = []
        positions = []
        if self._resident_data is not None:
            assert self._resident_route_starts is not None
            assert self._resident_route_ends is not None
            resident_positions = torch.nonzero(
                self._resident_dataset_mask[dataset_ids], as_tuple=False
            ).squeeze(-1)
            if resident_positions.numel():
                resident_ids = motion_ids[resident_positions]
                resident_starts = self._resident_route_starts[resident_ids]
                resident_ends = self._resident_route_ends[resident_ids]
                index = (
                    resident_starts + starts[resident_positions]
                ).unsqueeze(1) + steps.unsqueeze(0)
                index.clamp_max_(resident_ends.unsqueeze(1) - 1)
                index.clamp_min_(resident_starts.unsqueeze(1))
                resident_result = convert_motion_data(
                    self._resident_data[index], float_dtype=torch.float32
                )
                resident_result.motion_id = (
                    resident_result.motion_id
                    - self._resident_motion_id_offsets[
                        dataset_ids[resident_positions]
                    ].unsqueeze(1)
                )
                parts.append(resident_result)
                positions.append(resident_positions)

        for dataset_index, dataset in enumerate(self.datasets):
            if self._resident_data is not None and self._resident_flags[dataset_index]:
                continue
            member_positions = torch.nonzero(
                dataset_ids == dataset_index, as_tuple=False
            ).squeeze(-1)
            if not member_positions.numel():
                continue
            parts.append(
                dataset.get_slice(
                    motion_ids[member_positions] - self.route_offsets[dataset_index],
                    starts[member_positions],
                    steps,
                )
            )
            positions.append(member_positions)

        if not parts:
            raise IndexError("motion_ids do not route to any child dataset")
        if len(parts) == 1:
            return parts[0]
        merged_positions = torch.cat(positions)
        return torch.cat(parts, dim=0)[torch.argsort(merged_positions)]

    def sample_motion(
        self,
        env_ids: torch.Tensor,
        *,
        terminated_t: torch.Tensor,
        rewind_mask: torch.Tensor,
        rewind_steps: torch.Tensor,
    ) -> MotionSample:
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        terminated_t = terminated_t.to(device=self.device, dtype=torch.long)
        rewind_mask = rewind_mask.to(device=self.device, dtype=torch.bool)
        rewind_steps = rewind_steps.to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return MotionSample(
                motion_id=env_ids,
                motion_len=env_ids,
                start_t=env_ids,
            )

        self._has_sampled = True
        if self._fixed_env_dataset_ids is not None:
            dataset_ids = self._fixed_env_dataset_ids[env_ids]
        else:
            dataset_ids = self._current_env_dataset_ids[env_ids]
            new_dataset_mask = (~rewind_mask) | (dataset_ids < 0)
            if torch.any(new_dataset_mask):
                dataset_ids = dataset_ids.clone()
                dataset_ids[new_dataset_mask] = torch.multinomial(
                    self._dataset_probs,
                    int(new_dataset_mask.sum().item()),
                    replacement=True,
                )

        motion_ids = torch.empty_like(env_ids)
        motion_lengths = torch.empty_like(env_ids)
        start_ts = torch.empty_like(env_ids)
        for dataset_index, dataset in enumerate(self.datasets):
            positions = torch.nonzero(
                dataset_ids == dataset_index, as_tuple=False
            ).squeeze(-1)
            if not positions.numel():
                continue
            sampled = dataset.sample_motion(
                env_ids[positions],
                terminated_t=terminated_t[positions],
                rewind_mask=rewind_mask[positions],
                rewind_steps=rewind_steps[positions],
            )
            motion_ids[positions] = (
                sampled.motion_id.to(device=self.device, dtype=torch.long)
                + self.route_offsets[dataset_index]
            )
            motion_lengths[positions] = sampled.motion_len.to(self.device)
            start_ts[positions] = sampled.start_t.to(self.device)

        self._current_env_dataset_ids[env_ids] = dataset_ids
        return MotionSample(
            motion_id=motion_ids,
            motion_len=motion_lengths,
            start_t=start_ts,
        )
