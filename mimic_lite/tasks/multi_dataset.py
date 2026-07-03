from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

import torch
from any4hdmi import BaseDataset, DatasetIndex, MotionSample


@dataclass(frozen=True)
class MotionDatasetConfig:
    name: str
    path: str | list[str]
    weight: float
    full_motion: bool
    filenames: list[str] | None = None
    filenames_path: str | None = None


def _as_long_tensor(value: Any, *, name: str, device: torch.device) -> torch.Tensor:
    if value is None:
        raise TypeError(f"{name} must be present on child datasets")
    tensor = torch.as_tensor(value, device=device, dtype=torch.long)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be a 1D tensor, got shape {tuple(tensor.shape)}")
    return tensor


def _combine_dataset_metadata(
    datasets: Sequence[BaseDataset],
    *,
    motion_id_offsets: Sequence[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, DatasetIndex]:
    combined_starts: list[torch.Tensor] = []
    combined_ends: list[torch.Tensor] = []
    combined_motion_ids: list[torch.Tensor] = []
    combined_steps: list[torch.Tensor] = []
    current_step_offset = 0

    for dataset_index, (dataset, motion_id_offset) in enumerate(zip(datasets, motion_id_offsets)):
        starts = _as_long_tensor(
            getattr(dataset, "starts", None),
            name=f"datasets[{dataset_index}].starts",
            device=device,
        )
        ends = _as_long_tensor(
            getattr(dataset, "ends", None),
            name=f"datasets[{dataset_index}].ends",
            device=device,
        )
        if starts.shape != ends.shape:
            raise ValueError(
                f"datasets[{dataset_index}] starts/ends must share shape, got "
                f"{tuple(starts.shape)} and {tuple(ends.shape)}"
            )
        if starts.numel() != int(dataset.num_motions):
            raise ValueError(
                f"datasets[{dataset_index}] starts count {starts.numel()} does not match "
                f"num_motions={int(dataset.num_motions)}"
            )

        data = getattr(dataset, "data", None)
        if data is None:
            raise TypeError(f"datasets[{dataset_index}] must expose a data index")
        motion_ids = _as_long_tensor(
            getattr(data, "motion_id", None),
            name=f"datasets[{dataset_index}].data.motion_id",
            device=device,
        )
        steps = _as_long_tensor(
            getattr(data, "step", None),
            name=f"datasets[{dataset_index}].data.step",
            device=device,
        )
        if motion_ids.shape != steps.shape:
            raise ValueError(
                f"datasets[{dataset_index}] data.motion_id/data.step must share shape, got "
                f"{tuple(motion_ids.shape)} and {tuple(steps.shape)}"
            )
        if motion_ids.numel() != int(dataset.num_steps):
            raise ValueError(
                f"datasets[{dataset_index}] index length {motion_ids.numel()} does not match "
                f"num_steps={int(dataset.num_steps)}"
            )

        combined_starts.append(starts + current_step_offset)
        combined_ends.append(ends + current_step_offset)
        combined_motion_ids.append(motion_ids + int(motion_id_offset))
        combined_steps.append(steps)
        current_step_offset += int(dataset.num_steps)

    starts = torch.cat(combined_starts, dim=0)
    ends = torch.cat(combined_ends, dim=0)
    lengths = ends - starts
    data = DatasetIndex(
        motion_id=torch.cat(combined_motion_ids, dim=0),
        step=torch.cat(combined_steps, dim=0),
    )
    return starts, ends, lengths, data


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
    if isinstance(value, (str, bytes)):
        raise TypeError(f"motion_cfgs[{name!r}] filenames must be a sequence of strings, not a string")
    if not isinstance(value, Sequence):
        raise TypeError(f"motion_cfgs[{name!r}] filenames must be a sequence of strings")
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
        raise TypeError(
            "motion_cfgs must be a mapping of dataset name to "
            "{path, weight, full_motion, filenames, filenames_path}"
        )

    configs: list[MotionDatasetConfig] = []
    for raw_name, raw_cfg in motion_cfgs.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("motion_cfgs dataset names must not be empty")

        if not isinstance(raw_cfg, Mapping):
            raise TypeError(
                f"motion_cfgs[{name!r}] must be a mapping with keys "
                "{path, weight, full_motion, filenames, filenames_path}"
            )

        allowed_keys = {"path", "weight", "full_motion", "filenames", "filenames_path"}
        missing_keys = [key for key in ("path", "weight", "full_motion") if key not in raw_cfg]
        if missing_keys:
            raise ValueError(
                f"motion_cfgs[{name!r}] is missing required keys: {', '.join(missing_keys)}"
            )
        unexpected_keys = sorted(str(key) for key in raw_cfg.keys() if key not in allowed_keys)
        if unexpected_keys:
            raise ValueError(
                f"motion_cfgs[{name!r}] has unexpected keys: {', '.join(unexpected_keys)}"
            )

        data_path = _normalize_data_path(raw_cfg["path"])
        raw_weight = raw_cfg["weight"]
        if not isinstance(raw_weight, (int, float, str)):
            raise TypeError(
                f"motion_cfgs[{name!r}] weight must be numeric or numeric-string, got {type(raw_weight).__name__}"
            )
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(
                f"motion_cfgs[{name!r}] weight must be a positive finite number, got {raw_cfg['weight']!r}"
            )

        full_motion = raw_cfg["full_motion"]
        if not isinstance(full_motion, bool):
            raise TypeError(
                f"motion_cfgs[{name!r}] full_motion must be a boolean, got {type(full_motion).__name__}"
            )

        filenames = _normalize_motion_filenames(raw_cfg.get("filenames"), name=name)
        filenames_path = _normalize_optional_path(
            raw_cfg.get("filenames_path"),
            name=name,
            key="filenames_path",
        )
        if filenames is not None and filenames_path is not None:
            raise ValueError(f"motion_cfgs[{name!r}] must provide only one of filenames or filenames_path")

        configs.append(
            MotionDatasetConfig(
                name=name,
                path=data_path,
                weight=weight,
                full_motion=full_motion,
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
    serialized: dict[str, dict[str, str | list[str] | float | bool]] = {}
    for cfg in motion_cfgs:
        data_path: str | list[str]
        if isinstance(cfg.path, list):
            data_path = list(cfg.path)
        else:
            data_path = cfg.path
        serialized[cfg.name] = {
            "path": data_path,
            "weight": float(cfg.weight),
            "full_motion": cfg.full_motion,
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
            filenames=cfg.filenames,
            filenames_path=cfg.filenames_path,
            body_names=body_names,
            joint_names=joint_names,
            windowed_next_window_device=windowed_next_window_device,
            windowed_pin_window_load=windowed_pin_window_load,
        )
        for cfg in motion_cfgs
    ]
    if len(datasets) == 1:
        return datasets[0]
    return WeightedMultiMotionDataset(motion_cfgs=motion_cfgs, datasets=datasets, num_envs=num_envs)


class WeightedMultiMotionDataset(BaseDataset):
    dataset_kind = "weighted_multi"

    def __init__(
        self,
        *,
        motion_cfgs: Sequence[MotionDatasetConfig],
        datasets: Sequence[BaseDataset],
        num_envs: int,
    ) -> None:
        if len(motion_cfgs) != len(datasets):
            raise ValueError("motion_cfgs and datasets must have identical lengths")
        if not datasets:
            raise ValueError("WeightedMultiMotionDataset requires at least one dataset")

        self.motion_cfgs = list(motion_cfgs)
        self.datasets: list[BaseDataset] = list(datasets)
        self.num_envs = int(num_envs)

        reference_dataset = self.datasets[0]
        self.body_names = list(reference_dataset.body_names)
        self.joint_names = list(reference_dataset.joint_names)
        for cfg, dataset in zip(self.motion_cfgs[1:], self.datasets[1:]):
            if list(dataset.body_names) != self.body_names:
                raise ValueError(
                    f"Dataset {cfg.name!r} body_names do not match the first dataset"
                )
            if list(dataset.joint_names) != self.joint_names:
                raise ValueError(
                    f"Dataset {cfg.name!r} joint_names do not match the first dataset"
                )

        dataset_weights = torch.tensor(
            [cfg.weight for cfg in self.motion_cfgs],
            dtype=torch.float32,
        )
        self._dataset_probs = dataset_weights / dataset_weights.sum()
        self.device = torch.device(reference_dataset.device)
        self._dataset_probs = self._dataset_probs.to(self.device)
        self._env_dataset_id = torch.full(
            (self.num_envs,),
            -1,
            device=self.device,
            dtype=torch.long,
        )

        motion_id_offsets: list[int] = []
        motion_id_ends: list[int] = []
        motion_route_offsets: list[int] = []
        motion_route_ends: list[int] = []
        current_offset = 0
        current_route_offset = 0
        for dataset in self.datasets:
            motion_id_offsets.append(current_offset)
            current_offset += int(dataset.num_motions)
            motion_id_ends.append(current_offset)
            motion_route_offsets.append(current_route_offset)
            route_span = int(dataset.num_motions)
            if hasattr(dataset, "_current_window"):
                route_span = max(route_span, self.num_envs)
            current_route_offset += route_span
            motion_route_ends.append(current_route_offset)
        self._motion_id_offsets = torch.tensor(
            motion_id_offsets,
            device=self.device,
            dtype=torch.long,
        )
        self._motion_id_ends = torch.tensor(
            motion_id_ends,
            device=self.device,
            dtype=torch.long,
        )
        self._motion_route_offsets = torch.tensor(
            motion_route_offsets,
            device=self.device,
            dtype=torch.long,
        )
        self._motion_route_ends = torch.tensor(
            motion_route_ends,
            device=self.device,
            dtype=torch.long,
        )
        self.starts, self.ends, self.lengths, self.data = _combine_dataset_metadata(
            self.datasets,
            motion_id_offsets=motion_id_offsets,
            device=self.device,
        )

        self.motion_paths = []
        for dataset in self.datasets:
            self.motion_paths.extend(getattr(dataset, "motion_paths", []))

    @property
    def num_motions(self) -> int:
        if self._motion_id_ends.numel() == 0:
            return 0
        return int(self._motion_id_ends[-1].item())

    @property
    def num_steps(self) -> int:
        return sum(int(dataset.num_steps) for dataset in self.datasets)

    def to(self, device: torch.device | str):
        target_device = torch.device(device)
        self.datasets = [dataset.to(target_device) for dataset in self.datasets]
        self._dataset_probs = self._dataset_probs.to(target_device)
        self._env_dataset_id = self._env_dataset_id.to(target_device)
        self._motion_id_offsets = self._motion_id_offsets.to(target_device)
        self._motion_id_ends = self._motion_id_ends.to(target_device)
        self._motion_route_offsets = self._motion_route_offsets.to(target_device)
        self._motion_route_ends = self._motion_route_ends.to(target_device)
        self.starts = self.starts.to(target_device)
        self.ends = self.ends.to(target_device)
        self.lengths = self.lengths.to(target_device)
        self.data = self.data.to(target_device)
        self.device = target_device
        return self

    def get_slice(
        self,
        motion_ids: torch.Tensor,
        starts: torch.Tensor,
        steps: torch.Tensor,
        *,
        profile_name: str | None = None,
    ):
        motion_ids = motion_ids.to(device=self.device, dtype=torch.long)
        starts = starts.to(device=self.device, dtype=torch.long)
        steps = steps.to(device=self.device, dtype=torch.long)

        if motion_ids.numel() == 0:
            return self.datasets[0].get_slice(
                motion_ids,
                starts,
                steps,
                profile_name=profile_name,
            )

        dataset_ids = torch.bucketize(motion_ids, self._motion_route_ends, right=True)
        slice_parts = []
        slice_positions = []
        for dataset_index, dataset in enumerate(self.datasets):
            member_positions = torch.nonzero(
                dataset_ids == dataset_index,
                as_tuple=False,
            ).squeeze(-1)
            if member_positions.numel() == 0:
                continue
            local_motion_ids = (
                motion_ids.index_select(0, member_positions)
                - self._motion_route_offsets[dataset_index]
            )
            part = dataset.get_slice(
                local_motion_ids,
                starts.index_select(0, member_positions),
                steps,
                profile_name=profile_name,
            )
            slice_parts.append(part)
            slice_positions.append(member_positions)

        if len(slice_parts) == 1:
            return slice_parts[0]

        merged_positions = torch.cat(slice_positions, dim=0)
        merged_slices = torch.cat(slice_parts, dim=0)
        return merged_slices[torch.argsort(merged_positions)]

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
            empty = torch.empty((0,), device=self.device, dtype=torch.long)
            return MotionSample(motion_id=empty, motion_len=empty, start_t=empty)

        dataset_ids = self._env_dataset_id.index_select(0, env_ids)
        new_dataset_mask = (~rewind_mask) | (dataset_ids < 0)
        if bool(torch.any(new_dataset_mask).item()):
            dataset_ids = dataset_ids.clone()
            dataset_ids[new_dataset_mask] = torch.multinomial(
                self._dataset_probs,
                int(new_dataset_mask.sum().item()),
                replacement=True,
            )

        motion_ids = torch.empty_like(env_ids)
        motion_lens = torch.empty_like(env_ids)
        start_ts = torch.empty_like(env_ids)

        for dataset_index, dataset in enumerate(self.datasets):
            member_positions = torch.nonzero(
                dataset_ids == dataset_index,
                as_tuple=False,
            ).squeeze(-1)
            if member_positions.numel() == 0:
                continue

            sampled = dataset.sample_motion(
                env_ids.index_select(0, member_positions),
                terminated_t=terminated_t.index_select(0, member_positions),
                rewind_mask=rewind_mask.index_select(0, member_positions),
                rewind_steps=rewind_steps.index_select(0, member_positions),
            )
            motion_ids.index_copy_(
                0,
                member_positions,
                sampled.motion_id.to(device=self.device, dtype=torch.long)
                + self._motion_route_offsets[dataset_index],
            )
            motion_lens.index_copy_(
                0,
                member_positions,
                sampled.motion_len.to(device=self.device, dtype=torch.long),
            )
            start_ts.index_copy_(
                0,
                member_positions,
                sampled.start_t.to(device=self.device, dtype=torch.long),
            )

        self._env_dataset_id.index_copy_(0, env_ids, dataset_ids)
        return MotionSample(motion_id=motion_ids, motion_len=motion_lens, start_t=start_ts)

    def find_joints(self, *args, **kwargs):
        return self.datasets[0].find_joints(*args, **kwargs)

    def find_bodies(self, *args, **kwargs):
        return self.datasets[0].find_bodies(*args, **kwargs)
