from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

import torch
from any4hdmi import BaseDataset, MotionDatasetRouter, MotionSample


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
    if len(datasets) == 1:
        return datasets[0]
    return WeightedMultiMotionDataset(
        motion_cfgs=motion_cfgs,
        datasets=datasets,
        num_envs=num_envs,
    )


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
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        self.motion_cfgs = list(motion_cfgs)
        self.num_envs = int(num_envs)
        self.router = MotionDatasetRouter(datasets)
        self.datasets = self.router.datasets
        self.body_names = self.router.body_names
        self.joint_names = self.router.joint_names
        self.motion_paths = self.router.motion_paths
        self.starts = self.router.starts
        self.ends = self.router.ends
        self.lengths = self.router.lengths
        self.device = self.router.device
        weights = torch.tensor([cfg.weight for cfg in self.motion_cfgs])
        self._dataset_probs = (weights / weights.sum()).to(self.device)
        self._env_dataset_id = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )

    @property
    def num_motions(self) -> int:
        return self.router.num_motions

    @property
    def num_steps(self) -> int:
        return self.router.num_steps

    def to(self, device: torch.device | str) -> WeightedMultiMotionDataset:
        self.router.to(device)
        self.datasets = self.router.datasets
        self.starts = self.router.starts
        self.ends = self.router.ends
        self.lengths = self.router.lengths
        self.device = self.router.device
        self._dataset_probs = self._dataset_probs.to(self.device)
        self._env_dataset_id = self._env_dataset_id.to(self.device)
        return self

    def get_slice(
        self,
        motion_ids: torch.Tensor,
        starts: torch.Tensor,
        steps: torch.Tensor,
    ):
        return self.router.get_slice(motion_ids, starts, steps)

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

        dataset_ids = self._env_dataset_id[env_ids]
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
                + self.router.route_offsets[dataset_index]
            )
            motion_lengths[positions] = sampled.motion_len.to(self.device)
            start_ts[positions] = sampled.start_t.to(self.device)

        self._env_dataset_id[env_ids] = dataset_ids
        return MotionSample(
            motion_id=motion_ids,
            motion_len=motion_lengths,
            start_t=start_ts,
        )
