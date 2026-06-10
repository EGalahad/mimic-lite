"""Fixed-step rollout evaluation for mimic-lite policies.

This entry point is intentionally narrower than play.py:
- default headless execution
- 512 parallel environments by default
- fixed-step rollout, default 1000 steps
- TensorDict output for downstream metric aggregation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tqdm import tqdm

import active_adaptation as aa
from active_adaptation.learning.modules.vecnorm import VecNorm
from active_adaptation.utils.wandb import parse_checkpoint_path

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase


FILE_PATH = Path(__file__).resolve().parent
CONFIG_PATH = FILE_PATH.parent / "cfg"

ROLLOUT_KEYS = (
    ("next", "done"),
    ("next", "terminated"),
    ("next", "truncated"),
    ("next", "reward"),
    ("next", "stats"),
    ("next", "is_init"),
)


def _scalar_tensordict(values: dict[str, float | int], *, device: torch.device) -> TensorDict:
    return TensorDict(
        {key: torch.as_tensor(value, device=device) for key, value in values.items()},
        batch_size=[],
    )


def _jsonable(values: dict[str, float | int]) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for key, value in values.items():
        if isinstance(value, bool):
            result[key] = int(value)
        elif isinstance(value, int):
            result[key] = value
        else:
            result[key] = float(value)
    return result


def _initial_motion_length(carry: TensorDict, env: "_EnvBase") -> torch.Tensor:
    command_manager = getattr(env.base_env, "command_manager", None)
    if command_manager is not None:
        return command_manager.motion_len.detach().clone().to(torch.float32).squeeze(-1)
    try:
        motion_group = carry["_motion_length"]
        if isinstance(motion_group, TensorDict):
            motion_length = motion_group["motion_length"]
        else:
            motion_length = motion_group
    except KeyError:
        motion_length = env.base_env.command_manager.motion_len
    return motion_length.detach().clone().to(torch.float32).squeeze(-1)


@VecNorm.freeze()
def fixed_step_eval(cfg: DictConfig, env: "_EnvBase", policy) -> TensorDict:
    env.base_env.eval()
    assert not env.base_env.training

    rollout_policy = policy.get_rollout_policy("eval")
    steps = int(cfg.get("eval_steps", 1000))
    store_rollout = bool(cfg.get("store_rollout", True))
    carry = env.reset()
    motion_length = _initial_motion_length(carry, env)
    first_done_step = torch.full(
        (int(env.num_envs),),
        steps,
        dtype=torch.float32,
        device=motion_length.device,
    )
    first_done_seen = torch.zeros(
        (int(env.num_envs),), dtype=torch.bool, device=motion_length.device
    )

    termination_names = (
        "motion_timeout",
        "root_pos_error",
        "root_ori_error",
        "body_pos_error",
        "body_ori_error",
    )
    termination_counts = {name: 0.0 for name in termination_names}
    done_count = 0.0
    trajs = []

    with torch.inference_mode(), set_exploration_type(ExplorationType.DETERMINISTIC):
        for step_idx in tqdm(range(steps), desc="Eval rollout", unit="step"):
            carry = rollout_policy(carry)
            td, carry = env.step_and_maybe_reset(carry)
            next_td = td["next"]
            done = next_td["done"].squeeze(-1)
            newly_done = done & ~first_done_seen
            first_done_step[newly_done] = float(step_idx + 1)
            first_done_seen |= done
            done_count += float(done.sum().item())
            for name in termination_names:
                try:
                    term_value = next_td["stats", "termination", name].squeeze(-1)
                except KeyError:
                    continue
                termination_counts[name] += float(term_value[done].float().sum().item())
            if store_rollout:
                trajs.append(td.select(*ROLLOUT_KEYS, strict=False).cpu())

    rollout = (
        torch.stack(trajs, dim=1)
        if store_rollout
        else TensorDict({}, batch_size=[int(env.num_envs), 0])
    )
    motion_length_safe = motion_length.clamp_min(1.0)
    progress_horizon = torch.minimum(
        motion_length_safe,
        torch.full_like(motion_length_safe, float(steps)),
    )
    first_episode_length = torch.minimum(first_done_step, progress_horizon)
    lafan_progress = (first_episode_length / progress_horizon.clamp_min(1.0)).clamp(
        max=1.0
    )
    reward_stats = {key: float(value) for key, value in env.stats_ema.items()}
    episode_summary = {
        f"stats/termination/{name}": (
            termination_counts[name] / done_count if done_count > 0 else 0.0
        )
        for name in termination_names
    }

    success_rate = 0.0
    if done_count > 0:
        success_rate = (
            termination_counts["motion_timeout"] + termination_counts["root_pos_error"]
        ) / done_count

    summary = {
        "steps": steps,
        "num_envs": int(env.num_envs),
        "success_rate": float(success_rate),
        "joint_pos": reward_stats.get("reward.tracking_metrics/joint_pos", 0.0),
        "body_pos": reward_stats.get("reward.tracking_metrics/body_pos", 0.0),
        "body_ori": reward_stats.get("reward.tracking_metrics/body_ori", 0.0),
        "num_finished_episodes": int(done_count),
        "lafan_progress": float(lafan_progress.mean().item()),
        "lafan_progress_std": float(lafan_progress.std(unbiased=False).item()),
        "first_episode_length_mean": float(first_episode_length.mean().item()),
        "motion_length_mean": float(motion_length_safe.mean().item()),
        "progress_horizon_mean": float(progress_horizon.mean().item()),
        "num_first_episodes_finished": int(first_done_seen.sum().item()),
    }

    return TensorDict(
        {
            "rollout": rollout,
            "lafan_progress": TensorDict(
                {
                    "progress": lafan_progress.detach().cpu(),
                    "first_episode_length": first_episode_length.detach().cpu(),
                    "motion_length": motion_length_safe.detach().cpu(),
                    "first_done_seen": first_done_seen.detach().cpu(),
                },
                batch_size=[int(env.num_envs)],
            ),
            "summary": _scalar_tensordict(summary, device=torch.device("cpu")),
            "reward_stats": _scalar_tensordict(reward_stats, device=torch.device("cpu")),
            "episode_summary": _scalar_tensordict(
                episode_summary, device=torch.device("cpu")
            ),
        },
        batch_size=[],
    )


def _write_summary_json(path: Path, cfg: DictConfig, result: TensorDict) -> None:
    summary = _jsonable(
        {key: value.item() for key, value in result["summary"].items()}
    )
    reward_stats = _jsonable(
        {key: value.item() for key, value in result["reward_stats"].items()}
    )
    episode_summary = _jsonable(
        {key: value.item() for key, value in result["episode_summary"].items()}
    )
    payload = {
        "checkpoint_path": str(cfg.get("checkpoint_path", "")),
        "summary": summary,
        "reward_stats": reward_stats,
        "episode_summary": episode_summary,
        "tensordict_output": str(Path(cfg.eval_output).resolve()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"Saved eval summary to {path}")
    print("SUMMARY", json.dumps(summary, sort_keys=True))


@hydra.main(config_path=str(CONFIG_PATH), config_name="eval", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    cfg.headless = bool(cfg.get("headless", True))
    cfg.app.headless = cfg.headless

    aa.init(cfg, auto_rank=False)

    from active_adaptation.helpers import make_env_policy

    checkpoint_path = parse_checkpoint_path(cfg.get("checkpoint_path", None))
    if checkpoint_path is not None:
        cfg.checkpoint_path = checkpoint_path

    env, policy = make_env_policy(cfg)
    result = fixed_step_eval(cfg, env, policy)

    output_path = Path(cfg.get("eval_output", "eval_rollout.pt")).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    print(f"Saved eval TensorDict to {output_path}")

    summary_output = cfg.get("eval_summary_output", None)
    if summary_output is not None:
        _write_summary_json(Path(summary_output).resolve(), cfg, result)

    env.close()


if __name__ == "__main__":
    main()
