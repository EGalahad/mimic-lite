"""
Play and export policy for HDMI project.

This script mirrors the full export behavior from HDMI/scripts/play.py:
- export traced policy as .pt
- export ONNX as .onnx
- export deploy config as .yaml
"""

from __future__ import annotations

import copy
import datetime
import itertools
import os
import re
import secrets
import time
from pathlib import Path
from tqdm import tqdm
import yaml

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torchrl.envs.transforms import VecNorm as TorchRLVecNorm
from torchrl.envs.utils import ExplorationType, set_exploration_type
from active_adaptation.utils.profiling import ScopedTimer

import active_adaptation as aa
from active_adaptation.learning.modules.vecnorm import VecNorm
from active_adaptation.utils.export import export_onnx
from active_adaptation.utils.helpers import EpisodeStats
from active_adaptation.utils.timerfd import Timer
from active_adaptation.utils.wandb import parse_checkpoint_path

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase

FILE_PATH = Path(__file__).resolve().parent
CONFIG_PATH = FILE_PATH.parent / "cfg"


def _get_asset_meta(asset) -> dict:
    meta = {
        "joint_names": list(getattr(asset, "joint_names", [])),
        "joint_kp": {},
        "joint_kd": {},
        "default_joint_pos": {},
    }

    cfg = getattr(asset, "cfg", None)
    init_state = getattr(cfg, "init_state", None)
    if init_state is not None and hasattr(init_state, "joint_pos"):
        joint_pos = init_state.joint_pos
        if isinstance(joint_pos, dict):
            meta["default_joint_pos"] = dict(joint_pos)

    actuators = getattr(asset, "actuators", [])
    for actuator in actuators:
        acfg = getattr(actuator, "cfg", None)
        if acfg is None:
            continue

        names = (
            getattr(acfg, "target_names_expr", None)
            or getattr(acfg, "joint_names_expr", None)
            or []
        )
        stiffness = getattr(acfg, "stiffness", None)
        damping = getattr(acfg, "damping", None)

        for joint_name in names:
            if stiffness is not None:
                meta["joint_kp"][joint_name] = float(stiffness)
            if damping is not None:
                meta["joint_kd"][joint_name] = float(damping)

    return meta


def _checkpoint_tags(checkpoint_path: str | None) -> tuple[str, str]:
    wandb_run_id = "unknown"
    checkpoint_num = "unknown"

    if checkpoint_path is None:
        return wandb_run_id, checkpoint_num

    state_dict = torch.load(checkpoint_path, weights_only=False)
    if "wandb" in state_dict and "id" in state_dict["wandb"]:
        wandb_run_id = state_dict["wandb"]["id"]

    filename = os.path.basename(checkpoint_path)
    match = re.search(r"checkpoint_(\d+)", filename)
    if match:
        checkpoint_num = match.group(1)
    elif filename.endswith("_final.pt"):
        checkpoint_num = "final"

    return wandb_run_id, checkpoint_num


def _make_render_output_path() -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = secrets.token_hex(4)
    return Path.cwd() / f"{timestamp}-{suffix}.mp4"



@VecNorm.freeze()
def export_policy(cfg: DictConfig, env: "_EnvBase", policy) -> None:
    checkpoint_path = parse_checkpoint_path(cfg.checkpoint_path)
    wandb_run_id, checkpoint_num = _checkpoint_tags(checkpoint_path)

    deploy_policy = copy.deepcopy(policy.get_rollout_policy("deploy")).cpu()
    fake_input = env.observation_spec[0].rand().cpu()

    export_dir = FILE_PATH / "exports" / str(cfg.task.name)
    export_dir.mkdir(parents=True, exist_ok=True)
    base = export_dir / f"policy-{wandb_run_id}-{checkpoint_num}"

    onnx_path = str(base.with_suffix(".onnx"))
    yaml_path = str(base.with_suffix(".yaml"))

    export_onnx(deploy_policy, fake_input, onnx_path)

    dict_cfg = OmegaConf.to_container(cfg, resolve=True)
    policy_config = {}

    obs_cfg = policy_config.setdefault("observation", {})
    for k in deploy_policy.in_keys:
        obs_cfg[k] = dict_cfg["task"]["observation"][k]

    asset = env.scene.articulations["robot"]
    asset_meta = _get_asset_meta(asset)
    policy_config["joint_names_simulation"] = asset.cfg.joint_names_simulation
    policy_config["body_names_simulation"] = asset.cfg.body_names_simulation
    policy_config["joint_kp"] = asset_meta["joint_kp"]
    policy_config["joint_kd"] = asset_meta["joint_kd"]
    policy_config["default_joint_pos"] = asset_meta["default_joint_pos"]

    # Make joint observation order explicit for sim2real consumers.
    from mimic_lite.tasks.command import RobotTracking
    from mimic_lite.tasks.actions import JointPosition
    action_manager = cast(JointPosition, env.action_manager)
    policy_config["policy_joint_names"] = action_manager.joint_names
    policy_config["action_scale"] = action_manager.action_scaling.detach().cpu().tolist()

    command = cast(RobotTracking, env.command_manager)

    motion_cfg = policy_config.setdefault("motion", {})
    from mimic_lite.tasks.multi_dataset import motion_cfgs_to_dict

    motion_cfg["motion_cfgs"] = motion_cfgs_to_dict(command.motion_cfgs)
    if len(command.motion_cfgs) == 1 and isinstance(command.motion_cfgs[0].path, str):
        motion_cfg["motion_path"] = str(command.motion_cfgs[0].path)
    motion_cfg["future_steps"] = command.future_steps.tolist()
    motion_cfg["body_names"] = command.tracking_body_names
    motion_cfg["joint_names"] = command.tracking_joint_names
    motion_cfg["root_body_name"] = command.root_body_name
    motion_cfg["anchor_body_name"] = command.anchor_body_name

    with open(yaml_path, "w") as f:
        yaml.dump(policy_config, f, sort_keys=False)

    print(f"Exported deploy config to {yaml_path}")


@hydra.main(config_path=str(CONFIG_PATH), config_name="play", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    aa.init(cfg, auto_rank=True)

    from active_adaptation.helpers import make_env_policy

    checkpoint_path = parse_checkpoint_path(cfg.get("checkpoint_path", None))
    if checkpoint_path is not None:
        cfg.checkpoint_path = checkpoint_path

    env, policy = make_env_policy(cfg)

    if cfg.get("export_policy", False):
        export_policy(cfg, env, policy)

    stats_keys = [
        k
        for k in env.reward_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)
    rollout_policy = policy.get_rollout_policy("eval")

    env.base_env.eval()
    carry = env.reset()

    assert not env.base_env.training

    timer = Timer(env.step_dt)
    fps_window_start = time.perf_counter()
    fps_window_frames = 0
    render_seconds = float(cfg.get("render_seconds", 0.0))
    render_enabled = render_seconds != 0.0
    if render_enabled:
        max_steps = max(1, int(render_seconds / env.step_dt))
        progress = tqdm(range(max_steps), total=max_steps, desc="Playing", unit="step")
    else:
        progress = itertools.count()
    output_path = _make_render_output_path()

    with (
        env.get_recorder(output_path, enabled=render_enabled) as recorder,
        torch.inference_mode(),
        set_exploration_type(ExplorationType.DETERMINISTIC),
        # set_exploration_type(ExplorationType.RANDOM),
    ):
        for i in progress:
            with ScopedTimer("inference", sync=False):
                carry = rollout_policy(carry)
            with ScopedTimer("env_step", sync=False):
                td, carry = env.step_and_maybe_reset(carry)
            episode_stats.add(td)

            if len(episode_stats) >= env.num_envs:
                print("Step", i)
                for k, v in sorted(episode_stats.pop().items(True, True)):
                    print(k, torch.mean(v).item())

            if render_enabled:
                recorder.add_frame()

            fps_window_frames += 1
            window_elapsed = time.perf_counter() - fps_window_start
            if window_elapsed >= 1.0:
                if not render_enabled:
                    print(
                        f"Loop FPS: {fps_window_frames} frames in "
                        f"{window_elapsed:.2f}s"
                    )
                # ScopedTimer.print_summary(clear=True)
                fps_window_start = time.perf_counter()
                fps_window_frames = 0

            timer.sleep()

    env.close()


if __name__ == "__main__":
    main()
