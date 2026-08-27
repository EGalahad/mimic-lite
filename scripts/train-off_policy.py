import datetime
import logging
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from setproctitle import setproctitle
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tqdm import tqdm

import active_adaptation as aa

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


FILE_PATH = Path(__file__).resolve().parent
CONFIG_PATH = FILE_PATH.parent / "cfg"


@hydra.main(config_path=str(CONFIG_PATH), config_name="train", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    aa.init(cfg, auto_rank=True)
    print(
        f"is_distributed: {aa.is_distributed()}, "
        f"local_rank: {aa.get_local_rank()}/{aa.get_world_size()}"
    )

    from active_adaptation.helpers import make_env_policy
    from active_adaptation.utils.helpers import EpisodeStats

    env, policy = make_env_policy(
        cfg.task,
        cfg.algo,
        seed=cfg.seed,
        headless=cfg.headless,
        device=cfg.device,
        checkpoint_path=cfg.get("checkpoint_path", None),
    )

    frames_per_iter = env.num_envs * cfg.algo.collect_steps
    total_iters = cfg.get("total_iters", None)
    if total_iters is None:
        total_frames = cfg.get("total_frames", -1) // aa.get_world_size()
        total_frames = total_frames // frames_per_iter * frames_per_iter
        total_iters = total_frames // frames_per_iter

    checkpoint_interval = cfg.checkpoint_interval
    upload_interval = cfg.upload_interval
    log_interval = (cfg.task.max_episode_length // cfg.algo.collect_steps) + 1
    logging.info(f"Log interval: {log_interval} steps")

    stats_keys = [
        key
        for key in env.reward_spec.keys(True, True)
        if isinstance(key, tuple) and key[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)

    run = None

    def save(policy, checkpoint_name: str, *, upload_to_wandb: bool = True):
        assert run is not None
        run_dir = Path(run.dir)
        ckpt_path = run_dir / f"{checkpoint_name}.pt"
        state_dict = OrderedDict()
        state_dict["wandb"] = {"name": run.name, "id": run.id}
        state_dict["policy"] = policy.state_dict()
        state_dict["env"] = env.state_dict()
        state_dict["cfg"] = cfg

        torch.save(state_dict, ckpt_path)
        if upload_to_wandb:
            run.save(str(ckpt_path), policy="now", base_path=run.dir)

        latest_link = run_dir / "checkpoint_latest.pt"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(ckpt_path.name)
        logging.info(
            f"Saved checkpoint to {ckpt_path}" + (" (wandb)" if upload_to_wandb else "")
        )
        return str(ckpt_path)

    def should_save(iter_idx: int) -> bool:
        if not aa.is_main_process():
            return False
        return iter_idx % checkpoint_interval == 0 or iter_idx % upload_interval == 0

    assert env.training

    carry = env.reset()
    rollout_policy = policy.get_rollout_policy("train")
    progress_desc = str(cfg.algo.get("name", "off_policy"))

    next_saved_keys = [
        "done",
        "terminated",
        "truncated",
        "discount",
        "reward",
        "stats",
        "is_init",
        "episode_id",
    ]
    next_saved_keys.extend(policy.get_next_saved_keys())
    next_saved_keys = list(dict.fromkeys(next_saved_keys))

    env_frames = 0
    ckpt_path = None

    if aa.is_main_process():
        run = wandb.init(
            job_type=cfg.wandb.job_type,
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            tags=cfg.wandb.tags,
            id=cfg.wandb.get("id", None),
        )
        run.config.update(OmegaConf.to_container(cfg))
        run.config["world_size"] = aa.get_world_size()

        default_run_name = (
            f"{cfg.exp_name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        wandb_id = run.name.split("-")[-1]
        run.name = f"{wandb_id}-{default_run_name}"
        setproctitle(run.name)

        run_dir = Path(run.dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_save_path = run_dir / "cfg.yaml"
        OmegaConf.save(cfg, cfg_save_path)
        run.save(str(cfg_save_path), policy="now")
        run.save(str(run_dir / "config.yaml"), policy="now")

    progress = range(total_iters)
    if aa.is_main_process():
        progress = tqdm(progress, desc=progress_desc)

    start_iter = getattr(env, "current_iter", 0)
    for iter_idx in progress:
        if should_save(iter_idx):
            should_upload = iter_idx % upload_interval == 0
            checkpoint_name = f"checkpoint_{iter_idx}" if should_upload else "checkpoint_temp"
            ckpt_path = save(policy, checkpoint_name, upload_to_wandb=should_upload)
            if ckpt_path is not None:
                print(f"Latest checkpoint: {ckpt_path}")

        iter_start = time.perf_counter()
        rollout_time = 0.0
        training_time = 0.0
        metric_lists: dict[str, list[float]] = defaultdict(list)

        with set_exploration_type(ExplorationType.RANDOM):
            if hasattr(env, "set_progress"):
                env.set_progress(start_iter + iter_idx)
            for _ in range(cfg.algo.collect_steps):
                rollout_start = time.perf_counter()
                with torch.inference_mode():
                    carry = rollout_policy(carry)
                    td, carry = env.step_and_maybe_reset(carry)
                rollout_time += time.perf_counter() - rollout_start

                episode_stats.add(td)
                td["next"] = td["next"].select(*next_saved_keys, strict=False)

                train_start = time.perf_counter()
                policy.observe(td)
                step_info = policy.update()
                training_time += time.perf_counter() - train_start

                for key, value in step_info.items():
                    if isinstance(value, (float, int)):
                        metric_lists[key].append(float(value))

        env_frames += frames_per_iter

        info = {}
        if iter_idx % log_interval == 0 and len(episode_stats):
            for key, value in sorted(episode_stats.pop().items(True, True)):
                log_key = "train/" + ("/".join(key) if isinstance(key, tuple) else key)
                info[log_key] = torch.mean(value.float()).item()

        for key, values in metric_lists.items():
            if values:
                info[key] = sum(values) / len(values)

        info.update(env.extra)
        info.update(env.stats_ema)
        info["env_frames"] = env_frames * aa.get_world_size()
        info["performance/rollout_fps"] = (
            frames_per_iter / max(rollout_time, 1e-6) * aa.get_world_size()
        )
        info["performance/rollout_time"] = rollout_time
        info["performance/training_time"] = training_time
        info["performance/iter_time"] = time.perf_counter() - iter_start

        if aa.is_main_process() and run is not None:
            run.log(info)

    if aa.is_main_process():
        ckpt_path = save(policy, f"checkpoint_{total_iters}")
        wandb.finish()
        print(f"Final checkpoint: {ckpt_path}")

    raise SystemExit(0)


if __name__ == "__main__":
    main()
