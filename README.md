# MimicLite

## Setup

Clone the `active-adaptation` repository and the MimicLite project:

```bash
git clone -b dev/hdmi https://github.com/Agent-3154/active-adaptation.git
cd active-adaptation
git clone https://github.com/EGalahad/mimic-lite projects/mimic-lite
```

Setup uv venv directories and install dependencies:

```bash
mkdir -p venv/mjlab
cp projects/mimic-lite/pyproject-mjlab.toml venv/mjlab/pyproject.toml

mkdir -p venv/isaaclab
cp projects/mimic-lite/pyproject-isaaclab.toml venv/isaaclab/pyproject.toml
```

The repository should now look like this:

```text
active-adaptation/
├── venv/
│   ├── mjlab/
│   │   └── pyproject.toml
│   └── isaaclab/
│       └── pyproject.toml
├── active_adaptation/
├── projects/
│   └── mimic-lite/
└─ scripts/
```

Refresh project discovery:

```bash
uv --project venv/mjlab run aa-discover-projects
```

This command generates the project registry at `.cache/projects.json`. Open that file and make sure both MimicLite entries are enabled:

Set up environment variables:

```bash
export WANDB_API_KEY=<your_wandb_api_key>
export HF_TOKEN=<your_huggingface_token>
```

Training motion datasets are available in the [any4hdmi Hugging Face collection](https://huggingface.co/collections/elijahgalahad/any4hdmi). Dataset conversion and validation tools are maintained in [`EGalahad/any4hdmi`](https://github.com/EGalahad/any4hdmi).

## Released Checkpoints

The released checkpoint set contains three PPO policies trained for 4,000 iterations. Wall-clock times are reported on RTX 4090 GPUs.

| Policy | Actor hidden dimensions | Parallel environments | Checkpoint | Wall-clock time |
| --- | --- | ---: | --- | ---: |
| MimicLite-Huge | `[1024, 1024, 1024]` | `32 × 8192` | [`xua2csee`](https://wandb.ai/elijahgalahad/mimic_lite/runs/xua2csee) | 3 h 30 min |
| MimicLite-Base | `[256, 256, 256]` | `8 × 8192` | [`iij0q0b5`](https://wandb.ai/elijahgalahad/mimic_lite/runs/iij0q0b5) | 2 h 57 min |
| MimicLite-Small | `[128, 128, 128]` | `4 × 8192` | [`zb9e19ih`](https://wandb.ai/elijahgalahad/mimic_lite/runs/zb9e19ih) | 3 h 00 min |

Training-time sources: Huge [`55ie49o5`](https://wandb.ai/elijahgalahad/mimic_lite/runs/55ie49o5), Base [`07k900hl`](https://wandb.ai/elijahgalahad/mimic_lite/runs/07k900hl), and Small [`akq50h1n`](https://wandb.ai/elijahgalahad/mimic_lite/runs/akq50h1n).

## Train

Run single-stage PPO:

```bash
bash scripts/launch_ddp.sh 0,1,2,3,4,5,6,7 projects/mimic-lite/scripts/train.py venv/mjlab \
  task=tracking-base task/motion=g1/mixture +exp=ppo/train backend=mjlab
```

Each `motion_cfgs` entry controls partitioning and runtime storage independently:

- `shard: true` gives each distributed rank a disjoint, motion-aligned subset; it defaults to `false`.
- `full_motion: true` keeps the visible dataset resident in GPU memory, while `false` uses persistent window pools with asynchronous prefetching.

### Bumi tracker (MJLab)

Bumi uses a local MJCF/mesh cache and the 36 gait NPZ files from the Bumi MJLab reference repository. Generated assets and motion wrappers stay under `.cache/` and are not committed.

```bash
export BUMI_REF=/path/to/AMP_mjlab_bumi_worktree

uv --project venv/mjlab run projects/mimic-lite/scripts/prepare_bumi_assets.py \
  --source "$BUMI_REF/src/assets/robots/bumi/xmls"

uv --project venv/mjlab run projects/mimic-lite/scripts/prepare_bumi_motion_dataset.py \
  --source "$BUMI_REF/src/assets/motions/bumi/amp" \
  --link-mode symlink
```

The Bumi MJLab actuators model a 0–4 physics-step position-command delay. The task-level `JointPosition` delay is disabled so latency is applied exactly once. MJLab 1.3 fuses identical builtin actuator delay configurations, so this first version samples one lag per environment for the shared command path; it does not claim independent per-motor lag.

Run a 16-environment, one-update smoke test over the full 36-motion dataset:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv --project venv/mjlab run projects/mimic-lite/scripts/train.py \
  task=tracking-base-bumi task/motion=bumi/omni \
  +exp=ppo/train backend=mjlab task.num_envs=16 total_iters=1 \
  wandb.mode=disabled checkpoint_interval=100 upload_interval=100 \
  '~task.randomization'
```

For the current 36-motion dataset, the recommended single-GPU training budget is 1,000 PPO iterations with 8,192 environments. The entropy schedule is shortened to match this budget; the task randomization and actuator delay remain enabled. Save every 100 iterations and use per-motion evaluation to select the best checkpoint rather than assuming the final checkpoint is best.

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv --project venv/mjlab run python projects/mimic-lite/scripts/train.py \
  task=tracking-base-bumi task/motion=bumi/omni \
  +exp=ppo/train backend=mjlab \
  task.num_envs=8192 total_iters=1000 \
  algo.entropy_decay_start=600 algo.entropy_decay_end=800 \
  checkpoint_interval=100 upload_interval=200 \
  wandb.mode=online
```

For a single-motion mapping/overfit check, replace `task/motion=bumi/omni` with `task/motion=bumi/single`. Checkpoints intended for deployment should be trained with the actuator delay enabled; `~task.randomization` removes the other task randomizations only.

Evaluate a checkpoint and emit per-motion coverage/progress metrics:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv --project venv/mjlab run projects/mimic-lite/scripts/eval.py \
  task=tracking-base-bumi task/motion=bumi/omni backend=mjlab \
  task.num_envs=512 eval_steps=1000 checkpoint_path=/absolute/path/to/checkpoint.pt \
  eval_output=bumi_eval.pt eval_summary_output=bumi_eval.json \
  +store_rollout=false '~task.randomization'
```

The default G1 mixture enables sharding only for the full SONIC dataset.

Visualize the completed 1,000-iteration Bumi run with the repository-level
`scripts/play.py`. The project currently imports the Atom and G1 asset modules
alongside Bumi, so this command forces Hugging Face into offline mode and
removes malformed `socks://` proxy variables before resolving the already
cached asset snapshots:

```bash
env \
  -u ALL_PROXY -u all_proxy \
  -u HTTP_PROXY -u http_proxy \
  -u HTTPS_PROXY -u https_proxy \
  -u SOCKS_PROXY -u socks_proxy \
  CUDA_VISIBLE_DEVICES=0 \
  HF_HUB_OFFLINE=1 \
  HF_HUB_DISABLE_TELEMETRY=1 \
  HF_HUB_CACHE=/home/jun7.shi/.cache/huggingface/hub \
  uv --project venv/mjlab run python scripts/play.py \
    task=tracking-base-bumi task/motion=bumi/omni \
    +exp=ppo/train backend=mjlab task.num_envs=1 \
    +task.command.start_from_zero=true \
    task.command.init_joint_pos_noise=0 \
    task.command.init_joint_vel_noise=0 \
    task.termination.root_pos_error.enabled=false \
    checkpoint_path=outputs/2026-07-22/23-55-41-BumiTrackBase-mimic_lite_ppo/checkpoint_latest.pt \
    '~task.randomization'
```

`checkpoint_latest.pt` currently resolves to `checkpoint_1000.pt`. Each episode
selects one of the 36 motions and starts it from the beginning. This clean
visualization removes task randomization and initial joint-state noise, but it
keeps the actuator-configured 0--4 physics-step delay. Initial startup may spend
several seconds compiling CUDA kernels before the viewer appears.

Play a PPO checkpoint:

```bash
uv --project venv/mjlab run projects/mimic-lite/scripts/play.py \
  task=tracking-base task/motion=g1/lafan \
  +exp=ppo/train algo/ppo/module=huge \
  task.num_envs=4 task.termination.root_pos_error.enabled=false \
  checkpoint_path=run:elijahgalahad/mimic_lite/xua2csee
```

## Troubleshooting

### IsaacLab Warp cache

If IsaacLab picks up Isaac Sim's bundled Warp instead of the venv-installed
`warp-lang`, clear the cached Omni Warp extensions and retry:

```bash
rm -rf venv/isaaclab/.venv/lib/python3.11/site-packages/isaacsim/extscache/omni.warp*
rm -rf venv/isaaclab/.venv/lib/python3.11/site-packages/isaacsim/kit/data/Kit/Isaac-Sim/5.1/exts/3/omni.warp*
```

### mjlab MuJoCo compatibility

If mjlab training fails with an error like `mujoco.mjtEnableBit.mjENBL_MULTICCD` missing while importing `mujoco_warp`, your environment likely resolved `mujoco>=3.8`. Pin `mujoco<3.8` and resync the environment:

```bash
uv --project venv/mjlab add 'mujoco<3.8'
uv --project venv/mjlab sync
```

## Citation

If you find MimicLite useful in your research, please cite:

```bibtex
@misc{mimiclite2026,
  author       = {{RoboParty Lab Team}},
  title        = {MimicLite: Efficient and Effective General Humanoid Motion Tracking},
  year         = {2026},
  howpublished = {\url{https://github.com/EGalahad/mimic-lite}},
  note         = {Technical report: \url{https://github.com/Roboparty/MimicLite/blob/main/mimic-lite.pdf}}
}
```
