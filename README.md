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

## Train

Run single-stage PPO:

```bash
bash scripts/launch_ddp.sh 0,1,2,3,4,5,6,7 projects/mimic-lite/scripts/train.py venv/mjlab \
  task=lafan_sonic_100style_real +exp=ppo/train algo/ppo/module=residual backend=mjlab
```

Run SAC:

```bash
bash scripts/launch_ddp.sh 0,1,2,3,4,5,6,7 projects/mimic-lite/scripts/train.py venv/mjlab \
  task=lafan_sonic_100style_real +exp=sac/train backend=mjlab
```

Run sequential PPO:

```bash
uv --project venv/mjlab run projects/mimic-lite/scripts/train_sequential.py \
  nproc_per_node=8 task=lafan_100style_real stages=normal backend=mjlab
```

To play checkpoints:

```bash
uv --project venv/mjlab run projects/mimic-lite/scripts/play.py \
  task=lafan +exp=ppo/train backend=mjlab \
  checkpoint_path=run:elijahgalahad/hdmi/runs/kr0vxm4n \
  task.num_envs=4 task.termination.root_pos_error.enabled=false

uv --project venv/mjlab run projects/mimic-lite/scripts/play.py \
  task=lafan +exp=sac/train backend=mjlab \
  checkpoint_path=run:elijahgalahad/hdmi/runs/5jzn40za \
  task.num_envs=4 task.termination.root_pos_error.enabled=false

uv --project venv/mjlab run projects/mimic-lite/scripts/play.py \
  task=lafan +exp=ppo_roa/finetune backend=mjlab \
  checkpoint_path=run:elijahgalahad/hdmi/runs/a03f770b_lafan_100style_finetune \
  task.num_envs=4 task.termination.root_pos_error.enabled=false
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
