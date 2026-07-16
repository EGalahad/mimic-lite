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

The default G1 mixture enables sharding only for the full SONIC dataset.

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
