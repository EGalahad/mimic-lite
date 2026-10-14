# MimicLite

<video src="https://github.com/user-attachments/assets/24d16b09-0c8f-4371-9c0d-c07112a6ee0a" controls muted playsinline preload="metadata"></video>

Play the demo and click the speaker icon to enable sound.

## Setup

Clone the current main branches. Active Adaptation no longer embeds MimicLite
as a submodule, so clone the two repositories independently:

```bash
git clone https://github.com/Agent-3154/active-adaptation.git
cd active-adaptation
git clone https://github.com/EGalahad/mimic-lite projects/mimic-lite
```

Setup uv venv directories and install dependencies:

```bash
mkdir -p venv/mjlab
cp projects/mimic-lite/pyproject-mjlab.toml venv/mjlab/pyproject.toml
uv sync --project venv/mjlab

mkdir -p venv/isaaclab
cp projects/mimic-lite/pyproject-isaaclab.toml venv/isaaclab/pyproject.toml
uv sync --project venv/isaaclab
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
uv --project venv/mjlab run aa-project enable mimic_lite
uv --project venv/mjlab run aa-list-tasks
```

These commands refresh project discovery, enable MimicLite, and verify its tasks.

Set up environment variables:

```bash
export WANDB_API_KEY=<your_wandb_api_key>
export HF_TOKEN=<your_huggingface_token>
```

Training motion datasets are available in the [any4hdmi Hugging Face collection](https://huggingface.co/collections/elijahgalahad/any4hdmi). Dataset conversion and validation tools are maintained in [`EGalahad/any4hdmi`](https://github.com/EGalahad/any4hdmi).

## Released Checkpoints

The public releases include the 16x16384 G1 mixture Huge policies and the
8x8192 MimicLite v1.1 ROA student. GPU hours are measured training compute;
the PPO/ROA Huge runs used RTX 4090 GPUs, while v1.1 used RTX 4090 for Train
and Adapt and RTX PRO 6000 for Finetune.

| Policy | Actor hidden dimensions | Parallel environments | Checkpoint | GPU hours |
| --- | --- | ---: | --- | ---: |
| MimicLite-PPO | `[1024, 1024, 1024]` | `16 × 16384` | [`4234dd57`](https://wandb.ai/elijahgalahad/mimic_lite/runs/4234dd57) | 92.3 |
| MimicLite-ROA | `[1024, 1024, 1024]` | `16 × 16384` (`train -> adapt -> finetune`) | [`9287d8e0`](https://wandb.ai/elijahgalahad/mimic_lite/runs/9287d8e0) | 173.2 |
| MimicLite-v1.1 (ROA student) | `[512, 512, 512]` | `8 × 8192` (`train -> adapt -> finetune`) | [Deploy ONNX + YAML](https://drive.google.com/drive/folders/1lrPyiiy7anyG3P4wHNIQQQlydboLPd9e) (`checkpoints/mimic-lite/v1_1`) | 50.106 |

MimicLite also supports **H2, AgiBot A3, and RoboParty RP1 (28 or 24 DoF)** on mjlab and IsaacLab. See [supported robots and released checkpoints](docs/checkpoints.md) for model-matched downloads and motion profiles.

Download the deploy ONNX and YAML from the shared sim2real artifacts:
[MimicLite-PPO](https://drive.google.com/drive/folders/1xRmcOX0l-YIpqxUuCmW4s6Dl0HStbJSL)
and
[MimicLite-ROA](https://drive.google.com/drive/folders/1AFcvP4oDbEskx-wip5bJN-JBaUvwp8MH).
MimicLite v1.1 is a supported release, trained for 4,000 / 1,000 / 2,000
Train / Adapt / Finetune updates. Its original training used
`wandb.mode=disabled`, so it has no online W&B run; the table links to its
released deploy files instead. Teacher observations and reward targets are
clean, with reference-motion observation noise only on the student command.
See the [MotionDecode leaderboard](https://egalahad.github.io/sim2real/leaderboard/)
for its evaluation. Older Huge/Base releases remain in the Drive archive.

![Canonical cross-codebase tracking evaluation](assets/mimic_lite_cross_codebase_tracking_eval.png)

## Train

Run commands from the Active Adaptation repository. Select a robot with
`task/robot=g1|h2|a3|rp1|rp1_24dof|atom`; G1 is the default.
Each profile selects matching assets, joints, rewards and default motion data.
An explicit `task/motion=...` overrides that default. Use
`task=tracking-base task/robot=<robot>` for training, playback, and export.
For Atom, use `task/robot=atom`; its default motion profile is `atom/sonic`.

The base task includes command-reference noise and arm-keypoint rewards for
all robots. PPO training uses the four-second root XYZ window reward. ROA
stages add teacher references through frame 16, with absolute root XYZ in
Train and the window reward in Adapt/Finetune.

Run single-stage G1 PPO:

```bash
bash scripts/launch_ddp.sh 0,1,2,3,4,5,6,7 projects/mimic-lite/scripts/train.py venv/mjlab \
  task=tracking-base task/robot=g1 task/motion=g1/mixture +exp=ppo/train \
  algo/ppo/module=huge backend=mjlab wandb.mode=online
```

Run A3 through the three ROA stages (4,000 Train, 1,000 Adapt,
2,000 Finetune updates), with 8 GPUs and 8,192 environments per GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run --project venv/mjlab projects/mimic-lite/scripts/train_sequential.py \
  task=tracking-base task/robot=a3 backend=mjlab \
  stages=v1_1 algo/ppo_roa/module=large task.num_envs=8192 wandb.mode=online
```

Replace `a3` with another robot selector to use its default motions.
For IsaacLab, use `--project venv/isaaclab` and `backend=isaaclab`.
Cache the referenced HF assets and datasets once before a distributed offline
launch, then set `HF_HUB_OFFLINE=1` and `HF_HUB_DISABLE_TELEMETRY=1`.
Authenticate W&B before starting formal training.

The optional `task/motion=a3/soma` profile accepts a robot-matched,
filtered any4hdmi dataset through `A3_SOMA_MOTION_DATASET`. It uses the existing
512-frame windowed loader (`full_motion: false`); it does not download, retarget
or filter source data. Download the SOMA source yourself, convert it to the robot-matched any4hdmi
format, and filter it locally. Set `A3_SOMA_MOTION_DATASET` to the resulting
local dataset directory. SOMA motion data is not hosted on Hugging Face.

## Play and export

Play the G1 PPO release:

```bash
uv run --project venv/mjlab projects/mimic-lite/scripts/play.py \
  task=tracking-base task/robot=g1 task/motion=g1/lafan \
  +exp=ppo/train algo/ppo/module=huge \
  task.num_envs=4 task.termination.root_pos_error.enabled=false \
  checkpoint_path=run:elijahgalahad/mimic_lite/4234dd57
```

Play a downloaded multi-robot ROA finetune student using its saved configuration:

```bash
uv run --project venv/mjlab projects/mimic-lite/scripts/play.py \
  --config-path /absolute/path/to/release --config-name training-config \
  checkpoint_path=/absolute/path/to/release/finetune.pt \
  backend=mjlab task.num_envs=16 \
  task.termination.root_pos_error.enabled=false
```

Use the matching robot's motions and base/aligned termination when judging
tracking. ROA deploy uses the finetune student. Add `export_policy=true
++export_only=true` to export ONNX and its control YAML without playback.
Export initialization can use a small robot-matched motion configuration.

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
