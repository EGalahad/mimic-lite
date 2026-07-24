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

#### AMASS → HumanPose24 → GMR → Bumi

The AMASS conversion path uses the same XRobot/Pico `HumanPose24` contract and
`xrobot_to_bumi` GMR configuration as the online publisher. It preserves each
clip's native 60 or 120 Hz timeline through SMPL-X FK and causal GMR, then
resamples Bumi qpos exactly once to the tracker's 50 Hz grid. Body poses and
velocities are recomputed with the exact Bumi MJCF after resampling. GMR uses
the real source `dt` to enforce the same causal Bumi joint/root velocity limits
as the Pico publisher; the emitted state becomes the next IK warm start.

Set the local paths first:

```bash
export AMASS_ROOT=/data/jun7.shi/datasets/AMASS/AMASS
export GMR_ROOT=/data/jun7.shi/code/poc/github/UniLab/thirdparty/GMR
export SMPLX_MODEL_DIR=/data/jun7.shi/code/poc/hr/HoloMotion/thirdparties/smpl_models/models
export BUMI_MJCF="$PWD/.cache/aa-robot-models/bumi/bumi.xml"
export RETARGET_ROOT="$PWD/.cache/mimic-lite/retarget/bumi/amass"
```

Build SHA256-addressed manifests with a deterministic subject-level train/val
split. The checked local CMU inventory contains 1,983 clips: 1,787 train and
196 held out, with no rejected source files.

```bash
PYTHONPATH=projects/mimic-lite \
uv --project venv/mjlab run python \
  projects/mimic-lite/scripts/build_amass_manifest.py \
  --amass-root "$AMASS_ROOT" \
  --output "$RETARGET_ROOT/manifests" \
  --fixture-count 3 --pilot-count 30
```

Convert the three-clip fixture. `--actual-human-height=1.6` deliberately
matches the default Bumi Pico publisher setting; change both sides together if
the calibrated operator height differs.

```bash
PYTHONPATH=projects/mimic-lite \
uv --project venv/mjlab run \
  --with smplx --with mink --with loop-rate-limiters \
  --with 'qpsolvers[daqp]' \
  python projects/mimic-lite/scripts/convert_amass_to_bumi_tracker.py \
  --manifest "$RETARGET_ROOT/manifests/fixture_3.jsonl" \
  --smplx-model-dir "$SMPLX_MODEL_DIR" \
  --gmr-root "$GMR_ROOT" \
  --bumi-mjcf "$BUMI_MJCF" \
  --actual-human-height 1.6 --target-fps 50 \
  --output "$RETARGET_ROOT/fixture_3" \
  --workers 1 --resume --fail-on-reject
```

The tested fixture has 3,813 native frames and 2,029 final frames. All three
clips pass the root/foot position and orientation gates, both native and final
joint-velocity ratios stay at or below 1.0, and a second run is served entirely
from the provenance-checked cache.

Stage those files and run the tested AMASS-only smoke:

```bash
PYTHONPATH=projects/mimic-lite \
uv --project venv/mjlab run python \
  projects/mimic-lite/scripts/prepare_bumi_motion_dataset.py \
  --source "$RETARGET_ROOT/fixture_3/tracker_50hz" \
  --output .cache/mimic-lite/motions/bumi/amass_gmr_train \
  --link-mode symlink --force

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv --project venv/mjlab run python projects/mimic-lite/scripts/train.py \
  task=tracking-base-bumi task/motion=bumi/amass_gmr \
  +exp=ppo/train backend=mjlab task.num_envs=16 total_iters=1 \
  checkpoint_path=null wandb.mode=disabled '~task.randomization'
```

Run the pilot by changing the manifest/output above to `pilot_20_50.jsonl` and
`$RETARGET_ROOT/pilot_20_50`, then build its quality split:

```bash
PYTHONPATH=projects/mimic-lite \
uv --project venv/mjlab run python \
  projects/mimic-lite/scripts/report_bumi_retarget_quality.py \
  --conversion-root "$RETARGET_ROOT/pilot_20_50" \
  --manifest "$RETARGET_ROOT/manifests/pilot_20_50.jsonl"

PYTHONPATH=projects/mimic-lite \
uv --project venv/mjlab run python \
  projects/mimic-lite/scripts/prepare_bumi_motion_dataset.py \
  --source "$RETARGET_ROOT/pilot_20_50/tracker_50hz" \
  --quality-report "$RETARGET_ROOT/pilot_20_50/reports/quality_summary.json" \
  --output .cache/mimic-lite/motions/bumi/amass_gmr_pilot_v2 \
  --link-mode symlink
```

The tested 30-clip pilot converted 73,156 native frames into 32,334 tracker
frames with zero conversion rejects. `pipeline_integrity_ready=true`: native
and final velocity ratios are at most 1.0, quaternion error is below `5e-8`,
and the `motion_root` contract is exact. The conservative automatic split has
14 clips/14,501 frames; 6 clips require geometry review and 10 require dynamics
review. A 16-env, one-update PPO smoke on the 14 automatic clips completes
without NaNs.

The full train and held-out conversion has also completed. Each worker owns an
independent GMR/model instance, so causal warm-start state never crosses clip
boundaries. `--resume` validates provenance before reusing an output:

```bash
PYTHONPATH=projects/mimic-lite \
uv --project venv/mjlab run \
  --with smplx --with mink --with loop-rate-limiters \
  --with 'qpsolvers[daqp]' \
  python projects/mimic-lite/scripts/convert_amass_to_bumi_tracker.py \
  --manifest "$RETARGET_ROOT/manifests/train.jsonl" \
  --smplx-model-dir "$SMPLX_MODEL_DIR" \
  --gmr-root "$GMR_ROOT" --bumi-mjcf "$BUMI_MJCF" \
  --actual-human-height 1.6 --target-fps 50 \
  --output "$RETARGET_ROOT/train" \
  --workers 16 --torch-threads-per-worker 2 \
  --resume --fail-on-reject

PYTHONPATH=projects/mimic-lite \
uv --project venv/mjlab run \
  --with smplx --with mink --with loop-rate-limiters \
  --with 'qpsolvers[daqp]' \
  python projects/mimic-lite/scripts/convert_amass_to_bumi_tracker.py \
  --manifest "$RETARGET_ROOT/manifests/val.jsonl" \
  --smplx-model-dir "$SMPLX_MODEL_DIR" \
  --gmr-root "$GMR_ROOT" --bumi-mjcf "$BUMI_MJCF" \
  --actual-human-height 1.6 --target-fps 50 \
  --output "$RETARGET_ROOT/val" \
  --workers 2 --torch-threads-per-worker 2 \
  --resume --fail-on-reject

for split in train val; do
  PYTHONPATH=projects/mimic-lite \
  uv --project venv/mjlab run python \
    projects/mimic-lite/scripts/report_bumi_retarget_quality.py \
    --conversion-root "$RETARGET_ROOT/$split" \
    --manifest "$RETARGET_ROOT/manifests/$split.jsonl"

  PYTHONPATH=projects/mimic-lite \
  uv --project venv/mjlab run python \
    projects/mimic-lite/scripts/prepare_bumi_motion_dataset.py \
    --source "$RETARGET_ROOT/$split/tracker_50hz" \
    --quality-report "$RETARGET_ROOT/$split/reports/quality_summary.json" \
    --output ".cache/mimic-lite/motions/bumi/amass_gmr_$split" \
    --report-json "$RETARGET_ROOT/$split/reports/staging_summary.json" \
    --link-mode symlink --force
done
```

All 1,983 clips were converted, not only the clips admitted by the conservative
training gate:

- train: 1,787/1,787 clips, 3,188,842 native frames → 1,513,690 tracker frames;
- val: 196/196 clips, 262,893 native frames → 116,844 tracker frames;
- total: 1,983/1,983 clips and 1,630,534 final 50 Hz frames, with zero
  conversion rejects, missing clips, stale metadata, or train/val overlap.

##### Ground-origin parity (pipeline v4)

The counts above are the completed pipeline-v2 corpus. Pipeline v3 added one
post-GMR root-Z translation based on the lowest collision point in the entire
clip. It prevented penetration but made ordinary double-support poses float:
a pitched swing foot or heel/toe transition could lift every frame. Its first
ten-clip visual set was also CMU subject 01 ladder climbing, not a valid
flat-ground calibration set. Pipeline v3 is retained only as compatibility
code and is not called by the batch converter.

The G1 implementation does not contain a floor/collision clamp in GMR either.
`xrobot_to_g1.json` maps HumanPose24 feet to the two toe links and declares
`ground_height=0`; the live Pico publisher applies a fixed input-height offset
before GMR. Legacy G1 configurations inherit the existing 1 cm target and
freeze the last of 30 startup frames. On the same flat-walk source, bypassing
that publisher calibration leaves the physical G1 sole median about 1.86 cm
below Z=0. Thus 1 cm is a legacy HumanPose24 input target, not a universal
robot-foot clearance.

Pipeline v4 makes the offline and live Bumi paths share the robot-specific
`input_height_bootstrap` section generated in `xrobot_to_bumi.json`:

1. Use HumanPose24 timestamps, not a presumed source FPS. Over the first
   1.0 seconds, take the 10th percentile of the lowest of the 24 body points.
2. Apply one fixed input-space Z offset that makes that reference 4.0 cm, then
   run GMR. The raw HumanPose24 NPZ remains unchanged for provenance/Viser.
3. Map the SMPL `Left/Right_Foot` toe joints to the Bumi `l_foot/r_foot` MuJoCo
   sites with the fixed site-to-toe target offset `[-0.075, 0, 0.001]`.
   Reach scaling still ends at the actuated ankle chain, avoiding the 12 cm
   base residual produced by counting the rigid sole twice.
4. Do not translate, clamp, or foot-snap the resulting Bumi qpos. Resampling to
   50 Hz is still the only temporal resampling operation.

The 4 cm value is also an input-space target, not Bumi sole clearance. With
`actual_human_height=1.6`, its vertical response is scaled by about 0.529.
It is the smallest tested target for which every one of ten CMU subject 35
flat-walk clips passes the per-clip contact gate. Source contacts are inferred
on the native timeline (`foot_z <= clip p05 + 3 cm` and
`|vertical velocity| <= 0.20 m/s`); pitched swing-foot extrema remain in the
audit but cannot lift the stance foot.

The final pipeline-v4 pilot converted 10/10 clips (4,120 native frames to 1,718
tracker frames) with no reject or integrity failure. All ten pass the ground,
foot-position, and root-position gates; nine are automatically training-ready,
while `35_08` remains geometry review only because of its orientation gate.
Across clips, contact-foot median height is 3.6--7.4 mm, the worst contact-foot
p05 is -10.3 mm, and double-support low/high-foot median bounds are
-4.2/+10.5 mm. These are read-only FK/contact measurements; no output
postprocessing is hidden behind them.

Pipeline v4 invalidates pipeline-v2/v3 `--resume` entries. The 1,983-clip counts
and staging counts in this README remain historical pipeline-v2 results until
train/val are reconverted and re-reported with v4. The checked flat-walk pilot
is available for visual review:

```bash
PYTHONPATH=projects/mimic-lite \
uv --project venv/mjlab run --with mjviser==0.0.14 \
  python projects/mimic-lite/scripts/view_bumi_retarget_viser.py \
  --motion-dir \
  .cache/mimic-lite/retarget/bumi/amass/flat_walk_v4_10/tracker_50hz
```

The raw HumanPose24, native GMR qpos, final tracker NPZ, metadata, and review
sets remain under `$RETARGET_ROOT/{train,val}`. The default staging deliberately
contains only `automatic_training_ready_clip_ids`: train has 1,029 clips /
774,163 frames and val has 121 clips / 80,738 frames. The remaining 436
geometry-review + 322 dynamics-review train clips and 38 + 37 val clips are
retained for replay/review rather than silently discarded or automatically
trained.

AMASS-only, held-out val, and the AMASS + original 36-gait mixture have each
completed a 16-env, one-update GPU smoke on these production files. The
candidate mixture samples AMASS at 0.8 and the original 36 gait clips at 0.2.

##### Browse the retargeted motions with Viser

The batch viewer lazily loads only the selected clip and renders three
synchronized views: the Bumi MuJoCo mesh, the original SMPL-X FK source motion
as its 24 selected joints (purple/green), and an optional final Bumi FK debug
overlay (blue/orange). The 24 source positions are saved before GMR/IK and are
sampled for display by timestamp, so no 30 Hz assumption or frame-index
matching is introduced. By default the viewer opens the full production train
catalog and infers both its quality report and the sibling `human_pose24`
directory:

```bash
PYTHONPATH=projects/mimic-lite \
uv --project venv/mjlab run --with mjviser==0.0.14 \
  python projects/mimic-lite/scripts/view_bumi_retarget_viser.py
```

Open `http://localhost:8090`. The GUI provides clip selection, previous/next,
play/pause, frame stepping, speed, body-overlay visibility, and an end behavior
that defaults to automatically advancing through the catalog. `Playback` is the
default tab. `Follow root XY` recenters the source SMPL-X pelvis and Bumi
`base_link` independently in the horizontal plane for visualization only. It
does not remove either root Z, so the world ground remains at Z=0 and vertical
motion is preserved. Disable it (or start with `--no-recenter-root-xy`) to view
the original world trajectories. The source skeleton starts at Y=-1.0 m and
the blue/orange Bumi FK debug skeleton at Y=+0.6 m; their offsets can be changed
in the GUI. Set the Bumi `Overlay Y offset` to 0 to check exact mesh/FK
alignment. To inspect a review split or held-out motions:

```bash
# Only clips held for geometry review.
PYTHONPATH=projects/mimic-lite \
uv --project venv/mjlab run --with mjviser==0.0.14 \
  python projects/mimic-lite/scripts/view_bumi_retarget_viser.py \
  --quality-group geometry_review --start-paused

# Held-out val catalog; its quality report is also inferred automatically.
PYTHONPATH=projects/mimic-lite \
uv --project venv/mjlab run --with mjviser==0.0.14 \
  python projects/mimic-lite/scripts/view_bumi_retarget_viser.py \
  --motion-dir "$RETARGET_ROOT/val/tracker_50hz"
```

Use `--dry-run` to validate every selected file without starting a server.
The server binds to `127.0.0.1` by default; use SSH port forwarding, or pass
`--host 0.0.0.0` only when remote network exposure is intended.

The formal single-GPU mixture run starts from random weights and keeps the Bumi
actuator delay/randomization enabled:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv --project venv/mjlab run python projects/mimic-lite/scripts/train.py \
  task=tracking-base-bumi task/motion=bumi/amass_gmr_omni \
  +exp=ppo/train backend=mjlab \
  task.num_envs=8192 total_iters=4000 checkpoint_path=null \
  checkpoint_interval=500 upload_interval=500 wandb.mode=online
```

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
