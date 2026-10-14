# Supported robots and checkpoints

The three G1 releases remain in the [README checkpoint table](../README.md#released-checkpoints).
Additional robot profiles support both `backend=mjlab` and `backend=isaaclab`.
The checkpoints below were trained with mjlab; backend support does not imply
that a checkpoint can be deployed unchanged across simulators.

| Robot | `task/robot` | Motion profile / release data | Released training + deploy bundle |
| --- | --- | --- | --- |
| H2 | `h2` | [H2 seed-100](https://huggingface.co/datasets/elijahgalahad/any4hdmi-h2-seed-100) | [Seed ROA student](https://drive.google.com/open?id=12Bjw3hZSp7XHWqh85tzbd6LbK0ebl4IE), [mixture ROA student](https://drive.google.com/open?id=1eHO5DXvlHnfT_6QD0Q2NI1cK0xv6NGhA) |
| A3 | `a3` | LAFAN / 100style / seed mixture (local data) | [A3 ROA student](https://drive.google.com/open?id=10Lthw5p7hrLtIBB47xXPm3vHToAH2-1F) |
| RP1, 28 DoF | `rp1` | [RP1 LAFAN](https://huggingface.co/datasets/elijahgalahad/any4hdmi-rp1-lafan) | [RP1 ROA student](https://drive.google.com/open?id=10nzo4O9ZFJTLf0283RKns_Xmm4YxhjL1) |
| RP1, 24 DoF | `rp1_24dof` | [RP1 24 DoF LAFAN](https://huggingface.co/datasets/elijahgalahad/any4hdmi-rp1-lafan-24dof) | [RP1 24 DoF ROA student](https://drive.google.com/open?id=1QSMEzhPz_YJ29LB0LP6VC3V9lR73sDwP) |

A3 and RP1 bundles use a `512 × 3` actor and 4,000 / 1,000 / 2,000
Train / Adapt / Finetune updates. A3 uses 8 × 4,096 environments; RP1 uses
8 × 8,192. H2 bundles are existing
`1024 × 3` student releases; their original training dataset revisions were not
recorded. H2's current default seed-100 motions are for current training, not a
claim to reproduce those historical motion mixtures.

Each bundle includes `finetune.pt`, `policy.onnx`, `policy.yaml` and
`training-config.yaml`; its manifest identifies the source and model/data
revisions where available. `.pt` files support policy training continuation but
do not restore optimizer state for an exact interrupted-run restart. ONNX and
YAML are deploy artifacts and cannot resume training.

The A3 profile uses a pinned model with the official A3 T3.0 hand visual and
inertia. Match each checkpoint to its bundled robot and motion configuration.
The A3 release is the LAFAN / 100style / seed mixture student (weights
0.3 / 0.3 / 0.4), trained on 36,530 filtered motions (112.802 hours), seed 42.
Its checkpoint SHA256 is `a62a07d7b148edb9ba06b9196c891d75d948720124585ba754fe3d7b8809f2a7`.
CPU ORT export validation passed with maximum absolute error 2.62e-6. Native
full-motion diagnostics completed 67/72 motions without base failure, with
root-position termination disabled. Long motions exhibit global root drift;
integrated ONNX benchmark and sim2real deployment remain unverified.

The default A3 training profile still selects FullCover (146 short strike
motions); use the released bundle's training configuration and robot-matched
local mixture data to reproduce this checkpoint. No motion data is included
in the checkpoint bundle. The optional SOMA profile is separate and does not
have a released checkpoint.
