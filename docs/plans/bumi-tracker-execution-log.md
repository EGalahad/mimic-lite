# Bumi Tracker 执行记录

日期：2026-07-22
计划：`docs/plans/bumi-tracker-migration.md`

## 基线

- Active Adaptation commit：`faea67554b318798f5e8b1ca4f4d0f3d28df12e6`
- Active Adaptation 当时记录的 MimicLite gitlink：`089d0ad07619e47a6ce61393dcd69037d6aa32c9`
- MimicLite commit：`d8100588fd2e7cd4837545b0d6e0a67bab07c54c`
- Bumi reference commit：`6f7c3ab871b4846f0d50e870289aaf92875a74c3`
- MimicLite 初始状态：除本任务新增的 `docs/` 外无其他改动。
- 嵌套仓在任务开始前已比父仓 gitlink 前进 6 个既有提交，且 `d810058` 是当时的 `origin/main`；父仓最终更新 gitlink 时会同时包含这段已有上游历史和本次 Bumi commit。
- Bumi reference：只读；其原有未跟踪文件不纳入本任务。
- 首次基线测试误用了不存在的 venv 路径，落到系统 Python 后因缺少 `torch` 失败；这不是代码基线失败。随后按 README 创建 `venv/mjlab`，后续结果以该环境为准。

## 数据基线

- 源目录：`src/assets/motions/bumi/amp`
- motion 数：36
- 总帧数：7,535
- fps：50
- joint：21
- motion body：23（第 0 项为 metadata 中的 `motion_root`，第 1 项为 `base_link`）

## 固定实现决策

- MJLab 版本：1.3.0。
- actuator position-command delay：训练首版即启用 0 到 4 physics steps。
- delay 配置位置：`BuiltinPositionActuatorCfg.delay_min_lag/max_lag`。
- `JointPosition`：`min_delay=0`、`max_delay=0`、`alpha=1.0`，避免双重延迟。
- 首版 delay 粒度：MJLab 内建的 per-env 公共 lag；不宣称是逐电机独立 lag。
- 首版训练数据：上述全部 36 个 NPZ；single motion 仅用于映射/overfit gate。

## Gate 结果

| Gate | 状态 | 结果 |
| --- | --- | --- |
| 0 基线 | pass | 三个仓库 commit/status 已记录；本地 MJLab 1.3 环境已建立 |
| 1 asset cache | pass | 22 meshes、21 hinge joints、1 free joint、22 robot bodies；重复执行返回 `status=unchanged` |
| 2 asset adapter | pass | `bumi` registry、MJCF compile、actuator 参数与 0..4 delay、sensor、joint/body 顺序测试通过 |
| 3 simulation capacity | pass | Bumi resolve 为 nconmax=128、njmax=640、maxmatch=256、CCD=50；G1 不含 override |
| 4 motion cache | pass | 36 motions、7,535 frames、50 Hz；重复执行返回 `status=unchanged`；single/omni 均可加载且 invalid frame 为 0 |
| 5 task mapping | pass | runtime 解析为 14 tracking bodies、21 joints、21 actions；静态检查无 G1/Atom 名称残留 |
| 6 静态/配置 | pass with baseline typing debt | 20 个 unittest 全部通过；新增 Bumi、数据、eval 文件 pyright 为 0 error；见下方说明 |
| 7 PPO smoke | pass | single 和 36-motion omni 均完成 16-env rollout、PPO update 和 checkpoint 保存 |
| 8 single overfit | pending | 尚未运行 256-env 长时 overfit，因此不宣称达到 0.95 progress |
| 9 36-motion quality | pending | 已通过初始化/训练 smoke；尚未运行长训和全覆盖 deterministic quality eval |
| 10 DR | pending | 按计划在 nominal quality gate 之后逐项恢复 |
| 11 G1 runtime regression | blocked by local data | Hydra resolve 已通过；本机缺少 `../any4hdmi/output/g1/lafan`，未伪造 runtime 结果 |

## 最终验证命令与结果

单元测试：

~~~bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv --project venv/mjlab run \
  python -m unittest discover -s projects/mimic-lite/tests -v
~~~

结果：20 tests，0 failures。

新增/实质扩展文件的静态类型检查：

~~~bash
HF_HUB_OFFLINE=1 uv --project venv/mjlab run --with pyright pyright \
  projects/mimic-lite/mimic_lite/assets/bumi.py \
  projects/mimic-lite/scripts/prepare_bumi_assets.py \
  projects/mimic-lite/scripts/prepare_bumi_motion_dataset.py \
  projects/mimic-lite/tests/test_bumi_asset.py \
  projects/mimic-lite/tests/test_bumi_motion_dataset.py \
  projects/mimic-lite/tests/test_eval_motion_metrics.py \
  projects/mimic-lite/scripts/eval.py
~~~

结果：0 errors、0 warnings。若把两个已有动态适配文件 `active_adaptation/envs/backends/mjlab/env.py` 和 `mimic_lite/tasks/actions.py` 一并交给 pyright，仍会报告 10 个原有错误：MuJoCo `MjSpec/MjsGeom` stub、adapter abstract typing、可选 viewer、未安装 IsaacLab 的 fallback import，以及既有 Action/EntityCfg 动态属性。报错均位于本任务未修改的语句；本次新增的 simulation override 和 `expected_action_dim` 语句没有新报错。

最终 36-motion smoke：

~~~bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv --project venv/mjlab run python projects/mimic-lite/scripts/train.py \
  task=tracking-base-bumi task/motion=bumi/omni \
  +exp=ppo/train backend=mjlab task.num_envs=16 total_iters=1 \
  wandb.mode=disabled checkpoint_interval=100 upload_interval=100 \
  '~task.randomization'
~~~

结果：36 motions/7,535 frames 全部加载，14 bodies/21 joints 被保留，打印的 21 维 action 顺序与 MJCF 顺序完全一致；一次 rollout 和 PPO update 正常完成。临时 checkpoint 为 `/tmp/wandb/run--040trcrf/files/checkpoint_1.pt`，不提交。

最终 eval plumbing check：

~~~bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv --project venv/mjlab run python projects/mimic-lite/scripts/eval.py \
  task=tracking-base-bumi task/motion=bumi/omni backend=mjlab \
  task.num_envs=64 eval_steps=5 \
  checkpoint_path=/tmp/wandb/run--040trcrf/files/checkpoint_1.pt \
  eval_output=/tmp/bumi_eval_final.pt \
  eval_summary_output=/tmp/bumi_eval_final.json \
  +store_rollout=false '~task.randomization'
~~~

结果：TensorDict 和 JSON 均成功写出，JSON 包含 36 个总 motion、30 个本次随机采样到的 motion、coverage=0.8333，以及 30 组 path/name/count/progress 聚合。由于只运行 5 steps，`lafan_progress=1.0` 仅说明评估链路可执行，不能用于 Gate 8/9 的质量判断。

## 数据数值统计

- max |joint_pos|：1.3076304 rad
- max |joint_vel|：17.424585 rad/s
- max body z：0.9442099 m
- max body linear velocity norm：2.4701092 m/s
- max body angular velocity norm：18.16782 rad/s
- max quaternion norm error：1.1920929e-7

## API 差异记录

- MJLab 1.3 已移除参考仓使用的 `DelayedActuatorCfg`，delay 已迁移到 builtin actuator 的内联字段。
- MJLab 1.3 会把 delay 配置相同的 builtin position actuators 融合到一个 delay buffer；其 lag shape 是 `(num_envs,)`，因此当前 21 个关节在同一 env 内共用一次采样，不是逐电机独立采样。
- 默认 `delay_update_period=0`，表示每个 physics step 在 0..4 中重新采样。获得实机时延时间序列后，应再决定是否改为 episode 固定或低频更新。
- MJLab 1.3 builtin position actuator 没有 `velocity_limit` 参数；参考仓的构造也未传该字段。首版保留 nominal velocity limit 数据，但不宣称 simulator 已执行该限制。
