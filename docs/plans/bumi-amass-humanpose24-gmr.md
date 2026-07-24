# Bumi AMASS → HumanPose24 → GMR 数据与从头训练计划

状态：v1 已完成三条 fixture、30 条 pilot、AMASS 全量 1,983 条转换、production
质量分流/staging，以及 AMASS-only、AMASS+36 gait、held-out val 三个 16-env PPO
smoke；正式 4,000-iteration 训练、随机 replay 抽检与真实 Pico corruption 仍待执行

日期：2026-07-23

关联计划：`docs/plans/bumi-tracker-migration.md`

## 1. 目标与完成定义

目标是建立一条与 Pico 在线遥操作语义一致、可重复执行的离线数据链路：

~~~text
AMASS SMPL-X（源时间轴）
  → SMPL-X FK
  → XRobot HumanPose24（源时间轴）
  → GMR xrobot→Bumi（源时间轴、逐帧因果 warm-start）
  → Bumi qpos（源时间轴）
  → 唯一一次重采样到 tracker 频率 50 Hz
  → Bumi MuJoCo FK + velocity 重算
  → MimicLite tracking NPZ
  → Bumi tracker 从头训练
~~~

这里的完成不是“脚本能够跑完”，而是同时满足：

1. AMASS 的原始 FPS/时间戳在 HumanPose24 和 GMR 阶段被保留，不先转成 30 Hz。
2. 离线与在线都使用同一份 `xrobot_to_bumi` GMR 配置、同一 Bumi 模型坐标约定和同一关节顺序。
3. 最终 tracking NPZ 严格为 50 Hz，字段、body/joint 顺序和当前 36 条 Bumi 数据一致。
4. 最终 body pose/velocity 来自 50 Hz Bumi qpos 的 MuJoCo FK，不是独立插值旧 body 数组。
5. 代表性 AMASS clip 通过几何、IK、时序、足部和动力学质量 gate 后，才扩到全量转换。
6. 第 7 步的正式策略训练从随机初始化开始，不加载当前 36-motion checkpoint，也不加载 G1 checkpoint。
7. 当前 36 条 gait 数据和其已训练 checkpoint 只作为回归基线；正式训练数据可以保留这 36 条，但策略仍从头训练。

## 2. 已固定的设计决策

### 2.1 不经过离线 30 Hz 中间层

AMASS 文件中的 `mocap_frame_rate` 才是该 clip 的源采样率。当前本机抽查的
`CMU_SMPLX/78/78_10_stageii.npz` 是 120 Hz；不能假设 AMASS 都是 30 Hz，也不能用固定步长丢帧代替时间插值。

离线 v1 不复刻在线的“Pico 约 30 Hz publisher → 50 Hz `RealtimeMotionBuffer`”。在线的 30 Hz 是设备发布节奏，不是 HumanPose24 或 GMR 的数据格式约束。离线只有最终导出 tracker 数据时做一次时间归一化：

~~~text
native AMASS fps → native HumanPose24 → native GMR Bumi qpos → 50 Hz tracker qpos
~~~

因此：

- 不调用当前 `get_smplx_data_offline_fast(..., tgt_fps=30)` 的默认降采样路径。
- 新接口默认 `preserve_source_timeline=True`，或者单独提供名称明确的 native-timeline API；不能靠传一个很大的 `tgt_fps` 偶然绕过旧逻辑。
- GMR 按原始帧的时间顺序执行，上一帧解作为下一帧初值；clip 边界必须 reset。
- 若 GMR 新增平滑/速度正则，其权重必须使用真实 `dt`，不能按“每帧”定义而导致 60/120 Hz 的行为不同。
- 30 Hz 只保留在真实 Pico 在线回放测试中，不写入离线 AMASS 转换契约。

### 2.2 最终只重采样 Bumi configuration

设源 clip 有 `N` 帧，源 FPS 为 `f_src`：

~~~text
t_src[i] = i / f_src
duration = (N - 1) / f_src
t_dst[k] = k / 50, 且 t_dst[k] <= duration
N_dst = floor(duration * 50 + 1e-9) + 1
~~~

只对以下 configuration 量重采样：

- floating root position：线性插值；
- floating root quaternion：先做 hemisphere continuity，再做 SLERP；
- 21 个 Bumi hinge joint：按时间线性插值；若出现角度 wrap，先在 joint limit 允许的分支内连续化。

之后在 50 Hz 网格上重新运行 Bumi MuJoCo FK，并重新计算 `joint_vel`、`body_lin_vel_w` 和 `body_ang_vel_w`。禁止分别插值 `body_pos_w`、`body_quat_w` 和已经派生的 velocity，否则会产生互相不一致的训练 reference。

### 2.3 HumanPose24 是完整 pose，不只是 24 个点

HumanPose24 的每个 body 同时包含世界系位置和世界系四元数。canonical 顺序固定为：

~~~text
Pelvis
Left_Hip
Right_Hip
Spine1
Left_Knee
Right_Knee
Spine2
Left_Ankle
Right_Ankle
Spine3
Left_Foot
Right_Foot
Neck
Left_Collar
Right_Collar
Head
Left_Shoulder
Right_Shoulder
Left_Elbow
Right_Elbow
Left_Wrist
Right_Wrist
Left_Hand
Right_Hand
~~~

数值约定固定为：

- 坐标系：与 `XRobotStreamer.coordinate_transform_unity_data` 输出一致的右手、Z-up 世界系；
- 长度单位：米；
- quaternion：`wxyz`；
- position shape：`[T, 24, 3]`；
- quaternion shape：`[T, 24, 4]`；
- `timestamps_s` shape：`[T]`，严格单调；
- metadata 至少记录 `source_path`、`source_fps`、`gender`、`betas` hash、adapter version 和坐标系版本。

HumanPose24 clean conversion 中不加入 Pico 噪声、丢帧、延迟或滤波。reference corruption 属于训练输入增强，不属于几何转换。

### 2.4 在线部署路径不经过 SMPL

最终在线路径是：

~~~text
真实 Pico/XRobot 24-body pose
  → GMR(src_human="xrobot", tgt_robot="bumi")
  → Bumi motion publisher
  → RealtimeMotionBuffer（policy 50 Hz）
  → Bumi tracker
~~~

在线不做 `Pico → SMPL → Bumi`。SMPL-X 只在离线 AMASS 数据源一侧用于生成与 Pico 同构的 HumanPose24。这样训练 reference 和在线 reference 在 GMR 入口处对齐。

### 2.5 第 7 步全部从头训练

- clean AMASS baseline：从头训练；
- AMASS + 当前 36 gait mixture：从头训练；
- 加入 Pico reference corruption 的 robustness run：仍从头训练；
- 当前 `checkpoint_1000.pt` 只做 36-motion deterministic regression，不作为初始化。

正式大数据 run 先沿用 MimicLite 作者的 PPO 基线预算：单卡 8,192 env、4,000 iterations。四卡和单卡改变吞吐/墙钟时间，不应把 4,000 iterations 除以 GPU 数。最终 checkpoint 用 held-out/per-motion 指标选择，而不是默认取最后一轮。

## 3. 当前事实与必须修正的缺口

### 3.1 已有能力

- Bumi MimicLite task 已以 `step_dt=0.02` 运行，tracker 目标频率为 50 Hz。
- 当前 36 条 Bumi NPZ 都是 50 Hz，包含：
  `fps`、`joint_pos`、`joint_vel`、`body_pos_w`、`body_quat_w`、`body_lin_vel_w`、`body_ang_vel_w`。
- 单条现有数据的 shape 为 `[T, 21]` joint 和 `[T, 23, ...]` body。
- `prepare_bumi_motion_dataset.py` 当前会拒绝非 50 Hz 输入，这正好作为最终导出边界的强校验。
- GMR 已有 XRobot 24-body 输入和 `xrobot_to_g1.json`。
- sim2real 已直接把 Pico/XRobot 数据送进 GMR，并由 `RealtimeMotionBuffer` 在 50 Hz policy 时钟上做 position interpolation 和 quaternion SLERP。
- any4hdmi 已有经过测试的 `resampled_length`、linear interpolation 和 quaternion SLERP，可以复用其时间网格/SLERP测试思路。

### 3.2 实施前缺口

以下是计划冻结时的缺口；截至 2026-07-23，1--6 已由 GMR、sim2real 和
mimic-lite 的 v1 实现及测试关闭，第 7 项的 clean configs/smoke 已完成，真实 Pico
统计校准的 corruption 仍后置。

1. GMR 的 SMPL-X offline helper 默认 `tgt_fps=30`，不符合本计划的 native-timeline 决策。
2. `smpl_stream.compute_human_joints_np` 已能得到 24 个位置，但没有输出与 XRobot body frame 对齐的 24 个世界系 quaternion。
3. GMR registry 只有 `xrobot → unitree_g1`，没有 Bumi MJCF、root registry 和 `xrobot_to_bumi.json`。
4. sim2real publisher 虽有 `args.robot`，GMR target 仍硬编码为 `unitree_g1`；sim2real 也还没有 Bumi `RobotCfg`。
5. 还没有从 native Bumi qpos 单次采样到 50 Hz、再由 Bumi MJCF 生成标准 tracking NPZ 的 exporter。
6. 还没有 source-level train/validation split、批处理 resume、reject manifest 和数据质量报告。
7. 训练侧还没有 AMASS-only、AMASS+36 mixture 以及 Pico-noise robustness 配置。

## 4. 仓库职责与工作目录

执行时使用：

~~~bash
AA_ROOT=/data/jun7.shi/code/swe_codex/active-adaptation
MIMIC_ROOT=$AA_ROOT/projects/mimic-lite
UNILAB_ROOT=/data/jun7.shi/code/poc/github/UniLab
GMR_ROOT=$UNILAB_ROOT/thirdparty/GMR
SIM2REAL_ROOT=/data/jun7.shi/code/poc/github/EGalahad/sim2real
AMASS_ROOT=/data/jun7.shi/datasets/AMASS/AMASS
BUMI_MJCF=$AA_ROOT/.cache/aa-robot-models/bumi/bumi.xml
RETARGET_ROOT=$AA_ROOT/.cache/mimic-lite/retarget/bumi/amass
TRACKER_DATA_ROOT=$AA_ROOT/.cache/mimic-lite/motions/bumi/amass_gmr
~~~

职责固定如下：

| owner | 负责内容 | 不负责内容 |
| --- | --- | --- |
| GMR | HumanPose24 contract、SMPL-X adapter、Bumi kinematic model、`xrobot_to_bumi` IK 配置和纯 retarget API | MimicLite NPZ/训练配置 |
| sim2real | Pico wire/timestamp、Bumi `RobotCfg`、在线 publisher 选择 Bumi、online/offline parity fixture | AMASS 批处理和训练数据打包 |
| mimic-lite | AMASS batch orchestration、50 Hz Bumi exporter、质量报告、motion config、训练/eval | 重新实现 GMR IK |
| robot_retargeter | 不作为主要方案依据；最多只核对已有 Bumi constraint | 不作为本链路 runtime；不复制其 bidirectional/zero-phase 行为 |

实施前必须再次记录四个 repo 的 commit 和 dirty state。当前 `robot_retargeter` worktree 含其他任务的未提交改动，本计划不能在该 worktree 上继续叠加修改；GMR 改动应在其独立干净分支完成。

## 5. 文件级预期变更

具体名称允许在实现时按现有包结构微调，但 owner 和接口边界不能漂移。

### 5.1 GMR

| 文件 | 操作 | 目的 |
| --- | --- | --- |
| `general_motion_retargeting/human_pose24.py` | 新增 | canonical names、dataclass、schema/coordinate validation、frame dict adapter |
| `general_motion_retargeting/utils/smpl.py` | 修改 | 新增保留源时间轴的 SMPL-X FK API；旧 30 Hz API 保持兼容 |
| `general_motion_retargeting/utils/smplx_to_human_pose24.py` | 新增 | SMPL-X joint/frame mapping 和 quaternion frame correction |
| `general_motion_retargeting/params.py` | 修改 | 注册 Bumi XML、root、camera 和 `xrobot→bumi` config |
| `general_motion_retargeting/ik_configs/xrobot_to_bumi.json` | 新增 | Pico/AMASS 共用的 Bumi IK task |
| `assets/bumi/bumi_mocap.xml` | 新增 | 无大型 mesh 的 Bumi kinematic MJCF，frame/link/joint 必须与 tracker 模型一致 |
| `tests/test_human_pose24.py` | 新增 | schema、native timeline、quaternion 和坐标约定 |
| `tests/test_xrobot_to_bumi.py` | 新增 | Bumi registry、joint limits、静态 pose 和连续 frame IK |

### 5.2 sim2real

| 文件 | 操作 | 目的 |
| --- | --- | --- |
| `sim2real/config/robots/bumi.py` | 新增 | 21-joint、22-body、limits、default qpos、MJCF contract |
| `sim2real/config/robots/__init__.py` | 修改 | 注册 `bumi` |
| `sim2real/teleop/pico_retarget_pub.py` | 修改 | 用 robot config 选择 GMR target，移除 `unitree_g1` 硬编码 |
| `sim2real/teleop/smpl_stream.py` | 修改或复用 GMR adapter | 避免在线/离线各维护一份 24-body 名称和 quaternion convention |
| `tests/test_bumi_pico_retarget.py` | 新增 | prerecorded HumanPose24 的 offline/publisher parity |
| `README.md` / `README_zh.md` | 成对修改 | 在功能验证后记录 Bumi publisher 命令 |

### 5.3 mimic-lite

| 文件 | 操作 | 目的 |
| --- | --- | --- |
| `mimic_lite_conversion/bumi.py` | 新增 | qpos time resample、MuJoCo FK/velocity、tracking NPZ contract |
| `mimic_lite_conversion/amass_gmr.py` | 新增 | manifest、resume、cache、diagnostics 和 batch orchestration |
| `scripts/convert_amass_to_bumi_tracker.py` | 新增 | 薄 CLI；业务规则留在模块内 |
| `scripts/prepare_bumi_motion_dataset.py` | 小改 | 支持明确的 split/input manifest，同时继续强校验最终 50 Hz |
| `cfg/task/motion/bumi/amass_gmr.yaml` | 新增 | AMASS-only train dataset |
| `cfg/task/motion/bumi/amass_gmr_val.yaml` | 新增 | held-out eval dataset |
| `cfg/task/motion/bumi/amass_gmr_omni.yaml` | 新增 | AMASS + 当前 36 gait mixture |
| `cfg/task/reference_corruption/bumi_pico_v1.yaml` | 后置新增 | 由真实 Pico 统计校准的 actor reference corruption |
| `tests/test_bumi_amass_export.py` | 新增 | 时间轴、SLERP、FK/velocity 和 NPZ schema |
| `tests/test_bumi_amass_manifest.py` | 新增 | split、resume、reject 和 provenance |
| `README.md` | 最后修改 | 只写经过实测的转换、训练和 eval 命令 |

## 6. 七步执行计划

### 第 1 步：冻结基线、输入清单和版本

#### 1.1 操作

1. 记录 active-adaptation、mimic-lite、GMR、sim2real、robot_retargeter 的 commit、branch 和 dirty state。
2. 记录 Bumi full MJCF 的 SHA256、joint/body name/order、joint range 和 `nq/nv`。
3. 对当前 36 条 NPZ 生成只读 baseline report：帧数、总时长、joint/body shape、最大速度、quaternion norm 和 `motion_root` 约定。
4. 保存当前 36-motion checkpoint 的 deterministic eval summary，后续只做回归对照。
5. 从 AMASS 源文件生成 JSONL manifest；每条至少记录 relative path、dataset、subject、sequence、gender、source FPS、frames、duration 和 source hash。
6. 在 retarget 前按 source subject/sequence 做 train/validation split，禁止按 frame 随机切分造成泄漏。
7. 建立三个逐级 fixture 集合：
   - `fixture_3`：站立/慢走/上肢动作各一条；
   - `pilot_20_50`：覆盖不同动作、体型、时长和源 FPS；
   - `production`：通过 source schema gate 的完整清单。

建议输出：

~~~text
$RETARGET_ROOT/
├── manifests/
│   ├── fixture_3.jsonl
│   ├── pilot_20_50.jsonl
│   ├── train.jsonl
│   └── val.jsonl
└── reports/
    ├── source_inventory.json
    └── current_36_baseline.json
~~~

#### 1.2 Gate 1

- 每个 source clip 的 FPS 都来自文件 metadata，值为有限正数；不提供“默认 30”。
- `timestamps_s[-1]` 与 `(frames - 1) / source_fps` 一致。
- train/val 不共享 subject/sequence。
- Bumi full MJCF 和 GMR kinematic MJCF 的 21 joint、22 body 名称及静态 FK 对齐。
- 当前 36 条数据的 `motion_root` 已确认并锁定为 index 0 的常量：position=0、quaternion=identity、linear/angular velocity=0；真实 reset root 仍是 index 1 `base_link`。

### 第 2 步：实现 native-timeline SMPL-X → HumanPose24

#### 2.1 SMPL-X 输入

逐 clip 读取：

- `mocap_frame_rate`；
- `trans [T,3]`；
- `root_orient [T,3]`；
- `pose_body [T,63]`；
- `betas`、`gender`；
- 必要时读取 `pose_hand`，但 v1 的 XRobot 24-body contract 不依赖手指关节。

缺少必需字段、shape 不一致、FPS 非法或出现 NaN 时，写 reject reason，不用默认值静默修复。

#### 2.2 FK 和 body mapping

1. SMPL-X model 在原始 `T` 帧上一次性做 FK，不丢帧。
2. 位置 mapping 使用现有 sim2real 选择作为起点：SMPL-X joint `0..21` 加左右 hand endpoint `39/54`。
3. 对 `0..21` 按 SMPL-X parent tree 累乘 local rotation，得到世界系 quaternion。
4. `Left_Hand/Right_Hand` 的 orientation 从实际 SMPL-X parent chain 推导；不能只复制 wrist quaternion 而不记录假设。
5. 为每个 HumanPose24 body 定义显式 rest-frame correction；SMPL-X joint frame 与 XRobot body frame 不能仅靠同名假设为一致。
6. correction 表通过 neutral T-pose 和真实 Pico round-trip fixture 标定，并作为版本化常量写入代码。
7. 输出 `HumanPose24Sequence`，保留源 `T` 和 `timestamps_s`。

#### 2.3 测试

- 120 Hz、60 Hz 和非整数时长的 synthetic/source fixture 输出帧数与输入完全相同。
- quaternion norm error `< 1e-5`，相邻帧先做 hemisphere continuity。
- neutral pose 左右 limb length 对称，脚在 pelvis 下方，head/hand 方向无 90°/180° 系统偏转。
- 将一段真实 Pico 24 pose 经现有 `XRobot→SMPL` 再经新 adapter 回到 HumanPose24；对 GMR 实际使用的 body 比较 position/orientation，而不是要求不可观测手指完全可逆。
- adapter 相同输入重复运行时输出 deterministic。

#### 2.4 Gate 2

- 没有任何隐式 30 Hz 分支。
- HumanPose24 names、shape、coordinate、unit 和 quaternion order 全部通过 schema validator。
- T-pose、行走和上肢三个 fixture 可视化方向正确。
- round-trip 的 pelvis/feet/elbows/wrists 无系统性 frame offset；若存在，先修 correction 表，不进入 GMR tuning。

### 第 3 步：实现并调通 GMR `xrobot → Bumi`

#### 3.1 Bumi registry 和模型

1. 在 GMR 注册 `bumi` 的 kinematic MJCF、`ROBOT_BASE_DICT` 和 viewer 参数。
2. kinematic MJCF 不需要大型 render mesh，但 joint axis、range、body frame、link offset 和 21-joint qpos 顺序必须与 tracker 使用的 full MJCF 一致。
3. 初始化 qpos 使用 Bumi nominal standing pose。
4. 每帧 IK 强制 joint limits；超过数值 epsilon 的 violation 记为失败，不能导出后再大幅 clip。

#### 3.2 IK task 初版

| HumanPose24 target | Bumi target | 约束意图 |
| --- | --- | --- |
| Pelvis | `base_link` | floating root position + orientation |
| Spine3/Neck | `waist_yaw_link`/upper-body frame | 保持 torso 朝向；Bumi 只有 waist yaw，不追不可实现的 3-DoF waist |
| Hip/Knee/Ankle/Foot | 对应 leg links/foot sites | 腿方向、膝轨迹和足端位置优先；足端 orientation 次之 |
| Shoulder/Elbow | 对应 arm pitch/roll/yaw 和 elbow links | 保持肩肘方向与左右对称 |
| Wrist/Hand | elbow 后的 arm endpoint/proxy site | 用 position/direction 约束前臂；Bumi 无 wrist DoF，不强追 wrist orientation |

权重调节顺序固定为 pelvis/root → feet/legs → torso → elbows/arm endpoints。不要先靠高 smoothness 掩盖 frame/mapping 错误。

#### 3.3 时间行为

- clip 第 0 帧从 nominal standing qpos 初始化；
- 后续帧使用上一帧 qpos warm-start；
- 按 `timestamps_s` 顺序逐帧执行；
- clip 之间 reset，不能把前一条动作的末帧带到下一条；
- 不使用 robot_retargeter 的 backward pass、bidirectional initialization 或 zero-phase low-pass，因为在线 Pico 不能因果复现它们；
- 每帧保存 solver iterations、task residual、limit margin、success/failure 和 elapsed time。

#### 3.4 Gate 3

- 3 个 fixture 全帧无 NaN、无 solver exception、无 joint limit violation。
- neutral pose 左右对称且脚底高度合理。
- pelvis/feet 等高优先级 position target 的 p95 error 首轮要求 `< 0.10 m`，足端单独要求 `< 0.06 m`；若人体/机器人比例导致阈值不合理，必须先用 scale-normalized report 给出证据再修改阈值。
- 高优先级 orientation p95 error 首轮要求 `< 15°`；Bumi 不可实现的 wrist/waist roll-pitch 不纳入硬 gate，但仍输出报告。
- 连续帧 qpos 无单帧大跳；任一 joint 相邻帧跳变超过其物理可行速度对应增量时标红并停止扩批。
- GMR kinematic model qpos 放入 full Bumi MJCF 后，22 个对应 body 的 FK 在 tolerance 内一致。

### 第 4 步：建立 native Bumi qpos 批处理和在线/离线同源验证

#### 4.1 离线 batch runner

实现 `convert_amass_to_bumi_tracker.py` 的前半段：

1. 按 JSONL manifest 读取 AMASS；
2. 生成或从 content-addressed cache 读取 HumanPose24；
3. 调用同一个 GMR `xrobot→bumi` API 逐帧得到 native qpos；
4. 输出可选的中间文件：
   - `timestamps_s [T]`；
   - `root_pos [T,3]`；
   - `root_quat_wxyz [T,4]`；
   - `joint_pos [T,21]`；
   - solver diagnostics；
   - source/GMR/model/config hashes；
5. 支持 `--resume`，只有所有输入 hash 和版本都相同才复用 cache；
6. 每个 clip 原子写入，失败写 `rejected.jsonl`，不能留下被误认为成功的半文件。

这一阶段的 qpos 仍处于 AMASS 源时间轴；不能写 `fps=30`，也不能伪装成最终 tracker NPZ。

#### 4.2 sim2real 在线 Bumi 路径

1. 新增 Bumi `RobotCfg`，joint/body 顺序与训练完全一致。
2. 把 publisher 的 `tgt_robot="unitree_g1"` 改为由 `args.robot`/RobotCfg 映射；`--robot bumi` 必须实例化同一 GMR Bumi target。
3. 保留在线默认 `publish_hz=30` 和 policy `rl_rate=50`；这是 runtime 调度，不回灌到离线 AMASS pipeline。
4. 用录制的 HumanPose24 序列替代 Pico 硬件输入，分别走纯 GMR API 和 publisher worker，比较逐帧 qpos。

#### 4.3 Gate 4

- native qpos 帧数和 timestamps 与 HumanPose24 完全一致。
- 相同 HumanPose24、初始 qpos、GMR config 和 model 下，offline API 与 sim2real publisher 的 qpos 最大绝对差 `< 1e-5`。
- `--robot g1` 原路径 regression 通过；不能为了 Bumi 改坏 G1 publisher。
- `--robot bumi` 输出恰好 21 个 joint，并能由 Bumi full MJCF 完成 FK。

### 第 5 步：唯一一次采样到 50 Hz，并导出标准 tracking NPZ

#### 5.1 qpos resampling

1. 用第 2.2 节定义的时间网格生成 50 Hz timestamps。
2. root position、21 hinge joints 线性插值。
3. root quaternion 归一化、hemisphere continuity 后 SLERP。
4. 输入低于 50 Hz 时允许上采样；高于 50 Hz 时允许下采样；两者都使用真实时间，不用整数 `frame_skip`。
5. 首末帧时间不得超出源区间；导出时记录源/目标 duration 和误差。
6. 不静默大幅 clamp joint。只允许修正 `<=1e-6` 的浮点越界；更大的越界应拒绝 clip 并回到 GMR 修复。

#### 5.2 FK 和 velocity

在 exact full Bumi MJCF 上对每个 50 Hz qpos：

1. 用 `mujoco.mj_forward` 得到 22 个真实 body 的 `xpos/xquat`。
2. 用 `mujoco.mj_differentiatePos` 从相邻 qpos 计算 generalized velocity；内部帧用中心时间窗，端点用单边差分。
3. 将 hinge DoF velocity 按名字抽取为 `joint_vel [T,21]`。
4. 设置 qpos/qvel 后运行 velocity propagation，并用 MuJoCo object velocity 得到世界系 `body_lin_vel_w`、`body_ang_vel_w`。
5. 所有 quaternion 输出为 `wxyz` 且归一化。
6. 在 body 数组 index 0 插入 legacy `motion_root` 常量；index 1..22 才是 MuJoCo 真实 body。

#### 5.3 最终 NPZ contract

~~~text
fps                 [1]          = 50
joint_pos           [T, 21]      float32
joint_vel           [T, 21]      float32
body_pos_w          [T, 23, 3]   float32
body_quat_w         [T, 23, 4]   float32, wxyz
body_lin_vel_w      [T, 23, 3]   float32
body_ang_vel_w      [T, 23, 3]   float32
~~~

joint/body names 不塞进每个 NPZ，继续由 `meta.json` 提供；同时在 dataset-level manifest 中保存 names 和版本，避免单文件 provenance 丢失。

#### 5.4 Gate 5

- `prepare_bumi_motion_dataset.validate_motion` 通过。
- `fps` 必须恰好是 50；源 FPS 仅保存在 provenance，不覆盖最终 `fps`。
- 输出帧数符合 floor 时间网格公式；因目标点不得超出源区间，末尾截断
  duration drift 必须满足绝对值 `< 1 / 50 s`。不能用越界外推或不等间隔末帧
  人为满足半帧阈值。
- quaternion norm error `< 1e-5`，没有 NaN/Inf。
- 将保存的 50 Hz joint/root pose 重做 FK，与保存 body pose 的最大误差接近数值精度；position `< 1e-5 m`，orientation `< 1e-4 rad`。
- 从保存 pose 重算 velocity 与保存 velocity 一致；不出现由 double resampling 产生的相位差。
- `motion_root` 全程为零/identity，`base_link` 才承载真实 root motion。

### 第 6 步：分级扩批、质量筛选和训练配置

#### 6.1 分级执行

严格按以下顺序，不直接从 3 条跳到全量：

1. `fixture_3`：逐帧可视化 HumanPose24、GMR Bumi 和最终 50 Hz Bumi。
2. `pilot_20_50`：生成汇总分布，定位特定动作/体型/FPS 的系统失败。
3. `production train`：只转换 train manifest。
4. `production val`：独立转换 held-out manifest，不参与训练。

每一级必须保存：成功/失败数、总时长、源 FPS 分布、solver residual、joint limit margin、joint/body velocity、foot height/sliding、输出 hash 和 reject reason 分布。

#### 6.2 质量 gate

硬拒绝：

- schema/FPS/timestamp 非法；
- NaN/Inf 或 quaternion 非单位；
- GMR solver failure；
- joint limit 越界；
- FK contract 不一致；
- root/limb 出现单帧 teleport；
- feet 长时间明显穿地；
- 50 Hz velocity 超过 Bumi 可接受上限且不是可解释的短暂动态动作。

报告将这组结构/物理契约汇总为 `pipeline_integrity_ready`。因果 GMR 输出和最终
50 Hz 数据都按 joint family 对照电机限速，不能只看一个全局 rad/s 数字。

软标记但先不删除：

- foot sliding；
- 高手端误差；
- 动作超出 Bumi DoF 导致的 residual；
- 短时高速或低 crouch。

软标记数据先进入单独 view，经过物理 replay/eval 再决定是否训练，避免仅凭人工阈值删掉有价值的动态动作。v1 自动训练集是完整性通过、几何 gate 通过，并且同时满足保守 staging 门槛（joint velocity `<20 rad/s`、body linear velocity `<10 m/s`、body angular velocity `<30 rad/s`）的交集。这里 20 rad/s 是训练数据的保守门槛，不替代手臂电机 50 rad/s 等真实 joint-family 上限：介于两者之间的数据进入 `dynamics_review`，不是硬损坏。

#### 6.2.1 已执行的 30 条 pilot（2026-07-23）

- 输入：30 clips、73,156 native frames，含 24 条 120 Hz 和 6 条 60 Hz；
- 输出：30 clips、32,334 tracker frames，conversion reject 为 0；
- `pipeline_integrity_ready=true`，native/final 最大 joint velocity ratio 分别约
  `1.0000000000000022`/`1.0`，quaternion norm 最大误差 `<5e-8`，`motion_root`
  contract error 为 0；
- 自动训练集：14 clips、14,501 frames；
- `geometry_review`：6 clips；`dynamics_review`：10 clips；三组互斥且覆盖 30 条；
- 14 条自动训练集已通过 `prepare_bumi_motion_dataset.py`，并完成 16 env、1 PPO
  update、随机初始化 smoke，无 NaN。

14/30 不是保留率目标，只是这个覆盖性 pilot 在当前保守 gate 下的实测结果。扩到
production 后继续保存 review 集；不能为提高保留率而事后放松阈值。

#### 6.2.2 已执行的 production 全量转换（2026-07-23）

- manifest：train 1,787 clips，val 196 clips；按 subject 切分且没有交集；
- train：3,188,842 native frames → 1,513,690 个 50 Hz tracker frames；
- val：262,893 native frames → 116,844 个 50 Hz tracker frames；
- 总计：1,983/1,983 clips、3,451,735 native frames → 1,630,534 tracker
  frames，conversion reject、missing clip、stale metadata 都为 0；
- 两份报告均为 `pipeline_integrity_ready=true`；最大 native/final joint velocity
  ratio 分别约 `1.0000000000000027`/`1.0`，最大 quaternion norm error
  `<5.1e-8`，`motion_root` contract error 为 0；
- 逐关节位置按 Bumi MJCF range 检查，最大超限量为约 `1.05e-7 rad`，属于
  float32 边界误差；不再使用错误的全关节 `abs(q)<3` 假设；
- train 分流：1,029 automatic、436 geometry review、322 dynamics review；
- val 分流：121 automatic、38 geometry review、37 dynamics review；
- 默认 staging 为 train 1,029 clips / 774,163 frames、val 121 clips /
  80,738 frames；其余数据及 HumanPose24/native qpos/final NPZ 全部保留；
- AMASS-only、AMASS 0.8 + 36 gait 0.2 和 held-out val 均已用 production
  staging 完成 16 env、1 PPO update、随机初始化 smoke，无 NaN。
- `scripts/view_bumi_retarget_viser.py` 可惰性加载全量目录，在 Viser 中切换
  clip、连续播放/逐帧查看，并按 automatic/geometry/dynamics 质量分组筛选；MuJoCo
  mesh 与 NPZ 内保存的 body FK overlay 同时显示，用于后续随机 replay 目检。

这里“默认 staging 只含 automatic”不等于只转换了这些数据。1,983 条均已转换并可
追溯；review 集只是不会在未经 replay/人工复核前自动进入正式训练。

#### 6.3 数据布局和配置

~~~text
.cache/mimic-lite/motions/bumi/
├── omni/                         # 当前 36 条，不改
├── amass_gmr_train/              # 新 train split
└── amass_gmr_val/                # held-out split
~~~

新增配置：

- `bumi/amass_gmr`：AMASS train only，`full_motion=false`；
- `bumi/amass_gmr_val`：held-out eval；
- `bumi/amass_gmr_omni`：正式候选 mixture，初始 sampling weight 为 AMASS 0.8、现有 36 gait 0.2。

0.8/0.2 是首个受控对照，不直接宣称最优。必须比较 AMASS-only 与 mixture 的 held-out AMASS、36-gait 和 Pico replay 指标后再改权重。

#### 6.4 Pico reference corruption（clean baseline 之后）

clean pipeline 和 clean from-scratch baseline 通过后，再实现 actor-visible reference corruption：

- 低频 root drift/bias；
- temporally correlated joint/body jitter；
- frame hold/drop/duplicate；
- timestamp/latency jitter；
- reconnect 时的小概率 discontinuity。

幅度从真实 Pico recording 与同动作平滑 reference 的残差统计中标定，不凭经验写最终数值。训练 reward/reset 始终使用 clean latent reference，只有 actor 看到 corrupted command；若 actor 同时看到 body FK，corruption 后的 joint/root 必须重做 FK，不能制造内部不一致 observation。

这项增强不能替代在线 causal buffer/filter。它也不写回离线 clean NPZ。

#### 6.5 Gate 6

- `fixture_3` 和 `pilot_20_50` 的 `pipeline_integrity_ready=true`，且自动训练子集非空后才允许 production batch；soft review clip 不进入默认 staging。
- production report 可从 source relative path 追溯到 HumanPose24、GMR config/model hash 和最终 NPZ hash。
- train/val source 无泄漏。
- AMASS-only、mixture 和 val 配置都能 resolve 并完成 16-env、1-update smoke。
- 随机抽取至少 20 条最终 NPZ 做 MuJoCo replay，无 body explosion、明显坐标翻转或系统性穿地。

### 第 7 步：Bumi tracker 从头训练与选择 checkpoint

#### 7.1 训练矩阵

所有 run 都使用 `checkpoint_path=null`：

| run | 数据 | reference corruption | 目的 |
| --- | --- | --- | --- |
| A | AMASS-only | off | 验证新数据本身能学 |
| B | AMASS 0.8 + 36 gait 0.2 | off | 检查通用动作和稳定步态兼容性 |
| C | 与 B 相同 | Pico calibrated | 最终遥操作 robustness 候选 |

当前 36-motion checkpoint 只参与 eval 对照，不出现在 A/B/C 的初始化路径中。

#### 7.2 单卡正式命令

在 Gate 6 通过后，run B 的预期命令为：

~~~bash
cd "$AA_ROOT"

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv --project venv/mjlab run python projects/mimic-lite/scripts/train.py \
  task=tracking-base-bumi \
  task/motion=bumi/amass_gmr_omni \
  +exp=ppo/train backend=mjlab \
  task.num_envs=8192 total_iters=4000 \
  checkpoint_path=null \
  checkpoint_interval=500 upload_interval=500 \
  wandb.mode=online
~~~

run A 只替换 motion config。run C 再显式选择经过验证的 Bumi Pico corruption config。正式命令不使用 `~task.randomization`；Bumi actuator 0–4 physics-step delay 和部署所需物理随机化应保留。仅 smoke/映射定位时临时关闭其他 randomization。

#### 7.3 阶段 gate

1. **1 update smoke**：16 env，确认 dataset、observation、action、reward 和 optimizer 无 NaN。
2. **短 pilot**：检查 reward、episode length、root/joint/body error 确实改善；不因短 pilot 结果提前宣布收敛。
3. **正式 4,000 iterations**：单卡 8,192 env；按 500 iteration 保存，保持作者默认 3,250→3,500 entropy decay，除非曲线给出调整证据。
4. **deterministic eval**：分别评估 train sample、held-out AMASS、当前 36 gait 和 prerecorded Pico/GMR reference。
5. **checkpoint selection**：以 held-out/per-motion 指标选最优，不默认 `checkpoint_latest.pt`。

#### 7.4 Gate 7 / 完成标准

- A/B/C 都能从随机初始化稳定训练；至少 A 和 B 完整完成，C 在获得真实 Pico 统计后完成。
- held-out manifest 的 motion coverage 为 100%，没有只评到少数容易动作。
- B 的 36-gait deterministic mean progress 不低于当前 36-motion baseline 5 个百分点以上的退化容忍线；超出则调整 mixture，而不是加载旧 checkpoint 修补。
- B 的 held-out AMASS mean progress 首轮目标 `>=0.70`，train mean progress `>=0.80`；同时检查 per-motion 分布，不能用少数长 clip 掩盖失败动作。
- C 相比 B 在真实 Pico replay corruption 下显著减少 termination/error，同时 clean held-out 指标退化不超过 5 个百分点。
- 最终候选通过 1-env 可视化、512-env deterministic eval 和 sim2real prerecorded Pico end-to-end replay。
- README 只在命令实际跑通后更新；记录 data manifest hash、GMR commit/config hash、Bumi MJCF hash、训练 commit 和 checkpoint。

## 7. 预期执行命令

以下是实施后应成立的 CLI contract；脚本落地前不能把它们写成“已可运行”。

### 7.1 三条 fixture 转换

~~~bash
uv --project "$SIM2REAL_ROOT/venv/teleop" run python \
  "$MIMIC_ROOT/scripts/convert_amass_to_bumi_tracker.py" \
  --manifest "$RETARGET_ROOT/manifests/fixture_3.jsonl" \
  --smplx-model-dir /path/to/smplx/models \
  --gmr-root "$GMR_ROOT" \
  --bumi-mjcf "$BUMI_MJCF" \
  --target-fps 50 \
  --output "$RETARGET_ROOT/fixture_3" \
  --resume
~~~

### 7.2 production train/val 转换

~~~bash
PYTHONPATH="$MIMIC_ROOT" \
uv --project "$AA_ROOT/venv/mjlab" run \
  --with smplx --with mink --with loop-rate-limiters \
  --with 'qpsolvers[daqp]' \
  python \
  "$MIMIC_ROOT/scripts/convert_amass_to_bumi_tracker.py" \
  --manifest "$RETARGET_ROOT/manifests/train.jsonl" \
  --smplx-model-dir /path/to/smplx/models \
  --gmr-root "$GMR_ROOT" \
  --bumi-mjcf "$BUMI_MJCF" \
  --actual-human-height 1.6 \
  --target-fps 50 \
  --output "$RETARGET_ROOT/train" \
  --resume \
  --workers 16 \
  --torch-threads-per-worker 2 \
  --fail-on-reject

PYTHONPATH="$MIMIC_ROOT" \
uv --project "$AA_ROOT/venv/mjlab" run \
  --with smplx --with mink --with loop-rate-limiters \
  --with 'qpsolvers[daqp]' \
  python \
  "$MIMIC_ROOT/scripts/convert_amass_to_bumi_tracker.py" \
  --manifest "$RETARGET_ROOT/manifests/val.jsonl" \
  --smplx-model-dir /path/to/smplx/models \
  --gmr-root "$GMR_ROOT" \
  --bumi-mjcf "$BUMI_MJCF" \
  --actual-human-height 1.6 \
  --target-fps 50 \
  --output "$RETARGET_ROOT/val" \
  --resume \
  --workers 2 \
  --torch-threads-per-worker 2 \
  --fail-on-reject
~~~

已用 fixture 的单进程/两进程 resume 验证 determinism 和 cache。production 并行实现使用
spawn process，每个 worker 独占 GMR/model，clip 边界 reset，不能共享 warm-start
state。上面的 16/2 workers 是本次实际使用值；资源较小的机器可以降低，但不改变输出。

### 7.3 staging 和 smoke

~~~bash
uv --project "$AA_ROOT/venv/mjlab" run python \
  "$MIMIC_ROOT/scripts/prepare_bumi_motion_dataset.py" \
  --source "$RETARGET_ROOT/train/tracker_50hz" \
  --quality-report "$RETARGET_ROOT/train/reports/quality_summary.json" \
  --output "$AA_ROOT/.cache/mimic-lite/motions/bumi/amass_gmr_train" \
  --report-json "$RETARGET_ROOT/train/reports/staging_summary.json" \
  --link-mode symlink --force

cd "$AA_ROOT"
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv --project venv/mjlab run python projects/mimic-lite/scripts/train.py \
  task=tracking-base-bumi task/motion=bumi/amass_gmr \
  +exp=ppo/train backend=mjlab \
  task.num_envs=16 total_iters=1 checkpoint_path=null \
  wandb.mode=disabled '~task.randomization'
~~~

## 8. 测试矩阵

| 层 | 测试 | 失败时回到 |
| --- | --- | --- |
| AMASS loader | metadata、shape、native timestamps、不同 FPS | 第 1/2 步 |
| HumanPose24 | names、frame correction、quat norm、T-pose、round-trip | 第 2 步 |
| GMR Bumi | registry、limits、static pose、连续 warm-start、residual | 第 3 步 |
| online parity | pure GMR 与 sim2real publisher 相同输入相同 qpos | 第 3/4 步 |
| resampler | length/duration、linear、SLERP、hemisphere、低/高 FPS | 第 5 步 |
| exporter | MuJoCo FK/velocity、motion_root、NPZ schema | 第 5 步 |
| batch | resume、hash invalidation、atomic write、reject manifest | 第 4/6 步 |
| dataset | 50 Hz validator、name reorder、single/multi config | 第 5/6 步 |
| training | 16-env smoke、per-motion coverage、held-out eval | 第 7 步 |

最小测试命令预期为：

~~~bash
uv --project "$SIM2REAL_ROOT/venv/teleop" run python -m unittest discover \
  -s "$GMR_ROOT/tests" -p 'test_*.py'

uv --project "$SIM2REAL_ROOT/venv/teleop" run python -m unittest discover \
  -s "$SIM2REAL_ROOT/tests" -p 'test_*.py'

cd "$AA_ROOT"
uv --project venv/mjlab run python -m unittest discover \
  -s projects/mimic-lite/tests -p 'test_bumi_*.py'
~~~

GMR 必须在该 teleop 环境中以锁定 revision/editable path 安装；不依赖系统 Python 的偶然 package 状态。

## 9. 停止条件

出现任一情况，停止扩批或正式训练，修复最早失败的 owner layer：

- 发现任意 AMASS clip 被隐式当成 30 Hz；
- HumanPose24 quaternion frame correction 尚未由 T-pose/round-trip 验证；
- offline GMR 与 sim2real publisher 对相同输入输出不同；
- 同一 clip 出现两次时间重采样或 body 数组与 qpos 分别插值；
- Bumi kinematic/full MJCF 的 joint/body FK 不一致；
- solver 通过大幅 joint clip 才能满足 limits；
- source-level train/val 泄漏；
- fixture/pilot 尚未通过就启动全量转换；
- 训练 run 非随机初始化；
- 用加大网络、训练时长或噪声掩盖 mapping/FK/时序错误。

## 10. 提交和产物规则

提交顺序：

1. GMR：HumanPose24 native adapter；
2. GMR：Bumi registry、MJCF、IK config 和 tests；
3. sim2real：Bumi RobotCfg、publisher target 和 parity tests；
4. mimic-lite：native batch + 50 Hz exporter + tests；
5. mimic-lite：dataset configs、reference corruption 和文档；
6. 分别更新包含这些嵌套仓库的父仓引用。

每个 commit 都记录对应测试。以下生成物不提交：

- AMASS 原始数据；
- SMPL-X model weights；
- HumanPose24 cache；
- native Bumi qpos；
- 50 Hz tracking NPZ；
- reports 中包含大规模逐帧数据的文件；
- checkpoints、wandb 和 rollout。

必须提交的是代码、small synthetic/prerecorded fixture（确认许可和体积后）、manifest schema、配置、测试和不含原始动作内容的汇总报告模板。

## 11. ExtremControl 只读评估与 v2 边界

本地只读参考仓库为
`/data/jun7.shi/code/retarget/Genesis-Humanoid`。该 worktree 已有其他任务的未提交
改动，本工作不修改、不提交其中任何文件。

### 11.1 “50 ms”的正确含义

ExtremControl 所称约 50 ms 是实机遥操作的端到端响应延迟，不是一个
“HumanPose24 → robot qpos 的 50 ms retargeter”。其方案有三个耦合部分：

1. 不在在线闭环中求 full-body joint-space retarget；而是把人体参考快速映射成
   机器人 pelvis/torso、双足和双手等选定 link 的 SE(3) 目标。
2. tracker/policy 直接观察当前机器人 link 与这些 extremity targets 的误差，输出
   joint target；因此它改变 policy observation、reward、teacher/student 训练和部署
   contract，不是替换一个 GMR backend 就能获得。
3. 低层 PD 额外使用由连续 joint target 计算出的 target velocity feedforward；
   论文和实现都把这部分视为降低整体响应时间的关键。

项目页给出的量级是 Cartesian mapping 约 0.3 ms、定制 joint-space IK 约 10 ms，
而约 50 ms 来自完整硬件链路测量。这个数字不能直接外推到 Bumi/Pico/MimicLite。

### 11.2 v1 决策

当前 v1 不改为 ExtremControl：

- 离线仍是 `AMASS → HumanPose24 → GMR → Bumi qpos → 50 Hz tracker NPZ`；
- 在线仍是 `Pico HumanPose24 → 同一 GMR → Bumi qpos reference → MimicLite`；
- 不复制 `robot_retargeter` 的双向初始化或 zero-phase filter；
- GMR 的逐 task position/orientation residual 已加入 provenance 和质量报告，先完成
  pilot/production 数据闭环。

原因是当前 MimicLite actor 的参考是完整 Bumi joint/body motion。仅把在线 GMR 换成
Cartesian mapping 会使部署输入 contract 与训练 reference 不同，反而扩大 gap。

### 11.3 v2 建议接口

若 v1 的真实 Pico 端到端延迟或噪声测试不合格，再开独立 v2：

~~~text
HumanPose24
  → BumiCartesianMapper（纯代数、无逐帧 IK）
  → BumiExtremityCommand
      pelvis/torso pose
      left/right foot pose
      left/right arm-endpoint pose（Bumi 无 wrist DoF，需定义 elbow 后 proxy）
      timestamp + validity/confidence
  → Extremity-conditioned Bumi policy
  → target joint position + target joint velocity
  → velocity-feedforward actuator/deployment controller
~~~

`BumiExtremityCommand` 应成为离线 AMASS 和在线 Pico 共用的可替换 backend contract；
AMASS 直接从同一个 HumanPose24 adapter 生成 Cartesian command，不再先 GMR 成 qpos。
GMR qpos 可保留为 teacher/critic privileged target 或对照，但不能继续作为在线 actor 的
必需输入。

### 11.4 v2 启动 gate

只有以下测量完成后才值得迁移：

1. 在 Pico timestamp、publisher、GMR、buffer、policy、PD 和实机观测各边界打点，得到
   Bumi 的分段和端到端 p50/p95 latency；
2. 用 prerecorded Pico 同动作比较 GMR v1 与 Cartesian mapping 的 reference jitter、
   足端/手端误差和 drop/hold 行为；
3. 证明 GMR 计算或 qpos reference 确实是主要瓶颈，而不是 Pico 发布、网络、policy、
   电机 0--4 step delay 或 Bumi 机械响应；
4. 在仿真中实现 position-only 与 velocity-feedforward 的受控对照，并确认 Bumi 电机接口
   能安全接收 target velocity；
5. v2 从头训练新的 extremity-conditioned policy，不能直接把现有 MimicLite checkpoint
   当成兼容策略。

因此，ExtremControl 是合理的第二版体系候选，但不是当前 GMR 数据转换 v1 的局部优化项。

## 12. Pipeline v4：G1 地面链路对照与 Bumi 输入高度标定

状态（2026-07-24）：代码完成，10 条 CMU subject 35 平地步行 pilot 完成；全量
train/val 尚未按 v4 重转。

### 12.1 G1 实际实现

代码核对后的 G1 链路是：

~~~text
Pico/XRobot 24 点
  → publisher 输入高度 bootstrap
  → GMR xrobot_to_g1
      Left/Right_Foot → left/right_toe_link
      ground_height = 0
  → G1 qpos/FK
~~~

需要明确区分三件事：

1. `xrobot_to_g1.json` 的 `ground_height=0` 只是 IK target 的固定坐标偏移，不检查
   MuJoCo collision，也不限制脚底穿地。
2. GMR 内没有逐帧 floor clamp。live publisher 以往在 GMR 前把 24 点整体加一个
   固定 Z offset；legacy 配置默认目标为 1 cm，并在第 30 个输入帧冻结当帧 offset。
3. 同一批平地走路若绕过 publisher 输入标定，G1 实体脚底中位数约为
   -1.86 cm。因此 1 cm 是历史 HumanPose24 输入参数，不是 G1 脚底应离地 1 cm，
   也不能直接复制为所有机器人通用值。

### 12.2 为什么废弃 pipeline v3

Pipeline v3 在 GMR 后查整段动作所有 foot collision geom 的最低点，再给全部 qpos
加一次固定 root-Z。它虽然不是逐帧 snapping，但仍有两个问题：

- heel-strike、toe-off 或摆动脚大俯仰的一个低点会抬高整段动作，导致真正的支撑脚
  在双支撑时悬空；
- 最早用于 A/B 的十条 CMU subject 01 动作实际是爬梯子。没有梯子几何时，其脚本来
  就不应被强制压到世界 `Z=0`，所以这组数据不能标定平地原点。

因此 v4 的转换器不再调用 `align_bumi_qpos_to_ground`。旧函数和测试仅为读取/复现
pipeline-v3 产物保留，不能进入新批量转换路径。

### 12.3 v4 owner-layer contract

正式链路改为：

~~~text
raw HumanPose24（原样缓存）
  → 固定输入高度 bootstrap
      Bumi: 首 1.0 s，最低 24 点的 p10 → 4.0 cm
  → GMR toe-aware Bumi foot-site IK
  → 原生时间轴 Bumi qpos（不做 root-Z 后处理）
  → 唯一一次 50 Hz resampling
  → tracker NPZ
~~~

参数 owner 是 GMR 的 `xrobot_to_bumi.json`，而不是离线脚本或 Pico publisher：

~~~json
"input_height_bootstrap": {
  "target_min_body_height_m": 0.04,
  "bootstrap_duration_s": 1.0,
  "reference_percentile": 10.0
}
~~~

- Offline 使用 `HumanPose24Sequence.timestamps_s` 截取首 1 秒，所以 60/100/120 Hz
  AMASS 不会被误当成 30 Hz。
- Online 使用 Pico 纳秒 timestamp；无源 timestamp 时才使用接收时间。启动窗口内
  offset 因果更新，1 秒后冻结，不缓存 1 秒动作来增加遥操延迟。
- G1 配置没有该字段时走兼容默认：1 cm、30 帧、最后一帧值；本次没有静默改变 G1。
- Offline 为整条 clip 应用启动窗口最终得到的常量；online 的首 1 秒会逐步收敛，
  这是明确记录的启动期差异。冻结后两侧使用相同常量算法和同一配置。

### 12.4 Bumi 足端 frame 与动作完成度

HumanPose24 的 `Left_Foot/Right_Foot` 来自 SMPL toe joint。旧配置约束
`ankle_roll_link` 并用 sole-center offset，脚发生俯仰时 position residual 不直接对应
实体脚掌。v4 改为：

- IK frame：MuJoCo `l_foot/r_foot` site；
- Human target：`Left_Foot/Right_Foot`；
- site target 的局部 offset：`[-0.075, 0, 0.001]`，即从 Bumi toe contact point
  回到 sole-center site；
- position/orientation cost 保持 100/8，保留 heel/toe 动作完成度；
- reach/root scale 仍按 actuated ankle chain 计算（1.8 m 假设下 pelvis 约 0.595），
  rigid sole 几何只由 offset 表达。

曾实验把 rigid sole 也累计进 reach scale，pelvis scale 上升到约 0.649，静态
base position residual 由约 8.3 cm 恶化到约 12.2 cm。虽然 foot-site residual
更小，但躯干被迫为不可兼得的足端长度让步，因此按动作完成度回退到 ankle-chain
scale。

### 12.5 4 cm 的选择依据

4 cm 仍是原始 HumanPose24 最低点目标，不是 Bumi 脚底 clearance。
`actual_human_height=1.6` 时，输入 Z 变化到 Bumi root-Z 的比例约为 0.529。

使用 CMU subject 35 的十条平地 walk 做完整
`HumanPose24 → GMR → Bumi collision FK` 扫描。3.5 cm 与 4.0 cm 均实际重转，
不是在旧 qpos 上伪造最终产物：

| 输入目标 | ground gate | 最坏 contact-foot p05 | 最坏双支撑低脚中位数 | 最高双支撑高脚中位数 |
| --- | ---: | ---: | ---: | ---: |
| 3.5 cm | 9/10 | -12.9 mm | -6.9 mm | +7.9 mm |
| 4.0 cm | 10/10 | -10.3 mm | -4.2 mm | +10.5 mm |

4 cm 是扫描档位中满足逐 clip gate 的最小值。它只比 3.5 cm 抬高最终 Bumi
约 2.6 mm；双支撑高脚仍约 1 cm，未回到 pipeline-v3 约 3 cm 的明显悬空。

### 12.6 接触感知 audit，不参与动作修改

源接触在原生时间轴上由以下只读规则估计：

~~~text
foot_z <= per-clip foot_z p05 + 0.03 m
abs(foot_vertical_velocity) <= 0.20 m/s
~~~

Ground gate 检查 contact-foot p05/median 与双支撑低/高脚中位数。所有 collision
geom 的最低点、0 以下比例和 5 mm 以下比例仍写入 metadata/quality report，但摆动脚
的俯仰极值不再抬高 root，也不单独决定训练 gate。梯子、台阶等非平地动作在缺少对应
场景几何时应进入 review，而不是被后处理伪装成平地动作。

### 12.7 Pilot 结果与产物

最终 pilot：

- 10/10 converted，0 reject；
- 4,120 个 120 Hz native frames → 1,718 个 50 Hz tracker frames；
- pipeline integrity 通过，ground/foot-position/root-position gate 均 10/10；
- 9 条 automatic training-ready；
- `35_08` 仅因 orientation gate 进入 geometry review，与地面标定无关；
- contact-foot median 的逐 clip 范围为 3.6--7.4 mm。

产物目录：

~~~text
.cache/mimic-lite/retarget/bumi/amass/flat_walk_v4_10/
  human_pose24/
  native_qpos/
  tracker_50hz/
  metadata/
  reports/
~~~

查看命令：

~~~bash
cd /data/jun7.shi/code/poc/github/EGalahad/active-adaptation
PYTHONPATH=projects/mimic-lite \
uv --project venv/mjlab run --with mjviser==0.0.14 \
  python projects/mimic-lite/scripts/view_bumi_retarget_viser.py \
  --motion-dir \
  .cache/mimic-lite/retarget/bumi/amass/flat_walk_v4_10/tracker_50hz
~~~

全量 1,983 条数据的既有数字仍属于 pipeline v2。下一次正式训练前必须按 v4
重新转换 train/val、重新生成 quality report 和 staging；不能把旧 corpus 标记成
已经获得 online/offline 高度 parity。
