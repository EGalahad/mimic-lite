# Bumi Tracker 迁移实施计划

状态：MVP implemented；长时 overfit、36-motion quality gate 与完整 G1 runtime regression 待执行

负责人：Codex

日期：2026-07-22

目标后端：MJLab
目标任务：用 MimicLite PPO 训练 Bumi 21-DoF motion tracker

## 1. 目标与完成定义

本计划把 MimicLite 当前的 G1 tracker 链路迁移到 Bumi。迁移后的最小闭环必须做到：

1. Bumi MJCF、mesh、actuator、collision 和 contact sensor 能被 Active Adaptation 的 MJLab backend 正确加载。
2. 策略动作空间固定为 21 维，并采用 Bumi 现有部署链路使用的 MJCF/policy 关节顺序。
3. Bumi 现有 36 条、50 Hz tracking NPZ 能被 any4hdmi/MimicLite 无损加载。
4. 一条 motion 可以 overfit；全部 36 条 motion 可以启动并持续训练。
5. G1 原有 task 的配置和训练入口不回归。
6. 大型 XML/STL 和 motion 数据只写入缓存目录，不提交到代码仓库。

代码迁移完成的最低验收条件：

- 所有新增单元测试通过。
- G1 与 Bumi Hydra 配置均能完整 resolve。
- Bumi 16-env、1-iteration PPO smoke test 通过。
- 单 motion deterministic eval 的平均 motion progress 不低于 0.95。
- 全部 36 条 motion 的 deterministic eval 平均 progress 不低于 0.80。
- G1 16-env、1-iteration regression smoke test 通过。

完整训练质量不以“跑到固定迭代数”为判断依据。只有前一阶段的指标支持继续投入时，才扩大环境数、加入随机化或启动长训练。

## 2. 范围与非目标

### 2.1 本次范围

- MJLab-only Bumi asset adapter。
- Bumi tracking task 与 motion 配置。
- Bumi asset/motion 的本地缓存准备工具。
- MJLab simulation capacity 参数化。
- 资产、数据、配置、训练和 eval 验证。
- 当前 36 条 Bumi SONIC omni gait motion 的训练闭环。

### 2.2 本次非目标

- 不移植 AMP discriminator、conditional AMP、velocity command labels 或 AMP runner。
- 不实现 IsaacLab backend；当前 Bumi 参考仓库没有可用 USD/URDF。
- 不直接复用 G1 checkpoint。
- 不在首个闭环中追求 Bumi 参考仓库的完整 sim-to-real DR parity。
- 不在首个闭环中实现真机 tracker 部署。真机部署还需要 future-reference 输入和与训练一致的 observation history。
- 不把 G1 LAFAN、AMASS、100Style 等数据直接用于 Bumi。要获得通用 tracker 能力，需另行 retarget。

## 3. 固定设计决策

以下决策在执行中视为默认值；只有出现证据证明其不可行时才调整。

1. **资产存放**：Bumi XML/STL 放在 active-adaptation/.cache/aa-robot-models/bumi，不提交大型资产。
2. **资产注册**：在 mimic_lite/assets/bumi.py 注册名字 bumi，并使用 backend callable 返回 MJLab EntityCfg 与 sensors。
3. **MJLab-only**：请求 isaaclab backend 时明确抛出 NotImplementedError，不伪造 usd_path。
4. **policy 关节顺序**：采用 MJCF 顺序，即 waist、左臂、右臂、左腿、右腿。
5. **motion 关节顺序**：保留源 NPZ 的 BUMI_21DOF_NAMES 顺序，由 RobotTracking 按名字重排。
6. **根与 anchor**：root_body_name=base_link，anchor_body_name=waist_yaw_link。
7. **motion_root**：仅存在于 motion metadata；它不是机器人 body，也不能被用作 reset root。
8. **动作延迟**：从首版训练开始，由 MJLab 1.3 的 `BuiltinPositionActuatorCfg` 内联提供 0 到 4 个 physics step 的 position-command delay；MimicLite `JointPosition` 不再叠加 delay 或 smoothing。按当前 `physics_dt=0.005`，这对应 0 到 20 ms。
9. **动作数据格式**：MVP 使用 any4hdmi legacy wrapper；标准 qpos manifest 转换放到后续优化。
10. **训练初始化**：Bumi 从头训练；checkpoint_path 保持 null。
11. **奖励参数**：首轮沿用 MimicLite tracking reward 的 sigma/weight，先验证映射，再依据指标调参。
12. **delay 随机化语义**：MJLab 1.3 默认是每个 env 一个 lag，整组 command 共用；相同 delay 配置的 builtin actuators 可能共享 fused delay buffer。因此它模拟的是每台机器人/公共控制链路的随机延迟，不是 21 个电机各自独立的随机延迟。当前 Bumi 首版按公共链路延迟建模；若实机日志证实逐电机差异，再实现形状为 `(num_envs, 21)` 的 per-joint delay buffer。
13. **delay 时间相关性**：首版保持 MJLab 1.3 默认采样行为，以复现现有 0 到 4 step 假设；在获得实机时延日志后，再决定改为 episode 内固定或低频更新，避免凭经验制造不真实的高频 jitter。
14. **足部 airtime**：首轮关闭，因为 Bumi 没有 G1/Atom 的 toe body。
15. **随机化**：单 motion 和初始 multi-motion 阶段关闭；名义 tracking 通过后按家族逐项恢复。actuator delay 属于已确认的机器人链路模型，不随其他 DR 一起关闭。

## 4. 仓库与路径

执行时使用以下路径别名：

~~~bash
AA_ROOT=/data/jun7.shi/code/swe_codex/active-adaptation
MIMIC_ROOT=/data/jun7.shi/code/swe_codex/active-adaptation/projects/mimic-lite
BUMI_REF=/data/jun7.shi/code/poc/github/AMP_mjlab/.worktrees/bumi_sym_rough_quiet_dr
BUMI_ASSET_CACHE=/data/jun7.shi/code/swe_codex/active-adaptation/.cache/aa-robot-models/bumi
BUMI_MOTION_CACHE=/data/jun7.shi/code/swe_codex/active-adaptation/.cache/mimic-lite/motions/bumi/omni
~~~

这是两个代码仓库的联合改动：

- active-adaptation 主仓：MJLab simulation capacity 参数化。
- projects/mimic-lite 嵌套仓：Bumi asset、配置、数据准备工具、测试和文档。

参考 Bumi worktree 只读，不在其中修复或整理已有未跟踪文件。

## 5. 预期文件变更

| 仓库 | 文件 | 操作 | 目的 |
| --- | --- | --- | --- |
| mimic-lite | mimic_lite/assets/bumi.py | 新增 | Bumi constants、spec、actuator、sensor、registry |
| mimic-lite | mimic_lite/assets/__init__.py | 修改 | 导入 bumi 触发注册 |
| mimic-lite | scripts/prepare_bumi_assets.py | 新增 | 把 XML/STL 放入缓存并验证 |
| mimic-lite | scripts/prepare_bumi_motion_dataset.py | 新增 | 生成 legacy any4hdmi 数据布局 |
| mimic-lite | cfg/task/tracking-base-bumi.yaml | 新增 | Bumi tracker task |
| mimic-lite | cfg/task/motion/bumi/omni.yaml | 新增 | 全部 36 条 gait motion |
| mimic-lite | cfg/task/motion/bumi/single.yaml | 新增 | 单 motion overfit |
| mimic-lite | tests/test_bumi_asset.py | 新增 | 名称、映射、模型编译测试 |
| mimic-lite | tests/test_bumi_motion_dataset.py | 新增 | 数据转换与 metadata 测试 |
| mimic-lite | scripts/eval.py | 修改 | 输出初始 motion id、覆盖率和 per-motion progress |
| mimic-lite | tests/test_eval_motion_metrics.py | 新增 | 验证 coverage 与 per-motion 聚合 |
| mimic-lite | README.md | 最后修改 | 只在闭环通过后记录使用命令 |
| active-adaptation | active_adaptation/envs/backends/mjlab/env.py | 修改 | 从 task.sim.mjlab 读取 capacity |

不提交以下生成物：

- .cache/aa-robot-models/bumi 下的 XML/STL。
- .cache/mimic-lite/motions/bumi/omni 下的 NPZ、meta.json 和软链接。
- outputs、checkpoint、eval_rollout.pt、eval_summary.json。

## 6. 工作包 0：基线与执行记录

### 6.1 操作

1. 读取 active-adaptation/AGENTS.md。
2. 分别检查主仓、mimic-lite 和 Bumi reference 的 git 状态及 commit。
3. 确认 venv/mjlab 存在并已安装当前项目。
4. 运行现有 MimicLite 单元测试，保存基线输出。
5. 创建 docs/plans/bumi-tracker-execution-log.md，记录每个 gate 的命令、结果、checkpoint 和决定。

~~~bash
git -C "$AA_ROOT" status --short
git -C "$MIMIC_ROOT" status --short
git -C "$BUMI_REF" status --short
git -C "$AA_ROOT" rev-parse HEAD
git -C "$MIMIC_ROOT" rev-parse HEAD
git -C "$BUMI_REF" rev-parse HEAD

uv --project "$AA_ROOT/venv/mjlab" run \
  python -m unittest discover -s "$MIMIC_ROOT/tests"
~~~

### 6.2 Gate 0

- 已区分用户原有改动与本任务改动。
- 本任务即将修改的文件不存在未归属冲突。
- 现有测试结果已记录；若基线本身失败，先区分环境问题和代码问题，不把基线失败误归因于 Bumi。
- 若 Bumi reference commit 与本计划分析时不同，重新核对 actuator、joint order 和 motion inventory 后再继续。

## 7. 工作包 1：准备缓存资产

### 7.1 新增 prepare_bumi_assets.py

脚本接口：

~~~text
python scripts/prepare_bumi_assets.py \
  --source /path/to/bumi/xmls \
  --output /path/to/.cache/aa-robot-models/bumi
~~~

脚本职责：

1. 要求 source 下存在 bumi.xml 和 assets 目录。
2. 检查 XML compiler 的 meshdir 为 assets。
3. 解析 XML 中全部 mesh file 引用，逐个检查源 STL 存在。
4. 复制 bumi.xml 和 assets 目录到临时目录。
5. 在临时目录中用 mujoco.MjSpec.from_file 编译一次。
6. 编译成功后再原子替换目标缓存目录。
7. 输出 XML、mesh 数量、joint 数量、body 数量及源路径。
8. 默认拒绝覆盖已存在且内容不同的缓存；只有显式 --force 才覆盖。

目标布局：

~~~text
.cache/aa-robot-models/bumi/
├── bumi.xml
└── assets/
    ├── base_link.STL
    └── ...
~~~

执行命令：

~~~bash
uv --project "$AA_ROOT/venv/mjlab" run \
  python "$MIMIC_ROOT/scripts/prepare_bumi_assets.py" \
  --source "$BUMI_REF/src/assets/robots/bumi/xmls" \
  --output "$BUMI_ASSET_CACHE"
~~~

### 7.2 Gate 1

- MuJoCo spec 编译成功。
- 实际机器人 body 数为 22，hinge joint 数为 21，存在一个 floating_base freejoint。
- 所有 XML mesh 引用均可解析。
- 缓存目录不被 git 跟踪。

## 8. 工作包 2：实现 Bumi MJLab asset adapter

### 8.1 模块结构

mimic_lite/assets/bumi.py 至少包含：

- BUMI_XML
- BUMI_POLICY_JOINT_NAMES
- BUMI_BODY_NAMES
- BUMI_MOTION_JOINT_NAMES
- BUMI_MOTION_BODY_NAMES
- BUMI_ACTUATOR_SPECS
- BUMI_ACTION_SCALE
- BUMI_INIT_STATE
- get_bumi_spec
- make_mjlab_cfg
- make_cfg
- registry.register("asset", "bumi", make_cfg)

### 8.2 模型加载

get_bumi_spec 每次调用都返回一个新 spec：

1. 从 `ROBOT_MODEL_DIR/bumi/bumi.xml` 加载。
2. 使用 MJLab 1.3/MuJoCo 的 `MjSpec.from_file`，保留 `modelfiledir` 和相对 `meshdir=assets`；1.3 已移除 `mjlab.utils.os.update_assets`，不能沿用参考仓库的注入调用。
3. 若缓存不存在，错误信息必须给出 `prepare_bumi_assets.py` 的执行提示。

不要直接 import Bumi reference repo 的 Python 模块，因为参考仓库使用 mjlab 1.2.0，而 Active Adaptation 使用 mjlab 1.3.0。

### 8.3 actuator

按参考仓库逐项移植：

| actuator | kp | kd | effort | velocity | armature | frictionloss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| leg pitch | 60 | 3.0 | 60 | 12 | 0.007642512 | 0.2 |
| waist yaw | 53 | 3.4 | 27 | 9 | 0.0045024 | 0.2 |
| leg roll | 60 | 3.0 | 27 | 12 | 0.007642512 | 0.2 |
| leg yaw | 60 | 2.5 | 27 | 9 | 0.001860625 | 0.2 |
| knee pitch | 45 | 2.0 | 60 | 12 | 0.007642512 | 0.2 |
| ankle pitch/roll | 10 | 0.5 | 13 | 12 | 0.01 | 0.2 |
| arm pitch/roll/yaw | 12 | 0.4 | 5.5 | 50 | 0.001 | 0.2 |
| elbow pitch | 12 | 0.4 | 5.5 | 50 | 0.001 | 0.2 |

Active Adaptation 使用 MJLab 1.3.0；该版本已经移除 `DelayedActuatorCfg`。每组直接使用 `BuiltinPositionActuatorCfg`，并内联：

- `delay_min_lag=0`
- `delay_max_lag=4`

MJLab 1.3 的 `BuiltinPositionActuatorCfg` 没有 `velocity_limit` 参数，Bumi 参考仓的 MJLab 1.2 actuator 构造也未把表中的 velocity 值传给 simulator。因此首版保留这些值作为电机规格和后续显式限速的依据，但不宣称它们已由 builtin actuator 生效；当前 simulator 内实际生效的是 stiffness、damping、effort limit、armature、frictionloss 与 delay。

delay 自动作用于 position actuator 的 command field。不要设置已经移除的 `delay_target`，也不要 import Bumi reference 的 MJLab 1.2 actuator 类型。

当前内建 delay buffer 的 lag 粒度是每个 env，而不是每个 target joint。相同 delay 参数的 builtin actuator 可能被融合并共享 lag；这与“控制总线/公共链路延迟”的首版假设一致。若后续测量证明 21 个电机需要独立 lag，不通过拆成 21 个伪配置规避融合，而是实现和测试真正的 per-joint buffer。

验证每个 hinge joint 恰好匹配一个 actuator，不能漏配或重复覆盖。

### 8.4 初始状态

使用：

- base position z=0.472
- leg pitch=-0.1495
- knee pitch=0.3215
- ankle pitch=-0.172
- left arm roll=0.3
- right arm roll=-0.3
- 其余关节为 0
- 所有关节速度为 0

### 8.5 collision 与 sensor

collision：

- 启用名称匹配 .*_collision 的机器人 collision geom。
- 保留 Bumi 脚底 l_foot1_collision 到 l_foot7_collision 及右脚对应 geom。
- self-collision 行为与 Bumi reference 一致。

sensors：

1. contact_forces
   - primary：l_ankle_roll_link 或 r_ankle_roll_link 的 subtree
   - secondary：terrain body
   - fields：found、force
   - reduce：netforce
   - track_air_time：true
   - history_length：4
2. self_collision
   - primary/secondary：base_link subtree
   - fields：found、force
   - reduce：none
   - num_slots：1
   - history_length：4

### 8.6 名称与 symmetry

- joint_names_simulation 使用附录 A.1 的 21 项 policy 顺序。
- body_names_simulation 使用附录 A.3 的 22 项真实 body 顺序。
- joint symmetry 使用附录 A.4 的符号。
- spatial symmetry：base_link 与 waist_yaw_link 自映射，其余 l/r body 成对映射。

当前 mimic_lite_ppo 不依赖 symmetry 完成基线训练，但此处仍需定义正确，避免未来启用 symmetry 时重新修改资产接口。

### 8.7 测试

tests/test_bumi_asset.py 覆盖：

- 21 个 policy joint 名唯一。
- 22 个 body 名唯一。
- motion joint 集合与 policy joint 集合相同。
- motion body 去掉 motion_root 后与 asset body 集合相同。
- action scale 覆盖全部 21 个 joint。
- symmetry 映射是 involution。
- 11 组 actuator 均配置 `delay_min_lag=0`、`delay_max_lag=4`，且 `JointPosition` 不再配置 task-level delay。
- 缓存存在时，spec 可编译并得到预期 joint/body。

### 8.8 Gate 2

- 测试通过。
- make_cfg("mjlab") 返回 EntityCfg 和 sensor tuple。
- make_cfg("isaaclab") 明确失败并解释 MJLab-only。
- asset import 后 registry 中存在 bumi。

## 9. 工作包 3：参数化 MJLab simulation capacity

### 9.1 修改 env.py

保持所有现有 task 的默认行为不变，仅允许 task.sim.mjlab 覆盖：

- nconmax，默认 200
- njmax，默认 500
- contact_sensor_maxmatch，默认 80
- timestep，默认 0.005
- iterations，默认 10
- ls_iterations，默认 20
- ccd_iterations，仅在配置存在时传入

Bumi task 使用：

~~~yaml
sim:
  step_dt: 0.02
  isaac_physics_dt: 0.005
  mjlab:
    timestep: 0.005
    nconmax: 128
    njmax: 640
    contact_sensor_maxmatch: 256
    ccd_iterations: 50
    iterations: 10
    ls_iterations: 20
~~~

不要把 Bumi 参数改成全局默认值。

### 9.2 Gate 3

- 未配置 task.sim.mjlab 时，构造出的参数与修改前完全一致。
- Bumi config resolve 后显示上述覆盖值。
- G1 config resolve 不新增必填字段。

## 10. 工作包 4：准备 Bumi motion dataset

### 10.1 源数据基线

首版实现和测试固定使用 Bumi reference 当前目录中的全部数据：

- 36 个 NPZ。
- 共 7,535 帧。
- 全部 50 Hz。
- joint_pos/joint_vel 形状为 T x 21。
- body_pos_w/body_lin_vel_w/body_ang_vel_w 形状为 T x 23 x 3。
- body_quat_w 形状为 T x 23 x 4，四元数为 wxyz。
- 23 个 motion body 中 index 0 是 motion_root，index 1 是 base_link。

### 10.2 新增 prepare_bumi_motion_dataset.py

脚本接口：

~~~text
python scripts/prepare_bumi_motion_dataset.py \
  --source /path/to/bumi/amp \
  --output /path/to/.cache/mimic-lite/motions/bumi/omni \
  --link-mode symlink
~~~

支持 link-mode=symlink 或 copy，默认 symlink。脚本必须：

1. 只扫描 source 根目录下的 *.npz。
2. 要求每条 motion 都包含 fps、joint_pos、joint_vel、body_pos_w、body_quat_w、body_lin_vel_w、body_ang_vel_w。
3. 验证 fps=50、T 一致、21 joint、23 motion body。
4. 验证所有值 finite、四元数范数、MimicLite 当前数值阈值。
5. 为每条 motion 创建 output/MOTION_STEM/motion.npz。
6. 在每个 motion 目录写入内容完全一致的 meta.json。
7. meta.json 使用附录 A.2 motion joint 顺序和附录 A.3 前置 motion_root 的 body 顺序。
8. 先写临时目录，全部验证通过后再原子替换 output。
9. 默认拒绝覆盖内容不同的 output；显式 --force 才允许覆盖。
10. 输出 motion 数量、总帧数、fps 和所有字段的最大值。

执行：

~~~bash
uv --project "$AA_ROOT/venv/mjlab" run \
  python "$MIMIC_ROOT/scripts/prepare_bumi_motion_dataset.py" \
  --source "$BUMI_REF/src/assets/motions/bumi/amp" \
  --output "$BUMI_MOTION_CACHE" \
  --link-mode symlink
~~~

### 10.3 motion configs

cfg/task/motion/bumi/omni.yaml：

~~~yaml
# @package task.command

motion_cfgs:
  bumi_omni:
    path: ".cache/mimic-lite/motions/bumi/omni"
    weight: 1.0
    full_motion: true
    shard: false
~~~

cfg/task/motion/bumi/single.yaml 指向：

~~~text
.cache/mimic-lite/motions/bumi/omni/x0.5_8__steady
~~~

legacy dataset 不支持 filenames 过滤，所以 single config 指向只包含该动作的子目录。不要直接指向 symlink `motion.npz`：any4hdmi 会先 resolve 文件路径，从而看到源 NPZ 的原始文件名并拒绝其不是 `motion.npz`；目录扫描则会保留 legacy wrapper 文件名。

### 10.4 测试

tests/test_bumi_motion_dataset.py 使用 temporary directory 和微型假数据覆盖：

- 缺字段失败。
- fps 或 shape 错误失败。
- 非 finite 或异常 quaternion 失败。
- 正常转换产生 motion.npz 与一致的 meta.json。
- symlink 与 copy 两种模式。
- 默认不覆盖不同输出。

### 10.5 Gate 4

- 数据准备脚本报告 36 motions、7,535 frames、50 Hz。
- any4hdmi 能加载 single 和 omni 两个路径。
- MimicLite motion validation invalid_frames=0。
- dataset joint/body 名字覆盖 task 需要的全部名字。

## 11. 工作包 5：新增 Bumi tracking task

### 11.1 基础映射

从 tracking-base.yaml 复制为 tracking-base-bumi.yaml，并完成以下替换：

~~~yaml
name: BumiTrackBase

shared:
  root_body_name: base_link
  anchor_body_name: waist_yaw_link
  reward_root_body_name: waist_yaw_link
  termination_root_body_name: base_link

robot:
  name: bumi
~~~

obs_body_names：

~~~yaml
- base_link
- waist_yaw_link
- ".*_knee_pitch_link"
- ".*_ankle_roll_link"
- ".*_arm_yaw_link"
- ".*_elbow_pitch_link"
~~~

tracking_body_names：

~~~yaml
- base_link
- waist_yaw_link
- ".*_leg_roll_link"
- ".*_knee_pitch_link"
- ".*_ankle_roll_link"
- ".*_arm_roll_link"
- ".*_arm_yaw_link"
- ".*_elbow_pitch_link"
~~~

tracking_joint_names 与 reward_joint_names 均覆盖：

~~~yaml
- waist_yaw_joint
- ".*_leg_pitch_joint"
- ".*_leg_roll_joint"
- ".*_leg_yaw_joint"
- ".*_knee_pitch_joint"
- ".*_ankle_pitch_joint"
- ".*_ankle_roll_joint"
- ".*_arm_pitch_joint"
- ".*_arm_roll_joint"
- ".*_arm_yaw_joint"
- ".*_elbow_pitch_joint"
~~~

reward_root_body_name 和 termination_root_body_name 必须包含在 tracking_body_names 中。

### 11.2 action

使用以下不重叠的 pattern：

~~~yaml
input:
  action:
    _target_: mimic_lite.JointPosition
    action_scaling:
      ".*_leg_pitch_joint": 0.25
      waist_yaw_joint: 0.12735849
      ".*_leg_roll_joint": 0.1125
      ".*_arm_pitch_joint": 0.11458333
      ".*_leg_yaw_joint": 0.1125
      ".*_arm_roll_joint": 0.11458333
      ".*_knee_pitch_joint": 0.33333333
      ".*_arm_yaw_joint": 0.11458333
      ".*_ankle_pitch_joint": 0.325
      ".*_elbow_pitch_joint": 0.11458333
      ".*_ankle_roll_joint": 0.325
    min_delay: 0
    max_delay: 0
    alpha: 1.0
    expected_action_dim: 21
~~~

初始化环境时断言 action_dim=21，action manager 的 joint_names 与附录 A.1 完全相同。

### 11.3 reward 与 termination

第一版：

- 保持 G1 tracking reward 的 weight 和 sigma。
- feet_slip 保持 disabled。
- feet_air_time 设为 disabled，并删除 body2_names。
- self_collisions 保持 enabled，force_threshold 改为 10.0。
- action rate、joint velocity、joint limit、survival 保留。
- termination 的初始 threshold 保持现值，先用来暴露明显映射错误。

只有单 motion overfit 通过后，才根据 Bumi error 分布调整 tracking sigma 或 termination threshold。一次只改一个参数家族。

### 11.4 randomization

配置中先完成名称迁移，但 overfit/smoke 命令删除整个 randomization 节点。

MVP 名称与范围：

- material body：.*ankle_roll_link 与 .*elbow_pitch_link；范围沿用 G1。
- mass：base_link、waist_yaw_link，各 0.9 到 1.1。
- COM：base_link|waist_yaw_link，-0.03 到 0.03。
- arm/elbow stiffness、damping：0.9 到 1.1。
- waist/leg/knee/ankle stiffness、damping：0.8 到 1.2。
- armature：0.75 到 1.25。
- friction：0.05 到 0.25。
- joint offset：-0.01 到 0.01。
- root velocity perturbation：首轮保持 G1 间隔与范围。

完整 Bumi DR parity，包括 pseudo-inertia、effort-limit、默认关节偏置和结构化被动动力学，放在多 motion 名义训练通过之后。

### 11.5 静态检查

以下命令预期无输出：

~~~bash
rg -n \
  'pelvis|torso_link|hip_|shoulder_|wrist_|toe_link|left_|right_' \
  "$MIMIC_ROOT/cfg/task/tracking-base-bumi.yaml"
~~~

### 11.6 Gate 5

- task config 中不存在 G1/Atom 残留 body 或 joint 名。
- action、tracking joint 都解析为 21 项且没有重复。
- tracking body 解析为 14 项。
- root、anchor、reward root、termination root 均存在于 dataset 与 asset。
- feet reward 不引用不存在的 toe body。

## 12. 工作包 6：静态与配置验证

### 12.1 命令

~~~bash
uv --project "$AA_ROOT/venv/mjlab" run \
  python -m unittest discover -s "$MIMIC_ROOT/tests"

uv --project "$AA_ROOT/venv/mjlab" run \
  pyright "$AA_ROOT/active_adaptation" "$MIMIC_ROOT/mimic_lite"

uv --project "$AA_ROOT/venv/mjlab" run aa-discover-projects

uv --project "$AA_ROOT/venv/mjlab" run \
  "$MIMIC_ROOT/scripts/train.py" \
  task=tracking-base-bumi task/motion=bumi/single \
  +exp=ppo/train backend=mjlab \
  --cfg job --resolve

uv --project "$AA_ROOT/venv/mjlab" run \
  "$MIMIC_ROOT/scripts/train.py" \
  task=tracking-base task/motion=g1/lafan \
  +exp=ppo/train backend=mjlab \
  --cfg job --resolve
~~~

### 12.2 Gate 6

- 测试与 pyright 没有本任务引入的新错误。
- 两份 Hydra 配置 resolve 成功。
- Bumi resolved config 中 robot.name=bumi、action_dim 来源覆盖 21 joint、motion path 指向缓存。
- G1 resolved config 的原有 simulation 默认值不变。

## 13. 工作包 7：环境与 PPO smoke test

### 13.1 启动前检查

每次 GPU 运行前执行：

~~~bash
nvidia-smi
pgrep -af 'train.py|torchrun'
tmux ls
~~~

不得占用已有任务正在使用的 GPU。

### 13.2 16-env 单 motion smoke

~~~bash
cd "$AA_ROOT"

HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
bash scripts/launch_ddp.sh 0 \
  projects/mimic-lite/scripts/train.py venv/mjlab \
  task=tracking-base-bumi task/motion=bumi/single \
  +exp=ppo/train backend=mjlab \
  task.num_envs=16 total_iters=1 wandb.mode=disabled \
  '~task.randomization'
~~~

若 Hydra 不接受节点删除语法，停止并增加一个仅用于 smoke/overfit 的 task variant；不要带着随机化继续排查。

### 13.3 Gate 7

- asset、scene、sensor 和 motion dataset 初始化完成。
- 打印 action_dim=21。
- 运行至少一次 rollout 和一次 PPO update。
- 无 missing name、contact buffer overflow、NaN、CUDA illegal access 或 actuator overlap。
- motion reset 后 base_link 高度处于参考范围，关节没有瞬时爆炸。
- 进程正常退出且 GPU 释放。

## 14. 工作包 8：单 motion overfit

### 14.1 训练

从头训练，不加载 G1 checkpoint：

~~~bash
cd "$AA_ROOT"

HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
bash scripts/launch_ddp.sh 0 \
  projects/mimic-lite/scripts/train.py venv/mjlab \
  task=tracking-base-bumi task/motion=bumi/single \
  +exp=ppo/train backend=mjlab \
  task.num_envs=256 total_iters=1000 \
  wandb.mode=disabled checkpoint_interval=100 \
  '~task.randomization'
~~~

训练过程中首先判断 reward、episode length 和 termination 原因是否持续改善。若 300 到 500 iterations 已稳定达到 gate，可提前停止；若指标完全不改善，不为凑满 1000 iterations 而继续。

### 14.2 deterministic eval

使用最新 checkpoint：

~~~bash
uv --project "$AA_ROOT/venv/mjlab" run \
  "$MIMIC_ROOT/scripts/eval.py" \
  task=tracking-base-bumi task/motion=bumi/single \
  backend=mjlab task.num_envs=128 eval_steps=1000 \
  checkpoint_path=/absolute/path/to/model.pt \
  task.reward.tracking_metrics._enabled_=true \
  task.reward.tracking_metrics.joint_pos.enabled=true \
  task.reward.tracking_metrics.body_pos.enabled=true \
  task.reward.tracking_metrics.body_ori.enabled=true \
  eval_output=bumi_single_eval.pt \
  eval_summary_output=bumi_single_eval.json
~~~

### 14.3 Gate 8

- lafan_progress >= 0.95。
- mean joint_pos error <= 0.35 rad。
- mean local body_pos error <= 0.10 m。
- mean local body_ori error <= 0.35 rad。
- 主要结束原因为 motion_timeout，而不是 tracking error。
- deterministic rollout 无明显左右关节交换、根漂移、四元数翻转或脚底穿透。

若 Gate 8 失败，排查顺序固定为：

1. policy joint order 与 asset joint index。
2. motion joint metadata 与按名字重排结果。
3. base_link index 1 和 motion_root index 0。
4. quaternion 顺序 wxyz。
5. reset 后第 0 帧的 FK/body pose。
6. action scale、default pose、delay。
7. contact/collision。
8. 最后才调整 reward、network 或 PPO 超参数。

## 15. 工作包 9：全部 36 条 motion

### 15.1 名义训练

先不加随机化：

~~~bash
cd "$AA_ROOT"

HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
bash scripts/launch_ddp.sh 0 \
  projects/mimic-lite/scripts/train.py venv/mjlab \
  task=tracking-base-bumi task/motion=bumi/omni \
  +exp=ppo/train backend=mjlab \
  task.num_envs=1024 total_iters=1000 \
  wandb.mode=online checkpoint_interval=100 \
  '~task.randomization'
~~~

使用 eval.py 对 omni 数据运行 deterministic eval，并保存 JSON。为使“36 条动作都被评估”成为可验证条件，对 eval.py 做最小扩展：

- reset 后保存每个 env 的初始 motion id。
- 在结果中输出 num_unique_motions 和 motion_coverage。
- 按初始 motion id 聚合 lafan_progress 的 count、mean、std、min。
- JSON 中保留 motion id 到 motion path/name 的映射。
- 不改变现有全局 summary 字段和默认 rollout 行为。
- 用 synthetic motion id/progress tensor 测试 coverage、分组统计和空输入错误。

如果 512 env 仍未覆盖全部 motion，增加 eval env 数或对缺失 motion 单独补跑；不能用随机采样“理论上应该覆盖”代替实际 coverage。

### 15.2 Gate 9

- num_unique_motions=36 且 motion_coverage=1.0。
- mean progress >= 0.80。
- 每条 motion 都有 per-motion progress 记录。
- 没有单条 motion 因 name/shape/FK 问题必然失败；若某条 progress < 0.50，必须单独诊断。
- mean joint_pos error <= 0.45 rad。
- mean local body_pos error <= 0.15 m。
- 训练无 NaN，contact capacity 无 overflow。

若少量 motion 显著失败，先输出 per-motion 指标并判断是数据质量还是统一配置问题；不立即扩大 PPO sweep。

## 16. 工作包 10：逐步恢复稳健化

只有 Gate 9 通过后执行。每次只加入一个家族，并与固定的 nominal eval 对比：

1. joint offset。
2. actuator kp/kd。
3. armature/friction。
4. body mass/COM。
5. material friction。
6. root pushes。
7. feet airtime 或其他足部 shaping。

每个家族的保留条件：

- 训练稳定。
- nominal progress 下降不超过 0.05。
- 随机化 eval 或抗扰动指标有明确改善。

feet_air_time 恢复时：

- 只使用 l_ankle_roll_link 与 r_ankle_roll_link。
- 不设置 body2_names。
- 先基于现有 motion 的 ankle height 分布设定范围，再做一次有/无该项的决策实验。

完整 Bumi DR parity 是独立工作包。需要先确认 Active Adaptation 对 MJLab 1.3 builtin actuator delay、effort-limit 和 passive-dynamics randomization 的支持，不能直接复制参考仓库的 MJLab 1.2.0 event。

## 17. 工作包 11：扩大训练与 G1 regression

### 17.1 扩大训练

名义与稳健化 gate 均通过后，先单 GPU 4096 env，再根据显存和吞吐决定是否使用 8192 env 或 DDP。

长训练前：

- 再次检查 nvidia-smi、pgrep 和 tmux。
- 使用命名 tmux。
- stdout/stderr 通过 tee 保存。
- 记录准确命令、commit、GPU、seed、W&B id 和 checkpoint。
- 确认作业经过 asset resolution、environment creation、W&B init 并进入 iteration loop。

不以进程存在作为启动成功的证据。

### 17.2 G1 regression

由于 active-adaptation/env.py 被修改，必须运行：

~~~bash
cd "$AA_ROOT"

HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
bash scripts/launch_ddp.sh 0 \
  projects/mimic-lite/scripts/train.py venv/mjlab \
  task=tracking-base task/motion=g1/lafan \
  +exp=ppo/train backend=mjlab \
  task.num_envs=16 total_iters=1 wandb.mode=disabled
~~~

### 17.3 Gate 11

- Bumi 长训练正常进入迭代循环。
- GPU 显存和 contact buffer 有余量。
- G1 regression smoke 通过。
- README 中的 Bumi 数据准备、smoke、训练和 eval 命令与实测一致。

## 18. 风险与处置

| 风险 | 早期信号 | 处置 |
| --- | --- | --- |
| mesh 路径丢失 | XML 能找到但 scene compile 失败 | 使用 fresh `MjSpec.from_file` 并保留缓存目录中的相对 `assets/`；先过 Gate 1 |
| policy/motion 顺序混淆 | 能训练但肢体错位或剧烈摆动 | policy 固定 MJCF 顺序，motion 只按名字映射 |
| 把 motion_root 当真实根 | reset 到原点或根姿态异常 | root 固定 base_link；验证 index 1 |
| delay 叠加 | 响应极慢、动作严重滞后 | MJLab 1.3 builtin actuator delay 0..4；JointPosition delay=0、alpha=1 |
| 误以为 builtin delay 是逐电机独立 | 实机逐电机日志与仿真统计不一致 | 首版明确按每 env 公共 lag 建模；确有逐电机差异时实现 `(num_envs, 21)` buffer |
| contact capacity 不足 | sensor overflow、CUDA 错误 | Bumi 使用 njmax=640、maxmatch=256、CCD=50 |
| mjlab 版本差异 | actuator delay 或 sensor API 报错 | 使用 1.3 内联 delay API，移植语义，不 import reference Python |
| feet reward 引用 toe | 初始化 name resolution 失败 | 首轮关闭 feet airtime，移除 body2_names |
| G1 checkpoint 不兼容 | state_dict/VecNorm shape mismatch | 从头训练，不加载 G1 checkpoint |
| 36 条 gait 数据能力有限 | 步态可跟踪但通用动作失败 | 将全量 G1 motion 单独 retarget 到 Bumi |
| DR 掩盖映射错误 | smoke 不收敛且症状随机 | Gate 8 前删除全部 randomization |

## 19. 停止条件

出现以下情况必须停止当前阶段，不继续扩大训练：

- Asset/spec 或 motion metadata 尚未通过静态验证。
- 单 motion 500 iterations 后 reward、episode length 和 error 完全无改善。
- reset 第 0 帧与参考动作存在系统性 FK 偏差。
- 出现未解释的 joint permutation、quaternion convention 或 root index 问题。
- contact buffer overflow、NaN 或 CUDA 错误可重复发生。
- G1 regression 因 simulation 默认值改变而失败。
- GPU 已被其他作业占用。

停止后先修复最早失败的 gate，再从该 gate 重跑；不通过扩大网络、环境数或超参数 sweep 掩盖基础问题。

## 20. 建议提交顺序

active-adaptation 主仓：

1. make mjlab sim capacity configurable

mimic-lite 仓：

1. add bumi asset adapter
2. add bumi data preparation
3. add bumi tracking task
4. add bumi tracker validation
5. document bumi training workflow

每个提交只包含对应工作包。生成数据、checkpoint 和输出不进入提交。

## 附录 A：名称约定

### A.1 Policy/MJCF joint 顺序

~~~text
waist_yaw_joint
l_arm_pitch_joint
l_arm_roll_joint
l_arm_yaw_joint
l_elbow_pitch_joint
r_arm_pitch_joint
r_arm_roll_joint
r_arm_yaw_joint
r_elbow_pitch_joint
l_leg_pitch_joint
l_leg_roll_joint
l_leg_yaw_joint
l_knee_pitch_joint
l_ankle_pitch_joint
l_ankle_roll_joint
r_leg_pitch_joint
r_leg_roll_joint
r_leg_yaw_joint
r_knee_pitch_joint
r_ankle_pitch_joint
r_ankle_roll_joint
~~~

### A.2 Motion NPZ joint 顺序

~~~text
l_leg_pitch_joint
r_leg_pitch_joint
waist_yaw_joint
l_leg_roll_joint
r_leg_roll_joint
l_arm_pitch_joint
r_arm_pitch_joint
l_leg_yaw_joint
r_leg_yaw_joint
l_arm_roll_joint
r_arm_roll_joint
l_knee_pitch_joint
r_knee_pitch_joint
l_arm_yaw_joint
r_arm_yaw_joint
l_ankle_pitch_joint
r_ankle_pitch_joint
l_elbow_pitch_joint
r_elbow_pitch_joint
l_ankle_roll_joint
r_ankle_roll_joint
~~~

### A.3 Body 顺序

Asset 的 22 个真实 body：

~~~text
base_link
waist_yaw_link
l_arm_pitch_link
l_arm_roll_link
l_arm_yaw_link
l_elbow_pitch_link
r_arm_pitch_link
r_arm_roll_link
r_arm_yaw_link
r_elbow_pitch_link
l_leg_pitch_link
l_leg_roll_link
l_leg_yaw_link
l_knee_pitch_link
l_ankle_pitch_link
l_ankle_roll_link
r_leg_pitch_link
r_leg_roll_link
r_leg_yaw_link
r_knee_pitch_link
r_ankle_pitch_link
r_ankle_roll_link
~~~

Motion metadata 在上述列表最前面增加 motion_root，共 23 项。

### A.4 Joint mirror sign

| 左侧或自身 joint | 镜像 joint | sign |
| --- | --- | ---: |
| waist_yaw_joint | waist_yaw_joint | -1 |
| l_arm_pitch_joint | r_arm_pitch_joint | 1 |
| l_arm_roll_joint | r_arm_roll_joint | -1 |
| l_arm_yaw_joint | r_arm_yaw_joint | -1 |
| l_elbow_pitch_joint | r_elbow_pitch_joint | 1 |
| l_leg_pitch_joint | r_leg_pitch_joint | 1 |
| l_leg_roll_joint | r_leg_roll_joint | -1 |
| l_leg_yaw_joint | r_leg_yaw_joint | -1 |
| l_knee_pitch_joint | r_knee_pitch_joint | 1 |
| l_ankle_pitch_joint | r_ankle_pitch_joint | 1 |
| l_ankle_roll_joint | r_ankle_roll_joint | -1 |

## 附录 B：后续通用 tracker 数据

当前 36 条 motion 只覆盖由 G1 SONIC omni retarget 得到的 locomotion gait。要达到 G1 mixture 的通用 tracker 能力，另建数据工作流：

1. 对 LAFAN、100Style、AMASS、ground、extreme 等源动作逐库 retarget 到 Bumi。
2. 输出 Bumi qpos，使用 Bumi MJCF 重新计算 FK。
3. 对比 retarget 输出与 MuJoCo FK，确认 body pose、velocity 和 quaternion convention。
4. 生成标准 any4hdmi manifest 数据集。
5. 每个数据集单独做 single-dataset smoke，再组合权重。
6. 不在缺少单库证据时直接复制 G1 mixture 权重。
