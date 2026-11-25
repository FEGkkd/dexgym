# DexGym / Teleop 仓库审计

本页的现场目录、状态与资源记录属于原始 P00。Review Fix 的独立 `s0-5` checkout、解释器和资源见 [执行基线](EXECUTION_BASELINE.md)。原始发布快照仍可按 Git SHA 和清单复验。

Run ID `S05-P00-20261125-ubuntu-audit`。当前见证、历史记录与推测按主设计 M9 分类。所有本页行号取指定 checkout 的当时文件。完整的行号索引、摘要及 SHA-256 在 [证据目录](evidence/P00/README.md)。

## 1. Checkout 身份与宿主机

| 现场目录 | Git 身份 | 审计用途和工作树 |
|---|---|---|
| `/home/ubuntu/dexgym` | `main`，`21a813a1950891c712e421b5ced883763bc904cb` | 主仓库。tracked 修改 `source/dexgym/assets/marvin_wuji/__init__.py` 和 `source/dexgym/data/wuji_retarget.py`，tracked diff SHA-256 `ca0ca092d3120dc5249eccadf0c4c26eb56e22d22a4935ccd72481c4b555c2bc`，审计开始时未跟踪路径 954，涵盖历史 artifacts、源码和测试。S0-1、S0-2、S0-4 指定包在此 checkout 缺失 |
| `/tmp/dexgym-s04-main.Wwwqff99` | detached `e7f747e16e83aea4f8e6549e216cdd751bede229` | clean；发布形态 `wholebody_s04/train_aligned.py`、`sweep.py`、`servo_profile.json` 和任务文档；S0-2 仅精简接口 |
| `~/dexgym/data/processed/wholebody-s04/world_hand20_gravity_20000_20261125/checkout` | detached `b833eb6d69d67e7c10bcfbefa9eee13bce41e7e0` | tracked clean，42 未跟踪路径；包含完整 S0-1、S0-2 和 S0-4 历史调用基础。实际对齐 job 的同名脚本在 `world_hand20_aligned_from_scratch_20261125/`，脚本哈希单列 |
| `/home/ubuntu/teleop_ws` | `30cf13bdc496c43b3016462cfa0a206334d1ff7f` | dirty；含驱动、标定、Teleop 核心等用户差异。完整 tracked 路径与哈希见 host.json |
| Windows `implementation/dexgym` | `main`，`b7749f3eae84ddf820d9fde913eb6ab411a8decc` | 较早的本地 checkout，存在 8 处既有 tracked 改动。文档在同一相对路径中维护，代码证据以 Ubuntu 为准 |

Ubuntu 24.04.5 / 内核 7.0.0-42，i9-14900K 32 逻辑 CPU，125 GiB RAM，RTX 5090 32,607 MiB 显存；探查时显存已有约 23 GiB 占用。根盘约 29 GiB 可用，不以此推断 P01 有足够写入空间。ROS Jazzy 与现场 overlay 可用。审计范围检查了 README、pyproject、脚本、模型、docs、tests 与 data 组织。`pyproject.toml:9–10` 固定 Python 3.12 与 Isaac Lab 3.0.0。GitHub 公共仓库页面提供仓库身份，现场 checkout 版本以 SSH 证据为准。[仓库主页](https://github.com/FEGkkd/dexgym)

| 解释器 | 已见包与通过的检查 |
|---|---|
| Wuji Python 3.12.14 | Torch 2.10.0+cu128、MuJoCo 3.13.0 实际 import。Warp 1.17.0、RSL-RL 5.0.1 发现。Isaac Lab/Newton/mujoco_warp 无此路径 |
| 裸 Isaac Sim 6.0.1 Python 3.12.11 | Torch 2.11.0+cu130、MuJoCo 3.8.0 实际 import。Warp 1.13.0、Newton 1.2.1、mujoco_warp 3.8.0.3 发现。Isaac Lab 暴露另一来源 |
| 项目 overlay 下的 Isaac Sim | Torch 2.11.0+cu130、MuJoCo/Warp 后端 3.12.0、Warp 1.17.0、Isaac Lab 3.0.0、Newton 1.6.1、RSL-RL 5.5.1 actual import。import 见证无需启动完整 Isaac 应用 |

`scripts/supermimic/host.sh` 建立本机 overlay，`runtime.py` 先加载 Torch，再附加 venv，进行 `--pytest`、`-m` 或脚本分派。实际 S0-4 通过 job 脚本、训练 checkout source 和此 runtime 组合启动。源码或包名相同不代表二进制或模型相同。[环境和 Git](evidence/P00/host.json)

## 2. 原始 Teleop 调用入口

**CODE + OBSERVED。** `teleop_ws/src/whole_body_teleop/launch/whole_body_teleop.launch.py:15–18` 创建 Node 并叠加配置。`teleop_controller_node.cpp:734–745,853–881,925` 接入左右 calibrated Tracker 和 RobotState/JointState，按配置时钟发 `/tianji/joint_command`；`teleop_core.cpp:613,1478–1489` 处理 ARM IK/目标及腕注册路径。`tianji_real_driver/src/real_driver_node.cpp:948,998,1018` 接命令并校验。手路径独立，MANUS 原始 glove stream 经 Wuji retarget 节点得左右 `/left_hand/command` 和 `/right_hand/command`，`wuji_hand2_real_driver/src/node.cpp:88–98,134–135` 接收独立手命令并发手状态。原始 bag 同时有双手命令、反馈、左右 glove、raw/registered 双腕、臂命令、机器人状态及跟踪安全流。原始 B0 必须忠实保留这一上下文，不能单用模型手部 FK 代替 source intent。[行级见证](evidence/P00/source.json) · [原包内容](evidence/P00/raw.json) · [附加探查](evidence/P00/followup.json)

**频率。** C++ 配置默认 100 Hz（Node 530），已读 `bimanual_position_real_no_head.yaml:42` 对应 250 Hz 名义臂控制，手状态配置 120 Hz。2026-10-16 bag 记录 336.29 s、每只 MANUS 40,257 条、raw wrist 各 40,261 条，约 120 Hz 平均通道记录率。注册后的左右 wrist 分别 29,052 与 39,848 条。臂命令 76,261 条，机器人状态 40,141 条。录包平均速率不等于硬件生效频率；采集运行配置须与不同时间、消息状态和丢包对照。P01/P03/P10 分别核查时基、Servo 响应及端到端。50/500 Hz 是设计候选。

## 3. 数据及资产可访问性

| 层 | 现有证据 / 差异 |
|---|---|
| 原始 MCAP | `~/dexgym/data/raw/wholebody_s0/{2026-11-12,2026-11-13}` 有 30 个可解析路径的目录符号链接指向采集平台。取样 episode `18e74c646013f1e6-2156b79bc2482857` 有 manifest/effective_config/run_snapshot/events 与 7 类流。ROS 读取了多源 schema；有双手骨架和 Tracker 元数据。`~/teleop_ws/data/*/*.mcap` 另发现 86 个路径，含原包和 `s0_sim_*` 复放，86 不是确认的可训练 episode 数 |
| parsed.npz | `~/dexgym/data/processed/wholebody_s0/{date}/episodes/*` 30 套配对。样本保存每流 header/log/publish/receive_ros/monotonic/realtime/source 字段，MANUS pose、sensor 与 valid，Tracker pose、quality。审计读出实际 dtype、shape、SHA |
| aligned.npz | 同 episode 的 `time_ns`、`human_valid`、双手和 raw/registered wrist pose、`*_brackets` 与 `*_causal_index`。`sync.py:73,115–132` 计算因果索引并独立插值，默认 pose 数组为插值结果。P01 的在线路径须用原始时间信息验证和重建 |
| reference.npz | 同 episode 的 `q_ref`、`hand_q_unclipped`、腕目标与 native TCP、机器人 skeleton/tips、`reference_usable`、`rejection_bits` 和 `in_library`。现有最终几何不是独立人体 Ground Truth |
| Motion Library | 找到按日、batch 和 native replay/retiming 多份 manifest。初始合并 427 clips，native replay 561。manifest 存 clip 和 split、54 个 joint/firmware names、qpos/actuator indices、单位及 provenance。多个版本共享父录制，训练/验证/Test 须重新设计原始谱系 |
| S0-4 bank | 对齐 job 由 `contact_replay_filtered_6000_20261125/bank_round_2/manifest.json` 驱动。历史报告与代码明确可混 Teleop/EgoPro。不能作为 P01 唯一意图数据 |
| 模型 | main 的 `source/dexgym/assets/marvin_wuji/hand2_2/generated/dual_arm_marvin.xml` SHA 见 raw.json，MuJoCo CPU 载入，`nq=nv=nu=njnt=54`，重力 −9.81、2 ms，10 子步有限且 0 warning。S0-4 `launch.sh:6` 使用 Teleop workspace 的 `src/whole_body_mujoco_driver/model/dual_arm_marvin.xml`，训练档案 SHA `be69f387f5bc25c8011a992a88ac2c2449c2c0419e09318b740fcdcd1402871d` 与 main 的模型不同 |
| Checkpoint / profile | 对齐 job 的 `model_15000_actor.pt`、`model_19000_actor.pt` 可读取并 SHA。硬件对齐 profile、有重力的 parity JSON 可读。未来 B2 还需实际兼容 replay；未在 P00 载入 pickle 或运行旧训练 |

`~/dexgym_data` 亦可读取，含 Marvin54 旧 Motion Library、验证与依赖 overlay。GRAB/CMU/EgoPro 和 simulation rollouts 属于审计可见资源，不计作 S0-5 数据。`ros2 bag info` 确认存储与话题类型，未作全数据生产操作。[数组 schema](evidence/P00/data.json) · [源录包与模型](evidence/P00/raw.json)

## 4. 控制与训练路径及复用计划

| 模块及 Git | 已追踪源路径与行为 | 复用 / 改造或新建 |
|---|---|---|
| Wholebody S0 / main | `bag.py::read_episode`、`sync.py::align_episode` → `pipeline.py::reference_one:57`/`export_library:173`；`retarget.py::run_arm_tool / run_hand_sdk`；`pipeline.py::split_episodes:154` 原切分，产出参考和质量报告 | 复用 CDR 解析、元信息、可追溯校验和数据候选；P01 新建 source registry 和接收感知对齐，显式排除重复/派生 |
| S0 意图 / main | `intent.py::HumanIntentGeometry:39–86` 以几何基、方向和机器人骨长造目标，再转腕/世界 | 复用候选、验证 functional pinch 和 cross-hand 约束；新建三层目标、feasible/Oracle |
| S0 PPO / main | `ppo_task.py::learn:59` 使用 `TanhGaussian.log_prob:37–39` 和 RSL-RL；`learn:147` 传 `process_env_step(obs,reward,done,{})`。`residual_env.py:21–24,41–70` 构造 Domain/Actor 和 Critic 输入、RSI 和扰动；`evaluate_residual.py:32,102,382` 原生评估与 lag 诊断 | 复用稳定变换与导出思路；P04 分清时间截断、final_obs 和 RSL-RL 版本。不得假定空 info 的路径 bootstrap 正确。旧 lag 仅诊断 |
| S0 DR / main | `domain.py:7,14–26,55–104` 配置臂/手 kp、kd、effort、摩擦、延迟和偏置；`ResidualEnv:24,164–183` 后端写入及自检 | 候选复用字段；P08 逐项用物理或时序响应证明启用，不能推广到未走此路径的 S0-4 |
| S0-1 / b833eb6 | `targets.py::TargetMapper:59–86` 包装 HumanIntent；`kinematics.py::RobotKinematics:104, wrist_jacobian:141`；`control.py::Command:61–81`、`Observation:89–164`；`finger_reward.py::bilateral_tip_vectors:22、thumb_tip_vectors:28、opposing_fingertip_cost:39`；`bank.py`/`prepare.py` 的名义/目标导入 | 复用名称绑定、Torch FK、腕 Jacobian 和向量语义；全指 Jacobian、54 维联合约束与完整意图投影须独立验证/实现。原绝对动作不能直接当 residual |
| S0-2 / b833eb6 | `interface.py:15,138,216` 为 arm reference residual + hand command delta；`task.py::TrackingEnv:25,step:102–129` 经过命令与真实 FK 计分。存在 hand/arm servo 试验入口 | 利用手部驱动、限幅和日志样例；动作步进/历史参考与 C3 不同，不能直接设为 S0-5 |
| S0-4 / e7f747e + b833eb6 + job | `task.py::ArmCommand/Observation/tracking_value`，`hand_points.py` 5/20 点；`ServoTask.advance_memory` + `AlignedPhysics.interval`；`ReviewTrainer.assess`、`sweep`。见 `5 | 对齐 Servo、路径及 B2 候选复用；替换跟踪/数据/动作，重写评估入口和动态测试 |
| Marvin54 / main | `task.py::HandGeometry:74–139` 跨手几何及 tip 梯度，`HandCentricTask:140,248–257` 可选 Jacobian，`env.py::HandCentricEnv.step:72–118` 真实 interval、失败 flags、完成；`physics.py/warp_physics.py/newton_physics.py` 后端可选 | 复用 FK/梯度/物理监测，G/T 模型和 time semantics 必须校准。旧 motion goal library 不能替代因果输入 |
| SuperMimic / main 与发布/运行差异 | `source/dexgym/algorithms/supermimic/dex_reference/engine.py` 回放物理 transitions 与预测更新，S0-4 job Trainer 对 engine 走评估和 arm 滤波 memory | P09 通过相同 task adapter 对照，单独验证多步/静态/动作 Jacobian，跨版本加载式测试先行 |

## 5. S0-4 历史问题复核

| 问题 | 当前确认 | S0-5 所需证据 |
|---|---|---|
| 14/54 维失配 | **CODE＋HISTORICAL** `__init__.py:6–19` action 14、command 54；`task.py::ArmCommand.forward:52` 限 ±8° 并对 J6/J7 耦合投影，手直接取参考。`Observation.frame:88–99` 由实际与参考 FK 起误差，`tracking_value:106–109` 65% 平均和 35% 最差。job provenance 14/2880 与 `exact_sdk_command` witness 为归档历史 | P03/P04 用真实物理验证 54 轴与零残差，P05 单指与臂手补偿 |
| SDK 目标和人意图 | **CODE＋HISTORICAL** `hand_points.py::articulated_points/arm_landmark_cost:29–40` 用目标腕系的 SDK 参考手型构造 20 点。该目标衡量重放 S0-4 几何。不存在独立的 raw/mapped/feasible 三层评分 | P01 保留 L1，P02 复核 bone 和 pinch，P03/P09 外部共同目标 |
| 运动学和动力学混合 | **CODE** 当前 residual 加在臂 IK 上，手形直接执行，奖励看世界系误差，过滤/伺服影响闭环；现有评估只可推断效果，无法隔离关节可达性或映射误差 | P02 约束 Oracle 与臂/手自由度扫描；P03 B0/B1/B1+ 同物理 |
| 静态难姿态 | **HISTORICAL** 最新可读历史状态 19,000/20,000，暂停，记录被选 15,000。两者连续难段 P95 为 38.07/36.86 mm，难定点 P95 为 38.29/39.63 mm。测试 P95 为 12.03/12.23 mm，各 1 硬停。低速与位置分开，P00 未重跑轨迹 | P02/03 做可达性、限幅、Servo、标定与 FK 实验。唯一根因未确认 |
| 强基线不足 | **CODE＋HISTORICAL** `ReviewTrainer.assess` 比较 checkpoints、固定验证/测试及难段。无同契约且经确认充分优化的 54 DoF B1/B1+ 证据。15k 被选分公式见实际脚本 `train.py:257–264`，用 validation/hard/hold，测试每轮都评估。旧报告只用作历史证据 | P03 出具三类传统基线和冻结协议，P09 当前条件重评 B2 |
| RSI、长时与随机化 | **CODE** `sweep.py:37–43,81–105` 滑窗、首帧精确重置与硬停后恢复。job `aligned_task.py:65–78` 固定物理校准并采样随机起点，启用了难段混采与残差速度变化项。S0 robust 启用路径不能简单套到 S0-4。当前 profile 延迟参数 0，不能作为真实通讯见证 | P07 连续只初始 reset 及在线事件，P08 实效随机化和延迟实证 |

`docs/tasks/wholebody_s04.md` 来自发布 checkout。用户列的 dated 实验文档及 JSON 对应本地历史结果包 `results/2026-11-25/s04_main_20261125/`；其截至 11,000 的摘要比 live `train/state.json` 早，后者仅按历史文件观察。`docs/tasks/marvin54_hand_centric_s0.md` 是 Marvin54 历史 guide。历史 CPU/Warp 150 帧、两姿态最大臂角差 0.00034°，是指定配置仿真证据。[归档数据及 checkpoint 哈希](evidence/P00/s04.json)

## 6. P00 未运行或未确证

尚未验证全量来源谱系、数据到达时钟的正确性、每种 profile 的模型兼容和外测精度。未在 P00 评定难姿态根因、运行 GPU training rollout、评定 DR 实效、启动硬件或确定真实控制周期。这些在 [未决问题](OPEN_ISSUES.md) 中有对应所有者、证据和依赖。
