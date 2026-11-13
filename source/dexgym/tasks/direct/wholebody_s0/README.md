# wholebody_s0 双臂双手运动数据

这套流程从采集平台的 `dex-native-v2` 录制生成无物体双臂双手 Reference Motion。任务使用双 Marvin M6 和双 Wuji Hand 2.2 的 54 DoF 模型。

## 运行

在 Ubuntu 的 DexGym 目录执行。默认使用系统 Python、ROS 2 Jazzy、已构建的遥操和采集消息接口，以及数据目录中现成的 MCAP Python 包。C++ 工具复用遥操仓库的离线 CMake 目标，需要 Eigen3 和该仓库内的 Marvin `libKine`。

```bash
cd /home/ubuntu/dexgym
bash source/dexgym/tasks/direct/wholebody_s0/run.sh 2026-11-12 4
```

可以用 `WHOLEBODY_S0_TELEOP_ROOT`、`WHOLEBODY_S0_PLATFORM_ROOT`、`WHOLEBODY_S0_EPISODES` 和 `WHOLEBODY_S0_MCAP_PATH` 指向其他本地目录。原始目录按 `manifest.yaml` 的采集开始时间与 `Asia/Shanghai` 日期筛选。

## 数据位置

| 文件 | 内容 |
| --- | --- |
| `data/raw/wholebody_s0/<日期>/<episode>` | 指向原始录制的链接 |
| `data/processed/wholebody_s0/<日期>/summary.json` | 全批质检、同步、标定和数据划分汇总 |
| `episodes/<episode>/quality.json` | 每路消息数、速率、时间戳、估计缺帧、无效值、录制器计数和文件 SHA-256 |
| `episodes/<episode>/parsed.npz` | 原始采样顺序的结构化数据，包括 MANUS 骨架、ergonomics、raw sensor、腕部位姿及各时钟 |
| `episodes/<episode>/aligned.npz` | 50 Hz 人体数据、有效掩码、两端来源帧索引和按到达时间取得的因果索引 |
| `episodes/<episode>/reference.npz` | 全段 54 DoF 参考候选、目标和实解 TCP、接触、逐帧筛选原因及入库掩码 |
| `motion_library/manifest.json` | 训练、验证、测试索引与每段 NPZ 校验和，按源 episode 划分 |
| `viewer/index.html` | 可离线打开的同步审阅页面，需保留同目录文件 |

表内未列出完整前缀的条目均位于 `data/processed/wholebody_s0/<日期>/`。要在 Ubuntu 本机通过浏览器打开，也可以在 viewer 目录运行 `/usr/bin/python3 -m http.server 8765 --bind 127.0.0.1`。

## 时间和质量约定

- 逐一读取原始消息和采集观测，核对 MCAP、ROS bag 元数据和 episode manifest 的计数。校验文件包含的 CRC、记录观察时钟和 SHA-256。录制器丢弃、记录间隔估计的缺帧、MANUS 重复源帧分别统计。
- 同步轴使用 ROS 消息头时间。采集器观察时钟、SDK 原始时间和 source sequence 保存在解析数据里。MANUS 的设备字段用于识别重复帧与回退，其数值没有参与跨设备时间换算。
- 位移线性插值，姿态使用最短弧 SLERP。MANUS 的不同源帧最大间隔为 40 ms，Tracker 为 25 ms。无效样本和长缺口留下掩码，独立保存按到达时间可获得的因果索引。因果索引允许同一源样本被多次读取。
- 腕部配对使用已录制的标定后话题。用录制快照的外参和同时间戳的原始腕部位姿独立重算。前端的人体骨架在 MANUS 掌局部坐标下显示，腕部轨迹在录制的 Tracker 坐标系下显示。

## Reference Motion

手部按遥操节点的语义节点顺序与每侧 Z 轴旋转取 21 个关键点，在录制消息的频率调用同版 Wuji SDK，再按遥操配置平滑。机械臂链接原生 `TeleopCore` 和 Marvin IK，按录制快照的映射、限位、限速和回退策略离线求解。参考以第一对有效双腕为 START，使用模型 Home 中点放置起始双腕，并保留输入的相对腕向量。帧间以录制策略的控制频率做子步重放，机器人状态由上一控制目标提供理想跟随。报告中记录 START 假设、实际锚点方式与误差。

`q_ref[T,54]` 和 `qd_ref[T,54]` 的单位分别为 rad、rad/s。顺序固定为左臂 7、右臂 7、左手 20、右手 20。每只手按拇指、食指、中指、无名指、小指排列。关节名、固件名、MuJoCo qpos/actuator 映射均由模型校验后写入 manifest。位姿数组按 `[x,y,z,qx,qy,qz,qw]` 存储，位置单位是米。

每个入库 NPZ 包含 `q_ref`、`qd_ref`、源 episode 帧号、双腕和双手输入、目标 TCP、参考腕部和十指尖位姿。初筛条件记录在 `quality_filter`，包括双侧输入有效、IK 接受、腕部误差、静态穿透、关节限位和连续长度。全段参考候选可用来检查被筛除的位置。

## 原生 MuJoCo 闭环回放

在 DexGym 根目录运行。输出路径使用新的目录。

```bash
export PYTHONPATH="$PWD/source:$PWD/source/dexgym/tasks/direct"
data="$PWD/data/processed/wholebody_s0/2026-11-12"
OPENBLAS_NUM_THREADS=1 /usr/bin/python3 -m wholebody_s0.native_replay \
  --manifest "$data/motion_library/manifest.json" \
  --output "$data/native_replay" --workers 8
```

从每段首帧的 `q_ref` 和 `qd_ref` 重置。控制器使用 MJCF 编译的关节位置伺服和力限，目标每 20 ms 更新，物理步长为 2 ms。每个物理步结束后检测接触、关节越界和求解器状态。腕部和指尖误差在 50 Hz 时间边界与同帧机器人参考位姿比较。

回放通过须完整到达参考末帧，峰值穿透不超过 3 mm，且逐帧十指尖平均位置误差的 P95 不超过 85 mm。关节超出模型限位 0.05 rad 或求解异常会结束该段。另行统计超过 0.5 mm 的接触。该判据沿用 DexGym 的原生参考审计。

`native_replay/summary.json` 包含整体、按动作类型和按划分的完成率，以及双腕和十指尖误差与接触统计。逐段结果见 `clips.csv` 和 `replay_report.json`，每段的实际状态及物理步接触深度存入 `traces/`。

PPO 和 SuperMimic 共用 `native_replay/motion_library/manifest.json`。通过的参考 NPZ 按原始 episode 的训练、验证和测试划分收录，保留原参考值与文件校验和。

```python
from dexgym.tasks.direct.wholebody_s0.library import MotionLibrary

library = MotionLibrary.open("data/processed/wholebody_s0/2026-11-12/native_replay/motion_library/manifest.json")
clip = library.load_clip(library.entries("train")[0])
q_ref, qd_ref = clip["q_ref"], clip["qd_ref"]
```

## 两日合并与慢速参考

完成各日期的质检、参考生成和验证后，在 DexGym 根目录运行以下流程。输出目录应为新目录。

```bash
export PYTHONPATH="$PWD/source:$PWD/source/dexgym/tasks/direct"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
root="$PWD/data/processed/wholebody_s0"
batch="$root/batch_01_2026-11-12_13"
python=/usr/bin/python3
"$python" -m wholebody_s0.first_wave merge --processed-root "$root" --dates 2026-11-12 2026-11-13 --output "$batch"
"$python" -m wholebody_s0.first_wave verify --output "$batch"
"$python" -m wholebody_s0.native_replay --manifest "$batch/motion_library/manifest.json" --output "$batch/native_replay_original" --workers 10
"$python" -m wholebody_s0.retime --batch "$batch" --workers 8
"$python" -m wholebody_s0.native_replay --manifest "$batch/retimed_candidates/manifest.json" --output "$batch/native_replay_retimed" --workers 10
"$python" -m wholebody_s0.first_wave finalize --output "$batch"
"$python" -m wholebody_s0.first_wave viewer --output "$batch"
```

慢速参考在通过原速回放的区间以外寻找持续触发限速的有效动作，按 2 倍或 3 倍时长对原始
MANUS 和 Tracker 数据重新取样，并以同版离线 IK 生成腕部参考。采集时钟保存在
`source_time_ns`，`time_ns` 保持 50 Hz 播放间隔。慢速片段经原生回放后只进入训练集，
并与原速库按录制检查来源区间。训练使用批目录下 `native_replay/motion_library/manifest.json`。
`viewer/index.html` 可切换原始录制和通过回放的慢速参考，并显示对应的原始录制时间。

## Isaac Lab Newton 跟踪补偿训练

使用 Ubuntu 上 DexGym 的 `scripts/supermimic/host.sh`。两个独立入口分别定义任务身份
`DexGym-WholeBody-S0-PPO-Direct-v0` 和 `DexGym-WholeBody-S0-SuperMimic-Direct-v0`。
两个入口读取同一训练划分、状态、动作和奖励，学习器分别使用 RSL-RL PPO 和 DexGym
SuperMimic。每次运行使用新的输出目录。

```bash
cd /home/ubuntu/dexgym
data="$PWD/data/processed/wholebody_s0/2026-11-12"
manifest="$data/native_replay/motion_library/manifest.json"
run="$data/training_2026-11-13"
target="/home/ubuntu/teleop_ws/src/whole_body_mujoco_driver/model/dual_arm_marvin.xml"

# 小规模核对实际 Newton 采样、训练和 checkpoint。
bash scripts/supermimic/host.sh -m dexgym.tasks.direct.wholebody_s0.ppo_task \
  --manifest "$manifest" --output "$run/doc_ppo_smoke" --num-envs 64 --control-steps 48
bash scripts/supermimic/host.sh -m dexgym.tasks.direct.wholebody_s0.supermimic_task \
  --manifest "$manifest" --output "$run/doc_supermimic_smoke" --num-envs 64 --control-steps 48 \
  --warmup 16 --wm-only-iters 2 --horizon 8 --utd 0.5 --buffer-steps 96 \
  --wm-window 6 --wm-width 512 --wm-batches 2 --wm-batch 512 --actor-batch 192

# PPO 运行 16,777,216 次真实环境转移。
bash scripts/supermimic/host.sh -m dexgym.tasks.direct.wholebody_s0.ppo_task \
  --manifest "$manifest" --output "$run/ppo_nominal" --num-envs 4096 --control-steps 4096 --seed 42
```

机器人实测关节位置和速度与当前参考组成 Actor 的 324 维观测。顺序为相对 Home 的位置、
速度、参考位置误差、参考后向差分速度、上一次投影命令的位置误差和上一次动作，每组 54 维。
训练会按关节名把库内顺序换成模型顺序，`provenance.json` 保存实际顺序。
策略输出经 tanh 限幅，手臂和手指的原始增量上限分别为 0.035、0.060 rad。位置命令投影到
关节和执行器范围及每控制步的变化区间，变化上限为 0.027、0.10 rad。每个命令执行十个
2 ms 物理步。接触穿透超过 3 mm、关节越界超过 0.05 rad 或求解器异常会终止片段。

奖励从该命令执行后的测量状态计算。腕部位置与朝向、左掌内指尖和指节、右掌内指尖和指节、
双腕相对位姿分别归一化，另有弱关节项和动作变化、投影、越界与接触惩罚。
记录 `q_ref`、投影后的命令、延迟队列实际下发的目标和测量状态各自的语义。

用 `evaluate_residual` 在验证集评估多个 checkpoint。在同一调用中重复传入
`--checkpoint` 并使用 `--profile both`。随后用 `select_checkpoint` 按完整播放数、实测覆盖和分组 P95
误差选择 checkpoint，再开始鲁棒训练。例如已在 `val_ppo_nominal` 中保存验证结果时：

```bash
chosen=$(bash scripts/supermimic/host.sh -m dexgym.tasks.direct.wholebody_s0.select_checkpoint \
  "$run/val_ppo_nominal/comparison.json" "$run/val_ppo_nominal/selection.json")
bash scripts/supermimic/host.sh -m dexgym.tasks.direct.wholebody_s0.ppo_task \
  --manifest "$manifest" --output "$run/ppo_robust" --num-envs 4096 --control-steps 4096 --seed 43 \
  --init "$chosen" --robust --transfer-model "$target"
```

SuperMimic 使用自己的名义 checkpoint。既有小规模比较的显式配置保存在作业目录的
`config.json`。正式 32 步想象作业使用下方参数。
鲁棒训练随机化位置伺服、阻尼、力限、关节摩擦、接触摩擦、测量偏置、初始状态、一步控制延迟和速度扰动。
`--transfer-model` 在 Newton 中插值训练模型与遥操模型的伺服参数及重力补偿。Actor 沿用
名义训练权重，使用新增域参数的特权网络重新初始化。训练汇总与原始曲线分别写入
`training_proof.json` 和 `metrics.jsonl`。

## 原生 MuJoCo 策略回放

以下命令在原版遥操 XML 上比较零补偿和两个 checkpoint。验证时把划分设为
`validation`，冻结 checkpoint 后将划分设为 `test`。

```bash
bash scripts/supermimic/host.sh -m dexgym.tasks.direct.wholebody_s0.evaluate_residual \
  --manifest "$manifest" --model "$target" --output "$run/native_test" \
  --split test --profile both --seed 101 --workers 8 --baseline --trace \
  --checkpoint "$run/ppo_robust/model_final.pt" \
  --checkpoint "$run/supermimic_robust/model_final.pt"
```

`comparison.json` 和各策略的 `clips.json` 保存完成率、接触、原始时间轴跟踪误差与
单独估计的腕部时滞。掌内指尖误差同时用训练参考和目标模型上同一参考关节姿态计算。
`--reference-start` 使用参考首帧的关节位置与速度开始回放，可单独检查初始状态扰动的影响。
在扰动模式下保留相同种子的其他扰动。
`evaluate_newton` 可用相同的 `--manifest`、`--output`、`--split`、
`--baseline` 和 `--checkpoint` 在训练平台复核留出集。

`export_policy` 接收 `--manifest`、`--checkpoint` 和新的 `--output`，导出
CPU TorchScript 控制器及关节名和单位。每个 20 ms 控制周期输入六个 `[batch,54]`
有限值张量，顺序见 `contract.json`。上一条命令应在允许位置范围内。
控制器输出投影后的命令和供下一个周期使用的归一化动作。采集同步、重定向、控制链路时延及
外部安全停机由调用方提供。

## 掌内指尖精度训练

`--reward-profile fingertip` 适用于已有腕部策略的 PPO 和 SuperMimic。两种方法分别
继承各自的 `--init` 权重。训练固定腕部网络，仅更新左右手残差网络。Actor 输入仍为
原有的六组关节状态。手部网络还使用从同一输入推导的掌心朝向和上一命令的投影偏差。
补偿完全由训练权重输出，手指幅度上限为 0.10 rad，命令仍使用同一位置与速度约束。

奖励基于实测状态与参考在遥操模型中的掌内指尖几何。每只手增加 4 mm 尺度的指尖误差
与最差两指惩罚。腕部、相对位姿、指节、对捏和接触安全项继续参与同一任务。
`--transfer-model` 提供奖励几何和 Newton 域随机化参数。

```bash
previous="$data/training_2026-11-13"
run="$data/fingertip_precision_2026-11-13"
bash scripts/supermimic/host.sh -m dexgym.tasks.direct.wholebody_s0.ppo_task \
  --manifest "$manifest" --output "$run/ppo_precision" --num-envs 4096 --control-steps 4096 \
  --robust --reward-profile fingertip --transfer-model "$target" --seed 49 \
  --checkpoints 8 --init "$previous/ppo_robust/model_final.pt"

bash scripts/supermimic/host.sh -m dexgym.tasks.direct.wholebody_s0.supermimic_task \
  --manifest "$manifest" --output "$run/supermimic_h32_5000" --num-envs 4096 --iterations 5000 \
  --robust --reward-profile fingertip --transfer-model "$target" --seed 49 \
  --checkpoints 20 --init "$previous/supermimic_robust/model_final.pt" --warmup 64 \
  --wm-only-iters 500 --horizon 32 --wm-width 2048 --wm-window 8 --wm-batches 10 --wm-batch 2048 \
  --actor-batch 1024 --actor-lr 0.00008 --actor-bootstrap-weight 1 --critic-epochs 5 --utd 0.4 --buffer-steps 192
```

4096 环境下，1000 步 GPU replay 在当前工作站触发显存不足。192 步 replay 已通过完整
世界模型、32 步 Actor 和 Critic 更新烟测。5000 轮指 learner 迭代数，其中前 500 轮
训练世界模型，之后每轮更新 Actor。
验证应将 `--baseline`、对应的上一版策略与新策略放在同一命令里，比较
`target_local_tip_mm` 和 `target_local_tip_mm_fingers`，同时查看完整回放数和腕部误差。
`export_policy` 沿用相同的六个输入张量和两个输出。

## 人体意图优先训练

`--reward-profile human_intent` 使用采集的 Tracker 和 MANUS 作为主跟踪目标。腕部使用
`target_tcp_world` 中标定后的 TCP 目标。手部沿用已评估的骨段换算，将 MANUS 的骨段方向
投到 Wuji 掌坐标系，并按目标 MJCF 的指根和骨长重建指节与指尖。原始目标由采集数据生成，
`q_ref` 仍提供机器人参考和有界残差的中心。

两种学习器使用同一奖励，按双腕、左右手和双腕相对位姿分组计算人体目标误差，另加权重为
0.35 的参考执行误差、0.04 的动作及命令平滑项和 0.012 的残差幅度项。接触、越界和投影
另有惩罚。执行误差基于实际状态，包含左右腕、掌内手指、相对位姿和分组关节误差。
手臂和手指均参与学习。动作范围和每步命令变化仍由原来的投影处理。

Actor 读取原有 324 维关节观测，并增加 48 维人体目标相对 `q_ref` 几何的偏移。
可用早期指尖策略初始化，前提是其训练录制属于新库的训练集，并且模型与动作空间兼容。
新输入的权重从零开始。原始策略、PPO 第 1000 轮和 SuperMimic 第 3000 轮使用新库
留出集比较时，给 `evaluate_residual` 加 `--intent-metrics --compare-earlier-library`。

`export_policy` 为该配置额外导出人体和参考的腕位置、旋转矩阵及掌内指尖位置输入。
顺序、形状和换算版本见 `contract.json`。

## 重定向位姿跟踪

训练使用 `dual_arm_marvin_wholebody_s0.xml` 及同名 Newton USD。原生回放使用遥操工作区中的 `dual_arm_marvin.xml`。PPO 入口需要已验证的 RSL-RL 版本 `rsl-rl-lib==5.5.1`，可安装到运行 `scripts/supermimic/host.sh` 时可见的 Python 环境。原始录制、处理后的轨迹库和 checkpoint 由运行端提供。

`--reward-profile retarget_pose` 在遥操 MJCF 中对参考关节序列 `q_ref` 做 FK，得到三类目标。
双腕用世界坐标下 `tcp_L` 和 `tcp_R` 的位置及朝向。每个指尖 site 的位置及朝向先变换到
同侧 TCP 坐标系。双腕相对位姿使用右腕在左腕坐标系中的位姿。三项都比较同一控制周期的
实测状态和参考，按腕部、左右手和双腕相对位姿分组计分。

每只手的分数包含五指位置、五指朝向和各自误差最大的两指。位置尺度为腕部 16 mm、
指尖 4 mm、相对腕 25 mm，旋转尺度用旋转弦长 `2 sin(θ/2)` 表达。关节跟踪、动作变化、
残差幅度、投影和接触是辅助项。手臂和手指均训练有界残差，输入仍为原有的六组关节观测。

原生回放保存 `target_wrist_mm/deg`、`wrist_local_tip_mm/deg`、`target_relative_mm/deg`。
指尖有逐手、逐指标记。角度统计使用 SO(3) 最短旋转角，单位为度。回放命令仍以 50 Hz
更新，物理步长为 2 ms。

```bash
manifest="$PWD/data/processed/wholebody_s0/batch_01_2026-11-12_13/native_replay/motion_library/manifest.json"
model="$HOME/teleop_ws/src/whole_body_mujoco_driver/model/dual_arm_marvin.xml"
bash scripts/supermimic/host.sh -m dexgym.tasks.direct.wholebody_s0.ppo_task     --manifest "$manifest" --transfer-model "$model" --reward-profile retarget_pose --robust     --num-envs 2048 --iterations 1000 --output /path/to/new_ppo_run
bash scripts/supermimic/host.sh -m dexgym.tasks.direct.wholebody_s0.supermimic_task     --manifest "$manifest" --transfer-model "$model" --reward-profile retarget_pose --robust     --num-envs 2048 --iterations 3000 --horizon 16 --buffer-steps 500 --wm-only-iters 20     --output /path/to/new_supermimic_run
```

用 `--init` 可以加载同任务的兼容权重。参考清单须带有筛选过的 MuJoCo 回放记录与相符的机器人模型哈希。

## 腕内指尖位置精调

`--reward-profile tip_precision` 保留腕部位姿奖励，提高腕内指尖位置的平均和尾部误差权重。手部策略内部从关节输入计算腕内指尖位置误差和参考位移，输出有界关节残差。控制器沿用原有六组输入和命令约束。

首次运行用 `--arm-init` 指向已训练的 `retarget_pose` checkpoint，用 `--init` 指向同方法的 `fingertip` checkpoint。训练冻结腕部网络，手部继续学习。同 profile 的后续运行可只传 `--init`。PPO 可用 `--native-robot-usd` 选择由遥操 MJCF 转换的 Newton USD，微调指尖反馈网络。
