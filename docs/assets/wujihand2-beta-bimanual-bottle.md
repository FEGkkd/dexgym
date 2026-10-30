# WujiHand2 Beta 双手与螺纹瓶基础场景

本模块是独立 Isaac Lab 场景，不注册 RL 任务，也不定义策略 observation、reward 或训练配置。现有 `assets/wuji_hand.py` 及单手任务仍使用第一代 Wuji Hand 官方 USD；新模块只通过 `assets/wuji_hand2_beta.py` 接入 Hand 2 Beta。

## 来源、版本与硬件边界

- 官方资源：[wuji-description `v2026.9.11`](https://github.com/wuji-technology/wuji-description/releases/tag/v2026.9.11)，commit `c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95`，官方仓库 [MIT 许可证](https://github.com/wuji-technology/wuji-description/blob/v2026.9.11/LICENSE)。运行时读取外部 `WUJI_DESCRIPTION_ROOT`，核对 commit（若是 Git checkout）和每侧 USD SHA-256；不复制手模型到 DexGym。
- 默认选用无独立指尖传感框架的 `beta1`，可通过 `--variant beta2` 选择 Beta 2。真实设备只确认“WujiHand2 Beta、无触觉”，尚无序列号或厂商 BOM 能判定其几何版本。**Beta 1 是候选，不是实物同一性结论。** Beta 2 有额外指尖刚体，但场景也不需要触觉信号。
- 参考工程 `FEGkkd/whole-body-teleop` 的 `codex/whole-body-sim-phase1` 分支使用 Beta 2 `with_mount` MJCF，在 Marvin M6 上固定安装。该工程的 MuJoCo 验证不证明本场景的 PhysX 结果，也不证明实物是 Beta 2。参考工程 Git HTTP 匿名读取受限；本次通过 GitHub 连接器读取了 `model/README.md`、`model/VALIDATION.md` 和 `model_joints.hpp`，没有修改参考仓库。
- 独立自由腕使用官方 `body/usd/{left,right}` 原生 USD，不使用 `with_mount` 文件。官方 mount 的 MJCF 质量为 0.069 kg；若接机械臂需重新选择装配资产及基座惯量。

### Beta 1 / Beta 2 审计结果

运行 `scripts/environments/inspect_wuji_hand2_beta.py` 可重算下表。数值来自官方同一 tag 的左右 MJCF、STL 和原生 USD，未包含外接 mount。

| 项目 | Beta 1 | Beta 2 |
| --- | --- | --- |
| 每侧主动关节 | 20 | 20，名称、轴、范围、MJCF effort 与 Beta 1 相同 |
| 每侧 MJCF 刚体 | 21 | 26，新增五个 `*_tip_sensor_frame` |
| 五个指尖查询刚体 | `*_distal` | `*_tip_sensor_frame` |
| 每侧 MJCF 总质量 | 0.6207 kg | 0.6228 kg |
| 非拇指 distal 质量 | 各 0.0078 kg | 各 0.005 kg |
| 拇指 distal 质量 | 0.0159 kg | 0.0109 kg |
| STL 数量 | 27 | 32；五个 sensor mesh 新增，另外 19 个共有 mesh 的 SHA 改变 |
| MJCF 碰撞排除 | 10 对 | 10 对，列表相同；共有碰撞 mesh 内容可能改变 |

官方 MJCF `l_THJ0` 至 `l_LFJ3`、`r_THJ0` 至 `r_LFJ3` 是执行器名，**不是** USD 关节名。规范顺序为 thumb、index、middle、ring、pinky，每指 flex/abduction 或 cmc/mcp、pip、dip 四关节。例如 `l_THJ0 → l_thumb_cmc_flex`，`r_LFJ3 → r_pinky_dip`。全部 40 个映射、上下限及 MJCF effort 由 adapter 校验并导出。官方 USD 的 20 个 finger drive `maxForce` 均为 2 N·m，而 MJCF 中有 0.2、0.3、0.6、2.0 N·m；本场景保留 USD 值，两者均在 metadata 中暴露，硬件 effort 限制尚待厂商确认。左右侧各自加载官方对应文件，不做几何镜像。五个指尖刚体的位姿来自 Isaac Lab articulation body pose，均保留官方碰撞；本阶段场景实例化了左右手净接触传感器，并在食指附近放置瓶子验证接触。尚未提供每指独立的接触力矩阵。

## 坐标、控制与物理边界

- 世界为 Z-up、米、弧度、秒。左手初始腕约在世界 `(-0.30, 0, 0.62)`，右手在 `(0.30, 0, 0.62)`，面向瓶子。场景创建时禁用官方 USD 的 world fixed joint，把 `ArticulationRootAPI` 放到 `l_wrist` / `r_wrist`；每只手是独立自由 articulation，不按物理步改写 root pose。
- `HandCommand` 每侧 26 个值：世界坐标的 `Δx,Δy,Δz`，当前目标腕坐标的旋转向量 `Δrx,Δry,Δrz`，再按上述 canonical order 给 20 个手指位置目标。`BimanualCommand.from_flat()` 定义左右合计 52 值的未来接口；本阶段 CLI 逐侧输入，不注册 52 维 PPO action space。
- `Wrist6DoFController` 对自由腕施加世界坐标的有限力和力矩。单次位置目标增量上限 0.01 m、旋转增量 0.08 rad，目标半径 0.6 m、最低高度 0.1 m；力 40 N、力矩 2 N·m。线速 0.5 m/s、角速 3 rad/s 是 PD 软阻尼阈值。USD 上另配置 PhysX rigid body 最大线速 1 m/s、最大角速 360°/s；目前自由 articulation 的根角速度仍可能超过此值，因此**不能将该属性当作已经验证的硬限幅**。控制器可接收碰撞反作用；实际稳定性应看 GPU 验证数据。
- Isaac Lab 3.0 root pose 数组及本模块接口使用 `xyzw` quaternion；USD `Gf.Quatf` 在作者侧用实部加虚部 (`wxyz` 概念顺序)。外部手势或策略接入必须显式转换。官方 USD 没有独立 `palm` 刚体，掌部几何附着在 `l_wrist` / `r_wrist`；当前 `palm_body` 是腕部刚体别名，若需要掌心控制点须另行标定其相对腕部的固定变换。瓶子 `Bottle` 原点是瓶身质心，螺纹轴为其局部 +Z。
- 手部官方几何、质量、惯量、关节轴、原有 collision mesh 和 USD finger drive 未改写；只在场景层禁用固定根关节、设置 articulation root、启用 contact sensor 并绑定可配置的 PhysX 接触材质。每侧应独立检查关节和接触。

## 瓶与螺纹

`BottleConfig` 中全部尺寸是未标定的**工程样例**，单位 SI。默认瓶身半径 35 mm、高 200 mm，瓶颈外半径 12 mm，盖外半径 17 mm、高 25 mm，右旋螺距 3 mm、三圈、径向间隙 0.5 mm、初始啮合 18 mm，质量瓶身 30 g、盖 6 g。其余可配置项包括摩擦、松动阈值、运行阻力和有限约束力。真实瓶须测量 CAD/扫描、质量、惯量、螺纹牙型和扭矩后替换。

`generate_usd.py` 用 OpenUSD / `pxr` 从配置生成独立的 `bottle.usda`、`cap.usda`、`parameters.json`。视觉及碰撞由 cylinder、cube、capsule 组成，每个资产少于 60 个简单 collider；瓶颈外与盖内均有分段螺旋凸纹，没有高密度动态三角网格。瓶身和盖各有独立自由刚体、质量、惯量与塑料接触材质。生成文件约 44 KiB，随源码提交并可重建；没有第三方瓶模型，生成器依赖项目的 Isaac Sim/OpenUSD，不需要 MuJoCo。

瓶和盖间用 USD D6 joint 锁定横向两轴及倾斜两轴，保留瓶身局部螺纹轴的平移和转动，并限定轴向行程。当前 PhysX 不依赖任何假定存在的 Screw Joint API。`ThreadCoupler` 在每步由瓶与盖的相对姿态求连续角 `θ`，在瓶身局部轴计算 `z_target = z_initial + sign·pitch·θ/(2π)`，再用有限、等大反向力及力矩作顺从耦合。有限弹性静摩擦提供 `breakaway_torque`，速度阻力提供 `running_resistance_torque`。瓶盖运动始终来自 PhysX 刚体积分和约束，没有逐帧位姿瞬移。脚本测试力矩只作为确定性外力，不代表抓握或 RL 成功。

当前限制：螺纹只支持预啮合行程，**不支持完全拧脱及重新啮合**；分段凸纹与简化物理等效件尚未按真实瓶标定。低于松动阈值允许有限弹性角位移，不应解释为零位移。每步 Python wrench 耦合在大量并行环境的吞吐和长时间接触稳定性仍需测量。自由瓶会因重力下落，独立螺纹夹具只在瓶落地前的 120 步窗口验收；双手稳定抓握瓶体未实现。

## 启动与验收

先将官方仓库 checkout 到上述 commit，设 `WUJI_DESCRIPTION_ROOT`、`ISAACLAB_ROOT`、`ISAACLAB_PYTHON`。若目标工作站把 Isaac Lab 以源码方式安装，确保 `isaaclab` 在 Python 路径上。典型命令：

```bash
export WUJI_DESCRIPTION_ROOT=/path/to/wuji-description-v2026.9.11
export ISAACLAB_ROOT=/path/to/IsaacLab
export ISAACLAB_PYTHON=/path/to/isaacsim/python
./scripts/run_isaaclab.sh scripts/environments/inspect_wuji_hand2_beta.py
./scripts/run_isaaclab.sh scripts/environments/smoke_bimanual_scene.py --headless --num_envs 1 --steps 120
./scripts/run_isaaclab.sh scripts/environments/smoke_bimanual_scene.py --headless --num_envs 2 --steps 120 --bottle_tilt_deg 30 --test_torque 0.04
./scripts/run_isaaclab.sh scripts/environments/validate_wrist_6dof.py
./scripts/run_isaaclab.sh scripts/environments/validate_bottle_thread.py
./scripts/run_isaaclab.sh scripts/environments/validate_hand2_contact.py
```

GUI：

```bash
./scripts/run_isaaclab.sh scripts/environments/smoke_bimanual_scene.py --num_envs 1 --steps 300 --left_xyz 0.01 0 0 --right_rot 0 0 0.08 --left_grip 0.5 --contact_debug
```

GUI 创建左右腕（掌部几何使用同一刚体）与瓶局部轴的 frame marker，并显示当前角度、轴向位移及约束误差；结束时另由 `BIMANUAL_SCENE_RESULT` JSON 输出。`--left_fingers` / `--right_fingers` 接收各 20 个关节位置目标。GUI 窗口尚未在有显示器的工作站验收。上述脚本每次从相同初始条件复现；验证器要求明确 `*_PASS` 输出，Isaac Sim 进程退出码独自不足以判定成功。

无需 GPU 的契约测试：

```bash
PYTHONPATH=source python -m unittest discover -s tests -p 'test_bottle_thread.py' -v
PYTHONPATH=source python -m unittest discover -s tests -p 'test_wrist_6dof.py' -v
PYTHONPATH=source python -m unittest discover -s tests -p 'test_bimanual_commands.py' -v
```

`test_bottle_usd.py` 和官方 USD 审计需要 Isaac Sim 的 `pxr` Python 包，但不需要 GPU simulation。旧单手任务依然需要其旧版官方路径 `hand/body/usd/right/wujihand.usd`；若官方 Beta tag 不包含旧文件，则旧任务验证应使用含此路径的官方 checkout，避免混淆两种资产来源。

## 代码边界

| 路径 | 职责 |
| --- | --- |
| `assets/wuji_hand2_beta.py` | 官方 USD 解析、版本/SHA、左右关节映射、自由根场景 override |
| `assets/wrist_6dof.py` | 与任务无关的 6 轴有限 wrench 跟踪 |
| `assets/bottle/` | 米制工程配置、USD 生成、D6 joint 和螺纹力学 |
| `scenes/bimanual_bottle.py` | 双手、瓶身、盖、地面、灯、接触 sensor 场景配置 |
| `scenes/commands.py` | 未来可复用的 26+26 控制数据结构 |
| `scripts/environments/` | 独立检查、CLI 调试、GPU 验收，不进入 train/play 流程 |
