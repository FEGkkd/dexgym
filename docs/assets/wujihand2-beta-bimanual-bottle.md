# WujiHand2 Beta 双手与螺纹瓶基础场景

本模块是独立 Isaac Lab 场景，不注册 RL 任务，也不定义策略 observation、reward 或训练配置。现有 `assets/wuji_hand.py` 及单手任务仍使用第一代 Wuji Hand 官方 USD；新模块只通过 `assets/wuji_hand2_beta.py` 接入 Hand 2 Beta。

## 来源、版本与硬件边界

- 官方资源：[wuji-description `v2026.9.11`](https://github.com/wuji-technology/wuji-description/releases/tag/v2026.9.11)，commit `c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95`，官方仓库 [MIT 许可证](https://github.com/wuji-technology/wuji-description/blob/v2026.9.11/LICENSE)。运行时读取外部 `WUJI_DESCRIPTION_ROOT`，核对 commit（若是 Git checkout）、每侧顶层 USD SHA-256，以及该侧所有原生 USD 层、mesh、配套 MJCF 的路径与内容 SHA-256 清单；不复制手模型到 DexGym。
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
- `Wrist6DoFController` 对自由腕施加世界坐标的有限力和力矩。单次位置目标增量上限 0.01 m、旋转增量 0.08 rad，目标半径 0.6 m、目标姿态离初始姿态最多 1.2 rad、最低高度 0.1 m；力 40 N、力矩 2 N·m。线速 0.5 m/s、角速 3 rad/s 是 PD 软阻尼阈值。USD 上另配置 PhysX rigid body 最大线速 1 m/s、最大角速 360°/s；目前自由 articulation 的根角速度仍可能超过此值，因此**不能将该属性当作已经验证的硬限幅**。控制器可接收碰撞反作用；实际稳定性应看 GPU 验证数据。
- Isaac Lab 3.0 root pose 数组及本模块接口使用 `xyzw` quaternion；USD `Gf.Quatf` 在作者侧用实部加虚部 (`wxyz` 概念顺序)。外部手势或策略接入必须显式转换。官方 USD 没有独立 `palm` 刚体，掌部几何附着在 `l_wrist` / `r_wrist`；当前 `palm_body` 是腕部刚体别名，若需要掌心控制点须另行标定其相对腕部的固定变换。瓶子 `Bottle` 原点是瓶身质心，螺纹轴为其局部 +Z。
- 手部官方几何、质量、惯量、关节轴、原有 collision mesh 和 USD finger drive 未改写；只在场景层禁用固定根关节、设置 articulation root、启用 contact sensor 并绑定可配置的 PhysX 接触材质。每侧应独立检查关节和接触。

## 瓶与螺纹

`BottleConfig` 中全部尺寸是未标定的**工程样例**，单位 SI。默认瓶身半径 35 mm、高 200 mm，瓶颈外半径 12 mm，盖外半径 17 mm、高 25 mm，右旋螺距 3 mm、三圈、径向间隙 0.5 mm、初始啮合 9 mm，质量瓶身 30 g、盖 6 g。三圈轴向行程为 9 mm，退出后盖底缘可越过瓶颈顶端。其余可配置项包括摩擦、松动阈值、运行阻力和有限约束力。真实瓶须测量 CAD/扫描、质量、惯量、螺纹牙型和扭矩后替换。

另提供 `BottleConfig.pco1881_reference()` 与 CLI `--bottle_profile pco1881_reference`，用于更接近常见 28 mm 饮料瓶的**公开规格参考样例**。它不代表用户手中的某一款瓶子，尤其不代表经过实测的开盖力学。来源与工程假设分开如下：

| 参数 | 参考样例 | 来源与边界 |
| --- | --- | --- |
| 瓶口螺纹 | 单头右旋，螺距 2.7 mm，650°（1.806 圈） | [BERICAP PCO1881 产品资料](https://www.phperu.com/_files/ugd/2c48bc_53e4d3ea1ed3488a88d72c82416b8514.pdf?index=true)；[ISBT 螺纹规格目录](https://www.isbt.com/resources/isbt-threadspecs)列有 PCO1881 |
| 瓶口直径 | 螺纹牙顶 27.4 mm，牙底 24.94 mm | [NEXPET PCO1881 瓶口图纸](https://nexpet.co/resources/028_PCO1881_125.0_24.8_3.43_35.5.pdf)；目前用等高分段 capsule 表示牙型，并未复刻图纸的全部截面和公差 |
| 瓶盖外径、高、质量 | 29.6 mm、15.65 mm、2.34 g | [Delta El Nile PCO1881 瓶盖规格](https://www.deltaelnile.com/en/closure-pco-1881.html)；属于该供应商瓶盖，不保证与上述瓶口供应商形成实际配套 |
| 开启扭矩量级 | 供应商建议 5–14 in-lb（约 0.565–1.582 N·m）；配置名义紧盖阈值 0.90 N·m | [Delta El Nile 技术页](https://www.deltaelnile.com/en/sku/den-clo-pco-1881.html)。0.90 N·m 只是落在建议范围内的选点，未经本地实物测量 |
| 瓶身与内容物 | 半径 32.5 mm、高 190 mm、总质量 520 g | 工程近似：500 mL 水加约 20 g 塑料；不是与该盖配套的瓶身 CAD，未模拟液体晃动 |
| 径向间隙与阻力曲线 | 0.2 mm；松盖阈值 0.05 N·m，运行阻力 0.12→0.02 N·m | 工程假设，未找到该组合的实测值；没有密封垫预紧、残留内压或防盗环断裂模型 |

参考样例资产写入 `assets/bottle/generated/pco1881_reference/`，避免覆盖原工程样例。此配置的几何与螺距可由静态测试检查；**其开启阻力尚未通过物理验收**：在当前 120 步 GPU fixture 中，瓶盖质量 2.34 g、显式力矩更新与 PhysX 步长的组合使施加 0.6 N·m（低于 0.90 N·m 名义阈值）时瓶盖仍转过约 4.33 rad。曾测试 D6 角度驱动；它使 0.6 N·m 停住，但连 1.2 N·m 和 3.0 N·m 也几乎不转，故没有保留该方案。不能仅凭参数写了 0.90 N·m 就宣称仿真开启扭矩接近实物。未来需要实测目标瓶扭矩曲线，并采用通过低于/高于阈值双侧 GPU 测试的稳定接触/关节阻力实现。参考瓶口约 4.88 mm 的螺纹轴向行程小于盖与瓶颈的初始重叠，因此即使螺纹退出，盖可能仍套在瓶口；自然脱落或抓取提盖尚未验收。

`generate_usd.py` 用 OpenUSD / `pxr` 从配置生成独立的 `bottle.usda`、`cap.usda`、`parameters.json`。瓶身和盖的可见圆面是默认 64 段的 `UsdGeom.Mesh`，不带碰撞；隐藏的 cylinder、cube 和 capsule 负责碰撞，每个资产少于 60 个简单 collider。瓶颈外与盖内均有分段螺旋凸纹，没有高密度动态三角网格碰撞。瓶身和盖各有独立自由刚体、质量、惯量与塑料接触材质。生成文件随源码提交并可重建；没有第三方瓶模型，生成器依赖项目的 Isaac Sim/OpenUSD，不需要 MuJoCo。

瓶和盖间用 USD D6 joint 锁定横向两轴及倾斜两轴，保留瓶身局部螺纹轴的平移和转动，并限定轴向行程。当前 PhysX 不依赖任何假定存在的 Screw Joint API。`ThreadCoupler` 在每步由瓶与盖的相对姿态求连续角 `θ`，在瓶身局部轴计算 `z_target = z_initial + sign·pitch·θ/(2π)`，再用有限、等大反向力及力矩作顺从耦合。阻尼使用扣除了瓶身角速度引起的坐标搬运速度；有状态的粘着/滑动模型停转后重新锚定。松动阈值从完全啮合的 0.020 N·m 随轴向退出单调降至 0.004 N·m；运行阻力从 0.006 降至 0.001 N·m，指数和端点均可配置。这是按啮合重叠量定义的**工程阻力模型**，不是经实际瓶盖预紧力测试标定的扭矩曲线。瓶盖运动始终来自 PhysX 刚体积分和约束，没有逐帧位姿瞬移。脚本测试力矩只作为确定性外力，不代表抓握或 RL 成功。

默认保持三圈末端的有限止挡，以便重复进行已啮合螺纹测试。启用 `--allow_detachment` 后，在三圈行程结束、局部轴向误差和剩余啮合量满足阈值时，运行中禁用该环境的 D6 joint 并停止耦合力；盖子成为自由刚体。双环境 PhysX fixture 在第 510 步释放环境 0，施加 0.05 N 横向测试力后盖子离瓶 0.97 m 并落地，环境 1 保持连接。没有横向外力或抓握动作时，脱离的盖也可能落回瓶颈附近；它不会被脚本瞬移到地上。

当前限制：退出后支持单向释放，**不支持重新啮合**；分段凸纹与简化物理等效件尚未按真实瓶标定。低于松动阈值允许有限弹性角位移，不应解释为零位移。每步 Python wrench 耦合在大量并行环境的吞吐和长时间接触稳定性仍需测量。自由瓶会因重力下落，独立螺纹夹具只在瓶落地前的 120 步窗口验收；双手稳定抓握瓶体未实现。

## 启动与验收

先将官方仓库 checkout 到上述 commit，设 `WUJI_DESCRIPTION_ROOT`、`ISAACLAB_ROOT`、`ISAACLAB_PYTHON`。若目标工作站把 Isaac Lab 以源码方式安装，确保 `isaaclab` 在 Python 路径上。典型命令：

```bash
export WUJI_DESCRIPTION_ROOT=/path/to/wuji-description-v2026.9.11
export ISAACLAB_ROOT=/path/to/IsaacLab
export ISAACLAB_PYTHON=/path/to/isaacsim/python
./scripts/run_isaaclab.sh scripts/environments/inspect_wuji_hand2_beta.py
./scripts/run_isaaclab.sh scripts/environments/smoke_bimanual_scene.py --headless --num_envs 1 --steps 120
./scripts/run_isaaclab.sh scripts/environments/smoke_bimanual_scene.py --headless --num_envs 2 --steps 120 --bottle_tilt_deg 30 --test_torque 0.04
./scripts/run_isaaclab.sh scripts/environments/smoke_bimanual_scene.py --headless --num_envs 1 --steps 600 --bottle_center_z 0.1 --test_torque 0.04
./scripts/run_isaaclab.sh scripts/environments/validate_wrist_6dof.py
./scripts/run_isaaclab.sh scripts/environments/validate_bottle_thread.py
./scripts/run_isaaclab.sh scripts/environments/validate_bottle_detachment.py
./scripts/run_isaaclab.sh scripts/environments/validate_hand2_contact.py
./scripts/run_isaaclab.sh scripts/environments/validate_hand2_fingers.py
./scripts/run_isaaclab.sh scripts/environments/smoke_bimanual_scene.py --headless --bottle_profile pco1881_reference --bottle_center_z 0.095 --fixture_hold_bottle --steps 120 --test_torque 0.6
```

GUI：

```bash
./scripts/run_isaaclab.sh scripts/environments/smoke_bimanual_scene.py --viz kit --num_envs 1 --steps 100000 --bottle_center_z 0.1 --test_torque 0.04 --torque_cycle_steps 600 --contact_debug
```

GUI 创建左右腕（掌部几何使用同一刚体）与瓶局部轴的 frame marker，并显示当前角度、轴向位移及约束误差；结束时另由 `BIMANUAL_SCENE_RESULT` JSON 输出。当前 Isaac Lab 版本需要显式传 `--viz kit`，并使 `isaaclab_visualizers` 可导入；若没有该可选包，脚本只显示场景和状态文字，不显示 marker。`--left_fingers` / `--right_fingers` 接收各 20 个关节位置目标。上述脚本每次从相同初始条件复现；验证器要求明确 `*_PASS` 输出，Isaac Sim 进程退出码独自不足以判定成功。

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
