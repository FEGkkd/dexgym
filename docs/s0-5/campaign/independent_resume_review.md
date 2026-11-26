# S0-5 Campaign 恢复阶段独立复核

复核从 Campaign 启动提交 `999782a` 开始，核对固定 Teleop Git 对象 `6d0b0b5b2601618f1b0052d93ef2403eea9b064a`、本地测试和主代理提供的 Ubuntu 运行证据。我没有参与受审模块的原始实现。Ubuntu 统计、动态库定位和退出状态来自主代理执行记录。本地 pytest 及固定版本源代码由我独立复核。[启动复核](independent_launch_review.md) · [启动证据索引](initial_run_evidence.json)

## G-A 名义参考与目标覆盖

| 核验项 | 观察及适用范围 |
|---|---|
| Native SDK 身份 | Ubuntu 隔离 SDK 目录、Teleop 软件包及安装目录的 `ccs_m6_40.MvKDCfg` 一致，SHA-256 为 `ce20b1a80974c3ed58cdac4973152625096c8ad93a779e5fd4108c26ce19bd52`。`ldd` 指向预期的 vendor v1 库。桥接按左、右臂分别读取配置，SDK 内部单位为毫米和角度，桥接入口为米和弧度。 |
| `V1-A-ik-probe-03` | 退出 0。SDK FK→IK 每侧 8/8，通过 home 和逐轴脉冲正控。原 M1 映射样本左 8/8，右 0/8。右侧扣除工具与肩偏置后的目标半径为 0.73568–0.77548 m，SDK 外半径为 0.60208 m。该批右侧正式目标与 SDK 几何相差 133.60–173.40 mm。一次标准 workspace 投影后的直接 IK 也未产生成功命令。该批支持的结论限于指定腕姿和 SDK 包络。 |
| `V1-A-train-roots-05` | 退出 0。24 个 Train 根各取首分区前 120 拍，总计划 2,880 拍。HMB 联合有效 2,774 拍，实际 mapped 2,768 拍。记录左 IK 失败 2,092 次、右 2,689 次，双侧合法名义 0 拍，合法五帧历史 0。逐指独立必要界以 10 mm 排除 406 拍。计数 SHA-256 为 `8dd6b6d82bee03402333f83443328dadb3a88714e84f73272de1fbfe6e75f71b`。首分区样本用于故障定位，整库与 Validation 支持仍未测得。 |
| M1 连续性 | 启动样本观察到硬阈值切换额外世界指尖步长约 17.77 mm，超过冻结最大 15 mm，仍需进行独立连续性复验。按批准的顺序仅使用 Train 对 `[18,22]`、`[17,23]`、`[16,24]` mm 平滑候选做成对验证，对照原阈值及 M0，复核活跃捏合、局部手型、跨手、相对腕及指节长度。先锁定通过的最窄候选，再在原 Validation 独立复验一次。平滑仅改变近捏合手型规则，对右腕位置或腕姿的可达性问题需要独立诊断。 |

右腕半径与十指径向必要界对应不同连杆终点和用途。SDK home/pulse 正控与当前双侧失败分离了后端可用性和目标/命令链路问题。正式 G-A 仍需满足全量计数守恒、双 split 复验、合法名义覆盖、严格因果历史、见证和共同改善域支持。`STATIC_S5`、`DYNAMIC_D10`、共同改善域及正式评分分母均未冻结。[V1-A 实施规格](../specs/V1-A_offline_rmb.md) · [RMB 契约](../specs/V2_rmb_contract.md)

当前离线 `Nominal.step()` 同时要求四组有效及全身路径合法才更新 `self.last`。一个世代内没有新合法双侧命令时，每侧 SDK 解算继续使用初始合法姿态作为种子，即使该侧上一拍已取得独立解。逐侧失败数依赖这个状态机。后续诊断须记录单侧是否取得合法解、是否实际接受和提交、以及全身命令能否资格化，并保持持有、回退及局部解的原因分母。[离线名义状态机](../../../source/dexgym/tasks/direct/wholebody_s05/v1/rmb/nominal.py)

### 完整 Train 的原 M1 腕目标上界

`V1-A-train-radial-07` 退出 0。独立复核读取报告原始文件，SHA-256 为 `0658b1e8938b6402da5e001091321350a3a3655afd1718c253764759f084f13c`，逐分区复算 24 个 Train 根、228 分区、86,338 计划拍与 85,382 联合有效拍。按照冻结输入掩码进入探针的 85,243 拍均获得双侧 native workspace 响应。左腕排除 6,169 拍，右腕排除 61,348 拍，双侧同时超界 3,550 拍，并集为 63,967 拍。将 139 个未进入探针的联合有效拍全部按可能合法计，Train 双侧精确直接 IK 覆盖率仍至多为 **21,415 / 85,382 = 25.0814%**。24 个母源中无一达到 85% 的必要上界，最高为 81.3985%。[上界审计源码](../../../source/dexgym/tasks/direct/wholebody_s05/v1/rmb/audit_nominal_workspace.py)

受审桥接在 native base 系计算未投影请求的腕心半径 `||p_TCP − R_TCP·t_tool − p_shoulder||`，只在超过同配置两段有限连杆之和 `0.6020794064 m` 再加 `1 µm` 时作严格排除。该 epsilon 向保留样本的方向扩大上界。夹带实际控制器的 0.5 mm reserve 或 5 mm 平滑过渡会缩小可控半径，因此审计未用它们 For exclusion. 每侧 15 个原生 FK 正控的 MuJoCo 位置差小于 0.215 µm，姿态差小于 0.93 µrad。两个独立 Windows 统计正负控通过（0.77 s，退出 0）。该约束是给定腕姿经 SDK 适配后的姿态可达必要条件。P02 指尖必要界、实际 Servo 动态表现、硬件资格及指尖评分需分别验证。[Native CCS 几何](https://github.com/FEGkkd/whole-body-teleop/blob/6d0b0b5b2601618f1b0052d93ef2403eea9b064a/src/whole_body_teleop/src/marvin_kine_sdk.cpp#L425) · [P02 指尖必要界](../reports/P02_review_refinement.md)

精确腕姿超出本上界时，换 seed、增加因果子拍、START 锚定和 workspace 投影均无法让该原目标精确可达。平滑捏合也不会移动原 M1 腕目标。后续因果 waypoint 或 START 锚定方案可以独立证明命令可执行，并长期保留原 target、实现目标、投影差、硬安全和实际反馈。将这类命令计入原确切目标规格下的正式名义覆盖涉及资格语义变更，需要有限 DCR 和项目用户授权。该 DCR 如获批准，应明确保留冻结的 M1 独立评分目标、计划分母、85% 数值门、全套安全门、直接 IK 失败审计和残差能力域重验。在此之前 G-A 及其训练依赖继续阻塞。[冻结覆盖标准](../specs/V1-A_offline_rmb.md) · [冻结目标契约](../specs/V2_rmb_contract.md)

## 固定版本的 Teleop 恢复路径

- Coordinator launch 先读 `real.yaml` 再读 MuJoCo `sim.yaml`。该 overlay 选择 `manimux_analytic_ik_position`、Marvin、上一条已接受命令作为种子、START 锁存的 NSP 平面和独立双臂执行。运行参数为 250 Hz、步长 dt 上限 16 ms、平移限速 0.2592 m/s、旋转限速 1.085734 rad/s，重试顺序 `[1.0,0.5,0.25,0.1]` 和单轴投影。基底 YAML 提供 workspace 0.5 mm reserve 和 5 mm 平滑区，overlay 禁用 continuous Servo、radial discard、Ruckig 和 task projection。D1 仍须核对实际生效 YAML、工作区及二进制身份。[启动合并顺序](https://github.com/FEGkkd/whole-body-teleop/blob/6d0b0b5b2601618f1b0052d93ef2403eea9b064a/src/whole_body_teleop/launch/whole_body_teleop.launch.py#L15) · [MuJoCo overlay](https://github.com/FEGkkd/whole-body-teleop/blob/6d0b0b5b2601618f1b0052d93ef2403eea9b064a/src/whole_body_teleop/config/sim.yaml#L10) · [基底 workspace 设置](https://github.com/FEGkkd/whole-body-teleop/blob/6d0b0b5b2601618f1b0052d93ef2403eea9b064a/src/whole_body_teleop/config/real.yaml#L59)
- 生产代码分别保存原请求、平滑 workspace 目标、速率受限目标和本次 solver attempt。Workspace 饱和保留姿态，`TeleopCore` 依据上一合法 FK 限制当前位姿增量，再调用 `ik_nsp_angle` 和硬检查。可重试失败进入缩步和可选单轴处理。直接 IK 与一次投影后 IK 缺少这段因果推进和 START 状态。报告应分开计数投影生成失败、投影生成成功而 SDK 解失败、合法 waypoint 以及原目标误差。[Workspace 实现](https://github.com/FEGkkd/whole-body-teleop/blob/6d0b0b5b2601618f1b0052d93ef2403eea9b064a/src/whole_body_teleop/src/kinematics_adapter.cpp#L121) · [每步速率限制](https://github.com/FEGkkd/whole-body-teleop/blob/6d0b0b5b2601618f1b0052d93ef2403eea9b064a/src/whole_body_teleop/src/teleop_core.cpp#L824) · [原生 IK 缩步](https://github.com/FEGkkd/whole-body-teleop/blob/6d0b0b5b2601618f1b0052d93ef2403eea9b064a/src/whole_body_teleop/src/teleop_core.cpp#L1317)
- Teleop 默认相对位置来自分侧 OpenVR→base 标定与 START 锁存的机器人腕姿。P02 M1 则采用共享尺度 1 的 Train 拟合刚体注册与固定分侧安装。给定输入的绝对目标和名义关节序列需要按各自契约独立复算。后续诊断可在冻结的 1–5 因果子拍预算内复用原生 waypoint 路径，保持每个 20 ms HMB 计划拍的原始 M1 目标和来源，并保留实际求解目标、结果、种子、约束、限幅、失效及恢复。不得把持有回退计成合法新名义，或把 waypoint 成功计作原目标可达。获得 G-A 资格前须重新完成覆盖、前缀、路径和独立几何核验。改动 M1 注册、评分目标、失效口径或正式门槛超出现有授权。[腕位置变换](https://github.com/FEGkkd/whole-body-teleop/blob/6d0b0b5b2601618f1b0052d93ef2403eea9b064a/src/whole_body_teleop/src/teleop_core.cpp#L553) · [M1 历史拟合](../reports/P02_review_refinement.md) · [低误差能力域](../specs/V1-A_offline_rmb.md#13-预冻结的绝对精度能力域)

## G-B 本地证据与后端资格

独立复制三个测试目录与同一受审模型资产到新 ASCII 临时路径，使用 Windows Python 3.14.3、已安装 CPU 依赖、禁用 pytest 插件自动载入和缓存。`v1_learning`、`v1_rmb`、`v1_task` 共 27 项通过（8.86 s，退出 0）。单独对后续 `test_parity.py` 做判定器正负控，4 项通过（2.41 s，退出 0）。本地结果覆盖所选测试，不提供 Ubuntu 双后端运行、真实 Teleop 或数据阶段资格。核对的首份 Native 136 transitions 文件 SHA-256 为 `8ca101dbf9d7fd4ea39170be008fb35ec3ba1ca1ffd99ce1d0b1223b11c5c5e0`，启动索引中多出的一个字符已修正。两份 Native 小样本均使用合成目标。此前 Ubuntu 19 项通过的日志与退出 135 继续并列保留。[现有证据索引](initial_run_evidence.json)

自由响应判定检查位置、速度、指尖、Servo、实际滤波目标和控制量的有限性及容差，要求无接触/事件并完成计划子步。仍须独立验证软硬边界、自碰和跨手接触、硬停和回退。逐子步记录实际目标、控制量、扭矩、接触类别、透入和最早事件，并按相同规则输出有限状态的带符号间隙。旧 Warp 边界记录会在不测量的情形下填 0，不能当作 CPU 的实际有符号距离。边界安全与路径判定还须走 task 命令入口，覆盖 B0、非法动作与回退。正式 G-B 还要求签名 RMB 联调、54 轴正负与组合响应、CPU/Warp 规模爬梯和实际计数。[V1-B 实施规格](../specs/V1-B_residual_environment.md) · [Servo 与边界契约](../specs/V2_residual_task_contract.md)

Ubuntu 新的 MuJoCo、MuJoCo Warp 与 Warp 导入检查退出 0。PyTorch 精确重装后的导入仍观察到 SIGBUS。主代理用 `strace` 把 `BUS_ADRERR` 地址定位在原 `libcusparseLt.so.0` 文件映射偏移 `0x2e9f0c0`。共享库替换仍在进行，早期 NVMe READ 错误和长时间运行的持久化风险继续保留。成功导入、一个 CUDA 算子或一个 smoke 计数均不满足真实训练环境和持久化门。[启动阻塞记录](independent_launch_review.md#阻塞与恢复条件)

**独立意见**。保留诊断运行与失败收据。G-A 与 G-B 尚未通过，无资格发布数据、物理阶段或下游 WM/Actor 收据。正式物理评分、独立 Final Test 和真实机器人资格保持各自门禁。
