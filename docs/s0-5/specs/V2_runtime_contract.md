# S0-5 V1 运行时与仿真发布契约 v2.0.0

状态为 `DRAFT_READY_FOR_REVIEW`。本契约用于独立仿真包和今后实时 Teleop 集成的接口验收。历史 S0-2 快照中的 `wholebody_s0_policy/libtorch.py` 与 `native/controller.cpp` 可供加载、重置及前向接口复用，V1-D 须核对实际版本、设备和 ABI。[V1-D](V1-D_evaluation_deployment.md) · [任务契约](V2_residual_task_contract.md)

## R1. 发布目录及依赖

至少提供 Actor 权重、导出 Actor、训练冻结的归一化器和共享 Observation Builder 版本、Observation/Action schema、关节清单、每轴残差尺度、安全和回退接口、意图映射配置、仿真 Servo 契约和模型、推理运行时、可执行仿真 launcher 验证入口、依赖锁与 release manifest。TorchScript + LibTorch 是有历史工程基础的候选路径。后端选择在 V1-D 用实际目标主机可用依赖验证，保留版本、设备、线程、精度和 native ABI。发布包不包含 World Model、Critic、Failure Head、Replay Buffer 或训练数据库运行依赖。

`release_manifest.json` 必须可遍历验证全部文件 SHA-256、软硬件平台与零件/SDK/模型版本、代码 commit、导出命令、Actor checkpoint SHA、归一化器与有名 54 轴映射 SHA、一致的训练和部署 schema SHA、支持范围和等级及各证据路径。启动时校验 SHA、接口版本、dtype、shape、时基、所需 provider、线程设备参数和仿真模式。错误设备、缺依赖、路径越界、未知资产或版本、默认设备自动回退均纳入负控。

## R2. 流式调用与命令权

每拍锁存四路 MANUS/Tracker 有效输入及可见的机器人反馈，关联 capture、arrival、cutoff、tick 与世代，按认可时域选择到达前缀。由在线因果名义 provider（现有 Marvin IK + Wuji SDK 的适配器）生成 `q_nominal_online[54]` 及合法历史，独立映射适配器生成当拍 mapped intent。共享 Builder 编码当前和过去四拍，Actor 以 batch=1 输出有限的 54 维归一化残差。应用训练冻结的尺度和有名关节双射，路径安全服务处理请求，仿真 Servo 独占发令及驱动应用。发布和反馈按臂、手记录命令身份、持有/回退、应用和物理状态。实现不依赖 ROS 2 全图，仿真可采用隔离流式 provider 与独立进程。

每拍可用输入的原始来源顺序、hold、nominal/ref 世代与状态所有者有明确定义。独立运行在仿真模式、有隔离命名空间和唯一命令 owner 的过程中验证，不接入真实 driver、话题或设备。测试流可以回放历史人体包，但每次仅供给在当前 cutoff 前到达的样本，启动与运行通过移除全部训练目录和缓存名义轨迹的负控。POSIX 支持时只读文件白名单或进程文件访问记录应能证明只读取实时发布资产。

同一时钟计算 deadline。输入无效、版本错配、硬安全非法、过时、超时、NaN/Inf、发布失败或上轮结果迟到时拒绝该请求，按任务契约执行监督回退。超时的推理结果作废，禁止下一拍重用。仅当安全层判定上一个命令还能合法保持才允许短 HOLD，持续失效或无法证明保持合法时走仿真 STOP。重新进入 RUNNING 需要有效输入世代、重建合法历史和预登记恢复条件，以故障注入验证。Actor 无安全裁决权。

## R3. 三层等价和时间预算

1. **同输入算子等价。** 在训练、导出和独立进程使用冻结的合法 5 帧输入，各组校验字段顺序、归一化和历史掩码。确定性元数据比对按位，浮点特征、Actor 和位置命令按预登记容差。覆盖零残差旁路、组 mask、非法 reset、乱序、跨世代、padding、未来替换和删除。保存逐元素极值、误差分位及测试源 UID。
2. **命令旁路等价。** 在相同合法名义输入和仿真后端，禁用 residual 后与 B0 比较 requested、issued、应用/滤波、反馈、安全原因、失效 reset 和实际状态。安全模式及离散决策逐项相同。另用边界小扰动校验浮点误差不会跨越合法性门。
3. **缓存与在线 provider 差异。** 同一按到达顺序走的历史原始流，缓存路径与在线独立仿真 provider 需报告每轴 `q_nominal`、`qd_nominal`、腕/十指 FK 差异、稳定/困难/切换分层、有效率、失效原因、源世代和 deadline。报告预登记训练范围覆盖，凡输出语义不符或长期越界均触发固定回退、分析及重新验证。缓存对照的成绩与在线仿真的运行证据单列，完整实时 Teleop（实际 START、锁存和姿态映射、多频率控制、发布所有权、真实端到端延迟）另行验证。

软件时延采用目标主机单调时钟，记录 capture、receive、cutoff、build、inference start/end、projection、issue、apply、feedback 时间及归属。Actor batch=1 P99 ≤5 ms 为挑战。候选命令 deadline 为相对 tick 15 ms，迟到 0 次发令，统计失败与按时回退率。控制周期以 20 ms 为标称值，由仿真 Servo/profile 校核。CPU/Warp、运行时 benchmark 和在线应用环境分别记录，端到端真实时延需独立 C2-RT 和外部测量。[阈值和验证样本](V2_parameter_registry_evaluation.md)

## R4. 后续 Teleop、Shadow、HIL 及真机

正式集成需验证当前实际 START、映射、失效恢复、在线名义 provider 差异和 command ownership。比较原生 Teleop 与残差禁用旁路，再运行授权范围内的软件闭环。Shadow 需独立监测通道与禁发令证明，HIL 与真机按独立现场授权。真实部署还需物理基座注册、C2-RT、P01-T、真实 Servo 响应、安全约束责任、外部计量证书和现场负责人审核。每级应具备急停、看门狗、故障回退和中止条件，禁止自动恢复致动。编码器 FK、外测、物理应用时钟分别记录。物理科研主张仍需冻结正式评分域及独立 V1 之外的终评。[硬件联合测量](HARDWARE_SHARED_MEASUREMENT_PLAN.md)
