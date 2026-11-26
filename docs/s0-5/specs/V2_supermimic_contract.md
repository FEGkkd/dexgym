# S0-5 v2.0.1 SuperMimic 学习契约

状态 DRAFT_READY_FOR_REVIEW。资格门和阈值见 [V1-C0](V1-C0_world_model.md)，参数治理见 [参数登记](V2_parameter_registry_training.md)。

## 实际源码审计

以下文件均在 Git 对象 b8d8b401bdda59ad00a37f0a3bf6878b462ed756 中存在，按文件和符号审计。

| 源码 | 观察 | S0-5 适配 |
|---|---|---|
| [engine.py](../../../source/dexgym/algorithms/supermimic/dex_reference/engine.py) | collect 调 env.step，从 transition 读 c、u、x1，写 replay 后 reset。fit_world_model 用缓存执行量窗口递推。policy_update 冻结 WM、FailureHead、Critic 参数且保留输入梯度，产生 lambda-return 拟合目标。支持 command_dim、make_actor 与梯度平衡 hook | 增加外部资格锁、有效时钟、Servo 状态、pre-reset 末值、价值上下界和独立 watchdog |
| [buffer.py](../../../source/dexgym/algorithms/supermimic/dex_reference/buffer.py) | GPU 时间×环境环形张量，以 done 和 valid 遮罩连续窗口，保留有效 terminal 本拍；内部 holdout 按环境列切分，失败重采样含重要性校正 | 另建母源留出，追加部分区间、外扰和饱和标签 |
| [networks.py](../../../source/dexgym/algorithms/supermimic/dex_reference/networks.py) | WM 以动力学和执行量为输入，codec 重建标准化状态增量，输出 contact logits 和可选深度。FailureHead 独立。默认 Actor 宽度 512/256/128 | make_actor 提供 512/512/256 和共享输入，接触类别与标定固定 |
| [config.py](../../../source/dexgym/algorithms/supermimic/dex_reference/config.py) | 可配 window、batch、UTD、horizon、WM-only、max_step_reward 和 bootstrap | 配置与资格独立，迭代次数不代替 C0 结果 |
| [S0-1 adapter](../../../source/dexgym/tasks/direct/wholebody_s0_1/supermimic.py) | task.read/encode_memory/dynamics/context/control_target/reward/advance_memory 支持两条轨迹，参考由 memory 推进 | 帧号只留训练侧，Actor 只读当前及过去白名单 |
| [S0-4 train_aligned.py](../../../source/dexgym/tasks/direct/wholebody_s04/train_aligned.py) 与 [aligned_task.py](../../../source/dexgym/tasks/direct/wholebody_s04/aligned_task.py) | 物理和想象使用同一臂滤波区间均值、明确滤波慢状态。action_noise=0、max_step_reward=1，历史动作主要激励 14 轴 | 重新验证 54 轴的独立可控性与输入表示 |
| [Warp 物理](../../../source/dexgym/tasks/direct/marvin54_s0/warp_physics.py) 与 [aligned_physics.py](../../../source/dexgym/tasks/direct/wholebody_s04/aligned_physics.py) | 后者保存臂滤波目标并检查执行器限位映射 | 重验 CPU/Warp 同模同命令，覆盖全轴与硬约束 |

LF 内容 SHA-256。engine 为 8121c53a1bfa436bc8a5c1bb43289fd92cdf234850bfa4fb9a4d6bbfc1f10fa4，config 为 68376519564e170b0a510482bc1a54a291bf23c5685c4687d8db02c1eba773f4，train_aligned 为 d9634e9a420d68ec961942d0abfae1699423b83562a85c6b40505e114af210fb。

## 真实 transition

| 字段 | 形状 | 用途 |
|---|---|---|
| x、x1_pre_reset | 各 float32[B,108] | q[54] 后接 qd[54]，区间首末状态。外扰限明确注入时点，禁止用 reset 差充当扰动 |
| s、s_next | 各 float32[B,54] | Servo 滤波初末状态，显式确定递推 |
| action、requested、c | 各 float32[B,54] | 归一化动作、原始请求、硬投影下发值 |
| applied、ctrl | 各 float32[B,10,54] | 子步滤波目标和实际控制输入，扭矩事件同索引 |
| u | float32[B,54] | 完整区间已执行滤波目标的算术均值 |
| sim_context | float32[B,P] | 有名 profile、扰动和模型参数，精确字段与 P 在 B 冻结，Actor 只能看合法反馈 |
| m、goal、obs | 结构化，obs [B,8515] 或批准的紧凑版 | 训练侧时钟、两类目标、历史、有效与世代；Actor 只接收 obs |
| contact、contact_depth_m | float32[B,13]、[B,1] | 十指接触与同机器人局部、跨臂、跨手三组硬分类，源于每子步；最大贯穿深度另列 |
| terminated、truncated、valid、full_step、safety_failure | bool[B] | 真实终止、外部截断、数值有限且同世代、完整 10 子步、失败标签 |
| source、partition、tick、generation | int64 或 manifest 字典编码 | 来源和采样，独立于 Builder |

模型特征为 q、qd、s、sim_context 与 u（若开启 control_error_features，则最后通道为 u−q；要求动力学首 54 列确为 q）。没有目标和未来参考。输出 Δq/Δqd 108 维，通过统一 codec 重建下一状态，另有接触 logits 和可选深度。默认 u 与 s 在固定、单调位置 Servo 下可重建区间 q_issued 和全部滤波子步；将在 C0 从实测逐步验证。存在迟滞、延时或限位而使同 u 无法区分响应时，改用显式下发/慢状态或子步控制编码，重新资格化，不能以旧均值假设准入。

完整且有效 transition 才进入状态拟合，合法部分区间可纳入真实故障分类。叶节点真实 terminal 可保留，窗口不得跨 reset、失效、世代或源边界。外部时限截断保留相应 pre-reset 反馈，不能当失败。缺少可见末值上下文时标不可 bootstrap 并排除对应学习窗口，严禁用 reset 后状态补齐。

## 学习循环与梯度

1. 读取当前因果 obs，C0 用独立可重复探索或零动作，后续由 Actor 计算。硬安全和 Servo 驱动物理，先写实际 pre-reset transition，再 reset。独立母源的验证数据不进入归一化和任何参数拟合。
2. WM 从训练窗口最远递推 8 步，状态 loss 为按掩码归一化的 Δq/Δqd MSE，误差归一化下限初案 0.001 rad、0.01 rad/s。接触用 BCE，可选深度 Huber/FK 误差单列，接触/失败标签和 partial 状态损失使用不同遮罩。采样不断回到 MuJoCo 物理修正 WM。
3. 首次 Teleop-only 学习的 Actor/Critic 在 G-C0 及预登记低误差能力域证据锁定前保持锁定；混合域必须取得自己的当前 E2-WM 资格。之后从同一有效段且有连续可见参考的 replay 状态开始想象，预留 horizon 后一拍合法末值。环境可逐拍推进当前参考，Actor 历史仅更新该想象时刻合法可见字段，禁止把帧号或离线参考透传。
4. 想象使用可微投影候选与同一 Servo 时间定义计算 u，执行 WM 并对 FK、[几何奖励](V2_residual_task_contract.md) 求梯度。冻结 WM/Critic/FailureHead 参数的更新，保留 Actor 经 u、模型、几何与末值的输入梯度。非有限值停止该批，保留最大和失败计数。
5. 设 c_t 为存活指示或已验证的存活概率。Actor 最大化 Σγ^t(∏_{j<t}c_j)r_t + βγ^H(∏_{j<H}c_j)V_H，并可扣除预注册有界正则。γ=0.99，β=0.65。真实 terminal 的 c=0，截断使用自己合法 pre-reset 末值，参考段终止原因需预先指定。
6. Critic 用 G_t=r_t+γc_t[(1−λ)V_{t+1}+λG_{t+1}]，G_H=V_H、λ=0.95。真实 terminal 不 bootstrap，合法时限截断使用自己的末值。Critic 输入、reward/continue、回报与标签对 Actor/WM 梯度 detach，Critic 标注与 Actor bootstrap 均限制到 [0,100] 并记录夹值。验证闭式小序列、真实终止、合法截断、有限差分及参数组隔离。
7. 使用独立真实 MuJoCo rollout 选择 Actor 和监控 WM，参照同目标同轨迹 B0。watchdog 失败立即复锁 Actor，修复并重跑资格后才恢复。单列真实物理拍、imagined transitions、WM/Actor/Critic 更新数和失败。

默认 FailureHead 仅诊断。独立阳性和阴性各至少 128 个，并通过 AUROC≥0.85、Brier≤0.15、冻结阈值召回≥0.90、假阳性≤0.10 后，才允许模型存活项有界使用该预测，概率裁剪上限候选 0.25。证据不足保留禁用状态。物理硬安全与该分类器独立。

## Checkpoint 与恢复

保存四网络、各优化器、自动混合精度状态、归一化器、CPU/CUDA/NumPy/Python RNG、采样器与专用 RNG、课程、代码与依赖、数据资产、obs/action/codec/reward hashes、gate 证据、已提交的真实 transition 和 optimizer step、分阶段全部预算及日志游标。单写入者原子写入再登记 SHA，加载前查明兼容和来源。当前引擎保存网络、优化器、RNG 和采样器，但不保存完整 replay/环境，加载后补充采样。默认 RESUME_REFILL，重填计入同一预算，完成暖场和最小复验前锁 Actor；完整保存 replay、仿真物理、Servo 与时钟并通过逐拍对照后才可称 EXACT_CONTINUATION。Actor 导出包无法作完整恢复。中断最后 chunk 若无法核对实际消耗，按保守上界记账。

## 部署反馈与条件扩域

D2 先核对 START、映射、在线名义 IK/SDK、Builder、归一化与历史、时钟、Servo、约束介入、WM 和策略泛化。只在软件接口已定位并合格、剩余缺陷属于策略时用 D2 escrow 补训，保持诊断资料和训练来源物理隔离。变更涉及 Servo、输入分布、动作生效或观测语义时，检查根因影响，撤销过期的 WM、Actor、回归或部署证据。冻结的 Policy-T 和原实际数据可作历史参考，报告适用范围。

进入 EgoPro E2 的已批准候选率 80% Teleop /20% EgoPro 保存在采样登记和 Replay 元数据。只消费 E1 资格的数据、独立 geometry/masks 和因果参考。EgoPro 不模拟 MANUS/VIVE 硬件时间或反馈。训练真实状态、执行与 Servo 来自相同 MuJoCo；变量只在 source/mask/register 层区分，禁止把来源或特权字段塞入 Actor。指定 MCAP 的输入、执行反馈、派生目标及历史重叠身份均由 E0/D1 执行资格排查，回归来源不得用于训练、归一化或 WM 拟合。

原 Teleop G-C0 仅覆盖登记域。E2 先真实探索并 WM-only，分别按两域的 54 轴、1/4/8 步、动作敏感性、Servo/边界、真值和 anti-exploitation 重验。资格绑定数据与模型指纹、实际采样和 checkpoint，不以运行轮数或旧权重继承。Watchdog 失败停 Actor/Critic；基于冻结条件修复再资格后才恢复。E3 和 D3 的真实 offline/Teleop 比较决定选择；E3 的 Policy-E 在任何必要域失败时按资格回到 Policy-T。 [E2-WM 与 E3](V2_1_egopro_training_and_regression.md) · [实际部署回归](V1-D_teleop_replay_and_refinement.md)
