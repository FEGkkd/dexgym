# S0-5 接口契约 v0.1.0

阶段规格引用版本号。文中候选数值由指定阶段验证后冻结。

## C1 几何和数据

W 为机器人共同世界系，U_h 为参考或实测腕系，H 为原始采集系。使用右手系，列向量，T_AB 将 B 坐标变换到 A，p_W = R_WU p_U + t_WU。姿态存 3×3 旋转或注明 XYZW 的单位四元数。检查正交、行列式为 +1、四元数符号连续。SO(3) 误差用 Log(R_currentᵀ R_target)，近 π 用稳定实现。位置 m、速度 m/s、角 rad、关节速度 rad/s、时间 int64 ns。记录原始坐标单位、弧度转换、标定矩阵和方向。L/R 与拇指、食指、中指、无名指、小指排序显式，模型通过名称双射核验。

每个 raw source episode 按原始 recording GUID 和文件哈希登记，同一采集的重录包/裁剪/增广必须共用 parent lineage。对重命名/重新编码可能重复的源做时间区间与内容指纹检查，无法溯源的条目隔离并统计。Manifest 声明 source_family=teleop_manus_tracker，经 topic/schema 与采集记录复核。EgoPro、纯仿真衍生、控制器输出重录包、缺真原始来源项在训练来源准入前单独标注。Train/Validation/Test 按 lineage 分组，Test 清单冻结、访问记账，P09 冻结候选后评分。断言 split 交集为空并拒绝未经认证的转录来源。

| 层 | 每帧所需字段 | 来源或约束 |
|---|---|---|
| L1 | episode/parent ID，原始 MCAP/channel SHA，frame index，每流捕获/接收/记录时间与时钟域，帧 ID，双腕 tracker pose，双手 MANUS 节点 ID/父 ID/变换/有效位，标定版本 | 原样保留；缺失显式 mask，禁止伪造接收时间 |
| L2 | 原始与映射腕位姿、每指有名腕内关节与 tip、两手世界指尖、拇指–其余指对捏向量、跨手对应与异指向量、腕间 SE(3) | 人体骨长映射与标定；SDK 几何作为 B0 对照 |
| L3 | feasible target（同 L2 几何），original_sdk_ik_reference[54]，optimized_nominal_reference[54]，q̇_nom[54]，per-point projection_error，feasibility status，screening flags | 评估版本独立于所有被评估执行轨迹；保留投影误差和来源 |
| Index | raw/parsed/aligned/reference 哈希、父子索引、schema/tool/git/calibration hash，样本来源索引与年龄，处理/拒绝原因、split 及统计 | 文件与帧两级可追踪；所有单位、顺序、坐标系可机器校验 |

对捏向量 v_hf = p_local(h,f) − p_local(h,thumb)。跨手向量初版候选为机器人世界系 5 对同指及 25 对全指矩阵，实验预先登记所用子集和权重。双腕相对目标为 T_WL⁻¹ T_WR。有效性至少区分 VALID、STALE、DROPOUT、UNSYNC、CALIBRATION_INVALID、INFEASIBLE、SOLVER_TIMEOUT、UNKNOWN。不可行的原始目标保留，feasible 可缺失且原因编码独立于执行失败。

## C2 时钟、同步和缺测

设 t_k 为控制时刻，d_k 为该拍计算输入截止，记录原始捕获 c、控制端接收 r、对齐查询 t_k、发令 u_k、生效/物理计分 s_{k+1}。时间映射含版本、偏置、估计误差和启用时刻。在线选每流已到达的最新样本，条件 r≤d_k 且 c≤t_k，单流保持或用过去点做有界外推。离线双端插值只为可视化或离线标签，严禁送 Actor。对缺录包接收戳只可在仿真用显式已声明接收模型，线上延迟结论必须基于实测。年龄 t_k−c、偏离和有效掩码进入输入。观测历史按 old→new 均为过去可用的快照，reset 用有效初始值及 padding mask，禁止跨 episode、切换源或时钟跳变污染。过期阈值、失联回退及恢复滞回由 P01/P03/P07 依据时序测量冻结。

候选 Δt=20 ms，Δt_phys=2 ms，10 子步。读取 q_k、v_k 与 G_k，生成 u_k，积分子步得到 q_{k+1}，reward 与 G_k 绑定。应用态、发令、滤波等效目标分别记录。端到端指标按真实时钟对 s_{k+1} 的意图 G(s_{k+1}) 评分，禁止事后移位。所有控制器的延迟模型、时钟条件、状态采样和失败规则一致。

## C3 名义、动作与安全

`a∈[-1,1]^54`。逻辑顺序 left arm[0:7]、right arm[7:14]、left hand[14:34]、right hand[34:54]。每组模型关节/执行器名和映射、左右镜像和符号显式版本化。关节/执行器为非一一映射时必须提供校验过的转换，并验证 54 个可控命令维度。q_request=q_nom+diag(s)a，q_cmd=P_C(q_request;q_prev,q_meas,model,limits)。模拟候选 s 为臂与拇指 ±8°、其他手指 ±5°，真实限制由工程审批。

在线名义 IK 最小化 Σ_k w_k‖W_k r_k(q,G)‖² + λ‖q−q_posture‖² + μ‖q−q_prev‖²，r_k 为世界十指、腕内地标、对捏、跨手、腕平移和 SO(3) 对数。明确每项尺度与雅可比，按 P03 基线预实验冻结。安全投影优先保留合法零残差命令，其他请求最小化加权 ‖q−q_request‖²。C 含位置、每拍速度约束、双臂联动、手部耦合、执行器约束与运动段碰撞裕量。历史 Marvin 参数 1.025|J6|+|J7|≤110.5° 需结合实际型号校验，P03 验证对每臂应用。碰撞线性化只生成候选，发令前验证目标与运动路径。确定性超时或不可行分类；最后可行指令须按当前状态重新校核，安全保持/受控停机分别触发；骑缝路径、停机和看门狗边界由 P03/P10 验证。记录触发类别、主导约束、saturation、post-projection residual。

## C4 Actor observation

共享 builder 输入只有实测或因果估计的状态、已下发命令与共同实时意图流。输出 per-tick 动态特征和较少重复的当前几何。候选 schema 如下，手系按 C1，归一化上限与分组是否纳入由 P04 冻结并写入版本。

| 组 | 顺序与形状 | 单位与获取 |
|---|---|---|
| 历史动态 | q[54]、qd[54]、q_nom[54]、qd_nom[54]、last_issued[54]、last_applied_residual[54]、servo_state 按臂手分组、各流 age/valid、Δt、history mask | rad、rad/s、s，因果估计 Servo |
| 当前腕及全局指尖 | 各腕位置、6D rotation encoding、腕平移与 SO(3) 误差，tip target[2,5,3]、tip error 同形、目标与误差速度同形 | 米/米每秒；FD 向后差分带时间和有效位 |
| 当前局部与协同 | 腕内地标及误差[2,5,K,3]、pinch vec error[2,4,3]、cross-hand vec error[M,3]、relative wrist error[6] | 尺度用固定物理量或训练集统计，验证/测试只读 |
| 解析特征候选 | JᵀW e 或小型按臂手分组 Jacobian 摘要，按照 P04 消融方案冻结 | 仅用实时 q 与已知模型，记录精确 shape 和单位 |

K 初始候选四个有名手部地标（含 tip）。无效特征零填后单独 mask，归一化统计只在 training 更新。拒用未来帧/索引、episode ID、离线 Oracle 结果、仿真特权、隐藏扰动参数；Critic 可用独立有名输入和缓存。用前缀不变性、随机未来替换、字段白名单与训练/部署逐元素回归验证。

## C5 奖励和 PPO 时间语义

e 为 measured−feasible 的任务误差向量；C1 的 mapped 目标同步评估。S_g(||e||_mm)=Σ_i w_i/(1+(||e||/s_i)²)，(w,s)=(0.25,20),(0.35,5),(0.40,2)。全局分使用 0.65×十指均值 + 0.35×最差两指均值（候选，P04 冻结）。其它几何项 score 为 1/(1+mean(||e||²/L²))。候选尺度 local 5 mm、pinch 2 mm、跨手 5 mm、腕 10 mm/0.10 rad、tip velocity 0.05 m/s。双手项等权合并跨手点和腕间 SE(3)；腕自身分先归一位置和朝向再均值合成。大误差使用每拍最大指尖误差的 Huber tail，输入 (max(e_mm)−5)_+/20，阈值 1，超阈值线性增长。恢复由相同误差和尾部项提供优化信号，实验比较收敛与再捕获。

候选总分系数 global 4、local 2、pinch 2、bimanual 2、wrist 1、velocity 1。代价系数 tail 0.3、mean(a_applied²) 0.1、mean((a_applied−a_prev)²) 0.1、归一化关节加速度 0.03。加速度按 (qd_{k+1}−qd_k)/Δt 除以每轴物理尺度，P03 从能力测量给出。安全成本含投影差、剩余裕量、碰撞与终止事件，系数与终止条件 P04 经量纲和早退测试冻结。终止前最后分按实际轨迹处理，验证早退不会因为避开积累损失而占优；reward 对各可控几何方向做扰动检查，分别报告项级贡献。有效目标由预先冻结的 mask 决定，失效模式回退不计作控制成功。

PPO 使用有界策略变换 a=tanh(z)，logπ(a)=log N(z)−Σ log(1−tanh²(z))，训练储存 pre-tanh z 或等价稳定重建数学量。ratio 总合 54 维，禁止重新用投影值计算 logπ。环境每控制拍计 1 transition（并行环境分别计），区分真实终止（bootstrap=0）和截断（用 final_obs V），reset 后状态不作该拍 bootstrap。GAE 在 episode 结束边界截断递推且保留 truncation 的末值。采样、恢复、分布、超时用可控例子与真实仿真验证。

## C6 版本及运行证据

正式运行保存唯一 Run ID、Git SHA、tracked diff 与额外源码白名单哈希、各契约/评估协议版本、数据清单 SHA、模型和标定 SHA、Servo、算法与超参、随机种子、并行环境数、真实 transitions/预算、软硬件版本、checkpoint SHA、实际命令与 stdout/stderr/退出码、原始指标、覆盖率与失败轨迹。S0-5 checkpoint 拒绝不匹配版本，训练与部署共享 builder 或有逐字段、导出等价测试。交付 snapshot、如何重跑、适用范围和结果状态。无法运行标 NOT_RUN，缺必需资源标 BLOCKED。
