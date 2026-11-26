# V1-E EgoPro 条件扩展规格 v2.0.1

**DRAFT_READY_FOR_REVIEW · PRE_REGISTERED_CANDIDATE**。EgoPro 补充无物体环境的人体运动参考分布。Teleop-only 是主路线和主评价依据。E 分支需独立条件收据，受整体 Campaign 授权和资源上限约束。图像、物体状态、视觉策略、任务标签特征和自主操作超出当前范围。

## 1. 阶段、输入与角色

| 技术门 | 前置与本门输出 |
| --- | --- |
| V1-E0 Coverage Diagnosis | V1-D0 离线绝对及相对精度通过，V1-D1 现有 MuJoCo Teleop 功能通过。D2 故障分类与接口复验、当前合格 WM、最佳 Policy-T 和剩余预算可核对。独立 Reviewer 确认通用动作族存在可复验缺口和扩充假设，签出条件收据。 |
| V1-E1 Reference Bank Qualification | E0 通过、EgoPro 来源可追溯、运行获授权。生成独立 Bank，核对源、时序、名义、几何、掩码、可达及缺口覆盖。 |
| V1-E2 Supplemental Training | E1 Bank 技术通过，最佳 Policy-T 可读、哈希和 E0 相同，扩展域 WM 需重新通过资格。 |
| V1-E3 Mixed Policy Regression | 冻结 Teleop Validation 和指定 MCAP 均保留，分别报告 EgoPro Validation 和 Teleop 及指定回归。支持和收益通过时才提交 Policy-E，否则回退原 Policy-T。 |

Data Agent 对 E0/E1 源注册、准入、覆盖和产物负责。Deployment Agent 提供归因和回归记录，Training Agent 负责 E2/E3。独立 Reviewer 复算谱系、规则、实际数据资格、预算和下游读用。Teleop-only D3 可按本身证据推进。[路线](../ROADMAP.md) · [E2/E3 技术规格](V2_1_egopro_training_and_regression.md)

## 2. 缺口定义与 E0 判据

在读取 Policy-T 部署性能前，从 Teleop Train 与固定特征字典登记动作族、评估窗、分箱、阈值、缺失规则和版本。全部几何使用签名模型坐标和米/弧度。字典至少包含双腕在 W 中位置和腕 SO(3) 的测地角度，P 中有名局部手型、逐指因果速度、拇指对指向量与距离，U_L 中相对腕 SE(3)，W 中跨手向量，同步/异步双手运动及原速/停止/反向。SO(3) 使用固定 Train medoid 的测地角。候选量化宽度为手和捏合 5 mm、腕位移及相对位移 20 mm、旋转 10°、位置速度 50 mm/s，停止和反向沿主评估的冻结事件定义。时间差只读合法前驱，缺字段单独计数。

一个访问窗口需要至少三个相邻合格拍落在相同动作族的分箱。一个分箱由至少 2 个独立 Train 母源、合计至少 10 个互不重叠的 25 拍窗口访问，才算已有支持。计算每个动作族的无支持率，即特征有效却未获 Train 支持的拍数除以特征有效计划拍数，同时报告原始根等权和缺失率。触发候选要求同一动作族在 Teleop Validation 和实际指定录包组均有缺口，各组缺口占合格拍至少 10%，合计至少 250 个唯一合格拍、跨至少 2 个可证原始录制根及两个无重叠连续窗。重复回放、镜像、重采样和同源切片均不增加资格或根数。Reviewer 复算固定分母，并核查部署尾误差或失败恢复是否集中在该缺口，登记可检验的扩充目标、适用场景、方向和收益量。运动分布关联只能作为归因的一部分。

启动前须具备 D2 复验的 START、坐标、在线 IK/SDK、Builder/归一化、Servo、安全、时钟、输入和 WM 误差证据。可由接口解释的故障先修复并回归。若修复后仍有分布缺口，E0 只读取录包审计区释放的动作族、分箱计数及类别汇总。源事件、轨迹、时间戳、逐帧目标、片段和新旧反馈全部留在回归区。它们禁止进入训练、教师、检索或最近邻拟合、映射与归一化。聚合诊断访问记入台账。无法证明缺口则登记 NOT_REQUIRED，明确有缺口但不能获得数据或预算则 BLOCKED。未执行的比较记 NOT_RUN。

## 3. 历史来源与只读审计

在 f78ee27f4762fedca9529bf697560bb267f8c704 核对了下列源和历史证据。执行时核查 Ubuntu 路径、读取、规模、许可、母源和历史使用角色，旧档规模只用于追索。

| 可复用线索 | 复用条件 |
| --- | --- |
| Marvin54 EgoProStandard-body | Windows 旧报告记录 278 个历史声称验证的源及 Train/Val/Test 角色，保留 clip/frame 规模和未配齐的身份。旧任务使用独立可行投影训练。准入仍需从可追溯人体几何重新验证。 [旧任务](../../tasks/marvin54_hand_centric_s0.md) · [旧归档见证](../evidence/P00_review_fix/marvin54_archive_witness.json) |
| S0-1 bank | 查找旧 S0-1 provenance、源/库清单、Train/Val/Test、重定时与参考生成记录，关联原始 EgoPro 和 Teleop 根。经核验的采样、索引和分片 I/O 可复用。 |
| S0-4 混合 | 旧目标手型来自 Wuji SDK qref FK，沿用同一 50 Hz 参考。复用名义适配、loader 和使用历史，重新生成独立人体意图目标和当前资格。 [旧任务](../../tasks/wholebody_s04.md) · [旧评估](../../experiments/wholebody_s04_20261125.md) |
| 原始解码 | 既有代码记录 /pose/body 共 22 行、/pose/left_hand 与 /pose/right_hand 各 21 行及两组坏帧注释。只复用可独立核验的拓扑与解析。旧插值、Slerp 和居中 Gaussian 使用未来数据，历史处理产物仅作诊断，除非逐项可重建其因果输入。 [解码代码](../../../source/dexgym/data/egopro_poses.py) |
| 旧映射与约束 | 掌系、指方向、分段和真正执行的历史证据可以复核复用。旧联合 Solver 不进入名义链，历史 WXYZ 按契约转换 XYZW。 [重定向代码](../../../source/dexgym/data/egopro_handcentric.py) · [原角色登记](../HISTORICAL_DATA_USAGE.md) |

## 4. 谱系、split 与派生隔离

逐源登记 source_root_uid、原录制哈希、会话与主体证据、采集类型和许可、嵌入 schema、拓扑/单位/坐标转换 SHA、原始帧索引、时间资格、父/子关系和历史使用事件。做内容 SHA、规范化语义指纹、活动片段近邻、已知派生图四层交叉查重，人工复核近邻冲突。用母源亲缘连通分量划分，Train/Validation 重叠必须为 0。历史拟合、训练、教师和选型事件原样登记；曾用于训练、先验拟合的根只可考虑 Train。无法证明使用角色的隔离为 UNKNOWN，曾公开评估的 Validation 标 HISTORICAL_DEVELOPMENT_REUSED，独立终评依据另取。构建前冻结 Train 内的 WM-fit/calibration 及独立资格子组。候选最低 EgoPro Train 8 个合格根，其中至少 3 个 fit/calibration 和 3 个隔离 WM 资格根。EgoPro Validation 至少 4 个与 Train 无交集的合格根。

两个指定部署 MCAP、可证近亲及直接/间接派生只能进入隔离的回归审计区。Denylist 文件和哈希写入 Teleop/EgoPro 的新训练、课程、教师、WM pool、映射和归一化访问门；无法解决近亲冲突先隔离。已存在于历史训练或同根资产的重叠要公布，保持 HMB/RMB 和已有历史台账不变，另用冻结执行清单过滤后继新训练，复核剩余支持。指定回归包的角色仅为工程部署回归。

## 5. 独立 EgoPro Reference Motion Bank

使用 s05.egopro_reference_bank.v1 schema，存放于与 Teleop HMB/RMB 分开、分别 ACL 的目录。冻结独立 manifest、原源表、split、每个分片 SHA。历史解码器记录的 22 个 body 及每侧 21 个手部拓扑只是入口线索，实际入库查命名、手性、引用、位置及姿态单位。基于有证实际坐标转换到签名 W，使用 Train-only 参数校核腕装配和人体几何尺度，保留源、原几何与 mapped 几何的逐拍关联。禁止虚填 MANUS/VIVE 身份、反馈或真实采样时钟。无法辨别腕姿、拓扑或缺必需 SDK 输入的样本进入非资格掩码，非必要特征缺失走 Builder 约定的确定性占位和显式 mask。

| 层 | 数组与接口 |
| --- | --- |
| ego_source / source_clock | 类型 EGOPRO_DERIVED。记录 dataset_tick_ns[T]、source_pose_indices[T,3]（身体、左右手，-1 为缺失）、source_pose_valid[T,3]、嵌入 schema、坐标/单位、根证据与资格。V1 默认 DATASET_CLOCK_ONLY，20 ms 网格属 SIMULATED。未知真实接收和时钟对应单独存 false。 |
| causal_adapter | 按同世代最近可用过去采样，查询前事件才消费。附质量注释、重复与年龄，超过 60 ms 候选、时钟或会话切换切段并重置。真实时钟未知时记录独立模拟可用时间及来源依据。Reviewer 复核前缀并排查居中及双端插值。来源 ID、活动、对象标签及阶段不能作为 Actor 特征。 |
| mapped | 独立读取人体几何，输出双腕位置 [2,3]、XYZW [2,4]、四点局部地标 [2,5,4,3]、十指世界点 [2,5,3]、捏合 [2,4,3]、跨手 [5,5,3]、相对腕位姿及对应掩码，复用已锁定 V1 目标几何角色。独立原几何和适配资产保留证据。 |
| nominal / optional_feasible / domain | 同一签名 Marvin IK + Wuji SDK 输出 q_nominal[54]、qd_nominal[54]，记录合法步、因果 seed、世代及 SDK 原始量。统一 Servo、动作尺度 ±8°/±5°、关节顺序和硬约束。独立诊断严格排除、可改善、合法见证和 UNKNOWN。由物体维持的姿态、明确持握或必须接触物体的任务、无物体几何不可执行的片段单独标记并隔离，保留分母和失败。 |

EgoPro 适配只处理源坐标和可用字段，不更改冻结 Teleop 目标、Servo、安全、评分、数据根和既有归一化。允许共用 builder 和资格张量定义。证据保留每拍资格、同源 FK、旧几何、失败原因和 record-to-goal 路径。[RMB 几何与因果契约](V2_rmb_contract.md)

## 6. 正负控、量化和 E1 技术门

1. 全量处理前锁源允许表、历史角色、资格分组、目标和 E0 缺口。至少复核每个根的布局、时钟、开始与结束帧、异常样本、哈希和保留原因，独立复算几何、轴和命令。
2. 单元测试包括已知双腕变换、已知异步采样的因果 ZOH、十指重建及合法动作。负控覆盖虚假 HMB 角色、部署包及镜像/派生偷渡、split 重复、居中滤波读未来、时钟虚填、隐式拓扑、缺项变合法、非法残差/耦合、对象标签泄漏和无资格 bank 混采。集成验证可读性、运行权限、未来替换/删除和失效掩码。代码、输入及输出哈希逐项留档。
3. 可读性 100%，同源/泄漏/硬违规 0，Actor 合法前缀不变。按 E0 冻结缺口分箱，新增合法支持应覆盖至少 25% 的缺口分箱，新增支持仍按 ≥2 根及 ≥10 个窗口；EgoPro Validation 至少 500 个不重复有效拍且来自 ≥4 个根。报告未弥补、UNKNOWN、被隔离和实际闭环可读支持。审计发现无足额数据、计算或谱系证据时，E1 标 BLOCKED 或 FAILED，留已有合格 Teleop-only Policy；不得以低支持混合数据结果替代。
4. E3 延续 E0 同一覆盖判定和动作族定义。实际跟踪/恢复按独立 EgoPro Validation 比较 Policy-T/E，另完整复验 Teleop Validation 和两包指定 MCAP 的功能、失效和安全。总体混合均值无法改变既定 Teleop 主评分，任一必须门不通过即回退至封存 Policy-T。[训练及回归门](V2_1_egopro_training_and_regression.md)

## 7. 定标、预算、失败和交付

| 候选 | 允许范围、定标及冻结 | 审核与处理 |
| --- | --- | --- |
| E0 分箱和覆盖 | §2 候选；以 Teleop Train 合格轨迹 P5–P95 和两边开放箱固定箱界，代码、数据、支持下限及计数在看 Policy-T 部署结果前锁定。 | Data Agent 和 Reviewer 各自复算。冻结后修改阈值或采样依据须项目用户审议。 |
| E1 数据预算 | 至多 3 h 墙钟、4 CPU worker、20 GiB 净新增持久存储，从整体授权的 72 h/250 GiB 计。先用 Train ≤3,000 拍预检读取、I/O、进程和实际 SDK。 | 主代理按剩余额度预约，Reviewer 复核。资源耗尽阻断支线并保存原证据。提高上限须项目用户审议。 |
| 有效训练参考比例 | Teleop/EgoPro 初始 80/20，允许 70–90/10–30，按有效参考 transition 机会汇报。E 支线使用 C3 预算划出的 1M，包含 WM、失败、复验及可选匹配控制。 | E2 预登记比例和证据锁。预算不足或超范围须停止，Teleop-only 路线保留。 |

本轮只登记设计。正式运行须具备整体授权、批准动作族与允许工作包、冻结身份和现场可用预算。存储路径 v1/<campaign>/<stage>/<run>/。E0 交 coverage_receipt.json、feature_bins.json、gap_attribution.md 和签名 SHA。E1 交 egopro_manifest.json、source_registry.json、dedup_and_excluded_lineage.json、qualified_splits.json、causal_report.json、coverage_report.json、stage_budget.json 与独立 gate_receipt.json。保留实际输入目录及哈希、版本、后台进程/命令、失败尝试和退出。发生身份、因果或安全错误，暂停受影响工作和下游，重新取得全部受影响资格。


## 8. 实施批次与复现

工作包顺序为证据审计、根归并、冻结分组、Train 小样本读入、独立几何和名义正负控、合格根全量候选、独立 Validation 复核、只读 Bank 封存。固定所有运行根目录、输入映射、处理环境、所用 SDK/模型哈希、量化种子和并发方法。每个分片都使用原计划拍与当前输入事件 ID 检查行数守恒。未经确认的 E1 暂存目录不给训练可读权限。所有失败批次各保留日志和独立目录，恢复时只复用输入/产物 SHA、配置和分组完全相同的分片。V1-E2 只接受同一授权、同一冻结配置、带资格收据的 Bank manifest。

## 9. 风险与适用范围

历史任务多样性、旧帧格式中的平滑和重定时、缺手腕姿态、捕获域与当前机器人形态差异都可能缩小合格覆盖。被物体支撑的姿态或动作在无物体场景不能直接解释。训练采样比例达到门槛和几何覆盖增加分别验证，二者都不保证闭环精度。扩展域和既有 Teleop 域的结果分层报告，正式物理评分与独立终评由原硬件和研究门禁处理。
