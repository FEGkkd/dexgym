# S0-5 决策记录

以下为 P00 的方案决策，待人工接受 P00 后成为后续阶段依据。候选数值仍由各责任阶段冻结。

| ID | 决定与理由 | 备选及证据 | 影响 |
|---|---|---|---|
| D-001 | 以三层目标并行保留意图和执行。评估生成器状态独立于所有候选执行 | SDK FK 保留作 B0/B2，现有 S0-4 输入和权重的语义见审计 | 004/013，P01–P09 |
| D-002 | 54 维关节名核验的 residual，使用同一个 policy，名义优先 B1+ | S0-4 14 维无法直接单指补偿，S0-2 关节增量提供参考候选。候选限幅 P03 审核 | 002/005/010，P03–P05 |
| D-003 | 使用捕获和接收双时间戳，离线/在线输出分开，前缀不变性作为硬测试 | Wholebody S0 对齐文件含 brackets 与 causal index，索引仅为参考，P01 验证到达时钟 | 003/012/021，P01/P04/P07/P10 |
| D-004 | 保留硬件对齐的 MuJoCo CPU/Warp 接口候选，50/500 Hz 为仿真候选 | 已见 20 ms native step，历史 S0-4 profile；Teleop arm 频率可配置，手反馈另一路 | 009/015，P03/P04/P10 |
| D-005 | PPO 用物理 rollout 和有界动作语义，SuperMimic 后续按同 task 对照 | 现有 PPO 有 tanh 分布实现和超时语义待核，S0-4 有实际 rollout+imagination | 017–019/025，P04–P09 |
| D-006 | 统计失败同时保存覆盖与保守复合分位数，按源 episode 配对 | 旧 S0-4 sweep 有分段精确 reset、硬停后恢复。新协议保留旧指标仅作局部诊断 | 022/024/026，P03/P07/P09 |
| D-007 | 频率、约束及挑战目标按 P02/P03 定量，对不可达提交 DCR | 先冻结绝对口径，候选相对收益为动态和难段 20%，实际非退化边界须验证 | 030，P02/P03/P09 |
| D-008 | P00 用 Ubuntu main 与发布/运行快照共同审计，文档通过独立 S0-5 路径维护 | main 无 S0-4，修改两处用户源码，S0-4 运行脚本另有哈希；禁止覆盖既有改动 | 029，P00/P01 |
| D-009 | 难段由训练/验证固定，既有被用于选型或公开评测的 source 与其衍生不作 final Test | 历史 S0-4 逐检查点评两划分，推荐分仅用 val、hold、hard；未见源证明需 P01/P09 | 011/026，P01/P09 |
| D-010 | 奖励多尺度全局用最差两指候选、尾部 Huber、实测监督，P04 验证饱和及早退 | S0-4 只有 15 mm 单尺度与最差一指。各尺度保留为候选并分项验证 | 016，P04/P05 |

## 设计变更请求

当前无已批准的核心目标变更。提交 DCR 时记录 Original design、Discovered problem、Supporting evidence、Proposed change、Alternatives、Expected benefit、Risks、Affected requirements、Affected phases、Affected tests、Compatibility impact、Approval status。保持旧版本有效至人工批准。
