# P01 数据主线技术汇总，待人工签核

P01 整体为 **IN_PROGRESS**。A、B-HMB、C 历史 HMB 仿真资格、D、E 已获分阶段人工签核。F1 技术门为 PASS，P01-F READY_FOR_REVIEW。历史 B1 FAIL、B2 BLOCKED 与原 C 真实时钟 BLOCKED 保持在原记录中。[阶段状态](../phase_state.json) · [独立验收](P01_F_acceptance.md)

正式历史 HMB 包含 32 个真人母源、308 个分区和 116,613 个计划及发布拍。Train 24 根、Validation 8 根，母源交集为 0。独立从实际数组复核了 110,796 个五拍姿态历史、110,622 个五拍全核心流独立速度历史。原始重建覆盖 32 个母源、4,790 对载荷与观察器记录。

| 资格 | 当前结论 |
|---|---|
| 人体几何、历史模拟因果、离线诊断 | F1 技术验证通过，限对应历史仿真用途并等待人工签核 |
| 实时采样与控制端可用性 | C2-RT BLOCKED |
| 物理 robot-base 注册 | BLOCKED |
| 真机遥测和系统辨识数据 | P01-T BLOCKED |
| 独立 Final Test | BLOCKED |

P02 移交清单绑定获批 E1 manifest。下一节点是 P01-F、P01 主线人工签核及独立 P02 启动授权。P03 继续依赖 P02、P01-T、C2-RT 及物理 robot-base 的独立 ACCEPTED。后续训练与真机动作受各自门禁。[P02 读取范围](../evidence/P01_F/p02_handoff_manifest.json)
