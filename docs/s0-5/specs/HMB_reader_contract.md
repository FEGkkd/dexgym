# HMB Reader 与 Actor 特征契约 v1

入口 `dexgym.tasks.direct.wholebody_s05.hmb_reader.HMBReader` 需要指定批准的 E1 manifest、原生产计划、配置、归档根、唯一开发 split 及用途。它核验固定 manifest SHA、索引、生产范围、冻结参数、原始母源及标定版本、分区内容和来源引用。传入 A1 物理文件映射可复验原包内容 SHA。正式索引只覆盖 Train 或 Validation，Final Test、试片及历史 NPZ 拒绝。空的许可请求按签核和数据用途失败处理。[正式 manifest](../evidence/P01_E_E1/hmb_manifest.json)

| 用途 | 数据接口 |
|---|---|
| HMB_GEOMETRY_DEVELOPMENT | Causal 视图的 SDK common、分侧掌系及腕部 Frame，通过 source table 保留原始字段及节点顺序 |
| HMB_SIMULATED_CAUSAL | 具 availability bounds、来源索引、时龄、姿态/速度掩码、许可与版本的历史模拟控制输入 |
| HMB_OFFLINE_DIAGNOSTIC | 独立 Offline 视图，双端原索引、权重、时间跨度和诊断资格 |

节点按原始 MANUS 顺序。长度单位 m、旋转 XYZW、速度 m/s、时间 ns。六个 pose 流是左右 MANUS、左右注册腕、左右 raw Tracker。注册腕使用录制时软件 Frame，物理 robot-base 仍无资格。不同用途下的原始索引与字典来自同一批准分区。Actor 视图只保留四个在线核心流。

训练历史采样器要求连续同一有效段、相同 clock generation、相同标定与拓扑、同一分区且五拍均有许可和姿态资格。速度历史独立要求四路速度连续具资格。输入函数读取当拍及之前 4 拍并复制白名单字段。segment 身份、边界与历史索引只给采样器，推理特征不含离线结果、未来决定、pose 来源索引、文件、录制或绝对相位身份、完整轨迹统计和分段信息。

部署、真实时序、物理校准、遥测系统辨识和 Final Test 均有独立前置。P02 开工等待人工批准 P01 数据主线及移交范围。[Actor 特征白名单](../evidence/P01_F/actor_feature_allowlist.json)
