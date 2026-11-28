# S0-5 v2.1 阶段二预检与 Pilot 处置

运行 ID `HOME-ANCHORED-PILOT-DCR009-01`。用户批准阶段二，设计 SHA 为 `6808935d7588834ad875789a2ca6bbb779d2caa9`。已完成 Ubuntu 实际 Teleop 标定审计、两套 MuJoCo 资产的静态对照、现有 Marvin SDK 离线桥接和虚拟轴正负控。六根录包 Pilot 的技术入场状态是 `BLOCKED_CALIBRATION_AND_SAFETY`。[用户授权](../campaign/approvals/DCR-009_stage_two.md) · [独立审查](V2_1_stage_two_independent_review.md)

## 证据把三种结论分开

| 证据 | 实测结果 | 适用性 |
| --- | --- | --- |
| `teleop_ws` 历史独立朝向 | 三次有方向留出角为 +2.9464°、−4.9301°、+1.9837°，全数高于 1° 单点绝对设计上限。 | 缺少对历史采集工作区和录制世代有资格的物理映射。 |
| `teleop_ws` 历史腕掌安装 | 旧联合候选状态为 `candidate_failed_independent_holdouts`。代表配置的两个接触留出 P95 为 30.05、33.23 mm，接触推断的相对掌面角 P95 为 9.71°、8.85°。 | 缺录包注册腕到两侧对应解剖掌根的独立安装资格。现有腕话题已经过 raw→registered 外参，不能重复应用。 |
| MuJoCo 虚拟轴 | 三个独立指定有向 site 的点位和轴向误差均为 0。水平反向、左右互换和反射负控全部拒绝。 | 适用于具名虚拟坐标，录包朝向和安装仍不合格。 |
| 实际与签名 MuJoCo 资产 | 158 个配对 FK 探针。54 个有名局部轴的坐标一致，Home 全轴角度和双腕评分 TCP 一致。掌体最大差左 0.3001 mm、右 0.2502 mm，世界系指尖最大差左 6.7029 mm、右 5.8170 mm，左掌系手部地标最大差 6.7885 mm。 | 腕 TCP 可逐名比较。毫米级指尖、Servo 和碰撞一致性各需独立资格。 |
| 实际 Marvin FK 转接 | 用实际 Teleop 源码和生产依赖的 vendor `libKine.so` 重新构建离线 shim。每侧 14 个 Home 外扰动留出，位置最大左 0.000214 mm、右 0.000197 mm，姿态最大左 0.0000532°、右 0.0000347°。 | 只验证 MuJoCo 与 Marvin TCP 的运动学接口。 |

坐标对照图显示历史拟合留出、虚拟坐标约定差异和实际两模型 FK 差异。三组数据各有独立单位与支持集。[对照图](../campaign/evidence/dcr009-stage-two/calibration_preflight.png) · [标定门禁收据](../campaign/evidence/dcr009-stage-two/calibration_receipt.json) · [实际模型](../campaign/evidence/dcr009-stage-two/model_frame_receipt.json) · [Marvin FK](../campaign/evidence/dcr009-stage-two/model_native_fk_receipt.json)

## 实施和实际测试

落地了 Train-only 的 11 拍主稳定锚点和 21 拍灵敏度锚点、六根及两次冷启复测的固定选样规则、严格的单会话 Home 锚定入口、原始四流回放闸门和只允许既有 MuJoCo 控制归属的隔离启动器。实测 Home 稳定后锁定共享平移。原始时间轴需等严格 START 实测门达标才开始。运行记录区分名义、发布、实测和失效；当前最小 overlay 未捕获驱动逐物理子步的实际应用 ACK。

Ubuntu 使用现有系统 SciPy／MuJoCo、ROS2 Jazzy 和隔离 Python 包查找路径，18 项合成测试均过。旧离线 shim 缺少 FX_LOG_SWITCH 符号，使用项目已有构建脚本配合实际 Teleop vendor 库修复。Home→START C++ overlay 在隔离源码副本上以系统 Python 完成构建，生成的二进制 SHA-256 为 `9b1bda094accbc680c8af76bf22de30ca1008acfe5e50ab03064651343623aef`。实际标定校验脚本退出 2，表示历史记录 Pilot 被拒绝。启动器与严格回放入口都在 ROS 启动前拒绝不合格标定。正式原生回放还要求独立签名 54 轴扫掠路径安全门，旧 Native 试验的执行结果不能代替该门。[运行检查回执](../campaign/evidence/dcr009-stage-two/preflight_receipt.json)

```sh
python3 -B -m unittest discover -s scripts/s0_5_campaign_teleop -p 'test_home_anchored_*.py' -v
python3 -B scripts/s0_5_campaign_teleop/check_dcr009_frames.py --live LIVE_XML --signed SIGNED_XML --out model_frame_receipt.json
bash source/dexgym/tasks/direct/wholebody_s05/v1/rmb/build_native.sh TELEOP_WS NEW_BUILD_DIR
python3 -B scripts/s0_5_campaign_teleop/check_dcr009_frames.py --live LIVE_XML --signed SIGNED_XML --marvin-bridge NEW_BUILD_DIR/lib_s05_marvin.so --marvin-config LIVE_CFG --out model_native_fk_receipt.json
python3 -B scripts/s0_5_campaign_teleop/dcr009_virtual_axes.py --model-frame-receipt model_frame_receipt.json --out virtual_axis_receipt.json
python3 -B scripts/s0_5_campaign_teleop/audit_dcr009_calibration_archive.py --archive calibration_evidence.tar.gz --model-frame-receipt model_frame_receipt.json --virtual-axis-receipt virtual_axis_receipt.json --out calibration_receipt.json
python3 -B scripts/s0_5_campaign_teleop/plot_dcr009_calibration_preflight.py --calibration calibration_receipt.json --model model_frame_receipt.json --virtual-axes virtual_axis_receipt.json --out calibration_preflight.png
```

## 数据分母与停点

| Split | 母源 | 分区 | 原计划拍 | 原 HMB 联合有效 | 原签名五帧窗 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 24 | 228 | 86,338 | 85,382 | 82,559 |
| Validation | 8 | 80 | 30,275 | 29,150 | 28,237 |

保留 M0/M1 和所有原始计划拍。85% 门使用联合有效拍，70% 门使用原签名五帧窗，分别按 split 报告。六根正式样本已运行数为 0，录包名册的实际合格样本、START 到位、主动覆盖、五帧覆盖、不可达、HOLD、回退及安全路径结果均 `NOT_OBSERVED`。没有可用于计算这两道门新分子的执行证据。

本轮停在阶段二独立预检审查。DCR 人工批准仍有效。六根正式录包、全量 RMB、SuperMimic 训练、ROS 控制回放及真实机器人操作状态均为 `NOT_RUN`。需要与录包世代绑定的独立坐标及双侧腕掌安装见证、原生签名整身路径安全和 driver 应用取证，才可重新审查 Pilot 入场。
