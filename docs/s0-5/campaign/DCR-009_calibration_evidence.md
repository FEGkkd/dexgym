# DCR-009 标定证据与仿真坐标资格

阶段二从 Ubuntu 的实际 Teleop 工作区归档了 10 份标定、窗口和 Tracker 配置记录。归档检查器逐文件复算 SHA-256，保留历史原文件。源为 `/home/ubuntu/teleop_ws`。本轮虚拟标定用途为独立 MuJoCo 坐标和模型接口核对。实物配准、录制世代的人体标记安装和真实时钟仍具独立资格要求。

## 录制坐标及佩戴安装

Tracker 配置声明 `openvr_world`、`Standing` 及左右设备身份。静态腕安装 YAML 分别存在 `left_wrist_raw→left_wrist` 和 `right_wrist_raw→right_wrist` 的刚体参数。已归档 `tracker_pose_transformer_node.cpp`（SHA-256 `835a1c5e4dcf993c1cdd8f175a6fcf1d02aea9721da80ce07ebfebf6e354fcdc`）订阅 `/vive/raw/*_wrist`，应用左右固定边后发布 `/vive/tracker/*_wrist`。所选回放使用后者。现有静态固定边已应用于该话题，录制时含有此注册链，尚缺对应注册腕子帧到人体解剖根、标记实际装夹和外测见证。[历史七根采集验证](../reports/P01_C_acceptance.md)

三份 2026-11-19 heading 窗口均标为 `reference`。它们用 `/vive/tracker/left_wrist` 记录人工朝向机器人运腕。每个窗口包含独立的平掌记录。历史结果给出的三个拟合角为 17.70°、12.45°、17.06°，被标记为 held-out 的有向角差为 +2.95°、−4.93°、+1.98°，动作垂直幅度为 13.52–21.52 mm。归档没有两坐标系的外测共同基准、独立非共面夹具、每轮重装记录或相应不确定度预算。角差指方向复验值，用途和正式逐掌 SO(3) 误差分别记录。历史新旧朝向相差 17.06°，这类差异会改变原始运动在模拟世界中的方向。

左右腕联合标定以原始 MANUS 手骨、Tracker 姿态、带标签的接触和枢轴窗口拟合，保留两个独立接触录制。联合候选在两个录制的接触 P95 为 30.05 / 33.23 mm，接触推定的相对掌姿态 P95 为 9.71° / 8.85°。历史独立复查记录 `candidate_failed_independent_holdouts`。当前、仅五个旧拟合源、固定枢轴和联合拟合四个对照保留原误差。实际仿真启动记录可证明其运行时加载过历史腕参数，缺少逐侧绝对解剖安装和所选录制世代验证。

## MuJoCo 虚拟坐标与模型

预先声明共朝向虚拟操作员。H 坐标采用 OpenVR 的上 (+Y_H)、右 (+X_H) 及前 (-Z_H)。W 的上由 MuJoCo 重力确定，右由具名左基座朝向右基座确定。标准将 +Y_H 映至 +Z_W，将 +X_H 映至 −X_W，由叉积确定 +Z_H 映至 +Y_W。实际 Teleop `world_adapter_node.cpp`（SHA-256 `7498fef99877b99b8ea47507e7a27b0f27977b43a7fa8f17fc7f92a424fd7711`）使用同一个共朝向矩阵发布 `teleop_world←openvr_world` 静态转换。三个不对称位置和朝向在 MuJoCo 站点中独立枚举。测试覆盖正向、水平反向、左右交换和反射。虚拟标准与历史朝向候选相差 77.63°，其作用域为虚拟场景。

模型复核按准确 SHA 加载现有 Teleop 和 S0-5 签名 MJCF，分别枚举包括项和 mesh SHA。以同一具名 54 轴值进行 Home、单轴双向和混合位置 FK，记录关节轴、零点、作动器绑定和控制范围。通过三维欧氏距离与 SO(3) 角度分别比较具名基座、Wuji 掌部刚体、腕评分 TCP 及每指四地标。机器人虚拟掌根来自具名掌部刚体，纵向与拇侧向由四个非拇指 MCP 锚点构造，并验证指节扰动下不变。Marvin 原生工具桥接使用本侧 Home 和同侧双向单轴留出。历史 105 拍检查已见签名与现有模型之间约 6 mm 的手地标差；有限的腕部 FK 一致性与跨模型指尖毫米级计分、碰撞、Servo 和实际沿程安全分别审查。

## Pilot 处置

转移共朝向虚拟旋转以及机器人端桥接的数学验证通过时，可以据此进行受限虚拟夹具检查。真实录制回放仍需证明记录中既有 Tracker 刚体语义、会话适用的独立 H→W 朝向和逐侧 (T_{KA})。归档缺少这组证据时，校准收据为 `BLOCKED`，无可用于真实录制输入的七块矩阵。六条 Train 的原始录包与原有几何、计分范围、分子及分母继续保留。未运行的 Pilot 不记作 0% 覆盖，也不用于修改 85%／70% 门。

复核入口为 [归档证据检查器](../../../scripts/s0_5_campaign_teleop/audit_dcr009_calibration_archive.py)、[准确模型 FK 核验](../../../scripts/s0_5_campaign_teleop/check_dcr009_frames.py) 与 [虚拟轴夹具](../../../scripts/s0_5_campaign_teleop/dcr009_virtual_axes.py)。
