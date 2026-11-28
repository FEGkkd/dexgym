# DCR-009 扫掠守卫 V2 独立技术审查

本审查仅覆盖第 5 根 `capture:b63bc72d2e8b2ee395e966f6` 的一次隔离 MuJoCo 诊断。审查人员独立于 V2 实现人员。旧报告、真实运行回执和原生 STOP 记录保留。 [机器可读审查回执](../campaign/evidence/dcr009-swept-guard-v2/independent-technical-review.json)

**结论 `TECHNICAL_PASS`。** SHA 为 `35121d6540c3de06905cca4d22cdfb92b2af987e69c0c5e14d016455e0c781bb` 的 V2 可在 25 ms、63 个细分查询、深度 7 的原配置下进入本次单次诊断。原生发布控制器和 STOP 继续裁决实际运行。能否全程跟踪以及期限表现由该次诊断实测。 [原始负控输出](../campaign/evidence/dcr009-swept-guard-v2/independent-controls.txt)

## 连续几何证明

V2 继续检查包含 `q_issued`、`q_measured` 和 `q_proposed` 的完整 54 轴逐坐标包围盒。逐个碰撞对在共同运动坐标系计算偏移界。若两个几何体共享同一铰链祖先，该铰链对二者施加相同刚体运动，距离保持不变。其余铰链的位移界取旋转弦长。最近铰链到碰撞体中心的垂直距离加 `geom_rbound` 包住该碰撞体的旋转半径；更远的铰链沿用 V1 的链长界。逐轴合并这些保守界后，只有采样距离扣除位移上界仍满足原 0.500 mm 净空与 0.002 mm 数值余量的子区间才能获得证明。 [V2 源码](../../../scripts/s0_5_campaign_teleop/swept_live_guard_v2.py) · [原几何库](../../../source/dexgym/tasks/direct/wholebody_s05/v1/task/native_sweep.py)

每个未获证明的碰撞对选取对自身位移界贡献最大的可用细分轴，并进入覆盖原子区间的两个子区间。缓存只可用于已证盒的子集，或用于从已采样距离按相同数学上界外推。深度、查询预算、时间、几何数值异常均拒绝。完整请求在两模型均通过且期限仍有效时才可交给原生发布检查。 [V1 权限与期限检查](../../../scripts/s0_5_campaign_teleop/swept_live_guard.py) · [原生发布接口](../../../scripts/s0_5_campaign_teleop/dcr009_teleop_guard.hpp)

## 实际独立负控

在 Ubuntu 的 `/usr/bin/python3.12` 和 MuJoCo 3.13.0 上核对原模型身份。所加载的实际模型、签名模型及全部模型依赖与前期回执逐项匹配。9 个 Python 检查和 8 个编译执行的 C++ 检查通过，退出码为 0，原始输出 SHA-256 为 `f3d857d5c38a3b0775b9f582cb26b34ac9de2bf3c3ce53753051a9b4a23efb8f`。 [负控实现](../../../scripts/s0_5_campaign_teleop/test_dcr009_swept_guard_v2_independent.py) · [原始输出](../campaign/evidence/dcr009-swept-guard-v2/independent-controls.txt)

| 受检情形 | 实际结果 |
| --- | --- |
| 两端及中点均有足够间隙，两侧子区间各有穿透负控 | 两侧均返回 `COLLISION`，先缓存安全姿态也拒绝 |
| 实际双模型上的双臂关节路径中途穿透 | V1、V2 均拒绝 |
| 正间隙 0.250 mm，0.501 mm 和 0.510 mm 的门限检查 | 依次为 `COLLISION`、`SWEEP_UNPROVEN`、`PERMIT` |
| 共享铰链及无关轴，128 个带固定子体旋转的斜轴位移观测 | 共动间距保持，观测位移均小于几何上界 |
| 细分查询、深度预算耗尽及绝对超时 | 均不许可，已证明缓存亦服从期限 |
| 硬关节限位、端点各自合法但盒角不合法的 J6/J7、非有限值、关节组所有权、旧序号和错误下发基准 | 均按原请求合约拒绝 |
| 仅签名模型否决 | 整体拒绝，已下发状态链无推进 |
| 原生反馈混合时间戳、组重复、失效、未来戳、过期和时钟倒退 | 7 项反馈负控均返回 `kNoProof`，完整新鲜反馈到达刻意未配置的私有传输并返回 `kTransport` |

复现命令在 `/home/ubuntu/dexgym_worktrees/s05-dcr009-diagnostic` 下运行。`CAMP`、`DR` 与历史运行回执中的项目运行目录一致。

```sh
CAMP=/home/ubuntu/dexgym_data/s0-5/campaign/S05-V1-20261127-490770e
DR="$CAMP/runs/HOME-ANCHORED-DIAGNOSTIC-DCR009-01"
PILOT="$CAMP/runs/HOME-ANCHORED-PILOT-DCR009-01"
export PYTHONPATH="$PILOT/read_only_preflight/python_imports:$DR/preflight/python_imports:$PWD/source:$PWD/scripts/s0_5_campaign_teleop"
/usr/bin/python3.12 -B scripts/s0_5_campaign_teleop/test_dcr009_swept_guard_v2_independent.py \
  --live-model "$DR/native/install-supported/whole_body_mujoco_driver/share/whole_body_mujoco_driver/model/dual_arm_marvin.xml" \
  --signed-model source/dexgym/assets/marvin_wuji/hand2_2/generated/dual_arm_marvin.xml \
  --model-receipt docs/s0-5/campaign/evidence/dcr009-stage-two/model_frame_receipt.json
```

## 记录解释与适用范围

- 证明 6985、7020、7023 关联已记录的几何证明不足。6985 的现场模型右无名指远节与小指远节在关键采样姿态间隙约 5.949 mm，原证书需要 6.299 mm。观察点间隙、连续几何证明和运行中实际轨迹是三类证据。 [6985 见证](../campaign/evidence/dcr009-root-five-history/terminal-sweep-witness.json) · [7020 和 7023 历史报告](DCR-009_root_five_one_unproven_arm_results.md)
- 配对只读重放对上述三个请求均从 V2 的两个模型得到连续许可。高预算复核需要的 V1 细分次数分别为 127、255、2047，V2 分别为 4、4、6，现场与签名模型的计数一致。较早的原中心左右手请求 3、4 在两版正式预算内仍无足够证明。早期拒绝没有实测穿透见证，其原盒也未获安全证明。 [早期双手几何见证](../campaign/evidence/dcr009-root-five-segmented/original-hand-sweep-witness.json)
- 两套模型使用当前签名的碰撞目录。右手无名指远节、小指远节和小指指尖传感器框的碰撞几何均纳入检查。可视网格与实际碰撞网格角色已区分，当前许可继续按原目录及净空执行。 [签名模型](../../../source/dexgym/assets/marvin_wuji/hand2_2/generated/dual_arm_marvin.xml) · [模型身份](../campaign/evidence/dcr009-stage-two/model_frame_receipt.json)
- 原生配置为 500 Hz 驱动步进和 120 Hz 反馈发布目标，历史回放的 50 Hz 对齐用于评分。驱动把臂部延迟、滤波和限速后的 `ctrl` 与手部目标送进 `mj_step`，随后测得动态反馈。三个 (q) 向量的包围盒证明依然是原有几何契约。现有观测没有定标能包住每个物理子步、反馈迟滞、超调和异步更新的更小动态包络。 [历史时间和反馈证据](../campaign/evidence/dcr009-root-five-history/terminal-and-timing.jsonl) · [历史运行技术报告](DCR-009_root_five_recorded_history_results.md)
- 共同偏航、原中心、先腕后指、原时间轴和安全边界的诊断资格与腕掌外参物理资格分开。所有结论均为 `DIAGNOSTIC_ONLY`，正式 85%／70% 验收沿用原分母。 [历史 START 诊断](DCR-009_root_five_wrist_first_results.md)
