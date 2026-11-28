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

## 单次实际回放后的独立核验

只读核对 `HOME-ANCHORED-DIAGNOSTIC-DCR009-ROOT05-SWEPT-GUARD-V2-01`。运行清单与运行目录均只存在第 5 根的 `attempt-01`。该试次达到 START，历史段发送 1,692／46,112 包后保护停机。最后原始输入位于 START 后源时间 3.520408078 s，计划长度为 96.068267152 s。守卫执行的是已审查的 V2，实际预算及两套模型身份与技术回执匹配。 [启动绑定回执](../campaign/evidence/dcr009-swept-guard-v2/independent-technical-review.json)

- **发布许可完整。** 7,245 个唯一守卫请求含 7,236 个双模型许可和 9 个 `SWEEP_UNPROVEN`。7,236 条原生发布逐一匹配隔离守卫原始请求、命令组和两模型 `PERMIT`。未许可发布、例外记录和原生日志矛盾均为 0。9 条未证请求都留在拒绝记录中。
- **先出现手部拒绝，随后是双臂保护触发。** 左手请求 7232 先失败，实际模型通过，签名模型 63 次细分用尽。之后仍有合法命令获得发布。触发原生保护的是双臂命令 4945／证明 7245。实际模型用 48 次细分获得许可，签名模型达到 63 次查询上限，最小采样间隙 1.1321 mm，整体返回 `SWEEP_UNPROVEN`。从接收到判定耗时 6.675 ms，命令的发布标记为 0。
- **保护触发与看门狗记录顺序。** 原生拒绝的 ROS 时刻为 `1795865645.917478189`。控制器 `COMMAND_STEP` 的消息源戳为 `1795865645.917520782`，原因明确为双臂没有获得可发布的扫掠证明，触发锁存停止。随后驱动状态源戳 `1795865645.917634786`，比 `COMMAND_STEP` 晚 114.004 μs，报告 `command_allowed=false`、`position held; home required before next enable`。观察器先收到了驱动状态，回放器随后写入 `DRIVER_NOT_COMMANDABLE`。跨话题到达顺序与消息源戳逐项保留。失败保护的分类以守卫的查询上限及控制器源戳为准。
- **STOP 边界通过。** 独立重算得到 `STOP_BOUNDARY_PASS`。同一会话世代 2 的 `STOP_BARRIER` 原生单调时刻为 `176172216834772` ns，最后发布序号为双臂 4944、左手 1171、右手 1173。STOP 服务接受于 `176172221718883` ns，观察器完成边界于 `176172222238456` ns。边界后原生新发布、迟到观测和无法判定命令均为 0，拥有的进程已结束，隔离 ROS 域已清空。该审计验证发布边界，驱动消费每条命令的独立时间戳资格仍未建立。

以下 SHA-256 从实际只读运行目录直接计算。`stop-audit.json` 的 `recorded_phase` 读取了最终判定前的审计输入，最终试次回执将实际结果归档为 `HISTORICAL_REPLAY_GUARDED_STOP`。

| 原始文件 | SHA-256 |
| --- | --- |
| `run/run_receipt.json` | `a35180d1082163aa61fe9c6e2ee7717b0f7c4055d37d7f5f99c3837c5787d1ae` |
| `run/attempt-01/result.json` | `016eb34557c302954bbcd1d86c6d221b282411afc22b8a1265e3752bf5031e12` |
| `run/attempt-01/stop-audit.json` | `30eab7fe6b7c1a14ffd9932688c02f4f090528ec2dd0debb2b44a8b1fa31abbb` |
| `run/attempt-01/sim.log` | `f91cd5fcb454b980247eaadd8871ea152f2c27fa158943411b69faa7de748bd9` |
| `run/attempt-01/evidence/trace.jsonl` | `e70c540aaebcef205e095c81809c4d6820d1d7fe232cc1996758802723811ec8` |
| `run/attempt-01/sweep-trace.jsonl` | `70261964f49c6cf9c8c0c14c92f8a16273e3771f96456deca66335f93b4d3a2b` |
| `run/attempt-01/sweep-v2-metrics.jsonl` | `7945cbdc5b10249a4d32e664ec9e8a33926e3830f24e1d0df5ea6fbf02e8694d` |
| `run/attempt-01/sweep-v2-ready.json` | `0f2a9ebac523303e628b7b55e60f9e2755e7830275d2b6bc5026588256505e00` |
