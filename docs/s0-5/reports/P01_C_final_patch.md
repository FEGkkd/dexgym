# P01-C 最终边界补丁与人工签核申请

审查基线 `aecfd2deff75a564b2ef70538f09fbc997cbd4a3`。独立 Run ID 为 `P01-C-REVIEW-FINAL-20261126-aecfd2d`，最终复验为 `attempt-02`。本次仅补齐离线插值失效边界及同记录冲突处理。原审核报告、运行凭据、DCR-001/002、历史阶段事件和既有实验产物均保留。[原审查](P01_C_review_fix.md) · [本次运行凭据](../evidence/P01_C_final/final_receipt.json)

## 旧版本反例与补丁

| 上线前测试 | 基线上的实际结果 | 修订后 |
|---|---|---|
| SAFE t=0、Pose A t=10、可信 UNSAFE t=15 且 capture/arrival 未知、Pose B t=20 | t=15 插值错误有效 | `UNSAFE`，输出无效。出现新可信 SAFE 并取得其后的新鲜姿态才恢复 |
| Pose A/B 之间出现可信 Pose 失效，时间资格为 UNUSABLE | t=15 插值错误有效 | `UNSYNC`，姿态与速度均无效。新鲜可证端点重新建立后恢复 |
| Pose B 的 authenticated=false | t=15 插值错误有效 | B 被拒绝，缺少可信右端点。未认证的 SAFE 也没有许可效力 |
| 同来源、同记录身份，validity、freshness、calibration 分别冲突 | 三个子断言失败 | 离线隔离冲突记录。因果路径按确定性事件键处理，正反入队全状态一致且冲突无效 |

测试文件 `tests/s0_5/unit/test_p01c_offline_boundary.py` 在未修改的基线上执行了 8 项测试，退出 1，保留 7 个失败断言（含冲突子测试）。修订位于 `source/dexgym/tasks/direct/wholebody_s05/data/synchronize.py`。时间有界的可信失效用其已有界产生屏障。时间未知时从该流最后一个可信采样界开始保守隔离，缺先前界则隔离当前段。恢复须有可信、新鲜、时间有界的事件出现在已观测失效之后。处理过程中保留 capture/arrival 未知，观测时刻仅约束恢复。认证、源世代、标定和拓扑隔离同时适用。已知旧序姿态和真正重复包可丢弃，当前权威安全流中首次收到的晚到否决仍会生效。

`attempt-01` 拦截了新反例，但一项前序测试发现失效端点被误报为 `DROPOUT`。日志已保留。最终版本在允许的原有插值时间范围内保留端点失效原因，并重新运行全部 P01 单测及固定真包集成测试。

## 固定语料复验

重新读取同七个母源、23 个窗口、46 秒、2,300 拍，核对原 MCAP、消息配对和采集期标定 SHA。13,640 对已录软件腕位标定比对保持一致。age 60 ms、skew 35 ms、gap 60 ms、safety timeout 100 ms 和模拟到达 8 ms 均保持冻结。

| 验证项 | 审查基线 → 最终补丁 |
|---|---|
| 全部 P01 单测与固定真包集成 | 78 项单测、3 项集成全部通过 |
| 四路因果联合有效 | 2,248 → 2,248，逐拍零变化 |
| 四路离线联合有效 | 2,211 → 2,211，逐拍零变化 |
| 每流姿态与速度有效性 | 两种对齐均无掩码翻转。因果六流各 2,250 拍姿态有效、2,249 拍速度有效。离线双手和 raw 腕各 2,248 拍，注册腕各 2,249 拍，姿态与速度相同 |
| 失效原因 | 离线左注册腕及左 raw 腕各有 49 条既有无效记录由 `DROPOUT` 改为 `UNSAFE`。因果失效原因无变化 |
| 安全与恢复 | 因果许可保持 2,250 个 SAFE、50 个 UNSAFE，两版逐拍一致。负控覆盖跨未知失效、恢复前旧缓存和未认证许可 |
| 前缀不变性与批流等价 | 23/23 窗口通过未来数据修改、删除、锚点隔离及全状态批流等价，对比 1,173 个前缀拍 |
| 技术门 | C1-GEO 和 C2-SIM 均 PASS |

上述指标限于已审历史 HMB 仿真回放。真实 `VERIFIED` 控制拍仍为 0。C2-RT、物理 robot-base 注册、P01-T、Final Test 继续 `BLOCKED`。

## P01-D/E 数据交接边界

以下是生产数据契约的交接要求，等待 P01-D 单独批准并冻结。位姿、速度和安全均为独立字段。

| 字段 | 生产准入约定 |
|---|---|
| `pose_valid`、`velocity_valid` | 每拍、每流独立布尔掩码。速度禁止从姿态掩码或数值是否为零推定 |
| `velocity_age_ns`、`velocity_quality` | 以单调域纳秒记录。速度无效时用显式未知与对应质量枚举。离线插值斜率和模拟因果速度保留不同标记 |
| `safety_permission`、`joint_valid` | 均逐拍保存。安全许可来自认证且有界的安全证据；四路联合门检查两手和两个注册腕，raw Tracker 为诊断。离线产品另外保留整个插值区间的安全与跨度判定 |
| `clock_quality`、`clock_generation` | 保存同步时域、资格及世代，区分 SIMULATED、PROXY、VERIFIED、UNUSABLE。未知须显式保留 |
| `source_index` | 因果保留选中包的资产、物理索引及哈希，离线保留双端记录、跨度及权重。无源时标明原因 |
| `calibration_id`、`topology_id`、`reason_code` | 保存实际逐流版本、节点身份和版本化原因字典。丢失、重置和屏障不可默认为有效 |

原 `export_alignment` 的 NPZ 只保留部分 mask、数值和时间字段，缺独立速度资格、安全许可和时钟世代等必需证据。历史 `sync_audit` 入口保留 P01-C 有界诊断语义，不能作为生产构建入口。P01-D 应冻结 schema、修订核心的 Git/源码哈希、场景用途与原因字典。P01-E 的生产入口应绑定该版本，拒绝旧/未知 schema，并在写入前后用跨屏障、跨世代、缺认证、逐帧 traceability 及因果前缀测试校验。原 P01-C attempt-03 的旧 NPZ 仅作历史复验，不能进入正式训练数据。

## 复现、完整性与停止点

```bash
cd /home/ubuntu/dexgym_worktrees/s0-5
base=/home/ubuntu/dexgym_data/s0-5/p01/runs
run="$base/P01-C-REVIEW-FINAL-20261126-aecfd2d"
bash docs/s0-5/evidence/P01_C_review/run_review.sh "$run" \
  "$base/P01-C-20261126-8b0f399/attempt-03" \
  "$base/P01-A-20261126-09c1c9e-r03/registry" \
  "$base/P01-B-20261126-46ed5e2/attempt-04" \
  "$base/P01-B-REVIEW-20261126-f32d6bb-r02" attempt-02
python3 -B docs/s0-5/evidence/P01_C_final/compare_and_seal.py "$run" \
  "$base/P01-C-REVIEW-20261126-0d9d351/attempt-03" attempt-02
```

保留基线红测、首轮补丁失败、正式单测与集成命令、stdout/stderr/退出码。复现需用新目录和 attempt。封存时 Git HEAD 为审查基线并应用了补丁。源码、测试与实际执行入口的 SHA 在运行凭据，配对 `final_delta.json` 与逐拍追溯、阈值和技术门报告在 `$run/attempt-02/products/`。产物 SHA 清单为 `2b21ca708918c8cca47f71054523583ccfd48ec81031aea97f3da382f9289f27`。

申请对历史 HMB 仿真资格作最终人工签核，阶段仍为 `READY_FOR_REVIEW`。P01-D/E/F、P02/P03、全量数据处理、训练和硬件操作均未执行。
