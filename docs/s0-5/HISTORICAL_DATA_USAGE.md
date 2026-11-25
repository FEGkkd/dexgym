# 历史数据使用与 Final Test 封存 v0.1.0

## 现有使用证据

P00 Review Fix 只读枚举各家族少量 manifest／状态及其哈希。Marvin54 的旧 Ubuntu 数据目录在本轮缺失，本地既有报告归档仍可读取，记录了 EgoProStandard-body 使用和 Train/Val/Test 三种历史角色。样本列表不能穷尽被使用的录制或赋予任何母源独立性。[历史使用样本](evidence/P00_review_fix/historical.json) · [Marvin54 归档见证](evidence/P00_review_fix/marvin54_archive_witness.json)

| 家族 | P01-A 需要锁定的记录 | 默认用途约束 |
|---|---|---|
| Wholebody S0 | 日期与合并 Motion Library、native replay、参考筛查和 retime manifest，episode/span 映射及配置 | 与历史训练、验证、测试、回放交集的根仅入开发池 |
| S0-1 | targets bank、mixed/teleop 与稳定性或切换集、heldout 结果、拇指和硬样本 | 已分析难片段及其同根均不能作为独立终评 |
| S0-2 | 工作区中的 hand delta/servo 试验目录、训练／诊断包、原始 target bank 与慢放／重定向前驱 | 查明继承的 S0/S0-1 与衍生区间，未知先开发或隔离 |
| Marvin54 | 旧 motion library、生成源、flattened data、train/val/test、标定或消融分析 | Teleop 需有独立 source provenance。GRAB/CMU 等其他来源依法标记并排除 |
| S0-4 | 真实 train provenance 中的 bank、train/validation/test、refit/q_ref/SDK 重定向、collision/retime、hard pose 选择及评估快照 | 全部重复用过的测试和难段并入历史使用登记，EgoPro 先源级拒绝 |

发现的 manifest 略取样，P01-A 需追其父 manifest、run 配置、选型/评估/诊断产物和截取区间。缺历史文件或已无法唯一反查时状态为 `USAGE_UNKNOWN`。普通日志、路径缺失或未见记录不作为未使用的证明。

## 登记与传播

`historical_usage.v1` 每条含 `usage_id, artifact_family, artifact_path_or_uri, artifact_git/fingerprint, observed_role, method, date_or_UNKNOWN, episode_id, root_episode_id, event_interval/raw_index_span, transform_chain, source_evidence[], confidence, reviewed_by`。角色枚举 `TRAIN, VAL, MODEL_SELECTION, REPEATED_TEST, HARD_CASE_ANALYSIS, TUNING, CALIBRATION, INTEGRITY_ONLY, USAGE_UNKNOWN`。可用注册对象 ID、记录 SHA、规范化采样指纹、采集器 session 与时间区间、对照人工登记。根级传播取 lineage 连通分量及多父祖先的所有已知角色，禁止只根据文件名或当前 split 放行。

| 规格 | 方针 |
|---|---|
| 重命名/同字节 | SHA、逻辑 recording group 和角色继承，重复副本不计为独立录制 |
| 裁剪、慢放、增强、重编码、retarget、筛查 | 使用有证据映射还原父区间。未能排除重合时保留候选记录并隔离判断 |
| 旧数据缺独立性凭证 | 已能证明原始 Teleop 时可进入开发池；待补来源／反查不应默认为 Test |
| 基础完整性审计 | 批准后可看 metadata、schema、缺失、标定、哈希以及预注册质量窗。登记为 INTEGRITY_ONLY，隔离控制跟踪表现和调参 |

## 独立终评

P01-D 冻结 train、validation、test 家族和 split hash，根 episode 组两两交集为零。候选最终测试需要独立采集的录制清单、未用于以上历史用途的证明、与开发源的重复检测以及人工确认。当前 **`Final Test = BLOCKED / eligibility unproven`**，开发集可以推进。P09 开工前必须具备 `FINAL_TEST_SEALED` 状态并引用封存 manifest SHA、来源证明和访问日志。

若现有资料不足，由不同采集 session 重新录制，保存设备标定、操作者/日期分层与任务覆盖的预注册证明，分配互斥源/责任人。录制完成只由授权数据管理员做完整性/来源/字段检查，保留原件哈希和逻辑签名。封存目录或访问域限制 P02–P08 的训练与性能调参，写入 `access_log.jsonl`，字段含人/角色、时间、对象哈希、目的和读范围。P09 配置和策略冻结后只读开放，产生一次正式记录。任何后续调参需有新独立终评机制。
