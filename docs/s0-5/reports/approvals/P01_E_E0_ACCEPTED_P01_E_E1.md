# P01-E E0 人工签核与 E1 全量构建授权

2026-11-26，项目用户在独立审核后签核 `3f906526edccc108a1ff08e523da9559b16abcb0` 的 E0-1、E0-2、E0-3 及真包完整母源、八秒分区、自然尾段、恢复与幂等预检。运行凭据 SHA-256 为 `4dd4c3ea4c1b22a669c9b6d67ccbb3ec49b7653128ea88c728ed9562319337a4`。[E0 原凭据](../../evidence/P01_E_E0/review_receipt.json)

用户另行授权 Run ID `P01-E-E1-20261126-3f90652` 以单 worker 执行 `E1_PRODUCTION`。仅使用 32 个冻结母源、24/8 split、已批准同步器、生产入口和历史仿真 QC。

| 已批准对象 | SHA-256 |
|---|---|
| [配置](../../evidence/P01_E_E0/candidate_config.json) | `6aeaa203e39bea392bc20d5fceee26ea5f85752d219824f98f96b2c852846317` |
| [依赖锁](../../evidence/P01_E_E0/execution_lock.json) | `da3e1f0759ac010a989d9431f0337371df6fe208914139bd711be15354a453af` |

P01-E 开工状态为 IN_PROGRESS。执行身份出现差异时须暂停并交回人工审核。C2-RT、物理 robot-base 注册、P01-T 和 Final Test 保持 BLOCKED。P01-F、P02、P03、训练与真机动作需各自明确授权。
