# P00 可复现审计证据

Run ID `S05-P00-20261125-ubuntu-audit`。Ubuntu 运行的 JSON 记录观察时间，`argv/cwd/exit/stdout/stderr` 表示实际命令及结果。源码证据保留文件摘要与行号，原始数据仅保存元数据和哈希。

| 文件 | 证据范围 |
|---|---|
| [host.json](host.json) | 四个 checkout 的身份、tracked diff、未跟踪路径摘要、软硬件与实际导入 |
| [data.json](data.json) | NPZ schema 与 SHA、原始路径扫描、Motion Library manifests |
| [raw.json](raw.json) | 采集平台链接目录、ROS2 bag info、54 DoF CPU MuJoCo 模型步进 |
| [followup.json](followup.json) | 四个原始输入反序列化、wrapper 测试复试、S0-4 bank 与配置、Wuji retarget 源 |
| [s04.json](s04.json) | 历史 pause state、两组 checkpoint SHA、Servo parity、provenance |
| [source.json](source.json) · [code.json](code.json) | 远端源码片段及函数行索引，比对发布、运行、main 与 Teleop |
| [tests.json](tests.json) · [test_summary.txt](test_summary.txt) | 初次环境不匹配与 wrapper 复试/跳过原因 |
| [document_validation.txt](document_validation.txt) | 文档与证据一致性检查的实跑 |
| [probe.py](probe.py) · [runtime_probe.py](runtime_probe.py) · [check_documents.py](check_documents.py) | 只读探查与可重跑一致性检查 |

在现场 checkout 下使用 `~/miniconda3/envs/wuji/bin/python` 运行 `probe.py host`、`data`、`s04`、`code`，再运行 `runtime_probe.py source`、`raw`、`tests`、`followup`。探查时通过 SSH 将脚本送入 Python stdin 执行，证据可用 `--transport` 压缩传递并还原。解释器、ROS overlay、特定磁盘路径随现场改变，报告应记录新基线。`tests` 专门保留首次 Wuji Python 的导入失败，`followup` 内按项目 wrapper 尝试。`check_documents.py` 可在该文档目录所在 Git 仓库用 Python 3.12+ 运行，结构通过和真实验收的含义见 [报告](../../reports/P00_acceptance.md)。

历史记录仅在读取时冻结为证据，未启动历史训练作业，未发布 ROS 控制消息。
