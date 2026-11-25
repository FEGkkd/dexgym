# S0-5 P01 执行基线 v0.2.0

| 项目 | 本轮核验 |
|---|---|
| 原始审核发布 | GitHub `s0-5` 指向 `cc05f09847254ddae3b76581fb9e1feed4f31065` |
| 唯一 Ubuntu 开发 checkout | `/home/ubuntu/dexgym_worktrees/s0-5`，隔离 clone，分支 `s0-5`。预检时 HEAD = `cc05f09…`，tracked diff 空，status clean。Review Fix 文档在此 checkout 维护，任何后续生产变更按该分支 review |
| 历史工作区 | `~/dexgym` 的 `21a813a…` 与 `~/teleop_ws` 的 `30cf13b…` 有既有修改。保持其历史文件及运行目录。Windows 工作区只作为文档镜像和历史证据读取 |
| Python | `/home/ubuntu/dexgym_worktrees/s0-5/.venv/bin/python`，Ubuntu Python 3.12.3，独立依赖且显式复用系统 apt 包，`-P`、`PYTHONNOUSERSITE=1` |
| ROS 环境叠加顺序 | `/opt/ros/jazzy/setup.bash` → `~/teleop_ws/install/local_setup.bash` → `~/MANUS_STEAMVR_ws/dex_data_platform-cpp-collection/install/local_setup.bash`。清空继承环境后加载，再将 checkout 的 `source` 放在 Python 路径首位 |
| MCAP/CDR | ROS Jazzy `rosbag2_py` / MCAP storage 0.26.11、rclpy 7.1.12。锁定 `mcap 1.6.0` 与 `mcap-ros2-support 0.6.0`。分流 CDR 通过现有 ROS 消息类型并保留原始 wire clocks。旧合并 CDR 按录入 schema SHA 解码 |
| 系统数值依赖 | NumPy 1.26.4、SciPy 1.11.4、PyYAML 6.0.1、lz4 4.0.2+dfsg。独立 wheel 还含 gymnasium 1.2.0、cloudpickle 3.1.1、Farama-Notifications 0.0.4、zstandard 0.25.0 |
| 持久数据输出 | `/home/ubuntu/dexgym_data/s0-5/p01`，本轮创建空目录并核验可写。独立于两个 checkout 及源数据，与现有根盘同设备。正式 P01 仅在授权后写入按 Run ID 隔离的 staging 和发布目录 |
| 当前资源见证 | 输出目录约 684.43 GB，637.4 GiB 可用。同一次探查约 67.1 GiB RAM available，RTX 5090 总显存 32,607 MiB、占用 23,001 MiB。P01 CPU 抽样不需要 GPU。执行时刷新，不能作为全量空间或算力预留 |

实际 `sys.path`、解释器、精确 apt 版本和 `__file__` 见环境 JSON。`dexgym`、`wholebody_s0/bag.py`、`sync.py` 均解析到隔离 checkout，未观察到历史训练源码、runtime overrides 或临时历史 checkout 进入模块路径。Teleop 的 ROS 生成消息真实文件解析至其 `build/*/rosidl_generator_py`，采集器 `dex_collection_interfaces` 优先解析至 collector install。证据记录两套安装入口、有效的 package prefix、选用消息定义和 type-support SHA、核心 Python 文件及 setup 文件 SHA。项目依赖刷新后需复核解析位置与 hash。[环境与来源哈希](evidence/P00_review_fix/environment.json)

## 可复现实行

已用 `/usr/bin/python3 -m venv --without-pip --system-site-packages` 建独立环境。在可用 wheel 目录下用 `pip install --no-index --no-deps --require-hashes --find-links <wheelhouse> -r <requirements.lock>` 重建，验证 wheel 文件与锁匹配。MCAP、mcap_ros2_support、zstandard 来自已存在的 Ubuntu 工具 wheel 缓存，其余三个 wheel 从 PyPI 下载，保留首次超时和成功重试。系统 apt 包和自定义消息依赖本机受控安装，核验文件 SHA 与版本后才使用。旧 S0 `bag.py` 和 `sync.py` 的实际源码路径都已确认，P01 重新实现因果对齐。重复运行的入口是 `bash docs/s0-5/evidence/P00_review_fix/run_readiness.sh <probe.py> <new-audit-dir>`，脚本对输入和候选数据只读。[锁文件](evidence/P00_review_fix/requirements.lock) · [实际命令](evidence/P00_review_fix/commands.json)

空间预算方式见 [P01 `21](specs/P01_teleop_data.md#21-风险未定阈值及资源)。P01-D 需实测每格式注明读写与解码开销的受控样本，估算输出未压缩上界、临时分片并发、重试副本和空闲预留。实际写入前刷新可用字节与挂载。P00 旧 NPZ 的体量仅作样本参考，已保存本轮观察，尚未量测 P01 的生产膨胀率。[样本与大小证据](evidence/P00_review_fix/samples.json)

## 未解决版本差异

- ROS 生成消息/安装叠加与旧录包的嵌入 schema 不同。旧合并包中左右 HandState 用当前 ROS 类型尝试失败，已通过嵌入 schema 解码有限样本。P01-B 对允许的 schema 版本建立白名单及覆盖。
- 此基线依赖宿主机 ROS 配置和外部消息产物。复用的是已记录的消息文件，不从历史实验加载其任务源码。将来重建 overlay 或 apt 更新后须重新验证依赖身份。
- 主仓库、S0-4 发布与训练 checkout 之间的 MJCF/FK/Servo 差异保留在 P00 审计。P02/P03 冻结控制用途的模型与物理参数。P01 的原包解析不提前选择控制模型或硬件配置。
- 旧 recording 的方位标定与 controller receive/物理时钟无法由文件存在推断。在 P01-C 标注时序可证性，在 P03/P10 对完整链路做实际测量。
