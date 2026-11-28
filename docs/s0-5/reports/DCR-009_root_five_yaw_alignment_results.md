# DCR-009 第 5 根原中心共同偏航 START 对照

结论为 **DIAGNOSTIC_ONLY / START_NOT_REACHED**。唯一新增运行是第 5 根 `capture:b63bc72d2e8b2ee395e966f6` 的分段 Home→START。使用上一次 5° 左 J6 余量试次作为归档基线。同一控制器及驱动二进制、两模型扫掠守卫、原分段速率、原人体输入和零中心偏移通过核对。本次只对初始双腕目标施加共同世界 Z 轴旋转。 [运行回执](../campaign/evidence/dcr009-root-five-yaw/run_receipt.json) · [执行资产](../campaign/evidence/dcr009-root-five-yaw/runtime-assets.sha256)

从人体 START 的左→右腕连线到实测 Home 左→右掌根连线，在水平面计算有向最小转角。围绕 START 同一中点施加 **−85.68618°**，左右目标旋转同时左乘同一世界旋转。转后水平对齐误差低于 1e−12°，原间距 **564.809 mm**、中点及左右相对 SE(3) 在浮点容差内守恒。其余每侧变换和签名目标来源保持原值。此单例诊断并未建立录包朝向或人体腕掌安装的物理资格。 [独立重算与对照](../campaign/evidence/dcr009-root-five-yaw/comparison.json)

| 工程 TCP 指标 | 原方案 | 共同转向 |
| --- | ---: | ---: |
| 左腕从 Home 到 START 的完整 SO(3) 目标转角 | 79.239° | 22.757° |
| 右腕从 Home 到 START 的完整 SO(3) 目标转角 | 82.185° | 28.173° |
| 最后完整双腕实测位置误差，左／右 | 234.404／213.021 mm | 0.042／0.226 mm |
| 最后完整双腕实测朝向误差，左／右 | 32.662／35.341° | 0.015／0.282° |
| 原链路看门狗保护停机 | 3.1259 s | 1.9061 s |

转角比较在同一测得的 Home 下重算后，左减少 **56.482°**，右减少 **54.012°**。末次实测属于各自终止前有效样本，时间不同。转后共有 15 个双腕同时进入 10 mm／5° 范围的不同反馈样本，最后样本为 1.8841 s。以上为工程 TCP 误差。独立登记腕标记的对照字段保留单独口径，人体的真实掌面仍未定标。双手分别放行 6/86 与 4/86 次，终点缺少可核对的原 Wuji 手指目标反馈，完整 START 无交接。 [完整配对证据](../campaign/evidence/dcr009-root-five-yaw/comparison.json)

最终停机源为原整身守卫。臂组第 471 条、证明 643 的候选关节命令与上一次下发相同，守卫仍须覆盖实测与下发间的状态包络。实际加载模型许可，签名模型在第 5 层最后未决细分中，对**右腕与右无名指近节**得到 1.28581 mm 的中心间隙，未达到该子区间的 1.29311 mm 证明要求，耗尽 63 次细分查询上限。原请求及前缀复核无分歧，发布标志为 0。终止后没有观察到迟到命令，隔离 ROS 域已清空。 [守卫见证](../campaign/evidence/dcr009-root-five-yaw/arm-terminal-sweep-witness.json) · [终止命令](../campaign/evidence/dcr009-root-five-yaw/terminal-joint-delta.json) · [原生安全事件](../campaign/evidence/dcr009-root-five-yaw/terminal-events.jsonl)

短片左为原方案，右为共同转向。双方各 42 帧，取实际观察时间 0.22–1.86 s，慢放三倍，视频 SHA-256 `aac05c75b4c32b731bb77d2bbfb25a7df5211998636e485289214cc44740a1a6`。运行前 3 项合成控制通过，验证共同偏航的不变量、原始位移分支、START 后不回放。只执行本次单母源预定位，未执行历史回放、其他母源、正式 RMB 重建、训练和真机。 [原方案渲染](../campaign/evidence/dcr009-root-five-yaw/unaligned-render-receipt.json) · [对齐渲染](../campaign/evidence/dcr009-root-five-yaw/aligned-render-receipt.json) · [测试日志](../campaign/evidence/dcr009-root-five-yaw/preflight-tests.txt)
