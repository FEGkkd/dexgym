# Marvin 双臂 + Cube 同源 Hand2.2 双手升级

2026-11-11。默认整机资产已由旧 Beta2 手替换为 Hand2.2。右手复用已跑通
Cube Reorientation 的手模型和物理配置；左手使用同一官方版本的左手资产，
保留左手几何、惯量和关节轴，并按关节对应采用 Cube 的驱动参数。
本次没有训练策略，也没有修改单独 Cube 或 Hand21 PPO 的模型/任务。

## 当前入口与来源

- `source/dexgym/assets/marvin_wuji/robot_cfg.py`：默认整机入口，54 关节，默认开启手部接触。
- `source/dexgym/assets/marvin_wuji/hand2_2/generated/dual_arm_marvin.xml`：组合 MJCF。
- `source/dexgym/assets/marvin_wuji/hand2_2/generated/usd/dual_arm_marvin.usda`：当前整机 USD，使用 `mujoco` variant。
- `source/dexgym/assets/marvin_wuji/hand2_2/generated/metadata.json`：关节顺序、Home 等关键帧、SI 驱动参数、组合源文件哈希。
- `source/dexgym/assets/marvin_wuji/build_hand2_2.py`：可重建的组合/转换脚本。
- `source/dexgym/assets/marvin_wuji/hand2_2/provenance.json`：上游版本和输入文件哈希。

机械臂源版本 `5d4c345842f197bffeae7d7e08c4c3718230f9e0`；官方手源版本
`829c9aa27cb12c3a49d5b8d63c9869dd8f656702`。该官方版本的右手 XML 和全部右手
STL 都与当前 Cube 打包输入逐字节一致。
组合 XML SHA256：`939b3651e2426e596eed482e70a68c2382720df1fd55b124aa5f0893b640e92d`。

右手的关节限位、armature、质量/惯量、接触几何及接触参数、Kp/Kd 和力矩限幅
与 Cube 源模型对齐。左手的胶囊碰撞端点先在掌坐标系镜像，再转换到官方左手
各连杆的局部坐标，不能直接把右手局部坐标复制过去。Cube 的固定手掌世界位姿、
标记和展示夹具不属于机器人手本体，没有装到机械臂上。

机械臂几何、动力学参数、关节方向及 TCP-to-mount 安装变换保持不变。
全体驱动使用显式 SI 配置，修正旧 USD 导入路径的 180/pi 增益放大问题；
没有使用旧的高 Kp / 减小 Kd 实验控制器。导入前显式保存固定底座/支撑件的
推断惯量，避免 USD 导入器重新按网格估算。对网格碰撞采用与 Cube 一致的
高顶点数凸包设置。原 `usd/` 仅保留为历史资产，不由默认入口或驱动诊断选中。

## 验证结果

原始 JSON 见本目录的 `hand22_validation/`。

| 检查 | 结果 |
| --- | --- |
| 组合模型对独立源模型，100 组姿态、11,677 项检查 | 通过；右手相对掌位置误差最大 6.94e-16 m；左手 5.48e-8 m；机械臂 TCP 改变为 0 |
| Newton/MuJoCo FK，1,000 姿态、16 环境 | 通过；TCP 最大位置误差 0.000950 mm；全体连杆最大位置误差 0.00305 mm |
| 54 关节独立正反向控制，16 环境 | 全部响应；最小正反位移差 0.172883 rad，无失败通道 |
| 16 类保持/阶跃/斜坡/正弦动作，每类 500 个 2 ms 步 | 状态有限、无关节越界；受测关节最终后端差异最大 0.0005804 rad；所有关节全程差异最大 0.014311 rad |
| 实际 solver 中的 PD 增益 | 与配置的最大差异 0；确认位置/速度分拆驱动的增益和 bias 一致 |
| 实际质量、惯量、被动阻尼、armature、摩擦及手部接触参数 | 通过；质量误差最大 8.29e-7 kg，惯量张量误差最大 2.14e-6 kg m² |
| 总力矩限幅，4 环境 × 3 次重置 × 300 步极限目标 | 通过；最大总驱动力矩/限幅 1.00000004；重置位置误差 0 |
| 接触 | Home 两后端均零接触；手–支撑和双臂交叉都能在 Newton 中检测到对应接触 |
| 干净 wheel 安装 | 117 个新资产文件逐字节核验；独立导入/编译 54 关节；USD 未解析依赖为 0；未带回旧训练任务 |
| 仓库测试 | 66 passed，8 skipped；跳过项为外部源环境变量或本地数据依赖 |

动态验证的差异是两个仿真后端对相同模型/控制器的差异，不等于目标跟踪误差。
例如保持动作中受测关节仍有约 0.033 rad 的跟踪误差；没有添加重力补偿来掩盖
这一点。整机源模型保持相同的重力和有限 PD 力矩，其跟踪误差需要与实际
控制器要求分开评估。

## 复现

在仓库根目录运行，使用当前项目运行时：

```bash
bash scripts/supermimic/host.sh scripts/assets/validate_hand22_composition.py --output /tmp/composition.json
bash scripts/supermimic/host.sh scripts/assets/validate_hand22_runtime.py --output /tmp/runtime.json
bash scripts/supermimic/host.sh scripts/assets/validate_fk_parity.py \
  --usd source/dexgym/assets/marvin_wuji/hand2_2/generated/usd/dual_arm_marvin.usda \
  --samples 1000 --num-envs 16 --output /tmp/fk.json
bash scripts/supermimic/host.sh --pytest tests -q
```

## 结论边界

这些检查证明当前组合、单位、控制通道、有限力矩驱动、重置及已测试的仿真
响应一致性通过；不代表全部工况的接触轨迹逐位相同，更不代表已完成真机
辨识。左右手的实际电机响应仍需真机数据验证；左手使用的是右手 Cube 的对应
关节增益，不能称为已做左手独立标定。旧整机策略也不能直接视为适配新动力学。
单手 Cube 已训练策略不因此自动获得移动手腕或双臂协同能力。
