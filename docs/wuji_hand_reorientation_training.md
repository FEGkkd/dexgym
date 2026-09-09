# WUJI Hand Reorientation Training

训练和 play 使用仓库根目录的 `rsl_rl/` 入口。先执行：

```bash
source scripts/activate_wuji.sh
```

这个脚本激活 `wuji` Conda 环境，并指向本机的 `/home/ubuntu/IsaacLab`、
`/home/ubuntu/isaacsim-6.0.1` 和 `/home/ubuntu/wuji-description`。官方 WUJI
USD 是运行时唯一的手资产来源，任务实现不会转换或修改 USD 文件。

训练配置 `WujiHandReorientationDirectEnvCfg` 提供三个独立的显式开关：
`enable_curriculum`、`enable_domain_randomization` 和 `enable_disturbance`。
每个开关对应一个带参数的嵌套配置组 `curriculum`、
`domain_randomization` 和 `disturbance`，默认全部开启以匹配官方训练。

`scripts/train.sh` 会进入 `rsl_rl.base.train`，例如：

```bash
scripts/train.sh --task DexterousHands-Wuji-Hand-Reorientation-Direct-v0 --num_envs 256
```

`scripts/play.sh` 会关闭课程、域随机化、外力扰动和观测噪声，然后加载并导出
checkpoint：

```bash
scripts/play.sh --task DexterousHands-Wuji-Hand-Reorientation-Direct-v0 \
  --load_run dexteroushands_wuji_hand_reorientation_direct_isaac
```

策略契约保持 20 个 action、69 维单帧、3 帧 207 维 actor 输入和 413 维
asymmetric critic 输入。原始 action 历史独立于 EMA 和关节限幅控制目标。
PhysX 可应用的随机化值记录在 critic 的 `dr_params` 和运行 extras 中；
当前后端不支持的字段会被记录为 unsupported，不会伪装成已应用的参数。
