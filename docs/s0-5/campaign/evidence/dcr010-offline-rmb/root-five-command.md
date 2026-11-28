# 第 5 根实际运行命令与身份

候选执行使用 Ubuntu 已有的 `manus_wuji_real` Python 环境来载入 CPU NumPy、Torch 和 MuJoCo。工程名不代表连接真实机器人。入口直接使用 MuJoCo 和本地 C 接口，未启动 ROS 图、驱动或训练。首先在 `056be67` 运行 `ROOT05-01`，随后在 `c2b5720` 运行 `ROOT05-02`。直到第二次任务写完回执且退出，才切换 Ubuntu 工作树用于只读分析。 [首次原始文件 SHA](run01-raw.sha256) · [第二次原始文件 SHA](run02-raw.sha256)

```bash
cd /home/ubuntu/dexgym_worktrees/s05-dcr009-diagnostic
campaign=/home/ubuntu/dexgym_data/s0-5/campaign/S05-V1-20261127-490770e
diagnostic="$campaign/runs/HOME-ANCHORED-DIAGNOSTIC-DCR009-01"
phase1=/home/ubuntu/dexgym_worktrees/teleop-sim0-study
python=/home/ubuntu/miniconda3/envs/manus_wuji_real/bin/python
out="$campaign/runs/DCR010-OFFLINE-RMB-ROOT05-02"
env PYTHONPATH="$PWD/source:$PWD/scripts/s0_5_campaign_teleop" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    timeout 900 "$python" -B scripts/s0_5_campaign_teleop/build_offline_rmb_diagnostic.py \
    --plan "$diagnostic/plan/pilot_plan.json" \
    --cohort "$campaign/runs/TELEOP-SHARED-RIGID-START-HANDOFF-FULL-01/cohort.json" \
    --model "$PWD/source/dexgym/assets/marvin_wuji/hand2_2/generated/dual_arm_marvin.xml" \
    --model-receipt "$PWD/docs/s0-5/campaign/evidence/dcr009-stage-two/model_frame_receipt.json" \
    --frames "$diagnostic/plan/diagnostic_runtime_frames.txt" \
    --bridge /dev/shm/s05-resume-490770e/teleop-native-02/obj/libs05_teleop.so \
    --config "$phase1/third_party/marvin_kinematics/config/ccs_m6_40.MvKDCfg" \
    --phase1 "$phase1" --wuji "$phase1/third_party/wuji_c_api/lib/libwuji_sdk_c.so" \
    --out "$out"
```

初次将 `out` 设为 `DCR010-OFFLINE-RMB-ROOT05-01`。目录独占创建，已封存结果不覆盖。输入计划 SHA-256 为 `3c0268291bdb8916a9a520d3fa73a1340fffd7f9e8a7b7364a1ef9950f9a0004`，C++ 桥为 `510b1f10620ac69aa277d85e3399921a9050521aee10b934b62502dbd09ee5d4`，经动态链接核验的 `libKine.so` 为 `f2aec6ebc1f41302fe0126284076211c044d6c7e1c2276dc919767254e0c597f`。 [其他资产 SHA](run02-receipt.json) · [原生 C 接口预检](native-nominal-preflight.json)

只读检查运行 `audit_offline_rmb_diagnostic.py --run "$out" --out "$out/verified-audit.json"`，并用相同模型、`$diagnostic/plan/anchors/b63bc72d2e8b2ee395e966f6.json` 和同一回执执行 `render_offline_rmb_diagnostic.py`。视频没有生成额外机器人反馈。 [复算结果](run02-verified-audit.json) · [视频回执](run02-video.json)
