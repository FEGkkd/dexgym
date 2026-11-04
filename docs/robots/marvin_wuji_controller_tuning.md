# Marvin/Wuji Newton TCP controller experiment

**2026-11-04.** The 20–30 mm fixed-TCP wrist tracking error was largely caused
by the imported arm drive damping, not by FK. A bounded, Newton-only,
**opt-in** controller profile reduced the final error to about 0.56 mm on both
arms for the tested 14.324° in-place wrist rotation. The source
`dual_arm_marvin.xml`, converted USD, joint limits, force limits and real robot
settings were not edited. Full measurements are in
[marvin_wuji_controller_tuning.json](marvin_wuji_controller_tuning.json).

## What changed and why

The pinned source XML has arm Kp `400/400/440/340/200/120/85` and Kv
`45/45/40/34/23/16/12`. The converted USD authors these numeric values as
angular drives; Isaac Lab 3.0/Newton MJWarp exposes **both runtime Kp and Kv
at 180/π times the source numbers**. For example, J1 reads Kp 22,918.31 and
Kv 2,578.31 in Newton's SI joint-gain buffers, compared with source 400 and
45. J6 reads Kp 6,875.49 and Kv 916.73, compared with source 120 and 16.
This is the installed conversion path's observed unit behavior, checked
through the actual runtime arrays. The separate passive joint damping audit
in [the original validation](marvin_wuji_validation.md) concerns a different
parameter and remains valid.

The selected `newton_arm_tcp_tracking` profile keeps imported Kp and all
effort limits, and sets **only the 14 arm Kv values** to half the source XML
values (J1 22.5, J6 8.0 N·m·s/rad). Hands keep their imported drives. The
implementation checks the expected 54-joint model and imported gains before
writing, then verifies that Kp and effort limits did not change. The default
asset controller remains the imported USD configuration. This high-Kp profile
is an experimental simulation controller, **not** a motor calibration or a
setting for real hardware.

## Results

The same 750-step, 0.002 s fixed-TCP trajectory was run from Home for each
arm, with a 14.324° commanded world-Z rotation. FK command error remained
below 0.001 mm and 0.0001°. The table uses a fresh baseline and tuned run on
the packaged USD; Newton's GPU baseline varied by a few millimeters between
runs, while repeated tuned runs stayed around 0.56–0.60 mm.

| Controller | Arm | Final Newton TCP position error | Final orientation error | Arm drive samples clipped by effort limit |
| --- | --- | ---: | ---: | ---: |
| Imported USD | Left | 25.274 mm | 6.411° | 96.76% |
| Imported USD | Right | 23.008 mm | 5.335° | 96.70% |
| Tuned Kv = 0.5 × source | Left | **0.565 mm** | **0.0668°** | **0%** |
| Tuned Kv = 0.5 × source | Right | **0.562 mm** | **0.0672°** | **0%** |

Effort clipping was measured from Isaac Lab's implicit-actuator
`computed_effort` against the unchanged joint effort limits. Baseline applied
effort reached 100% of the limit; tuned peak applied effort was 43.7% (left)
and 43.5% (right). Maximum step-to-step joint jumps were below 0.014 rad in
both cases; no nonfinite state or joint-limit violation appeared. In a
separate 18-environment damping sweep, the selected profile's mean TCP
position error after reaching the final command was approximately 0.60 mm,
with P95 approximately 0.64 mm. Tested Kv scales 0.25–1.0 × source produced
similar submillimeter results; 0.5 was chosen as a central value, rather than
optimizing one endpoint sample.

The original MuJoCo controller still has approximately 33 mm final TCP
position error. To compare the **same controller law** in both backends, the
validator can alter only the compiled in-memory `MjModel` arm gains to match
the tuned Newton values. With this option, MuJoCo final TCP error was
0.598/0.593 mm (left/right), and Newton versus MuJoCo final TCP position
difference was **0.036/0.034 mm** and orientation difference
**0.024/0.023°**. The source XML file remains byte-for-byte unchanged.
Comparing tuned Newton against the original MuJoCo controller gives a large
backend difference because their controllers are intentionally different.

The tuned profile also passed three Home resets × 200 steps in 16 environments
(maximum joint jump 0.0189 rad), all 54 independently opposed joint channels
in 16 environments (minimum response separation 0.157 rad), and 16 bounded
static/step/ramp/sine dynamics scenarios. No NaN/Inf or joint-limit violation
was observed. Under matched arm gains, final J5/J6 and combined-wrist joint
differences were near zero, but the **maximum transient difference across all
54 joints reached 0.418 rad** in an arm step. The full robot's dynamics and
contact response are therefore **not** generally calibrated to MuJoCo or
real hardware.

## Reproduce

Set `DEXGYM_MARVIN_MJCF` to the authorized source XML and run from the DexGym
root. On the verified host, prepend
`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` to Python commands if
the Conda C++ runtime takes precedence.

```bash
export DEXGYM_MARVIN_MJCF=/path/to/whole-body-teleop/src/whole_body_mujoco_driver/model/dual_arm_marvin.xml
USD=source/dexgym/assets/marvin_wuji/usd/dual_arm_marvin.usda

# Original imported controller and original MuJoCo controller.
uv run --no-sync python scripts/assets/validate_wrist_rotation.py --usd "$USD" \
  --output artifacts/marvin_wuji/wrist_baseline.json --csv artifacts/marvin_wuji/wrist_baseline.csv

# Tuned Newton controller, with an equal-gain in-memory MuJoCo comparison.
uv run --no-sync python scripts/assets/validate_wrist_rotation.py --usd "$USD" \
  --damping-scale-source 0.5 --match-mujoco-controller \
  --output artifacts/marvin_wuji/wrist_tuned.json --csv artifacts/marvin_wuji/wrist_tuned.csv

uv run --no-sync python scripts/assets/validate_dynamics.py --usd "$USD" \
  --mode smoke --num-envs 16 --steps 200 --resets 3 \
  --damping-scale-source 0.5 --output artifacts/marvin_wuji/tuned_smoke16.json
uv run --no-sync python scripts/assets/validate_dynamics.py --usd "$USD" \
  --mode channels --num-envs 16 --steps 150 \
  --damping-scale-source 0.5 --output artifacts/marvin_wuji/tuned_channels54.json
uv run --no-sync python scripts/assets/validate_dynamics.py --usd "$USD" \
  --mode dynamics --num-envs 1 --steps 250 \
  --damping-scale-source 0.5 --match-mujoco-controller \
  --output artifacts/marvin_wuji/tuned_dynamics.json --csv artifacts/marvin_wuji/tuned_dynamics.csv
```

For another standalone Newton test, call
`apply_newton_arm_tracking_profile(scene["robot"], damping_scale_source=0.5)`
after the Isaac Lab scene has initialized or reset. It does not register a Gym
task or modify the default articulation config. The remaining roughly 0.6 mm
error is a measurement for this one feasible Home-centered wrist path; it is
not a full-workspace accuracy guarantee. Arm S0 task definition can use this
profile as an experimental baseline, while broader motion, load and contact
validation remains necessary before training or deployment claims.
