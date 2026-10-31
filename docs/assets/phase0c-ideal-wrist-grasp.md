# Phase 0C: ideal wrist and physical finger grasp

**Measured outcome:** fixed and commanded wrists work with the official Beta 1 hand in PhysX. The 70 mm bottle still fails the one-second left-only suspended hold. There are **0** validated initial states; no grasp bank was promoted. All success claims below are limited to the listed tests.

## Models and control

The formal hand asset is the official left/right WujiHand2 Beta 1 USD from `wuji-description` tag `v2026.9.11`, commit `c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95`. The measured USD SHA-256 values are `8d1b83a6130b64fd759ea3c0720d6b255da8c1e76abe5a4226788183058dbfda` (left) and `22600d3f0aeb185552bb6eeb454b5b7199966a60fc844d4a4b5e4791ec77f0ed` (right). Both articulations exposed 20 finger joints. The previously measured Beta 1 marker mass correction remains: the five zero-mass fixed tip frames receive a 1 g / `1e-6 kg m²` positive scene-layer proxy each. This is a PhysX compatibility override, not a measured hardware property. Beta 1 remains an **unverified** match for the user's non-tactile hand.

`fixed_wrist` keeps the official `root_joint` world fixed joint and its articulation-root API. Finger joints retain the official USD position drives, 2 N·m maximum drive force, inertia, limits and collision geometry. Bottle and cap remain dynamic, with gravity and their D6/thread coupling active.

`pose_command_wrist` also keeps that world joint. [Isaac Lab 3.0's indexed root pose writer](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/api/lab_physx/isaaclab_physx.assets.html) updates the fixed-root anchor at each 2 ms physics substep; `WristPoseController` limits the commanded path to 0.15 m/s and 1 rad/s and writes a consistent world root velocity. The result does **not** integrate contact reaction into wrist drift. [Isaac Sim 6.0's fixed-base semantics](https://docs.isaacsim.omniverse.nvidia.com/latest/migration_guides/isaac_sim_6_0/urdf_mjcf_importer_exporter_pipeline.html) also require preserving the world-to-root fixed joint for an anchored articulation. We rejected the floating-root-write variant after its otherwise identical static contact produced only 1.15 N peak versus 23.95 N with a fixed root. With the world anchor retained, the peak was 23.56 N. This is a tested PhysX mechanism, not a finite-wrench PD. The old `floating_wrench_pd` remains an explicit legacy experimental mode and remains the default of old fixture call sites for compatibility; new ideal-wrist scene users should request `pose_command_wrist`.

Both hand actions are `[world Δx, Δy, Δz (m), wrist-local Δrx, Δry, Δrz (rad), 20 canonical finger position targets (rad)]`, so one row is 52D. `BimanualPhase0Fixture.command_action52(env_ids, action_rows)` checks workspace and official joint limits before submitting indexed targets. Absolute position and `xyzw` quaternion targets are supported by `WristPoseController.set_target`; relative rotation is left-composed in the local wrist frame. The controller keeps per-environment target/current trajectory state and Reset clears command velocity. A fixed wrist accepts zero wrist increments plus normal finger targets. All scene/world and action quaternions use `xyzw`; the checked-in MJCF proposal file uses `wxyz` and the synthesis runner converts it explicitly.

The five official `*_tip` marker **rigid-body frames** are read from the current `body_link_pose_w` PhysX tensor after `scene.update()`. Their measured movement during finger closure was 5.9–8.6 mm. They are distinct from distal link origins and collision contact points. `FingertipSiteTracker` reports bottle-local axial and radial distances, signed distance to the cylinder surface, the related distal collider and exact-pair PhysX contact points. The optional GUI overlay shows all five sites, their target surface points, bottle axis, contact points and normal directions. Contact force is a signed PhysX **scalar** per point; the friction vector available from this backend is an aggregate **per pair**, so it must not be assigned to one point. Force-resultant direction is diagnostic until the backend's source/partner sign convention is independently calibrated.

The engineering bottle has radius 35 mm, body mass 30 g and cap mass 6 g. Those are not measurements of a real bottle. The contact peak guard (100 N) and penetration guard (2 mm) screen anomalies; they are not certified grasp-force thresholds. The user has not provided actual bottle geometry or torque data.

## Native GPU measurements

All results used Isaac Sim 6.0.1, Isaac Lab 3.0 and PhysX GPU. The compact numeric record, finger torque vectors, tip positions, contact points/normals, pair friction and short static/moving trajectory are in [phase0c_results.json](phase0c_results.json). The [raw GPU evidence bundle](phase0c_raw_physx_evidence.tar.gz) contains the complete 18 JSON/log traces used for the table (`tar -xzf docs/assets/phase0c_raw_physx_evidence.tar.gz -C /tmp`). The official-MJCF proposal and 5 mm preshape input are [reach proposals](phase0c_reach_35mm_mjcf.json) and [preshape proposals](phase0c_preshape_35mm_mjcf.json). MJCF is only the **offline geometry proposer**; native USD/PhysX determines pass/fail.

| Experiment | Measurement | Status |
| --- | --- | --- |
| Fixed wrist, 70 mm bottle, proposal 0 | 0 root translation drift; initial contact separation `-0.055 mm`; peak 23.95 N; after support removal bottle moves 0.496 m / 1 s; left-bottle contact ratio 9.6%. | **Hold Fail** |
| Pose wrist, same preshape and bottle | Static contact peak 23.56 N; separation `-0.057 mm`; release moves 0.496 m / 1 s; contact ratio 8.4%. | **Contact parity Pass; hold Fail** |
| Pose wrist, 2 environments, 52D actions | Both sides independently translate/rotate, max root position error `5.65e-8 m`, max orientation error `2.07e-7 rad`; driven left/right finger motion 0.2001/0.2000 rad; command speed at 0.15 m/s and 1 rad/s bounds. | **Pass** |
| Static versus +5 mm X commanded wrist, same reset contact state, 50 ms, support removed | Contact present on all 25 steps in both trials; bottle motion differs 6.35 mm. Static/moving peaks 1.57/2.49 N; moving deepest separation `-0.254 mm`; moving wrist error `<5e-8 m`. | **Pass for tested contact motion** |
| Fixed proposals 1 and 4 | Proposal 1 exceeds force guard at 127 N. Proposal 4 exceeds force and penetration guards at 530 N / `-2.11 mm`, despite its 5 mm MJCF clearance prediction. | **Rejected** |
| Fixed proposals 2 and 3 | No guard abort, but bottle moves 0.487/0.492 m; release contact ratio 0.2%/2.4%. | **Hold Fail** |
| Contact-point-directed proposals 6 and 1 | Official MJCF closest collision points were optimized for thumb opposition and given 5 mm preshapes. Native PhysX rejected them at 4/6 ms: 241/124 N distal contact peaks, finger speeds 14.6/13.5 rad/s and 12 drives near their 2 N·m limit, with fixed wrist speed zero. | **Rejected** |
| Pose selective Reset, 1/2/16 environments | 100 random selective resets each; unselected state and controller caches unchanged before physics step; detached thread re-engaged; contact history invalidated; max root restore error `0` / `5.96e-8` / `2.38e-7`. | **Pass** |
| GUI marker launch | `--viz kit --visualize_grasp` completed without marker API errors. | **Launch Pass; visual readability not formally rated** |
| Legacy 1 s hold validator | Seed 0 runs with the new opposition metric and remains failed: bottle displacement 0.545 m, left-only 0.692 m, opposition coverage 0.2%. | **No false promotion** |

The proposal-0 support stage did create real finger contact. At the last settle sample, thumb distal force was 21.13 N, index distal 1.06 N, and ring/middle distal contacts were on nearly the **same side** of the cylinder. Their bottle-local contact angles were about `142°`, `66°`, `-142°` and `-145°`. The maximum uncovered angular arc was `3.635 rad / 208°`, exceeding the 180° opposition criterion; the radial force imbalance ratio was 0.85. The tip and physical contact coordinates differ as expected. The applied-drive torque buffer reached the official 2 N·m cap across all 20 joints at the last settle sample. Once support was removed, correct contact collapsed and the bottle fell. These data identify missing opposing force closure in the tested posture; they do **not** prove that the 70 mm bottle is unreachable, nor isolate torque limit from geometry as the only cause. The MJCF/official USD clearance disagreement for proposal 4 also remains a concrete geometry-model limitation.

The old Phase 0B ~4 cm first-step penetration came from resetting a half-closed hand directly into the 35 mm-radius bottle. This phase starts from a collision-checked preshape, closes finger joints through normal drives and rejects deep penetration or large impulses. It does not cure the missing opposition.

The finite targeted follow-up is in [contact-point proposals](phase0c_antipodal_proposals_35mm_mjcf.json), [their preshapes](phase0c_antipodal_preshape_35mm_mjcf.json) and [the stricter geometric screen](phase0c_antipodal_strict_35mm_mjcf.json). It optimized the *closest collision points* instead of tip-site origins from eight fixed seeds. A stricter offline objective could create a 169° maximum angular gap only with 6.7 mm MJCF penetration, so that posture was not promoted. The two best clearance-oriented proposals then produced the measured native PhysX drive/contact impulses above. Their finger speeds were already 12.9–18.1 rad/s on the first contact-free step, despite targets initially matching the reset joint positions. `GraspSynthesisFixture` now rejects finger speed above 8.5 rad/s in the stationary preclose approach phase and above 25 rad/s during driven closing as anomaly guards. This narrows the current obstruction to the collision/drive dynamics of the tested approaches and the absence of safe opposing force closure. It does not prove an inherent geometric impossibility for the hand or bottle.

The raw 124/241 N impulse traces predate the new preclose speed guard; a current run stops these postures on the first high-speed step, before the large collision. The original traces remain in the evidence bundle to explain the guard. The guard does not make either candidate valid.

`GraspSynthesisFixture` captures the **measured** engaged cap angle, axial overlap, stick reference and sliding state. `BimanualResetManager` restores these at an episode boundary and checks engagement consistency. The old reset check incorrectly demanded nominal cap height after the cap had already rotated; that prevented replay of a physically formed state. Reset never steps global simulation, and the bottle is not moved between ordinary steps. A contact state now replays for the short wrist comparison. There is still no stable state to perform the required independent one-second hold replay.

## Gates

| Gate | Status | Evidence or limit |
| --- | --- | --- |
| A — official hands, mass and collision | **Pass** | Left/right native USD, 20+20 active joints, marker-mass override retained; contact and finger motion measured. |
| B — fixed wrist and physical fingers | **Pass** | Root drift zero, finger torques bounded, real contact influences bottle. |
| C — dynamic tip sites | **Pass** | Five nonempty current PhysX frames move 5.9–8.6 mm. |
| D — opposing contact measurement | **Pass** | Exact points/normals, coverage, pair friction and joint torque logged; actual candidate lacks opposition. |
| E — one-second left-only hold | **Fail** | 0/5 targeted 70 mm candidates; 0 validated states. |
| F — six-axis pose command and contact | **Pass for tested scenario** | Two-environment 52D motion and static/moving free-bottle contact comparison. Broader contact envelopes remain untested. |
| G — pose-command grasp hold/motion | **Fail** | Static one-second hold fails; moving a successfully held bottle could not be tested. |
| H — independent stable-state Reset replay | **Not Tested** | No candidate passes the prerequisite hold. Contact-state replay works. |
| I — Reset and old-task regression | **Pass for scoped regression** | Pose reset 1/2/16 ×100 and focused contracts pass. Full suite has two unrelated pre-existing user-worktree Allegro/MANO failures. |

The complete repository pytest run was **98 passed, 2 failed, 1 skipped** with Isaac paths configured. The two failures were `test_allegro_config_declares_contract_and_resolves_reference` (missing user-worktree Allegro asset path) and `test_build_21_keypoints_uses_mano_order_and_fingertips` (unrelated MANO ordering assertion). Phase 0C's focused offline wrist, contact, asset, thread, grasp and reset contracts were **44 passed, 5 skipped**. Existing single-hand Wuji contract tests were among the 98 passing tests. Neither unrelated user-worktree area was edited by Phase 0C.

## Reproduce

Use a GPU workstation with the pinned official USD/MJCF checkout, Isaac Sim 6.0.1 and Isaac Lab 3.0. Set `WUJI_DESCRIPTION_ROOT` and `ISAACLAB_PYTHON` for that installation. These commands write raw traces outside the source tree; the compact checked-in record above is from this run.

```bash
./scripts/run_isaaclab.sh scripts/environments/prepare_collision_free_preshape.py --reach_scan docs/assets/phase0c_reach_35mm_mjcf.json --output /tmp/phase0c_preshape.json
./scripts/run_isaaclab.sh scripts/environments/optimize_antipodal_contacts.py --reach_scan docs/assets/phase0c_reach_35mm_mjcf.json --max_nfev 60 --output /tmp/phase0c_antipodal.json
./scripts/run_isaaclab.sh scripts/environments/generate_grasp_synthesis.py --headless --reach_scan docs/assets/phase0c_preshape_35mm_mjcf.json --proposal_index 0 --candidate_count 1 --control_mode fixed_wrist --fixed_support --start_at_goal --output /tmp/fixed.json
./scripts/run_isaaclab.sh scripts/environments/generate_grasp_synthesis.py --headless --reach_scan docs/assets/phase0c_preshape_35mm_mjcf.json --proposal_index 0 --candidate_count 1 --control_mode pose_command_wrist --fixed_support --start_at_goal --output /tmp/pose.json
./scripts/run_isaaclab.sh scripts/environments/validate_wrist_pose_motion.py --headless --num_envs 2 --output /tmp/motion.json
./scripts/run_isaaclab.sh scripts/environments/validate_wrist_pose_contact.py --headless --reach_scan docs/assets/phase0c_preshape_35mm_mjcf.json --proposal_index 0 --output /tmp/contact.json
./scripts/run_isaaclab.sh scripts/environments/validate_bimanual_contacts.py --headless --output /tmp/contacts.json
./scripts/run_isaaclab.sh scripts/environments/validate_bimanual_reset.py --headless --control_mode pose_command_wrist --num_envs 16 --iterations 100 --output /tmp/reset.json
./scripts/run_isaaclab.sh scripts/environments/generate_grasp_synthesis.py --viz kit --visualize_grasp --reach_scan docs/assets/phase0c_preshape_35mm_mjcf.json --proposal_index 0 --candidate_count 1 --control_mode fixed_wrist --fixed_support --start_at_goal
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./scripts/run_isaaclab.sh -m pytest -q tests
```

The minimum next experiment is a **contact-angle constrained** official-USD search or controlled teleoperation of the thumb opposite the long-finger patches, with USD collision clearance and actual normal force distribution checked before release. Then repeat the 1 s fixed-wrist test and an independent Reset replay. If the 70 mm cylinder still fails, separate the effects of hand span, 2 N·m joint saturation, friction and cap/thread angular motion with controlled ablations. Do not promote geometry-only or support-stage states to `GraspInitBank`.
