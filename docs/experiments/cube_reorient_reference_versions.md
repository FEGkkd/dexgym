# Cube reference source versus published model

The native run follows reference main `2becb7380fed86171056f9bf4338b2a6a0c14b01`.
The public 8000-iteration model is an older trained artifact. Its successful
playback validates useful behavior in the native scene, but does not establish
that the current reference source trains to that behavior from scratch.

This audit reads tensor shapes and configuration directly from the published
model and our native iteration-2000 checkpoint. Machine-readable evidence,
checkpoint hashes, and native run provenance are in
[cube_reorient_validation/release_version_audit.json](cube_reorient_validation/release_version_audit.json).

| Item | Published model | Current source / native run |
| --- | --- | --- |
| Actor normalization | RunningNorm: sqrt(var + eps squared) | EmpiricalNorm: sqrt(var) + eps |
| Actor hidden widths | 512, 256, 128 | 512, 256, 128 |
| Critic hidden widths | 512, 256, 128 | 512, 512, 256, 128 |
| Physical context | 66 geometry-friction values + 40 gain ratios | 6 DR ratios + 40 gain ratios |
| WM input dimension including control target | 193 | 133 |
| Critic input dimension | 268 | 211 |
| Failure-head input dimension | 326 | 269 |
| Parallel environments | 1024 | 4096 (user requested) |

Both checkpoints agree on all shared algorithm configuration keys other than
logging and saving intervals: WM width 2048 x 3, window 8, 10 WM batches of 2048,
policy batch 1024, horizon 32, UTD 0.4, 64 initial collection steps, 500 WM-only
iterations, learning rate 0.0002 and 250000-step decay, gamma 0.99, lambda 0.95,
and exploration noise 0.1. Environment count lives in native run provenance,
not the native algorithm configuration. Equal vector-step UTD does not mean
equal individual-transition reuse when environment count changes.

Source history comparison uses release-v1 tag
`1fe2d744892debd05f6eb8034509cb5873f14ff1`, release-v2 tag
`9d70d45501b82ddf0a2d715c05e1d2f3db271979`, and pinned main. The published
checkpoint identifies its source asset as the v1 model; it does not record the
exact training source commit. A tag is therefore a comparison point, not proof
of the training commit.

The source diff also replaces uniform multinomial replay draws with randint,
caches failure pools, and batches recorded failure features across time. These
retain the intended distributions/features, but do not preserve random-number
streams. Old diagnostic and horizon-validation helpers were removed; their
presence in old source alone is not evidence that they affected the released
fixed-recipe training run.

No training code or running configuration was changed for this audit. Current
native learning remains unproven (one completed goal across 256 episodes at
iteration 2000). Continue the scheduled 4000/8000 evaluations. If the run fails,
these verified version differences are candidates for controlled comparison,
not established causes of failure or permission to introduce new mechanisms.
