# Cube final evaluation: reference across evaluation seeds

The public actor is a diagnostic baseline in the native scene, not a native
training result. Each evaluation uses 64 environments and four complete episodes
per environment, deterministic actions, observation noise, startup/reset
randomization, and interval pushes with a fresh reference curriculum.

| Public actor evaluation seed | Goals/episode | Episodes with >=1 goal | Episodes with >=3 goals | Drop fraction | Mean angle (rad) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 123 | 6.50391 | 0.86328 | 0.64453 | 0.66016 | 0.83461 |
| 456 | 6.44141 | 0.81641 | 0.64453 | 0.67969 | 0.84803 |

The same published checkpoint is used in both evaluations. These two seeds
probe evaluation randomization, not training-seed variability. The checkpoint
hash and per-episode records are stored in the corresponding JSON reports.

The native scene-v2 run is still training from random initialization, seed 42,
4096 environments. Its iteration-8000 checkpoint will be evaluated with the
same two evaluation seeds. The seed-456 evaluation is queued after the primary
seed-123 final evaluation to avoid concurrent evaluation processes. Native final
performance is not yet available; no claim of completed reproduction follows
from these reference-model results.
