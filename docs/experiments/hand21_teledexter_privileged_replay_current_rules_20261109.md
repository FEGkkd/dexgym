# Current-rule Hand 2.1 privileged replay (2026-11-09)

## Method

Run the current `DexGym-Wuji-Hand21-TeleDexter-Direct-v0` in Newton with 4,096 environments per split and curriculum stage, 500 policy steps at 30 Hz per run, seed 20261109. The controller reads the current `q_ref` and converts it to the existing deadzone residual action; all goal sampling, 5–15-step dwell, 20 mm all-finger threshold, cross-trajectory switching (probability 0.1 after success), failure budget, and automatic reset use the active task implementation. Every segment appears at least once in the first reset; later starts use normal random RSI. This is a privileged controller diagnostic, not PPO.

`initial` starts at the curriculum minimum (upper span 20); `full` starts after the 25,600-step curriculum ramp (upper span 40). The ramp then advances normally within each 500-step run. Span 0 is a cross-trajectory target with a 300-frame failure budget. Forward span `s` has failure budget `1.5s` out-of-tolerance policy frames. `first` is the first policy step on which all five fingertips enter the 20 mm threshold, and `full` is the step on which the required consecutive hold completes. Both are shown in policy steps (one step = 33.33 ms), with p50/p90 calculated over successful goals. A budget timeout is recorded when the task terminates with `n_fail > budget`; other termination and episode truncation are separate.

## Overall results

| Split | Stage | Settled goals | Reached | Budget timeout | Other termination | Cross target reached | Span=1 reached / tried |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | initial | 165,063 | 165,021 | 42 | 0 | 22,635/22,638 | 7,831/7,831 |
| train | full | 147,890 | 147,856 | 34 | 0 | 26,134/26,139 | 3,772/3,772 |
| validation | initial | 175,372 | 175,351 | 21 | 0 | 25,204/25,204 | 8,351/8,351 |
| validation | full | 161,672 | 161,653 | 18 | 1 | 31,062/31,062 | 4,006/4,006 |

All four runs had zero episode truncations during these 500-step windows. Each run ended with 4,096 active goals censored by the measurement window. Across the four runs, span=1 reached **23,960/23,960** with zero budget timeouts; its first-threshold p50/p90 were 1/1 step and full-hold p50/p90 were 10/14 steps in every split/stage. This is strong sampled evidence that the two-outside-frame span=1 budget is attainable for the privileged controller, but it does not establish PPO reachability.
Across all spans and runs, **649,881/649,997** settled goals succeeded; 115
hit the failure budget and one ended for another reason.

## Results by exact target span

Each cell is `settled / budget timeout / other termination; first p50/p90; full p50/p90`. The first and full times use only reached goals. All 4,096 starting environments per run were distributed across the 942 train or 446 validation segments.

### initial curriculum

| Span | Train: N / timeout / other; first; full | Validation: N / timeout / other; first; full |
| ---: | --- | --- |
| 0 | 22,638 / 3 / 0; 9/17; 19/27 | 25,204 / 0 / 0; 7/12; 16/23 |
| 1 | 7,831 / 0 / 0; 1/1; 10/14 | 8,351 / 0 / 0; 1/1; 10/14 |
| 2 | 7,829 / 0 / 0; 1/1; 10/14 | 8,271 / 0 / 0; 1/1; 10/14 |
| 3 | 7,603 / 1 / 0; 1/1; 10/14 | 8,121 / 0 / 0; 1/1; 10/14 |
| 4 | 7,589 / 1 / 0; 1/1; 10/15 | 8,087 / 1 / 0; 1/1; 10/14 |
| 5 | 7,590 / 6 / 0; 1/2; 10/15 | 8,002 / 3 / 0; 1/1; 10/14 |
| 6 | 7,457 / 0 / 0; 1/2; 10/15 | 7,831 / 4 / 0; 1/1; 10/15 |
| 7 | 7,284 / 5 / 0; 1/3; 10/15 | 7,785 / 3 / 0; 1/2; 10/15 |
| 8 | 7,240 / 7 / 0; 1/3; 11/15 | 7,721 / 5 / 0; 1/2; 10/15 |
| 9 | 7,059 / 9 / 0; 1/4; 11/15 | 7,618 / 3 / 0; 1/3; 10/15 |
| 10 | 6,987 / 0 / 0; 1/5; 11/15 | 7,385 / 1 / 0; 1/3; 11/15 |
| 11 | 7,042 / 4 / 0; 1/5; 11/16 | 7,409 / 0 / 0; 1/4; 11/15 |
| 12 | 7,130 / 0 / 0; 1/6; 11/16 | 7,397 / 1 / 0; 1/4; 11/15 |
| 13 | 6,908 / 4 / 0; 1/6; 11/16 | 7,261 / 0 / 0; 1/5; 11/15 |
| 14 | 6,814 / 0 / 0; 1/7; 12/17 | 7,115 / 0 / 0; 1/5; 11/16 |
| 15 | 6,821 / 0 / 0; 1/7; 12/18 | 7,149 / 0 / 0; 1/5; 11/16 |
| 16 | 6,725 / 0 / 0; 2/7; 12/18 | 6,960 / 0 / 0; 1/5; 11/16 |
| 17 | 6,750 / 0 / 0; 2/8; 12/18 | 7,127 / 0 / 0; 1/6; 11/16 |
| 18 | 6,640 / 0 / 0; 2/8; 12/19 | 6,919 / 0 / 0; 1/6; 11/17 |
| 19 | 6,618 / 0 / 0; 2/9; 12/19 | 6,860 / 0 / 0; 1/6; 12/17 |
| 20 | 6,508 / 2 / 0; 2/9; 12/19 | 6,799 / 0 / 0; 2/7; 12/17 |

### full curriculum

| Span | Train: N / timeout / other; first; full | Validation: N / timeout / other; first; full |
| ---: | --- | --- |
| 0 | 26,139 / 5 / 0; 10/18; 19/28 | 31,062 / 0 / 0; 7/12; 16/23 |
| 1 | 3,772 / 0 / 0; 1/1; 10/14 | 4,006 / 0 / 0; 1/1; 10/14 |
| 2 | 3,730 / 1 / 0; 1/1; 10/14 | 4,106 / 0 / 0; 1/1; 10/14 |
| 3 | 3,704 / 0 / 0; 1/1; 10/14 | 4,052 / 0 / 0; 1/1; 10/14 |
| 4 | 3,649 / 1 / 0; 1/1; 10/14 | 4,163 / 4 / 0; 1/1; 10/14 |
| 5 | 3,590 / 5 / 0; 1/2; 10/15 | 3,929 / 0 / 0; 1/1; 10/15 |
| 6 | 3,484 / 4 / 0; 1/2; 10/15 | 3,948 / 5 / 0; 1/1; 10/15 |
| 7 | 3,518 / 3 / 0; 1/3; 10/15 | 3,922 / 5 / 0; 1/2; 10/15 |
| 8 | 3,437 / 2 / 0; 1/3; 11/15 | 3,954 / 2 / 0; 1/2; 10/15 |
| 9 | 3,548 / 2 / 0; 1/4; 11/15 | 3,874 / 1 / 0; 1/3; 10/15 |
| 10 | 3,377 / 0 / 0; 1/5; 11/15 | 3,650 / 0 / 0; 1/3; 11/15 |
| 11 | 3,262 / 2 / 0; 1/5; 11/15 | 3,567 / 1 / 0; 1/3; 11/15 |
| 12 | 3,389 / 0 / 0; 1/6; 11/16 | 3,612 / 0 / 0; 1/4; 11/15 |
| 13 | 3,292 / 1 / 0; 1/6; 11/17 | 3,598 / 0 / 0; 1/4; 11/15 |
| 14 | 3,147 / 0 / 0; 1/6; 11/17 | 3,457 / 0 / 0; 1/5; 11/15 |
| 15 | 3,186 / 0 / 0; 1/7; 11/17 | 3,433 / 0 / 0; 1/5; 11/16 |
| 16 | 3,153 / 0 / 0; 1/7; 12/17 | 3,456 / 0 / 0; 1/6; 11/16 |
| 17 | 3,185 / 0 / 0; 2/8; 12/18 | 3,337 / 0 / 0; 1/6; 11/16 |
| 18 | 3,057 / 0 / 0; 2/8; 12/18 | 3,312 / 0 / 0; 1/6; 11/16 |
| 19 | 3,103 / 0 / 0; 2/9; 12/18 | 3,244 / 0 / 0; 1/6; 12/17 |
| 20 | 2,970 / 1 / 0; 2/9; 12/19 | 3,164 / 0 / 0; 1/7; 12/17 |
| 21 | 2,992 / 0 / 0; 2/9; 12/19 | 3,234 / 0 / 0; 1/7; 12/17 |
| 22 | 2,897 / 1 / 0; 3/10; 13/20 | 3,176 / 0 / 0; 1/7; 11/17 |
| 23 | 2,950 / 0 / 0; 3/10; 13/20 | 3,149 / 0 / 0; 2/7; 12/17 |
| 24 | 2,985 / 0 / 0; 3/10; 13/20 | 3,210 / 0 / 0; 2/7; 12/18 |
| 25 | 2,814 / 0 / 0; 3/10; 13/20 | 3,066 / 0 / 0; 2/8; 12/18 |
| 26 | 2,840 / 0 / 0; 3/11; 13/21 | 2,974 / 0 / 0; 2/8; 12/18 |
| 27 | 2,832 / 0 / 0; 3/11; 13/21 | 2,930 / 0 / 0; 2/8; 12/19 |
| 28 | 2,751 / 0 / 0; 3/11; 13/21 | 2,911 / 0 / 0; 2/8; 12/18 |
| 29 | 2,716 / 0 / 0; 3/11; 13/21 | 2,806 / 0 / 0; 2/8; 12/18 |
| 30 | 2,738 / 2 / 0; 4/11; 13/21 | 2,898 / 0 / 0; 2/9; 12/19 |
| 31 | 2,751 / 1 / 0; 4/12; 14/22 | 2,831 / 0 / 0; 3/8; 12/19 |
| 32 | 2,654 / 0 / 0; 4/12; 14/22 | 2,780 / 0 / 0; 3/9; 13/19 |
| 33 | 2,649 / 2 / 0; 4/12; 14/22 | 2,742 / 0 / 0; 3/9; 13/19 |
| 34 | 2,575 / 0 / 0; 4/13; 14/22 | 2,685 / 0 / 0; 3/9; 13/20 |
| 35 | 2,635 / 0 / 0; 5/13; 14/22 | 2,655 / 0 / 0; 3/9; 13/20 |
| 36 | 2,555 / 0 / 0; 4/13; 14/23 | 2,646 / 0 / 0; 3/9; 13/20 |
| 37 | 2,528 / 0 / 0; 5/13; 14/23 | 2,590 / 0 / 0; 3/10; 13/20 |
| 38 | 2,471 / 0 / 0; 5/13; 14/23 | 2,539 / 0 / 0; 3/10; 13/20 |
| 39 | 2,442 / 0 / 0; 5/13; 14/23 | 2,556 / 0 / 0; 3/9; 13/20 |
| 40 | 2,423 / 1 / 0; 5/14; 15/23 | 2,448 / 0 / 1; 3/10; 13/20 |

## Interpretation and limits

The direct `q_ref` controller reached almost every sampled target under the current random-span, dwell, and cross-trajectory rules. Most failures were true budget timeouts, not the 30-second episode cap. This means the new task rules do not make the sampled reference goals broadly infeasible at the controller level. It does not imply that the current PPO observation, action distribution, or reward can learn this controller.

This experiment samples 500 steps per environment, rather than replaying every target of every segment to its end. The first reset covers all segment IDs; later target and cross-target frequencies depend on task sampling. The exact event CSVs and four JSON summaries are local at `artifacts/hand21/privileged_replay_current_rules/` and are excluded from Git. Reproduce with `scripts/evaluate_hand21_teledexter_privileged_replay.py --split {train,validation} --stage {initial,full} --num-envs 4096 --steps 500 --seed 20261109 --output artifacts/hand21/privileged_replay_current_rules` in the configured Isaac Lab runtime.
