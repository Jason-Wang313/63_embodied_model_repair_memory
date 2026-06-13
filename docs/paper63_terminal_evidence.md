# Paper 63 Terminal Evidence

Decision: KILL_ARCHIVE.

ICLR main ready: no.

Submission-hardening version: v4 real MuJoCo rebuild.

## What Changed

- Replaced synthetic probability tables with a real sequential MuJoCo repair-memory benchmark.
- Added persistent hidden deployment streams with recurring mass, friction, actuation, and obstacle shifts.
- Added implemented baselines: random, nominal MPC, robust MPC, last-repair memory, global-average repair, and oracle hidden deployment.
- Added six stress splits, five seeds, 24 episodes per split/method, confidence intervals, ablations, paired deltas, learning curves, figures, and reproducible CSVs.

## Main Evidence

Embodied repair memory success rates:
- nominal: 0.225 +/- 0.075
- low_friction_shift: 0.058 +/- 0.042
- high_friction_shift: 0.142 +/- 0.063
- heavy_object_shift: 0.175 +/- 0.068
- actuation_bias_shift: 0.217 +/- 0.074
- combined_shift: 0.117 +/- 0.058

The proposed method does not beat nominal MPC or robust worst-case MPC on the decisive splits, and it is not consistently better than global-average or last-repair memory.

## Ablation Evidence

On combined shift, embodied repair memory reaches 0.117 success and 0.134 regret. Nominal/reset memory reaches 0.158 success, while oracle reaches 0.258 success and zero regret. This means persistent embodied repair memory is not the component that rescues performance.

## Terminal Reason

The benchmark is real, but the idea fails its own test. The correct terminal state is KILL_ARCHIVE, not STRONG_REVISE and not ICLR_MAIN_READY.
