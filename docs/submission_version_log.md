# Submission Version Log

## v1 - Generated Draft

- Original continuation-batch generated paper and toy single-seed experiment.

## v2 - Submission Hardening

- Added hostile reviewer attack log and response docs.
- Replaced the toy experiment with seven-seed synthetic metrics, stronger synthetic baselines, ablations, stress tests, and negative cases.
- Terminal decision: WORKSHOP_ONLY.

## v3 - ICLR Main Gate Archive

- Applied the stricter ICLR-main-conference standard.
- Determined that missing real-robot/high-fidelity evidence, template-generated experiments, and unresolved novelty threats were not recoverable from local artifacts.
- Terminal decision: KILL_ARCHIVE.

## v4 - Real MuJoCo Negative-Evidence Rebuild

- Added concrete rebuild plan.
- Replaced synthetic scaffold with real sequential MuJoCo repair-memory benchmark.
- Added six stress splits, persistent deployment streams, implemented baselines, ablations, paired statistics, learning curves, and figures.
- Found that embodied repair memory fails to beat nominal/robust MPC.
- Terminal decision: KILL_ARCHIVE.
