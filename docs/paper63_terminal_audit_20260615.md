# Paper 63 Terminal Audit

Date: 2026-06-15

Paper: `63_embodied_model_repair_memory`

Decision: `KILL_ARCHIVE`

ICLR-main ready: no

## Commands Executed

- `python -m py_compile src\run_experiment.py`
- CSV finite/schema audit over `results/repair_memory_raw.csv`, `results/repair_memory_metrics.csv`, `results/repair_memory_pairwise.csv`, `results/repair_memory_ablation.csv`, `results/repair_memory_seed_metrics.csv`, `results/repair_memory_learning_curve.csv`, `results/negative_cases.csv`, and compatibility CSVs.
- `pdflatex`, `pdflatex` in `paper`
- `Copy-Item paper\main.pdf C:\Users\wangz\Downloads\63.pdf -Force`

## Verified Evidence

- Real sequential MuJoCo repair-memory benchmark is implemented in `src/run_experiment.py`.
- Main evidence contains 5,040 paired rows: 6 stress splits, 5 seeds, 24 sequential episodes per seed/split/method, and 7 methods.
- Ablation evidence contains 840 rows on the combined-shift split.
- Learning-curve evidence contains 126 rows over early/middle/late deployment phases.
- Baselines include random candidate, nominal MPC, robust worst-case MPC, last-repair memory, global-average repair, and oracle hidden deployment.
- The rebuilt PDF is `C:/Users/wangz/Downloads/63.pdf`.
- `C:/Users/wangz/Desktop/63.pdf` is absent.

## Fatal Results

The proposed embodied repair memory fails the decisive-baseline gate:

- Nominal split: repair memory `0.225 +/- 0.075`, nominal MPC `0.242 +/- 0.077`, robust MPC `0.242 +/- 0.077`, global repair `0.250 +/- 0.078`.
- High-friction shift: repair memory `0.142 +/- 0.063`, nominal MPC `0.200 +/- 0.072`, robust MPC `0.208 +/- 0.073`, global repair `0.225 +/- 0.075`.
- Heavy-object shift: repair memory `0.175 +/- 0.068`, robust MPC `0.200 +/- 0.072`.
- Combined shift: repair memory `0.117 +/- 0.058`, nominal MPC `0.158 +/- 0.066`, robust MPC `0.150 +/- 0.064`, global repair `0.150 +/- 0.064`.
- Combined-shift ablations are worse for the claimed mechanism: reset memory and nominal MPC reach `0.158 +/- 0.066`, while embodied repair memory reaches only `0.117 +/- 0.058`.
- Violation rates remain high under shift, including `0.425` for embodied repair memory on combined shift.

## Gate Decision

This paper satisfies the local evidence-package requirements for a real negative result: high-fidelity simulator evidence, persistent deployment streams, paired baselines, ablations, learning curves, stress tests, uncertainty, negative cases, rebuilt PDF, and a public repository.

It does not satisfy `STRONG_REVISE` because the current mechanism is not merely incomplete; it fails nominal, robust, and simple repair baselines. The correct terminal state remains `KILL_ARCHIVE`.

Required revival work:

- replace kNN residual memory with a learned repair model;
- add public long-horizon benchmark or hardware validation;
- show statistically clear gains over nominal/robust MPC and simple repair baselines;
- add stronger safety handling for obstacle and violation residuals;
- perform manual related-work synthesis.
