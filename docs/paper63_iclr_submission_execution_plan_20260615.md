# Paper 63 ICLR-Main Execution Plan

Date: 2026-06-15

Paper: `63_embodied_model_repair_memory`

Goal: verify whether the current real sequential MuJoCo evidence justifies keeping the paper at `KILL_ARCHIVE`, or whether any evidence supports revival to `STRONG_REVISE` or ICLR-main readiness.

## Execution Gates

1. Reproducibility gate:
   - Compile `src/run_experiment.py`.
   - Confirm main, seed, paired, ablation, learning-curve, stress, and negative-case CSV outputs exist.
   - Confirm all CSV outputs are non-empty and finite.
   - Rebuild the PDF from `paper/main.tex`.

2. Evidence gate:
   - Confirm the benchmark uses real MuJoCo sequential deployment streams rather than synthetic tables.
   - Confirm six stress splits, multiple seeds, repeated episodes per deployment stream, uncertainty estimates, paired comparisons, learning curves, and ablations.
   - Confirm baselines include nominal MPC, robust worst-case MPC, last-repair memory, global-average repair, random candidate, and oracle hidden deployment.

3. Negative-claim gate:
   - Compare embodied repair memory to nominal MPC.
   - Compare embodied repair memory to robust worst-case MPC.
   - Compare embodied repair memory to simple last/global repair baselines.
   - Check whether memory persistence, context features, or capacity ablations isolate the claimed mechanism.
   - Check whether safety/violation behavior improves enough to justify the method.

4. Artifact gate:
   - Rebuild `paper/main.pdf`.
   - Copy only `C:/Users/wangz/Downloads/63.pdf`.
   - Confirm `C:/Users/wangz/Desktop/63.pdf` is absent.
   - Confirm the GitHub repository is public and pushed.

## Decision Rule

Upgrade only if repair memory clearly beats nominal MPC, robust MPC, and simple repair baselines with acceptable violation rates and mechanism-isolating ablations. If it fails those gates, keep the terminal decision as `KILL_ARCHIVE` and document the exact negative evidence.
