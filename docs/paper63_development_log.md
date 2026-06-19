# Paper 63 Development Log

Date: 2026-06-19

This log records pre-freeze development only. The final frozen run must not be tuned after results are observed except for recoverable infrastructure failures.

## Starting Point

- Repository started from the v4 Paper 63 package.
- Existing v4 decision was `KILL_ARCHIVE`.
- Existing evidence falsified the old kNN residual repair memory: it did not reliably beat nominal MPC, robust MPC, global repair, or last-repair baselines.
- Existing PDF was too short for a real submission and did not contain enough theory, stress testing, or failure analysis.

## Planned Rebuild

The old thesis was narrowed into a more defensible claim:

> Repeated deployment streams can contain reusable physical residual structure, but repair memories should be reused through calibrated online residual models with uncertainty and safety checks. Uncalibrated episodic retrieval can be unsafe or worse than simple robust control.

Implemented method family:

- `knn_repair_memory_v4`: old nearest-neighbor memory retained as a baseline.
- `online_ridge_residual`: online linear residual model without CORM calibration.
- `corm_repair_memory_v5`: calibrated online residual memory with uncertainty-aware trust and selective safety shielding.
- `corm_no_safety`, `corm_no_uncertainty`, `corm_no_context`, and `corm_limited_memory`: ablation variants.

Added frozen stress splits:

- `nominal`
- `low_friction_shift`
- `high_friction_shift`
- `heavy_object_shift`
- `light_object_shift`
- `actuation_bias_shift`
- `obstacle_shift`
- `nonstationary_deployment_shift`
- `combined_shift`

## Smoke Test

Command:

```powershell
python src\run_experiment.py --seeds 1 --episodes 1 --splits nominal --ablation-splits combined_shift --results-dir results\dev_smoke --figures-dir figures\dev_smoke
```

Result:

- Initial smoke found a plotting edge case: one-episode runs had empty middle/late phase bins.
- Fixed plotting to tolerate empty diagnostic bins.
- Fixed a kNN local-spread calculation to use a NumPy array before weighting.
- Re-ran smoke successfully.

## Medium Development Run 1

Command:

```powershell
python src\run_experiment.py --seeds 2 --episodes 6 --splits nominal actuation_bias_shift nonstationary_deployment_shift combined_shift --ablation-splits combined_shift nonstationary_deployment_shift --results-dir results\dev_medium --figures-dir figures\dev_medium
```

Result:

- Terminal decision: `KILL_ARCHIVE`.
- CORM improved some success/regret metrics but the safety gate was too noisy.
- The original continuous risk penalty changed action scores even when no shield was activated.
- On combined/nonstationary shifts, this increased or failed to reduce violations relative to robust and ridge baselines.

Interpretation:

- This was a recoverable method-development failure, not a result to hide.
- The weakness exposed by stress tests was that a safety penalty should be selective, not always-on.

## Method Fix Before Freeze

Change:

- Made the CORM shield selective.
- The shield activates only under high predicted violation risk.
- The safety penalty is applied only when the shield is active.
- This prevents low-confidence safety estimates from constantly perturbing the residual controller.

Rationale:

- This is a principled fix to a pre-freeze development failure.
- It does not use final test results.
- It makes the safety gate less likely to create false-positive action changes.

## Medium Development Run 2

Command:

```powershell
python src\run_experiment.py --seeds 2 --episodes 6 --splits nominal actuation_bias_shift nonstationary_deployment_shift combined_shift --ablation-splits combined_shift nonstationary_deployment_shift --results-dir results\dev_medium2 --figures-dir figures\dev_medium2
```

Result:

- Terminal decision on the small development protocol: `STRONG_REVISE`.
- Aggregate CORM deltas in the development run:
  - vs nominal MPC: higher success, lower regret, lower violation.
  - vs robust MPC: higher success, lower regret, matched violation.
  - vs old kNN memory: higher success, lower regret, lower violation.
  - vs online ridge residual: modestly higher success, lower regret, matched violation.
- The development run is not the final evidence.

## Freeze Boundary

After this log and the protocol-freeze document are written, final results must be reported honestly. If the frozen 8-seed, 32-episode protocol returns `KILL_ARCHIVE`, the manuscript must present a rigorous negative result.
