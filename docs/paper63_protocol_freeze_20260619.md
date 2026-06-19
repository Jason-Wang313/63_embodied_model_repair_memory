# Paper 63 Frozen Protocol

Date frozen: 2026-06-19

This protocol is frozen after smoke testing and the two documented pre-freeze development runs in `docs/paper63_development_log.md`.

## Objective

Evaluate whether Calibrated Online Repair Memory (CORM v5) survives strong CPU-only MuJoCo repeated-deployment stress tests. The result may be positive or negative. The final manuscript must report the frozen evidence honestly.

## Final Command

```powershell
python src\run_experiment.py --seeds 8 --episodes 32 --splits nominal low_friction_shift high_friction_shift heavy_object_shift light_object_shift actuation_bias_shift obstacle_shift nonstationary_deployment_shift combined_shift --ablation-splits combined_shift nonstationary_deployment_shift --results-dir results --figures-dir figures
```

## Main Methods

- `random_candidate`
- `nominal_mpc`
- `robust_worst_case_mpc`
- `last_repair_memory`
- `global_average_repair`
- `knn_repair_memory_v4`
- `online_ridge_residual`
- `corm_repair_memory_v5`
- `corm_no_safety`
- `corm_no_uncertainty`
- `oracle_hidden_deployment`

## Ablation Methods

- `corm_repair_memory_v5`
- `corm_no_safety`
- `corm_no_uncertainty`
- `corm_no_context`
- `corm_limited_memory`
- `global_average_repair`
- `last_repair_memory`
- `knn_repair_memory_v4`
- `nominal_mpc`
- `robust_worst_case_mpc`
- `oracle_hidden_deployment`

## Splits

Main splits:

- `nominal`
- `low_friction_shift`
- `high_friction_shift`
- `heavy_object_shift`
- `light_object_shift`
- `actuation_bias_shift`
- `obstacle_shift`
- `nonstationary_deployment_shift`
- `combined_shift`

Ablation splits:

- `combined_shift`
- `nonstationary_deployment_shift`

## Evidence Size

Expected main raw rows:

- 8 seeds x 32 episodes x 9 splits x 11 methods = 25,344 rows.

Expected ablation raw rows:

- 8 seeds x 32 episodes x 2 splits x 11 methods = 5,632 rows.

Expected total raw decision rows:

- 30,976 rows.

## Metrics

Report:

- success rate;
- final distance;
- violation rate;
- energy regret to hidden-deployment oracle;
- 95 percent confidence intervals;
- paired bootstrap intervals;
- sign-flip p-values;
- early/middle/late learning curves;
- trust and uncertainty diagnostics;
- shield activation;
- repair prediction error;
- nonstationary stale-memory behavior.

## Decision Rule

`STRONG_REVISE` is allowed only if the frozen run shows CORM is at least tied on safety and better on success/regret against the strong baseline suite:

- nominal MPC;
- robust worst-case MPC;
- last repair memory;
- global repair memory;
- old kNN repair memory;
- online ridge residual.

`KILL_ARCHIVE` is required if CORM fails the strong baseline suite, increases violations on combined or nonstationary shifts, or if ablations show that simple baselines match the full method.

`ICLR_MAIN_TARGET_READY` is not available for Paper 63 without hardware or public benchmark evidence.

## Post-Freeze Rule

After the final run starts, only recoverable infrastructure failures may be fixed. No method, hyperparameter, split, seed, gate, or baseline may be changed in response to final results.
