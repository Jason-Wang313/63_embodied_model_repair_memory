# Paper 63 Final Frozen Evidence Summary

Decision: `KILL_ARCHIVE`.

ICLR main readiness: no. The best permissible state is `STRONG_REVISE` because the evidence is CPU-only MuJoCo without hardware or public benchmark confirmation.

## Frozen Evidence Size

- Main raw rows: 25344.
- Ablation raw rows: 5632.
- Metric rows: 99.
- Ablation metric rows: 22.
- Pairwise rows: 90.
- Frozen run: 8 seeds, 32 episodes, 9 main stress splits, 2 ablation splits, 11 main methods.

## CORM Aggregate

- Success: 0.166233.
- Energy regret: 0.148914.
- Violation rate: 0.288194.
- Mean trust: 0.434946.
- Shield activation rate: 0.013889.

## Strong-Baseline Deltas

- vs `nominal_mpc`: success delta 0.0356, regret improvement 0.0241, violation delta -0.0586.
- vs `robust_worst_case_mpc`: success delta 0.0451, regret improvement -0.0043, violation delta 0.0273.
- vs `last_repair_memory`: success delta 0.0360, regret improvement 0.0227, violation delta -0.0586.
- vs `global_average_repair`: success delta 0.0056, regret improvement 0.0137, violation delta -0.0582.
- vs `knn_repair_memory_v4`: success delta -0.0056, regret improvement 0.0059, violation delta -0.0391.
- vs `online_ridge_residual`: success delta -0.0143, regret improvement -0.0001, violation delta -0.0143.

## Stress Pairwise Rows

- `combined_shift` vs `nominal_mpc`: success delta 0.031250, regret improvement 0.031601, violation delta -0.093750, regret sign-flip p 0.000244.
- `combined_shift` vs `robust_worst_case_mpc`: success delta 0.019531, regret improvement -0.005824, violation delta 0.011719, regret sign-flip p 0.531120.
- `combined_shift` vs `knn_repair_memory_v4`: success delta 0.011719, regret improvement 0.006485, violation delta -0.054688, regret sign-flip p 0.341958.
- `combined_shift` vs `online_ridge_residual`: success delta -0.011719, regret improvement -0.005694, violation delta -0.042969, regret sign-flip p 0.479131.
- `combined_shift` vs `oracle_hidden_deployment`: success delta -0.371094, regret improvement -0.206791, violation delta 0.371094, regret sign-flip p 0.000244.
- `nonstationary_deployment_shift` vs `nominal_mpc`: success delta 0.007812, regret improvement 0.002546, violation delta -0.003906, regret sign-flip p 0.710032.
- `nonstationary_deployment_shift` vs `robust_worst_case_mpc`: success delta 0.000000, regret improvement -0.030239, violation delta 0.074219, regret sign-flip p 0.000488.
- `nonstationary_deployment_shift` vs `knn_repair_memory_v4`: success delta -0.015625, regret improvement 0.002805, violation delta -0.027344, regret sign-flip p 0.649500.
- `nonstationary_deployment_shift` vs `online_ridge_residual`: success delta 0.007812, regret improvement -0.007008, violation delta 0.007812, regret sign-flip p 0.385160.
- `nonstationary_deployment_shift` vs `oracle_hidden_deployment`: success delta -0.347656, regret improvement -0.203943, violation delta 0.394531, regret sign-flip p 0.000244.

## Honest Interpretation

The final decision must follow the frozen protocol. If the decision is `STRONG_REVISE`, the result is promising but still not submission-ready for ICLR main without broader benchmarks or hardware. If the decision is `KILL_ARCHIVE`, the manuscript should present the failure as the main contribution.
