# Paper 63 Rebuild Plan

Paper: Embodied Model Repair Memory.

Goal: rebuild the v3 archive into a real ICLR-main-target evidence package, or terminate honestly as STRONG_REVISE / KILL_ARCHIVE.

## Salvageable Thesis

Store repairs to embodied model beliefs as reusable state, not transient corrective prompts. The contribution is only meaningful if a robot can use past model-error repairs to improve later physical decisions under recurring deployment shifts.

## Real Evidence Target

Build a MuJoCo sequential contact-pushing benchmark:

- Task: push a puck to varying targets under hidden deployment-specific dynamics.
- Hidden recurring deployment factors: mass, friction, actuation gain/bias, and obstacle placement shift.
- Episode stream: each seed/split is a sequence of tasks from the same hidden deployment condition.
- Observation after each action: nominal predicted outcome versus actual MuJoCo outcome.
- Repair signal: residual displacement, obstacle-violation surprise, and action/context features.

## Proposed Method

Embodied repair memory:

- Maintains a persistent memory of residual corrections from previous tasks.
- Retrieves nearest repairs by action/geometry/context features.
- Corrects nominal candidate rollouts before action selection.
- Writes new repairs after every executed action.

This is not a language prompt memory. It is an embodied residual memory attached to physical action models.

## Baselines

- random_candidate
- nominal_mpc
- robust_worst_case_mpc
- online_last_repair_reset_each_episode
- global_average_repair
- embodied_repair_memory (proposed)
- oracle_hidden_deployment

## Ablations

- no_memory_nominal_mpc
- reset_memory_every_episode
- global_memory_no_retrieval
- no_context_features
- limited_memory_capacity
- oracle_hidden_deployment

## Metrics

- Success rate and 95 percent confidence interval.
- Final distance to target.
- Obstacle/unsafe-contact violation rate.
- Regret versus oracle.
- Learning curve over deployment episode index.
- Paired deltas against nominal, robust, and global-average repair.

## Stress Splits

- nominal
- low_friction_shift
- high_friction_shift
- heavy_object_shift
- actuation_bias_shift
- combined_shift

## Terminal Criteria

ICLR_MAIN_TARGET_READY requires:

- Real MuJoCo sequential evidence.
- Multiple seeds and stress splits.
- Proposed repair memory clearly beats nominal, robust, and non-retrieval repair baselines.
- Ablations show retrieval/context/memory persistence are necessary.
- Honest hostile prior-work synthesis and limitations.

STRONG_REVISE if:

- The benchmark and repair mechanism are real and reproducible, but gains are mixed, custom-only, or not strong enough for ICLR main.

KILL_ARCHIVE if:

- Repair memory fails to beat simple baselines or cannot be made reproducible with real MuJoCo evidence.
