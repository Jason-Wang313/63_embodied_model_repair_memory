# Novelty Boundary Map

## Crowded Territory
- Bigger data/model scaling.
- New benchmark only.
- Generic active learning or uncertainty.
- Combining a planner with a learned policy without a new state/action object.

## Claimed Boundary
Embodied model repair memory keeps action-critical alternatives explicit until a physical observation collapses them.

## What Would Falsify The Claim
If observed-only baselines match the adverse-mode coverage and closed-loop success of the proposed branch-aware mechanism, the paper should be revised or killed.

## v4 Outcome
The v4 sequential MuJoCo rebuild falsifies the current claim. Embodied repair memory does not outperform nominal MPC, robust MPC, or simple repair baselines strongly enough to support a paper. The idea is killed/archived unless a substantially stronger learned repair model is introduced.
