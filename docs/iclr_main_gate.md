# ICLR Main Gate

Paper: 63 embodied_model_repair_memory

Previous v3 decision: KILL_ARCHIVE.

v4 gate verdict: KILL_ARCHIVE.

ICLR main ready: no.

Evidence digest: real MuJoCo sequential repair-memory benchmark with 5 seeds, 6 stress splits, 7 main methods, ablations, confidence intervals, paired deltas, and figures.

Fatal blockers after rebuild:
- Proposed repair memory does not beat nominal MPC.
- Proposed repair memory does not beat robust worst-case MPC.
- Proposed repair memory is not consistently better than global/last repair baselines.
- Violation rates remain high under shift.
- No learned repair model, public benchmark, hardware, or manual full-paper related-work synthesis.

The only honest main-conference-safe decision is KILL_ARCHIVE.
