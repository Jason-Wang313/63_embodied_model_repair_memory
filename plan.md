# Plan

Paper 63 was rebuilt from the archive scaffold into a real sequential MuJoCo repair-memory evidence package before moving to Paper 64.

Terminal state: KILL_ARCHIVE.

Reason: the proposed memory mechanism fails the decisive-baseline test.

Future revival would require:
- A learned repair model rather than kNN residual memory.
- A public long-horizon benchmark or hardware validation.
- Clear gains over nominal/robust MPC and simple global repair.
- Better safety handling for obstacle/violation residuals.
