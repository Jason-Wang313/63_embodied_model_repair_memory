# Plan

Paper 63 was rebuilt from the archive scaffold into a real sequential MuJoCo repair-memory evidence package before moving to Paper 64.

Terminal state: KILL_ARCHIVE.

2026-06-15 continuation audit: plan-first verification was executed in `docs/paper63_iclr_submission_execution_plan_20260615.md`; terminal evidence is recorded in `docs/paper63_terminal_audit_20260615.md`. The paper remains KILL_ARCHIVE, not ICLR-main-ready.

Reason: the proposed memory mechanism fails the decisive-baseline test.

Future revival would require:
- A learned repair model rather than kNN residual memory.
- A public long-horizon benchmark or hardware validation.
- Clear gains over nominal/robust MPC and simple global repair.
- Better safety handling for obstacle/violation residuals.
