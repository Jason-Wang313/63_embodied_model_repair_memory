# Final Audit

1. Chosen thesis: Embodied Model Repair Memory stores physical model-error repairs as reusable state across repeated deployment tasks.
2. ICLR-main decision: KILL_ARCHIVE.
3. Submission-hardening version: v4.
4. Evidence: real MuJoCo sequential benchmark with 5,040 main rows and 840 ablation rows.
5. Main result: repair memory does not beat nominal or robust MPC and is not consistently better than simple repair baselines.
6. Reproducibility: code, CSVs, paired stats, learning curves, figures, and PDF reproduce locally.
7. Closest hostile prior work: memory-augmented long-horizon robot policies, scene memory transformers, CALVIN/FurnitureBench-style long-horizon manipulation.
8. Claim-validity status: killed for ICLR main; real negative evidence retained.
9. Exact Downloads PDF path: `C:/Users/wangz/Downloads/63.pdf`
10. GitHub URL: https://github.com/Jason-Wang313/63_embodied_model_repair_memory
11. Confirmation: no visible Desktop copy was requested or made.

## 2026-06-15 Continuation Audit

Executed `docs/paper63_iclr_submission_execution_plan_20260615.md`.

Additional verification:
- Python compile passed for `src/run_experiment.py`.
- CSV finite/schema audit passed for main, paired, ablation, seed, learning-curve, stress, and negative-case result files.
- LaTeX/PDF rebuild completed and `C:/Users/wangz/Downloads/63.pdf` was refreshed.
- `C:/Users/wangz/Desktop/63.pdf` is absent.

Decision remains `KILL_ARCHIVE`, not ICLR-main-ready. See `docs/paper63_terminal_audit_20260615.md`.
