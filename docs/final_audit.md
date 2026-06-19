# Final Audit

1. Chosen thesis: repeated robot deployments contain reusable physical residual structure, but repair memory must be calibrated and safety-aware to be useful.
2. Rebuilt method: Calibrated Online Repair Memory (CORM v5), with online ridge residual modeling, uncertainty-aware trust, shock-based stale-memory damping, and selective safety shielding.
3. ICLR-main decision: KILL_ARCHIVE.
4. Submission-hardening version: v5 expanded.
5. Evidence: frozen MuJoCo repeated-deployment benchmark with 25,344 main rows and 5,632 ablation rows.
6. Main result: CORM improves over nominal/global/last baselines in some metrics but fails the decisive gates against old kNN memory, online ridge residual, and robust MPC safety on nonstationary stress.
7. Reproducibility: code, raw CSVs, paired bootstrap/sign-flip stats, learning curves, trust calibration, generated tables, build script, validation script, and PDF reproduce locally.
8. Theory additions: repeated-deployment formalization, residual-trust bound, sufficient improvement condition, stale-memory negative theorem, and safety-risk decomposition.
9. Claim-validity status: killed for ICLR main; rigorous negative evidence retained.
10. Exact Downloads PDF path: `C:/Users/wangz/Downloads/63.pdf`
11. PDF SHA256: `E4FD93FAE7E60EF0260DDC75512183E19C80FBB1EFD097DCD542EE972DC02978`
12. GitHub URL: https://github.com/Jason-Wang313/63_embodied_model_repair_memory
13. Confirmation: no visible Desktop copy was requested or made.

## 2026-06-20 Expanded-Standard Audit

Executed `docs/paper63_expanded_submission_plan_20260619.md` and froze protocol in `docs/paper63_protocol_freeze_20260619.md`.

Additional verification:

- Python compile passed for `src/run_experiment.py` and helper scripts.
- Frozen experiment completed with expected row counts.
- LaTeX/PDF build completed and `C:/Users/wangz/Downloads/63.pdf` was refreshed.
- Validation script passed all row-count, page-count, PDF-placement, and Desktop-absence checks.
- LaTeX scan found no undefined references/citations, fatal errors, or emergency stops.

Decision remains `KILL_ARCHIVE`, not ICLR-main-ready. See `docs/paper63_final_v5_evidence_summary.md`.
