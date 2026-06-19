# Submission Readiness Decision

Decision: KILL_ARCHIVE.

ICLR main-conference readiness: NO.

Submission-hardening version: v5 expanded frozen MuJoCo rebuild.

Final update: 2026-06-20 01:01:40 +08:00.

## Evidence

- 25,344 main raw rows.
- 5,632 ablation raw rows.
- 99 main metric rows.
- 792 seed metric rows.
- 22 ablation metric rows.
- 90 paired CORM-vs-baseline rows.
- 25-page ICLR-style PDF with bright boxed clickable citation links.
- Canonical PDF: `C:/Users/wangz/Downloads/63.pdf`.
- PDF SHA256: `E4FD93FAE7E60EF0260DDC75512183E19C80FBB1EFD097DCD542EE972DC02978`.
- No visible Desktop PDF.

## Reason

CORM v5 is a serious improvement over the old kNN-only repair memory, but it does not survive the frozen hostile-review gate.

Frozen aggregate failures:

- CORM success is below old kNN repair memory: 0.1662 vs 0.1719.
- CORM success is below online ridge residual: 0.1662 vs 0.1806.
- CORM regret does not improve over robust MPC: 0.1489 vs 0.1446.
- CORM regret is essentially tied/slightly worse than online ridge residual: 0.1489 vs 0.1488.
- Nonstationary stress increases violation relative to robust MPC by 0.0742.

Honest terminal action: archive/kill for ICLR main. Do not submit this paper in its current form.

Revival condition: add explicit stale-memory/change-point detection, repair scene geometry rather than only outcome residuals, strengthen robust baselines, and show clear gains on public benchmarks or hardware.
