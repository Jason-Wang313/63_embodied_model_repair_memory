# Experiment Rigor Checklist

## v4 Real Rigor

- [x] High-fidelity MuJoCo sequential benchmark.
- [x] Persistent hidden deployment streams.
- [x] Multiple seeds: 5.
- [x] Episodes per split/method: 120.
- [x] Six stress splits.
- [x] Implemented baselines.
- [x] Ablations.
- [x] Learning curve.
- [x] Confidence intervals and paired deltas.
- [x] Reproducible CSVs and figures.

## ICLR Main Bar

- [ ] Beats nominal MPC.
- [ ] Beats robust MPC.
- [ ] Beats simple global/last repair baselines.
- [ ] Real-robot validation.
- [ ] External long-horizon manipulation benchmark.
- [ ] Learned repair-memory model.
- [ ] Manual full-paper related-work synthesis.

Decision: KILL_ARCHIVE. The evidence is real, but the method fails.
