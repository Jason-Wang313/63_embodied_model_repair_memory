# Hostile Reviewer Response

Paper: 63 Embodied Model Repair Memory

## Strongest Technical Threats

- PI-VLA: Adaptive Symmetry-Aware Decision-Making for Long-Horizon Vision-Language-Action Manipulation (2026)
- Scene Memory Transformer for Embodied Agents in Long-Horizon Tasks (2019)
- From Real World to Logic and Back: Learning Generalizable Relational Concepts For Long Horizon Robot Planning (2024)
- MAP-VLA: Memory-Augmented Prompting for Vision-Language-Action Model in Robotic Manipulation (2025)
- CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks (2021)
- FurnitureBench: Reproducible real-world benchmark for long-horizon complex manipulation (2025)
- Rethinking Progression of Memory State in Robotic Manipulation: An Object-Centric Perspective (2026)

## ICLR Main Response

A hostile ICLR reviewer would no longer be correct to reject the paper for synthetic-only evidence. The v4 rebuild contains a real MuJoCo sequential benchmark, implemented repair-memory variants, strong baselines, stress splits, ablations, and confidence intervals.

The reviewer would still be correct to reject the paper because the evidence is negative. Embodied repair memory does not outperform nominal or robust MPC and does not isolate a useful memory mechanism.

## Honest Action

The paper is marked `KILL_ARCHIVE`. This preserves the real negative result and prevents converting a failed mechanism into a polished-but-misleading submission.

## What Would Be Needed To Revive

- Learned repair model rather than kNN residual retrieval.
- Public benchmark or hardware validation.
- Statistically significant gains over nominal/robust MPC.
- Clear memory-persistence ablation.
- Manual related-work synthesis.
