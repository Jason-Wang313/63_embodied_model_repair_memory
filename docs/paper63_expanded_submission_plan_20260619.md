# Paper 63 Expanded Submission Plan

Date: 2026-06-19

Paper: `63_embodied_model_repair_memory`

Target venue posture: ICLR main-conference candidate only if the rebuilt method escapes the current KILL_ARCHIVE result under frozen hostile-review gates. If it does not, produce a 25+ page kill/archive paper with real negative evidence and no rhetorical rescue.

Operating constraints:

- CPU only.
- Keep RAM light with compact tabular residual memories and streaming CSV writes.
- Do not compromise quality because of RAM limits. Prefer more seeds/splits and stronger baselines over larger neural networks.
- Do not pad to 25 pages. Added length must be theory, protocol, evidence, failure analysis, ablations, related work, or reproducibility detail.
- Keep `C:/Users/wangz/Downloads/63.pdf` as the only numbered PDF artifact. Never copy to Desktop.
- Freeze the final protocol before the terminal run.

## 1. Starting Diagnosis

The current v4 repository is clean but terminally negative.

Verified starting facts:

- Current PDF is 3 pages.
- Existing main evidence has 5,040 rows: 6 splits, 5 seeds, 24 sequential episodes, 7 methods.
- Existing ablation evidence has 840 rows.
- Existing terminal decision is `KILL_ARCHIVE`.
- The current kNN residual repair memory loses to nominal MPC, robust MPC, and simple global/last repair baselines on decisive splits.
- Combined shift has high violation rate for embodied repair memory.

Core failure:

The current memory stores residuals but does not distinguish reusable deployment structure from noisy task-specific residuals. kNN retrieval can import irrelevant repairs, cannot detect latent deployment modes reliably, and has no safety-calibrated way to decide when memory should be trusted.

## 2. Revised Thesis

The original thesis is only salvageable if repair memory becomes an online system-identification object, not a nearest-neighbor cache.

Revised testable thesis:

> Repeated deployment streams contain reusable physical residual structure, but a robot should reuse repair memories only through calibrated online residual models that estimate deployment-specific correction and uncertainty; uncalibrated episodic retrieval is unsafe and often worse than robust MPC.

This thesis allows two possible honest outcomes:

- If calibrated residual repair beats nominal/robust/global baselines without increasing violations, Paper 63 can move from KILL_ARCHIVE to STRONG_REVISE.
- If it still fails, the paper becomes a rigorous 25-page kill/archive study explaining why naive embodied repair memory should not be submitted.

## 3. Method Rebuild: CORM v5

Implement a new proposed method: Calibrated Online Repair Memory (CORM).

Required components:

- Expanded action library:
  - add lateral offsets and more push distances;
  - keep the same candidate set for every method;
  - preserve CPU-light MuJoCo rollouts.

- Residual observation:
  - after each executed action, record nominal predicted final displacement, true final displacement, violation surprise, selected primitive parameters, obstacle geometry, target geometry, and split/deployment context.

- Online residual model:
  - use recursive ridge or Bayesian linear residual regression over compact features;
  - predict 2D displacement residual and violation-risk residual;
  - maintain uncertainty from residual covariance or recent prediction error;
  - update after every episode in the deployment stream.

- Trust calibration:
  - blend nominal predictions and repair predictions according to uncertainty;
  - reduce trust after large residual errors or suspected deployment change;
  - report trust weights over early/middle/late stream phases.

- Safety gate:
  - penalize candidates whose repaired prediction increases modeled violation risk;
  - include shield activation and violation-rate metrics;
  - compare against shielded simple baselines so safety is not a hidden trick.

Fairness constraints:

- Include the old kNN repair memory as a baseline.
- Include global-average and last-repair baselines.
- Include nominal MPC, robust worst-case MPC, and oracle hidden deployment.
- Include an online residual-system-ID baseline if it differs from CORM.

## 4. Frozen Baseline Suite

Main methods:

- random_candidate
- nominal_mpc
- robust_worst_case_mpc
- last_repair_memory
- global_average_repair
- knn_repair_memory_v4
- online_ridge_residual
- corm_repair_memory_v5
- corm_no_safety
- corm_no_uncertainty
- oracle_hidden_deployment

Optional if implementation remains clean:

- reset_every_episode
- shielded_global_repair

## 5. Stress Splits

Main frozen splits:

- nominal
- low_friction_shift
- high_friction_shift
- heavy_object_shift
- light_object_shift
- actuation_bias_shift
- obstacle_shift
- nonstationary_deployment_shift
- combined_shift

The nonstationary split should deliberately change the hidden deployment halfway through the stream. It tests whether memory can stop trusting stale repairs.

## 6. Ablations

Run ablations on combined shift and nonstationary shift:

- full CORM;
- no safety gate;
- no uncertainty/trust calibration;
- no context features;
- global-only residual;
- last-only residual;
- limited memory/window;
- old kNN v4 memory;
- nominal MPC;
- robust MPC;
- oracle.

Gate: the full method must beat or statistically tie while improving safety over nominal MPC, robust MPC, global repair, last repair, old kNN, and online ridge residual. If not, terminal decision stays KILL_ARCHIVE or at most STRONG_REVISE.

## 7. Statistical Protocol

Development phase:

- Run tiny smoke tests only for crashes and schema bugs.
- Run one medium pre-freeze development run to check whether CORM is sane.
- Log all pre-freeze fixes in `docs/paper63_development_log.md`.

Final freeze:

- Write `docs/paper63_protocol_freeze_20260619.md`.
- Freeze seeds, episodes, splits, methods, ablations, hyperparameters, and gates.
- After freeze, only fix recoverable infrastructure failures.

Final evidence target:

- At least 8 seeds.
- At least 32 sequential episodes per seed/split.
- At least 9 stress splits.
- Raw main rows expected: 8 x 32 x 9 x 11 = 25,344 if all main methods are kept.
- Ablation rows expected: at least 2 x 8 x 32 x 11 = 5,632.

Statistics:

- Success/final-distance/violation/regret means and 95 percent CIs.
- Paired bootstrap intervals.
- Sign-flip p-values.
- Early/middle/late learning curves.
- Trust-calibration diagnostics.
- Stale-memory failure rates on nonstationary deployment shift.

## 8. Theory Additions

Required theory sections:

- Formal repeated-deployment problem with hidden but recurring latent parameters.
- Residual-memory objective and update rule.
- Bias-variance decomposition: when residual memory helps versus hurts.
- Trust-calibration lemma for blending nominal and repair predictions under bounded residual estimation error.
- Stale-memory negative theorem: if deployments are nonstationary and change points are unobserved, persistent memory can be worse than reset/robust baselines.
- Safety-risk decomposition separating residual prediction error from action-library discretization.

## 9. Related Work and Citations

Replace or expand the bibliography with primary sources on:

- model-based RL and model error correction;
- adaptive and robust MPC;
- online system identification;
- episodic memory and case-based control;
- lifelong/meta robot learning;
- long-horizon manipulation benchmarks;
- MuJoCo and contact simulation.

Citation UX requirement:

- Use `hyperref` with `colorlinks=false`.
- Use bright boxed citation borders, e.g. `citebordercolor={1 0.48 0}`.
- Verify citations route to the bibliography and LaTeX has no undefined references.

## 10. Manuscript Rebuild

Minimum manuscript contents:

- Abstract with honest terminal decision.
- Introduction explaining the KILL_ARCHIVE starting point.
- Formal repeated-deployment setup.
- CORM method.
- Theory and negative results.
- Frozen protocol.
- Main results.
- Learning-curve analysis.
- Ablations.
- Nonstationary/stale-memory stress test.
- Failure analysis.
- Related work.
- Limitations and revival conditions.
- Reproducibility appendix.
- Full generated result tables.

Length requirement:

- At least 25 pages.
- If the final evidence remains negative, the 25 pages should make the kill decision more rigorous, not more optimistic.

## 11. Validation Gates

Before commit:

- `python -m py_compile src/run_experiment.py`
- Frozen experiment complete with expected row counts.
- Analysis/tables generated.
- 25+ page PDF built with BibTeX and enough LaTeX passes.
- `C:/Users/wangz/Downloads/63.pdf` exists.
- `C:/Users/wangz/Desktop/63.pdf` absent.
- Downloads PDF matches repo PDF.
- LaTeX scan has no undefined citations/references or fatal errors.
- `git diff --check` passes.
- Paper repo committed and pushed to public GitHub.
- Root ledgers updated locally.

## 12. Terminal Decision Gates

`ICLR_MAIN_TARGET_READY` is not realistic for Paper 63 without hardware/public benchmark evidence.

`STRONG_REVISE` requires:

- CORM clearly improves over nominal MPC, robust MPC, global repair, last repair, old kNN, and online residual baselines on aggregate success/regret;
- no violation increase on combined or nonstationary shifts;
- ablations show uncertainty calibration and safety are necessary;
- 25+ page paper and reproducible artifacts pass validation.

`KILL_ARCHIVE` if:

- CORM fails the strong baselines;
- safety remains worse than robust/reset baselines;
- memory gains vanish under nonstationary deployment shift;
- ablations show simple global/last repair matches full memory.

