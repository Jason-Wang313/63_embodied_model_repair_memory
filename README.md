# 63 Embodied Model Repair Memory

Submission-hardening version: v4 real MuJoCo rebuild.

Terminal decision: KILL_ARCHIVE for ICLR main conference.

This repository now contains a real sequential MuJoCo repair-memory benchmark. The result is negative: embodied repair memory does not beat nominal MPC or robust worst-case MPC, and simple global/last repair baselines are competitive or better on several splits. The idea should not be submitted as an ICLR-main paper in this form.

## Evidence Summary

- Main run: 5 seeds, 24 sequential episodes per seed/split/method.
- Splits: nominal, low-friction shift, high-friction shift, heavy-object shift, actuation-bias shift, combined shift.
- Baselines: random candidate, nominal MPC, robust worst-case MPC, last-repair memory, global-average repair, oracle hidden deployment.
- Ablations: reset memory, global memory, no context features, limited capacity, nominal, oracle.
- Terminal state: kill/archive with real negative evidence.

## Reproduce

```powershell
python src\run_experiment.py --seeds 5 --episodes 24
```

## Build PDF

```powershell
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Canonical local PDF: `C:/Users/wangz/Downloads/63.pdf`

GitHub: https://github.com/Jason-Wang313/63_embodied_model_repair_memory
