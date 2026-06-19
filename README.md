# 63 Embodied Model Repair Memory

Submission-hardening version: v5 expanded frozen MuJoCo rebuild.

Terminal decision: KILL_ARCHIVE for ICLR main conference.

This repository contains a rigorous repeated-deployment MuJoCo repair-memory benchmark. The original kNN repair-memory thesis failed in v4. The v5 rebuild tests a stronger method, Calibrated Online Repair Memory (CORM), with online ridge residual modeling, uncertainty-aware trust, selective safety shielding, strong baselines, nonstationary stress, ablations, paired statistics, and generated manuscript tables.

The result is still negative. CORM improves over nominal/global/last baselines in some aggregate metrics, but it does not clear the frozen gates against old kNN repair memory, online ridge residual, or robust MPC safety on nonstationary stress. The honest terminal state is KILL_ARCHIVE, not ICLR-main ready.

## Evidence Summary

- Main run: 8 seeds, 32 sequential episodes per seed/split/method.
- Main raw rows: 25,344.
- Ablation raw rows: 5,632.
- Splits: nominal, low-friction shift, high-friction shift, heavy-object shift, light-object shift, actuation-bias shift, obstacle shift, nonstationary deployment shift, combined shift.
- Main methods: random, nominal MPC, robust worst-case MPC, last repair, global repair, old kNN repair memory, online ridge residual, CORM v5, CORM no safety, CORM no uncertainty, hidden-deployment oracle.
- Terminal PDF: `C:/Users/wangz/Downloads/63.pdf`
- PDF SHA256: `E4FD93FAE7E60EF0260DDC75512183E19C80FBB1EFD097DCD542EE972DC02978`
- Desktop copy: absent by validation.

## Reproduce Frozen Evidence

```powershell
python src\run_experiment.py --seeds 8 --episodes 32 --splits nominal low_friction_shift high_friction_shift heavy_object_shift light_object_shift actuation_bias_shift obstacle_shift nonstationary_deployment_shift combined_shift --ablation-splits combined_shift nonstationary_deployment_shift --results-dir results --figures-dir figures
```

## Build PDF

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_submission_pdf.ps1
```

## Validate

```powershell
python scripts\validate_submission_artifacts.py
```

GitHub: https://github.com/Jason-Wang313/63_embodied_model_repair_memory
