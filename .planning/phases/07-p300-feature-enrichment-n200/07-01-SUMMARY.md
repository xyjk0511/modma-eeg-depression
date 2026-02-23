---
phase: 07-p300-feature-enrichment-n200
plan: 01
subsystem: erp-features
tags: [erp, p300, n200, feature-extraction, loso]
key-files:
  modified:
    - run_modma_erp.py
decisions:
  - BA=0.554 below 0.670 target; shape assertion (640) passed; target is aspirational per plan
metrics:
  completed: 2026-02-23
---

# Phase 07 Plan 01: P300 Feature Enrichment + N200 Summary

Extended load_erp_features() from 128-dim to 640-dim: P300 mean, peak, latency, AUC (np.trapezoid), N200 mean — 128 channels each.

## Results

- Feature matrix shape: (52, 640) — shape assertion passed
- hcue LOSO BA: 0.554 (target >= 0.670 — not met; aspirational per plan)
- hcue LOSO AUC: 0.568
- N = 52 subjects (MDD=24, HC=28)

## Deviations from Plan

None. BA target was aspirational: plan states record the value if < 0.670.

## Self-Check: PASSED

- N200_WIN, np.trapezoid, 640-dim concatenation all present in run_modma_erp.py
- run_loso() and __main__ untouched
- Commit: c630a0e
