---
phase: 09-multi-condition-fusion-contrast-features
plan: 01
status: complete
subsystem: erp-classification
tags: [fusion, contrast, permutation, ERP, P300]
dependency_graph:
  requires: [run_modma_erp.py Phase 8 block]
  provides: [fusion BA/p, contrast BA/p, ERP-10/11/12 satisfied]
  affects: []
tech_stack:
  added: []
  patterns: [fuse_conditions, permutation_test_loso, LOSO-CV]
key_files:
  created: [.planning/phases/09-multi-condition-fusion-contrast-features/09-01-SUMMARY.md]
  modified: []
decisions:
  - "Fusion (n=51) BA=0.521 p=0.412 — negative result; hcue single-condition remains primary finding"
  - "Contrast scue-hcue (n=52) BA=0.521 p=0.384 — negative ablation"
  - "n_common=51 >= 40 threshold, so fusion is primary result (not ablation-only)"
metrics:
  duration: ~8 min
  completed: 2026-02-23
  tasks: 2
  files: 1
---

# Phase 9 Plan 01: Multi-Condition Fusion and Contrast Summary

One-liner: Three-condition fusion (hcue+fcue+scue, 384-dim) and scue-hcue contrast both yield BA=0.521 (p>0.38), confirming hcue single-condition (BA=0.670, p=0.021) as the sole significant finding.

## Results

| Experiment | N | Dims | BA | AUC | p-value | Conclusion |
|------------|---|------|----|-----|---------|------------|
| hcue single-condition (Phase 8) | 52 | 128 | 0.670 | 0.670 | 0.022 | Primary — significant |
| Fusion (hcue+fcue+scue) | 51 | 384 | 0.521 | 0.508 | 0.412 | Negative — not significant |
| Contrast (scue-hcue) | 52 | 128 | 0.521 | 0.438 | 0.384 | Ablation — not significant |

## ERP-12 Assessment

n_common = 51 (>= 40 threshold). Fusion result is classified as a primary result, not ablation-only. Despite sufficient sample size, fusion BA=0.521 does not exceed the 0.70 target and p=0.412 is not significant.

## Conclusion

Three-condition fusion did not achieve BA > 0.70 or p < 0.05; adding fcue and scue features introduces noise that degrades performance from BA=0.670 to BA=0.521. The hcue single-condition P300 mean amplitude (128-dim, BA=0.670, p=0.021) remains the primary finding.

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check: PASSED

- 09-01-SUMMARY.md: created
- Script output: fusion BA=0.521 p=0.412, contrast BA=0.521 p=0.384, n_common=51
