---
phase: 08-permutation-test-validation
plan: "01"
subsystem: erp-classification
tags: [permutation-test, hcue, p300, loso, statistical-validation]
dependency_graph:
  requires: []
  provides: [hcue-pvalue, statistical-significance-confirmed]
  affects: [STATE.md, ROADMAP.md]
tech_stack:
  added: []
  patterns: [subject-level-permutation, loso-cv]
key_files:
  created: [.planning/phases/08-permutation-test-validation/08-01-SUMMARY.md]
  modified: []
decisions:
  - "hcue BA=0.670 is statistically significant (p=0.021 < 0.05) — ERP-07/08/09 satisfied"
metrics:
  duration: "~8 min"
  completed: "2026-02-23"
---

# Phase 08 Plan 01: Permutation Test Validation Summary

hcue P300 128-dim LOSO classification confirmed statistically significant: BA=0.670, p=0.021 via 1000-iteration subject-level permutation test.

## Result

| Metric | Value |
|--------|-------|
| Condition | hcue |
| Features | 128-dim P300 mean amplitude (250-500 ms) |
| N subjects | 52 (MDD=24, HC=28) |
| Observed BA | 0.670 |
| AUC | 0.670 |
| Permutations | 1000 |
| Perm mean BA | 0.494 |
| p-value | 0.0210 |
| Conclusion | Statistically significant (p < 0.05) |

## Interpretation

hcue P300 mean amplitude achieves BA=0.670 with p=0.021, meeting the p<0.05 threshold and satisfying requirements ERP-07, ERP-08, ERP-09, ERP-NF-01, ERP-NF-02.

## Artifacts

- run_modma_erp.py — no changes (permutation_test_loso already implemented)

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check: PASSED
