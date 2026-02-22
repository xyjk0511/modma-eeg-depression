# Phase 4: Statistical Validation - Context

**Gathered:** 2026-02-22
**Status:** Ready for planning

## Phase Boundary

Achieve BA>0.60 + p<0.05 through small feature adjustments and full validation run.
Baseline: 5-dim Logistic C=0.1, BA=0.558, p=0.401, 43 subjects.

## Decisions

### Feature exploration
- Small-scope tuning only: swap regions (e.g. parietal instead of frontal), add back 1-2 high F-score features
- NOT grid search over many combos (multiple comparison inflation)
- Run diagnostic script pattern: quick CV comparison, pick best, then full 1000-perm

### Success criteria
- Keep original: BA>0.60 + p<0.05
- No relaxation — if not met, record negative conclusion honestly

### Reporting
- Full experiment report: all configs tested, BA/AUC/F1/CI/p for each
- Final metrics.json with conclusion object
- ROC curve + confusion matrix + permutation distribution plot

### Claude's Discretion
- Which feature variants to test in diagnostic
- How many variants (keep small, ~5-8 configs max)
- Report format details

## Deferred
- pyprep RANSAC, ICLabel — not in scope for Phase 4
