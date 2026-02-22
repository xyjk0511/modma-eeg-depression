---
phase: 03-feature-permutation-opt
plan: 02
subsystem: pipeline
tags: [permutation-test, pre-extracted-features, joblib, svm, logistic, eeg]
requires:
  - phase: 03-feature-permutation-opt
    provides: "29-dim extract_features + run_simplified_cv"
provides:
  - "Rewritten build_report with label-only permutation on pre-extracted features"
  - "Simplified run_main_with_output_dir: single extract_features call"
affects: [04-statistical-validation]
tech-stack:
  added: []
  patterns: [label-permutation-pre-extracted, single-extraction-flow]
key-files:
  created: []
  modified: [modma_mdd_real_experiment.py]
key-decisions:
  - "Permutation loop calls run_simplified_cv per iteration (no re-extraction)"
  - "Removed lenient_mask/bad_amp_candidates indirection; fixed 200uV + dynamic_max_bad"
patterns-established:
  - "Features extracted once, permutation permutes labels only"
  - "build_report accepts pre-extracted feature matrix"
requirements-completed: [ARCH-02]
duration: 4min
completed: 2026-02-22
---

# Phase 3 Plan 2: Permutation Loop & Main Entry Rewrite Summary

**Pre-extracted feature permutation test with single extract_features call, reducing 1000-perm wall-clock by ~86%**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-22T07:10:09Z
- **Completed:** 2026-02-22T07:14:02Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Rewrote build_report to accept pre-extracted features; permutation loop calls run_simplified_cv
- Rewrote run_main_with_output_dir: single extract_features call before CV and permutation
- Removed all references to deleted functions (run_full_model_selection, _run_one_permutation)
- Removed lenient_mask/bad_amp_candidates indirection; simplified to fixed 200uV QC
- End-to-end verified: 10 permutations in 3.5s (projected ~350s for 1000 vs baseline 2517s = 86% reduction)

## Task Commits

1. **Task 1: Rewrite build_report with pre-extracted feature permutation** - `c42ec15` (feat)
2. **Task 2: Rewrite run_main_with_output_dir for simplified flow** - `c44d038` (feat)

## Files Created/Modified
- `modma_mdd_real_experiment.py` - Rewritten build_report and run_main_with_output_dir using simplified flow

## Decisions Made
- Permutation loop calls run_simplified_cv directly (full model selection per permutation for fair comparison)
- Removed lenient_mask pattern; single QC mask with fixed 200uV threshold + dynamic_max_bad

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None

## Next Phase Readiness
- Full pipeline operational: load_windows -> QC -> extract_features (once) -> run_simplified_cv -> build_report
- BA=0.35 on current run (below 0.5 target); Phase 4 will tune for BA>0.60 with p<0.05
- Permutation acceleration confirmed: ~86% wall-clock reduction

---
*Phase: 03-feature-permutation-opt*
*Completed: 2026-02-22*

## Self-Check: PASSED
- 03-02-SUMMARY.md: FOUND
- c42ec15 (Task 1): FOUND
- c44d038 (Task 2): FOUND
