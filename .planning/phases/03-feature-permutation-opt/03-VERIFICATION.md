---
phase: 03-feature-permutation-opt
verified: 2026-02-22T07:24:01Z
status: gaps_found
score: 10/11 must-haves verified
re_verification: false
gaps:
  - truth: "BA >= 0.5 on real MODMA data"
    status: failed
    reason: "Actual BA=0.351 from results_phase03_verify/metrics.json. Pipeline is structurally correct but classification performance is below chance."
    artifacts:
      - path: "results_phase03_verify/metrics.json"
        issue: "balanced_accuracy=0.351, below 0.5 target"
    missing:
      - "BA >= 0.5 not achieved; Phase 4 must address feature quality or model tuning"
---

# Phase 3: Feature Permutation Opt Verification Report

**Phase Goal:** Reduce feature dimensionality from 139->29, simplify pipeline, achieve BA >= 0.5
**Verified:** 2026-02-22T07:24:01Z
**Status:** gaps_found
**Re-verification:** No

## Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | extract_features returns exactly 29 dims | VERIFIED | feats.shape=(3,29) confirmed by live execution |
| 2 | Theta/Beta ratio per region (5 dims) | VERIFIED | 5 TBR names; epsilon guard /(beta+1e-10) present |
| 3 | Alpha asymmetry frontal+temporal (2 dims) | VERIFIED | alpha_asymmetry_frontal, alpha_asymmetry_temporal in names |
| 4 | Only top-2 Riemannian features | VERIFIED | riem_central_temporal, riem_central_parietal only |
| 5 | Pipeline uses SVM + Logistic only | VERIFIED | build_feature_model_pipeline raises ValueError for others |
| 6 | run_simplified_cv on pre-extracted features | VERIFIED | No extract_features call inside; signature (features,y,groups,...) |
| 7 | Features extracted exactly once before permutation | VERIFIED | Single call in run_main_with_output_dir line 70 |
| 8 | Permutation loop shuffles labels, calls run_simplified_cv | VERIFIED | build_report -> _run_one_perm_cv -> run_simplified_cv |
| 9 | run_main_with_output_dir uses simplified flow | VERIFIED | extract_features(once)->run_simplified_cv->build_report |
| 10 | Dead feature functions removed | VERIFIED | compute_de/hjorth/lateral_asymmetry/connectivity all absent |
| 11 | BA >= 0.5 on real MODMA data | FAILED | metrics.json: balanced_accuracy=0.351 |

**Score:** 10/11

## Required Artifacts

| Artifact | Status | Details |
|----------|--------|---------|
| modma_mdd_real_experiment.py | VERIFIED | 761 lines; imports cleanly; all functions present |

## Key Link Verification

| From | To | Via | Status |
|------|----|-----|--------|
| extract_features | theta/beta ratio | /(beta+1e-10) | WIRED |
| run_simplified_cv | build_feature_model_pipeline | called per fold | WIRED |
| build_report | run_simplified_cv | _run_one_perm_cv wrapper | WIRED |
| run_main_with_output_dir | extract_features | single call line 70 | WIRED |

## Requirements Coverage

| Requirement | Plan | Status | Evidence |
|-------------|------|--------|----------|
| FEAT-01 | 03-01 | SATISFIED | 5 TBR features; epsilon guard present |
| ARCH-02 | 03-02 | SATISFIED | Single extract_features; permutation receives pre-extracted matrix |

## Anti-Patterns Found

None.

## Gaps Summary

10/11 structural must-haves verified. The pipeline architecture is correct.

The single gap is BA >= 0.5. The 10-perm run recorded BA=0.351 vs permuted_bas_mean=0.505 (permuted baseline at chance, real classifier below chance). This is a feature quality issue, not a structural defect. SUMMARY notes "BA=0.35 on current run (below 0.5 target); Phase 4 will tune for BA>0.60".

FEAT-01 and ARCH-02 are both satisfied at the code level.

---
_Verified: 2026-02-22T07:24:01Z_
_Verifier: Claude (gsd-verifier)_
