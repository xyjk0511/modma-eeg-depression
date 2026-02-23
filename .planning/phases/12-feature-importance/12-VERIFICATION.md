---
phase: 12-feature-importance
verified: 2026-02-23T09:24:30Z
status: passed
score: 3/3 must-haves verified
---

# Phase 12: Feature Importance Verification Report

**Phase Goal:** 将 LOSO 各折 LR coef_ 反投影到通道空间，输出可解释的 128 通道权重排名
**Verified:** 2026-02-23T09:24:30Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | 每折 coef_ 通过 pca.components_.T @ coef_ 反投影为 128-dim 通道权重 | VERIFIED | run_modma_erp.py:457 — exact expression inside LOSO loop, returns (n_folds, 128) |
| 2 | 跨折平均绝对权重排名输出，top-10 通道名打印 | VERIFIED | lines 498-514 — avg_abs computed, top-10 printed; CSVs 128 rows, hcue max=0.7281, bin250 max=0.3027 |
| 3 | 128 通道权重地形图生成并保存 | VERIFIED | out_phase12/topomap_hcue_vs_bin.png exists, 348 KB, RdBu_r, top-5 labels |

**Score:** 3/3 truths verified

### Required Artifacts

| Artifact | Status | Details |
|----------|--------|---------|
| run_modma_erp.py | VERIFIED | _loso_collect_coefs (line 447) + run_feature_importance (line 461), called from __main__ line 573 |
| out_phase12/feature_importance_hcue.csv | VERIFIED | 128 rows, cols: rank/channel/avg_abs_weight/avg_weight/sign, top=E55 w=0.7281 |
| out_phase12/feature_importance_bin250.csv | VERIFIED | 128 rows, same columns, top=E55 w=0.3027 |
| out_phase12/topomap_hcue_vs_bin.png | VERIFIED | 348,473 bytes, RdBu_r, top-5 channel labels wired |

### Key Link Verification

| From | To | Via | Status |
|------|----|-----|--------|
| _loso_collect_coefs | pca.components_.T @ clf.coef_[0] | back-projection per fold | WIRED — line 457 |
| run_feature_importance | mne.viz.plot_topomap | GSN-HydroCel-128 Info object | WIRED — lines 484-528 |
| run_feature_importance | __main__ | direct call | WIRED — line 573 |

### Requirements Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| FIMP-01 | SATISFIED | per-fold pca.components_.T @ clf.coef_[0] implemented; two 128-row ranked CSVs produced |

### Anti-Patterns Found

None.

### Human Verification Required

None.

---
_Verified: 2026-02-23T09:24:30Z_
_Verifier: Kiro (gsd-verifier)_
