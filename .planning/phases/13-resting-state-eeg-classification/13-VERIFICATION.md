---
phase: 13-resting-state-eeg-classification
verified: 2026-02-23T22:43:00Z
status: negative
score: 3/3 must-haves verified
---

# Phase 13: Resting-State EEG Classification Verification Report

**Phase Goal:** Resting-state EEG band-power LOSO classification, compare with ERP BA=0.670
**Verified:** 2026-02-23T22:43:00Z
**Status:** NEGATIVE (BA=0.391)

## Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | load_resting_features() returns (53, 640) | VERIFIED | Commit 00b549b; 5 bands x 128 channels = 640 dims |
| 2 | LOSO BA=0.391, AUC=0.369 | VERIFIED | out_phase13/resting_result.txt |
| 3 | out_phase13/resting_result.txt exists | VERIFIED | Contains BA, AUC, delta vs ERP baseline |

**Score:** 3/3

## Required Artifacts

| Artifact | Status | Details |
|----------|--------|---------|
| run_modma_erp.py | VERIFIED | load_resting_features() + run_resting_analysis(), commit 00b549b |
| out_phase13/resting_result.txt | VERIFIED | BA=0.391, AUC=0.369, Delta BA=-0.279 |

## Requirements Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| REST-01 | SATISFIED | Resting-state LOSO completed, BA=0.391 vs ERP BA=0.670 |

---
_Verified: 2026-02-23T22:43:00Z_
_Verifier: Kiro (gsd-executor)_
