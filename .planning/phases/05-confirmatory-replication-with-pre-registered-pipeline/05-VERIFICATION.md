---
phase: 05-confirmatory-replication-with-pre-registered-pipeline
verified: 2026-02-23T22:41:00Z
status: negative
score: 4/4 must-haves verified
---

# Phase 5: Confirmatory Replication Verification Report

**Phase Goal:** Pre-register locked pipeline, train on MODMA, test on TDBRAIN
**Verified:** 2026-02-23T22:41:00Z
**Status:** NEGATIVE (BA=0.500, p=1.0)

## Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Pre-registration doc committed and git-tagged before external data | VERIFIED | Tag pre-registration-v1 on commit c614296 |
| 2 | External adapter loads TDBRAIN BrainVision files (356 subjects) | VERIFIED | external_data_adapter.py; 312 MDD + 47 HC - 11 no EEG = 356 |
| 3 | Model trained on MODMA 43 subjects, tested on TDBRAIN frozen | VERIFIED | 05-02-SUMMARY: train 43/196 windows, test 356/1820 windows |
| 4 | BA=0.500, p=1.0 — definitive negative | VERIFIED | 05-02-SUMMARY: BA=0.500, F1=0.0 |

**Score:** 4/4

## Required Artifacts

| Artifact | Status | Details |
|----------|--------|---------|
| docs/PRE_REGISTRATION.md | VERIFIED | Locked pipeline params, commit c614296 |
| external_data_adapter.py | VERIFIED | TDBRAIN loader with channel mapping and spectrum sanity check |
| Tag pre-registration-v1 | VERIFIED | On c614296 before any external data processing |

## Key Link Verification

| From | To | Via | Status |
|------|----|-----|--------|
| PRE_REGISTRATION.md | git tag | pre-registration-v1 on c614296 | WIRED |
| external_data_adapter.py | load_windows() format | Same return signature | WIRED |

## Requirements Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| REP-01 | SATISFIED | Pre-registration committed and tagged (c614296) |
| REP-02 | SATISFIED | external_data_adapter.py loads TDBRAIN with 10-20 mapping |
| REP-03 | SATISFIED | Train MODMA 43 subjects, test TDBRAIN frozen |
| REP-04 | SATISFIED | Report: BA=0.500, p=1.0, conclusion=negative |

---
_Verified: 2026-02-23T22:41:00Z_
_Verifier: Kiro (gsd-executor)_
