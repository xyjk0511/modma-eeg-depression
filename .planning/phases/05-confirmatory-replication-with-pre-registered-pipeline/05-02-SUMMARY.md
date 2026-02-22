---
phase: 05-confirmatory-replication-with-pre-registered-pipeline
plan: 02
tags: [replication, tdbrain, cross-dataset, negative-result]
requirements-completed: [REP-03, REP-04]
completed: 2026-02-22
---

# Plan 02: Cross-Dataset Replication Summary

**Conclusion: NEGATIVE** — BA=0.500, p=1.0, F1=0.0

Train: MODMA 43 subjects/196 windows. Test: TDBRAIN 356 subjects/1820 windows.
Pipeline does not generalize cross-dataset. Consistent with Phase 4 (BA=0.613, p=0.135).

## Commits
- 04d1272: replication runner
- 15a6d81: results (metrics.json)

## Self-Check: PASSED
