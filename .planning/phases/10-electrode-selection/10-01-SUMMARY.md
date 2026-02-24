---
phase: 10-electrode-selection
plan: "01"
subsystem: electrode-selection
tags: [erp, p300, electrode-selection, parietal, egi-128]
dependency_graph:
  requires: [run_modma_erp.py, GSN-HydroCel-128 montage]
  provides: [parietal channel indices, 3-dim/5-dim subset classifiers]
  affects: [run_modma_erp.py]
tech_stack:
  added: []
  patterns: [coordinate-distance fallback for EGI channel lookup, PCA-capped LOSO for low-dim subsets]
key_files:
  modified: [run_modma_erp.py]
decisions:
  - EGI HydroCel-128 uses E1-E128 naming; all 10-20 targets require coordinate-distance fallback
  - Cz and CPz both map to E55 (idx=54) -- nearest GSN channel is the same electrode
  - 5-dim subset (BA=0.634, p=0.045) is significant; 3-dim (BA=0.521, p=0.430) is not
metrics:
  duration: 8 min
  completed: 2026-02-23
  tasks_completed: 2
  files_modified: 1
---

# Phase 10 Plan 01: Electrode Selection Summary

Identified EGI HydroCel-128 parietal channel indices via coordinate-distance fallback and ran 3-dim/5-dim LOSO classifiers; 5-dim (Pz/P3/P4/Cz/CPz) achieves BA=0.634 p=0.045 with 96% fewer channels than the 128-dim baseline.

## Results

| Model | Dims | BA | p-value | Significant |
|-------|------|----|---------|-------------|
| 128-dim baseline | 128 | 0.670 | 0.022 | Yes |
| 3-dim (Pz/P3/P4) | 3 | 0.521 | 0.430 | No |
| 5-dim (+Cz/CPz) | 5 | 0.634 | 0.045 | Yes |

Channel mapping (EGI HydroCel-128):

| 10-20 Name | Nearest GSN | Index |
|------------|-------------|-------|
| Pz | E62 | 61 |
| P3 | E60 | 59 |
| P4 | E85 | 84 |
| Cz | E55 | 54 |
| CPz | E55 | 54 |

Note: Cz and CPz both map to E55 -- effectively 4 unique channels in the 5-dim model.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | add get_parietal_indices() | 5fc50b6 | run_modma_erp.py |
| 2 | add subset classifiers + run_electrode_selection | f97f462 | run_modma_erp.py |

## Decisions Made

1. EGI HydroCel-128 uses E1-E128 naming -- coordinate-distance fallback always used for 10-20 targets.
2. _loso_ba_subset caps PCA n_components at min(X.shape[1], len(y)-1) for low-dim inputs.
3. 5-dim subset is statistically significant (p=0.045 < 0.05) -- satisfies ELEC-01/02/03.

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check: PASSED
