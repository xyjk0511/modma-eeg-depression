# Project Research Summary

**Project:** ERP Task-State MDD Classification — v3.0 ERP Deep Analysis
**Domain:** EEG/ERP biomarker research, MDD vs HC binary classification (MODMA dot-probe, N=52)
**Researched:** 2026-02-23
**Confidence:** HIGH

## Executive Summary

This is a scientific EEG/ERP analysis pipeline built on a validated v2.0 baseline (BA=0.670, p=0.021 on hcue P300 mean amplitude, 128-dim, LOSO). The v3.0 milestone is not about improving classification accuracy through more features — v2.0 proved that adding features (640-dim Phase 7, 384-dim Phase 9) consistently degrades performance due to the curse of dimensionality at N=52. The v3.0 goal is dimensionality reduction via electrode selection and scientific interpretation of what drives the existing BA=0.670 result.

The recommended approach is a strict build order dictated by data dependencies: (1) extend `load_erp_features()` with LPP window and verify BA does not regress, (2) add time-window t-test as standalone descriptive analysis, (3) add electrode selection to reduce 128 channels to a parietal ROI (Pz/P3/P4), (4) add feature importance via SVM weights back-projected through PCA. All new analysis branches feed into the unchanged `run_loso()` and `permutation_test_loso()` functions — these are locked and must not be modified. Zero new library installs are required.

The dominant risk is data leakage in three forms: electrode selection on the full dataset before LOSO, time-window t-test results feeding back into the LOSO feature window, and feature importance weights reported in PCA component space rather than channel space. The mitigation is strict separation: use literature-defined windows and channel sets for classification; report data-driven analyses (t-test, importance) as descriptive only.

## Key Findings

### Recommended Stack

The existing stack (MNE 1.11.0, NumPy 2.4.2, SciPy 1.17.0, scikit-learn 1.8.0, joblib 1.5.3) covers 100% of v3.0 requirements. No new installs are needed.

**Core technologies:**
- MNE 1.11.0: ERP epoching, baseline correction, evoked averaging — already in use
- NumPy 2.4.2: windowed mean amplitude extraction, array concatenation — already in use
- SciPy 1.17.0: `ttest_ind` for time-window and electrode t-tests — already installed
- scikit-learn 1.8.0: `LinearSVC`, `permutation_importance`, `f_classif` — already installed
- joblib 1.5.3: parallel permutation testing — already in use

**Critical config change (not a library change):** `TMAX` must be extended from 0.5 to 0.8 to accommodate the LPP window (500-800ms). Must assert `times[-1] >= 0.8` before extracting LPP features.

### Expected Features

**Must have (table stakes for v3.0):**
- Electrode subset selection (Pz/P3/P4 parietal ROI) — P300 is maximal at parietal sites; reduces 128-dim to 3-dim, directly addressing the dimensionality root cause
- Time-window t-test (per 4ms bin, 250-500ms) — standard in ERP clinical literature; identifies which milliseconds are discriminative
- LR coef_ interpretation across LOSO folds — zero implementation cost; maps which channels drive BA=0.670

**Should have (differentiators):**
- Parietal-only classifier (3-dim: Pz/P3/P4 mean) — if BA >= 0.670 with 3 channels, result is clinically interpretable and publishable
- Time-window ablation (50ms bins) — classifier-based evidence for peak discriminative window
- Permutation importance (model-agnostic) — corrects for correlated features where raw coef_ misleads

**Defer to v3.2+:**
- LPP 500-800ms full 128-channel — medrxiv 2020 found no evidence for reduced LPP in MDD; low prior probability
- N200 parietal 6-dim — only worth testing if 3-dim parietal P300 achieves BA >= 0.670 first
- Multi-condition fusion — Phase 9 confirmed this degrades performance (BA=0.521)

### Architecture Approach

Single-file pipeline (`run_modma_erp.py`) with a strict layered structure. All v3.0 additions are either modifications to `load_erp_features()` or new standalone functions. The classification and evaluation layers (`run_loso`, `permutation_test_loso`) are locked.

**Major components:**
1. `load_erp_features()` (modified) — extend to return 384-dim (P300+N200+LPP mean per channel) and `avg_erps` array
2. `select_electrodes(X, y, k)` (new) — rank 128 channels by univariate t-stat, return top-k indices
3. `time_window_ttest(avg_erps, y, times)` (new) — MDD vs HC t-stat at each time point, descriptive only
4. `feature_importance(X, y, feat_names)` (new) — LinearSVC coef_ weights ranked by absolute value
5. `run_loso()` / `permutation_test_loso()` (unchanged) — locked since Phase 8 validation

### Critical Pitfalls

1. **Feature enrichment dimensionality regression** — confirmed Phase 7: 640-dim dropped BA from 0.670 to 0.554. Keep total features < N_train/5 ≈ 10 after any reduction. Use mean amplitude only (no peak, latency, AUC).

2. **Electrode selection leakage** — selecting top-K channels on full dataset then using in LOSO is leakage. Use literature-defined sets (P300: Pz/Cz/P3/P4/CPz) or select inside LOSO loop.

3. **Time-window double-dipping** — running t-test on all 52 subjects to find the best window, then using that window in LOSO, is circular. Keep t-test strictly descriptive; use literature windows (250-500ms) for classification.

4. **PCA weight back-projection missing** — `clf.coef_` is in PCA component space (shape 1x20), not channel space. Back-project via `pca.components_.T @ clf.coef_.flatten()` to get (128,) channel weights.

5. **LPP epoch duration** — current `TMAX=0.5` truncates the LPP window (500-800ms). Must change `TMAX=0.8` and assert `times[-1] >= 0.8` before extracting LPP features.

## Implications for Roadmap

### Phase 10: LPP Extension + Epoch Duration Fix
**Rationale:** Foundation for all v3.0 work. Must extend `TMAX` to 0.8 and verify BA does not regress before any other analysis.
**Delivers:** 384-dim feature matrix (P300+N200+LPP mean per channel); `avg_erps` array for t-test; confirmed non-regression of BA=0.670.
**Addresses:** LPP component extraction (table stakes), epoch duration fix (Pitfall 5).
**Avoids:** Pitfall 1 (dimensionality) — PCA(20) absorbs the increase; Pitfall 5 (epoch truncation).

### Phase 11: Time-Window t-test (Descriptive)
**Rationale:** Independent of classifier; runs immediately after Phase 10 provides `avg_erps`. No leakage risk if kept strictly descriptive.
**Delivers:** t-statistic vs time plot; identification of peak discriminative window within 250-500ms.
**Addresses:** Time-window t-test analysis (table stakes feature).
**Avoids:** Pitfall 3 (double-dipping) — results reported separately, never fed back into LOSO feature window.

### Phase 12: Electrode Selection + Parietal ROI Classifier
**Rationale:** Core v3.0 scientific question: can 3 parietal channels (Pz/P3/P4) match or exceed BA=0.670? Requires EGI montage channel lookup (non-trivial).
**Delivers:** Parietal-only 3-dim classifier with BA and permutation p-value; comparison vs 128-dim baseline.
**Addresses:** Electrode subset selection (P1 feature), parietal-only classifier (differentiator).
**Avoids:** Pitfall 2 (electrode selection leakage) — literature-defined channel sets only.

### Phase 13: Feature Importance
**Rationale:** Post-hoc interpretation after LOSO significance confirmed. Last because it is descriptive, not a classification step.
**Delivers:** Channel importance map (back-projected LR/SVM weights); top-20 features by absolute weight.
**Addresses:** LR coef_ interpretation (table stakes), permutation importance (differentiator).
**Avoids:** Pitfall 4 (PCA back-projection) — use `pca.components_.T @ clf.coef_.flatten()`.

### Phase Ordering Rationale

- Phase 10 before all others: `avg_erps` array and `TMAX=0.8` are prerequisites for LPP and t-test.
- Phase 11 before Phase 12: t-test is independent and low-risk; validates 250-500ms window before electrode selection.
- Phase 12 before Phase 13: importance analysis should run on the validated, reduced feature matrix.
- Strict separation of descriptive (Phases 11, 13) from classification (Phases 10, 12) prevents leakage.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 12:** EGI 128-channel montage lookup is non-trivial. Channel names are numeric (E1-E128), not standard 10-20. Requires `mne.channels.make_standard_montage("GSN-HydroCel-128")` and spatial coordinate matching. Validate on 2-3 subjects before full run.

Phases with standard patterns (skip research-phase):
- **Phase 10:** LPP extraction is a direct extension of existing P300/N200 pattern.
- **Phase 11:** `scipy.stats.ttest_ind` per time point is a standard ERP analysis pattern.
- **Phase 13:** `LinearSVC.coef_` + PCA back-projection is a documented sklearn pattern.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All libraries verified installed; zero new installs required |
| Features | MEDIUM | P1/P2 features well-supported; LPP deferral based on single preprint |
| Architecture | HIGH | Based on direct code analysis + empirical Phase 7/8/9 results |
| Pitfalls | HIGH | Pitfalls 1, 8, 10, 11 empirically confirmed, not just theoretical |

**Overall confidence:** HIGH

### Gaps to Address

- **EGI montage channel mapping:** Exact indices for Pz/P3/P4 in EGI HydroCel 128 not pre-computed. Must resolve in Phase 12 via `mne.channels.make_standard_montage("GSN-HydroCel-128")`. Failure silently produces wrong features.
- **LPP epoch duration:** Whether MODMA .raw files support `TMAX=0.8` is unverified. Assert `times[-1] >= 0.8` at runtime in Phase 10.
- **N200 grand-average:** Whether a distinct negative deflection exists in 100-250ms in MODMA dot-probe data is unverified. If not present, skip N200 features entirely.

## Sources

### Primary (HIGH confidence)
- `run_modma_erp.py` — direct code analysis; current pipeline structure
- `.planning/phases/07-p300-feature-enrichment-n200/07-01-SUMMARY.md` — 640-dim regression BA=0.554
- `.planning/phases/08-permutation-test-validation/08-01-SUMMARY.md` — BA=0.670, p=0.021 baseline
- `.planning/phases/09-multi-condition-fusion-contrast-features/09-VERIFICATION.md` — fusion BA=0.521
- Brookshire et al., Frontiers in Neuroscience 2024 — data leakage in translational EEG
- Bioinformatics 2010 — permutation importance as corrected feature importance measure

### Secondary (MEDIUM confidence)
- arXiv 2510.21969 (2024) — parietal ROI for small-sample P300 classification
- arXiv 2511.02735 (2024) — data-driven ROI selection for P300 BCIs
- Frontiers Psychiatry 2022 — auditory P300 at Pz/Cz for MDD
- MDPI Applied Sciences 2024 — SVM-RFE feature selection on EEG depression

### Tertiary (LOW confidence)
- medrxiv 2020 — lack of evidence for reduced LPP in MDD (preprint; drives LPP deferral)

---
*Research completed: 2026-02-23*
*Ready for roadmap: yes*
