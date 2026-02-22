# Phase 3: Feature Reduction & Pipeline Optimization - Research

**Researched:** 2026-02-22
**Domain:** EEG feature dimensionality reduction, permutation test optimization, pipeline simplification
**Confidence:** HIGH

## Summary

Phase 3 addresses the core problem identified in baseline v4: 139 features on 43 subjects causes overfitting (BA=0.436, worse than chance). The fix is aggressive dimensionality reduction to ~29 features backed by domain knowledge, combined with pipeline simplification (fixed QC, fewer classifiers) and permutation acceleration (extract features once, permute labels only).

The baseline permutation test took 2517s for 200 permutations (~12.6s each) because each permutation re-runs the full pipeline including feature extraction. At 1000 permutations, that would be ~3.5 hours. Pre-extracting features and permuting only labels should reduce per-permutation cost to ~0.1-0.5s (just CV fitting on a 29-dim matrix), achieving >95% wall-clock reduction.

**Primary recommendation:** Rewrite `extract_features()` to produce only 29 dims, create a new simplified `run_simplified_cv()` that takes pre-extracted features with fixed QC, and rewrite `build_report()` to permute labels on the pre-extracted feature matrix.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Feature set: PSD-rel(20) + Alpha-asym(2) + Theta/Beta-ratio(5) + Riemannian-top2(2) = 29 dims
- Remove: Hjorth(15), DE(20), ImCoh(40), remaining Riemannian(13), PSD-abs(20), region-std(5), non-alpha lateral asymmetry(7)
- Fixed QC threshold 200uV -- remove inner QC threshold search
- Drop LightGBM -- keep SVM + Logistic only
- No PCA/SelectKBest -- just Scaler + Classifier
- Keep hyperparameter grids: SVM C=[0.1,1,10], Logistic C=[0.01,0.1,1]
- Extract features once, permute only subject labels, 1000 permutations
- Keep n_jobs=4
- build_report receives pre-extracted feature matrix, not raw windows

### Claude's Discretion
- Exact refactoring of extract_features internals
- How to pass pre-extracted features through permutation loop
- Whether to keep _apply_qc_and_extract or inline simplified logic
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FEAT-01 | Theta/Beta ratio per region | Compute as mean(theta_band_power) / mean(beta_band_power) per region. 5 regions = 5 dims. Theta=[4,8]Hz, Beta=[13,30]Hz. Use existing Welch PSD already computed in extract_features. |
| ARCH-02 | Permutation test extracts features once, not 1000x | Pre-extract 29-dim feature matrix once. Permutation loop only shuffles subject-level labels and runs CV on the fixed feature matrix. Eliminates ~99% of permutation cost. |
</phase_requirements>

## Codebase Analysis

### Current Feature Extraction (139-145 dims)

From `extract_features()` (line 314-383 of `modma_mdd_real_experiment.py`):

| Feature Family | Dims | Function | Keep/Remove |
|----------------|------|----------|-------------|
| Region std | 5 | inline in extract_features | REMOVE |
| PSD absolute band power | 20 | inline (5 regions x 4 bands) | REMOVE |
| PSD relative band power | 20 | inline (5 regions x 4 bands) | KEEP |
| Differential Entropy | 20 | compute_de_features() | REMOVE |
| Hjorth (activity/mobility/complexity) | 15 | compute_hjorth_features() | REMOVE |
| Lateral asymmetry (all bands) | 8 | compute_lateral_asymmetry() | REMOVE |
| Alpha asymmetry (frontal) | 1 | _alpha_asymmetry() | KEEP (expand to 2: frontal+temporal) |
| Imaginary coherence | 40 | compute_connectivity_features() | REMOVE |
| Riemannian log-cov | 15 | compute_riemannian_features() | KEEP (top 2 only) |
| **Theta/Beta ratio** | **5** | **NEW** | **ADD** |

### Current Permutation Bottleneck

The call chain for each permutation iteration:

```
_run_one_permutation()
  -> run_full_model_selection()        # outer CV loop
       -> _apply_qc_and_extract()      # per fold, per QC threshold
            -> build_quality_mask()     # QC filtering
            -> extract_features()      # EXPENSIVE: Welch PSD x2, CSD, covariance
       -> GridSearchCV.fit()           # inner CV
```

Cost breakdown per permutation (~12.6s):
- Feature extraction: ~10s (Welch PSD computed twice, CSD for 10 region pairs, covariance matrices)
- QC filtering + CV fitting: ~2.6s
- With 29 dims and no re-extraction: ~0.1-0.5s (just StandardScaler + SVM/Logistic fit on small matrix)

### Current Pipeline Complexity

`run_full_model_selection()` (line 585-715) has:
- Outer loop over `bad_amp_candidates` (currently just (200,) but the loop structure remains)
- Inner loop over model candidates (SVM, Logistic, optionally LightGBM)
- Inner GridSearchCV with StratifiedGroupKFold
- Re-extraction of features per fold per QC threshold

This entire structure can be replaced with a simpler function that:
1. Takes pre-extracted features (29 dims)
2. Runs outer StratifiedGroupKFold
3. For each fold: GridSearchCV over SVM + Logistic hyperparams
4. Returns subject-level BA

## Feature Details

### PSD Relative Power (20 dims) -- KEEP

Already computed in `extract_features()`. Formula: `mean(band_power[:, region_idx]) / mean(total_power[:, region_idx])`.

5 regions x 4 bands (delta, theta, alpha, beta) = 20 features.

Relative power normalizes for individual amplitude differences, making it robust to the MODMA dataset's amplitude variability. From baseline F-scores:
- parietal_theta_rel: F=12.95
- temporal_theta_rel: F=11.94
- occipital_theta_rel: F=9.73
- parietal_beta_rel: F=8.19
- temporal_delta_rel: F=7.61

### Alpha Asymmetry (2 dims) -- KEEP, expand

Currently 1 dim (frontal only). Expand to 2: frontal + temporal.

From baseline: `alpha_asymmetry` F=9.35, `temporal_alpha_asym` F=4.60. Both are well-established MDD biomarkers. Generalize `_alpha_asymmetry()` to accept a region parameter.

### Theta/Beta Ratio (5 dims) -- NEW (FEAT-01)

Compute per region: `mean(theta_power[:, region_idx]) / mean(beta_power[:, region_idx])`.

Theta=[4,8]Hz, Beta=[13,30]Hz. Both band powers already computed by existing Welch PSD. TBR is elevated in depression (frontal regions especially). The 2024 Frontiers study confirmed significant TBR differences in frontal lobe between depressive state groups.

Implementation: reuse `band_power["theta"]` and `band_power["beta"]`, compute ratio per region. Add epsilon to denominator: `theta / (beta + 1e-10)`.

### Riemannian Top-2 (2 dims) -- KEEP, subset

From baseline feature importance:
- riem_central_temporal: F=20.78 (highest F-score globally)
- riem_central_parietal: F=15.43 (second highest Riemannian)

Recommendation: Inline the 2-feature computation rather than computing all 15 then selecting. Compute `log(|cov(central, temporal)|)` and `log(|cov(central, parietal)|)` directly from region-averaged signals.

Riemannian index mapping (REGIONS order: frontal=0, central=1, temporal=2, parietal=3, occipital=4):
- central_temporal = triu index for (1,2)
- central_parietal = triu index for (1,3)

## Architecture: Simplified Pipeline

### New extract_features() Structure

```python
def extract_features(X, sfreq, ch_names=None):
    # 1. Welch PSD (computed ONCE)
    freqs, psd = welch(X, fs=sfreq, axis=2, nperseg=nperseg)
    # 2. Band power + total power per channel
    # 3. PSD relative power per region (20 dims)
    # 4. Alpha asymmetry: frontal + temporal (2 dims)
    # 5. Theta/Beta ratio per region (5 dims)
    # 6. Riemannian top-2: central_temporal + central_parietal (2 dims)
    return features_29dim, feature_names
```

Key simplification: Welch PSD computed only once (currently computed twice -- once in main body, once in `compute_de_features()`). No CSD computation. No per-channel Hjorth.

### New Simplified CV Function

Replace `run_full_model_selection()` with a simpler function:

```python
def run_simplified_cv(features, y, groups, n_splits=5, seed=42):
    """CV with pre-extracted features. No QC search, no feature re-extraction."""
    cv = StratifiedGroupKFold(n_splits=n_splits)
    candidates = [
        ("svm", {"clf__C": [0.1, 1.0, 10.0]}),
        ("logistic", {"clf__C": [0.01, 0.1, 1.0]}),
    ]
    # For each fold: GridSearchCV over candidates, pick best
    # Aggregate subject-level predictions
    return subject_level_metrics
```

Eliminates: QC threshold loop, _apply_qc_and_extract per fold, LightGBM, PCA/SelectKBest.

### New Permutation Loop

```python
def build_report(features, y, groups, cv_results, n_permutations=1000,
                 seed=42, n_jobs=4):
    # features is the fixed 29-dim matrix (extracted once)
    rng = np.random.RandomState(seed)
    unique_groups, first_idx = np.unique(groups, return_index=True)
    group_labels = y[first_idx]

    def _one_perm(perm_labels):
        label_map = dict(zip(unique_groups, perm_labels))
        y_perm = np.array([label_map[g] for g in groups])
        return run_simplified_cv(features, y_perm, groups)["ba"]

    jobs = [rng.permutation(group_labels) for _ in range(n_permutations)]
    results = Parallel(n_jobs=n_jobs)(delayed(_one_perm)(j) for j in jobs)
```

Key change: `build_report` receives `features` (29-dim) instead of `X_raw` (3D windows).

### Estimated Speedup

| Component | Baseline (per perm) | Phase 3 (per perm) | Reduction |
|-----------|--------------------|--------------------|-----------|
| Feature extraction | ~10s | 0s (pre-extracted) | 100% |
| QC filtering | ~1s | 0s (pre-applied) | 100% |
| CV fitting (139d, 3 models) | ~1.6s | ~0.3s (29d, 2 models) | ~80% |
| **Total per permutation** | **~12.6s** | **~0.3s** | **~97%** |
| **1000 permutations** | **~3.5 hours** | **~5 minutes** | **~97%** |

Exceeds the 50% wall-clock reduction target by a wide margin.

## Pipeline Flow: Before vs After

### Before (baseline v4)
```
load_windows() -> X_raw (3D)
  -> run_full_model_selection(X_raw, ...)
       -> for each QC threshold:
            _apply_qc_and_extract(X_raw) -> features (139 dims)
            -> GridSearchCV (SVM, Logistic, LightGBM)
  -> build_report(X_raw, ...)
       -> for each permutation:
            _run_one_permutation(X_raw, y_perm)
              -> run_full_model_selection(X_raw, y_perm)  # re-extracts!
```

### After (Phase 3)
```
load_windows() -> X_raw (3D)
  -> apply QC (fixed 200uV, dynamic max_bad)
  -> extract_features(X_qc) -> features (29 dims)  # ONCE
  -> run_simplified_cv(features, y, groups)
  -> build_report(features, y, groups, ...)
       -> for each permutation:
            run_simplified_cv(features, y_perm, groups)  # no re-extraction
```

## Pitfalls

### Pitfall 1: Theta/Beta Ratio Division by Zero
**What goes wrong:** Beta power near-zero causes inf/NaN in TBR.
**How to avoid:** Add epsilon: `theta / (beta + 1e-10)`. Same pattern used for relative power.

### Pitfall 2: Riemannian Feature Index Mismatch
**What goes wrong:** Selecting wrong indices from 15-dim Riemannian vector.
**How to avoid:** REGIONS order: frontal=0, central=1, temporal=2, parietal=3, occipital=4. central_temporal=(1,2), central_parietal=(1,3). Verify by checking feature names match.

### Pitfall 3: Permutation Labels Must Be Subject-Level
**What goes wrong:** Permuting window-level labels breaks subject grouping.
**How to avoid:** Shuffle `group_labels` (one per subject), map back to windows via `groups` array. Already done correctly in current code -- preserve this pattern.

### Pitfall 4: Pre-Extraction Invalidates QC Search
**What goes wrong:** Inconsistent QC between real eval and permutations.
**How to avoid:** Apply QC once before feature extraction. Simplified CV must not re-apply QC.

### Pitfall 5: Temporal Alpha Asymmetry Channel Lookup
**What goes wrong:** Temporal alpha asymmetry returns zeros if L/R channels not found.
**How to avoid:** Use `EGI128_LEFT["temporal"]` and `EGI128_RIGHT["temporal"]` (already defined in codebase). Verify non-zero output on real data.

## Implementation Strategy

### Modify In-Place (single file: modma_mdd_real_experiment.py)

**Functions to Remove:**
- `compute_de_features()` (lines 405-423)
- `compute_hjorth_features()` (lines 426-447)
- `compute_lateral_asymmetry()` (lines 450-468)
- `compute_connectivity_features()` (lines 471-497)
- `_apply_qc_and_extract()` (lines 557-582)
- `run_full_model_selection()` (lines 585-715)
- `run_nested_group_cv()` (lines 718-784) -- already unused
- `_run_one_permutation()` (lines 800-814)

**Functions to Modify:**
- `extract_features()` -- rewrite to 29 dims only
- `_alpha_asymmetry()` -- generalize to accept region parameter
- `build_feature_model_pipeline()` -- remove LightGBM, PCA, SelectKBest
- `_get_model_candidates()` -- remove LightGBM
- `build_report()` -- new signature with pre-extracted features
- `run_main_with_output_dir()` -- simplified flow

**Functions to Add:**
- `run_simplified_cv()` -- new simplified cross-validation

## Verification Strategy

### Success Criteria Checks

1. **29-dim features**: Assert `features.shape[1] == 29` after extraction
2. **Pipeline simplified**: No QC threshold search, SVM+Logistic only, no PCA/SelectKBest
3. **Permutation pre-extraction**: `extract_features` called exactly once (not inside permutation loop)
4. **Wall-clock >= 50% reduction**: Compare `elapsed_seconds` vs baseline 2517s. Expected: ~300-600s total
5. **BA >= 0.5**: Check `balanced_accuracy` in metrics.json

### Regression Checks

- Subject count still >= 35 (QC unchanged)
- Subject-level evaluation preserved (StratifiedGroupKFold + subject aggregation)
- No data leakage (groups array correctly used)
- Permutation p-value formula unchanged: `(sum(perm_ba >= actual_ba) + 1) / (n_perm + 1)`

## Open Questions

1. **Inline Riemannian or keep full computation?**
   - Recommendation: Inline. Compute only 2 covariance entries directly instead of all 15.
   - Confidence: HIGH

2. **Keep `_apply_qc_and_extract`?**
   - Recommendation: Remove. QC applied once before extraction.

3. **Test file updates?**
   - Tests for removed functions need removal. Tests for extract_features need 29-dim update.
   - Per user rules: only modify tests when explicitly requested.

## Sources

### Primary (HIGH confidence)
- Baseline v4 metrics.json: BA=0.436, p=0.886, 139 features, elapsed=2517s for 200 permutations
- Baseline v4 feature_importance.csv: F-scores confirming riem_central_temporal (F=20.78) and riem_central_parietal (F=15.43) as top discriminators
- Codebase analysis: `modma_mdd_real_experiment.py` (1076 lines), full call chain traced

### Secondary (MEDIUM confidence)
- [Theta/Beta ratio in depression (Frontiers 2024)](https://www.frontiersin.org/articles/10.3389/fnhum.2024.1384330/full)
- [EEG markers in MDD (Frontiers Psychiatry 2024)](https://doi.org/10.3389/fpsyt.2024.1480228)
- [sklearn permutation_test_score](https://scikit-learn.org/1.1/modules/generated/sklearn.model_selection.permutation_test_score.html)
- [Dimensionality reduction for EEG (Springer 2024)](https://link.springer.com/10.1007/s10462-024-10711-8)
- [scipy.signal.welch](https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.signal.welch.html)

## Metadata

**Confidence breakdown:**
- Feature reduction strategy: HIGH -- backed by F-score analysis on real data
- Permutation optimization: HIGH -- standard pre-extraction pattern
- Pipeline simplification: HIGH -- removing unused code paths
- Theta/Beta ratio: HIGH -- trivial computation using existing band power
- Speedup estimate: MEDIUM -- 97% is theoretical, actual depends on CV overhead

**Research date:** 2026-02-22
**Valid until:** 2026-03-22

---
*Phase: 03-feature-permutation-opt*
*Research completed: 2026-02-22*
