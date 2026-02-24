# Domain Pitfalls: v2.0 Stacking + Functional Connectivity

**Domain:** EEG psychiatric classification — adding stacking meta-learner and connectivity features to existing pipeline
**Researched:** 2026-02-24
**Confidence:** HIGH (stacking/leakage pitfalls verified against sklearn 1.8.0 source), MEDIUM (connectivity-specific)

---

## Critical Pitfalls

### Pitfall 1: StackingClassifier Silently Ignores Subject Groups (DATA LEAKAGE)

**What goes wrong:**
sklearn's `StackingClassifier` uses `cross_val_predict` internally to generate out-of-fold meta-features. In sklearn 1.8.0, `_BaseStacking.fit()` calls `cross_val_predict()` but does NOT pass the `groups` parameter. Even if you set `cv=StratifiedGroupKFold(5)`, the splitter receives no `groups`, so it ignores group constraints entirely. Same subject can appear in both training and prediction sets within the stacking layer.

**Verified by source inspection:**
```python
# sklearn 1.8.0: sklearn/ensemble/_stacking.py, _BaseStacking.fit()
predictions = Parallel(n_jobs=self.n_jobs)(
    delayed(cross_val_predict)(
        clone(est), X, y,
        cv=deepcopy(cv),
        method=meth,
        n_jobs=self.n_jobs,
        params=routed_params[name]["fit"],  # <-- no groups here
        verbose=self.verbose,
    )
    ...
)
```

**Consequences:**
Subject leakage in meta-feature generation. AUC inflation of 0.02-0.08 typical.

**Prevention:**
Do NOT use `sklearn.ensemble.StackingClassifier`. Implement stacking manually:
```python
meta_features = np.zeros((len(X), n_base_models))
for train_idx, val_idx in stacking_cv.split(X, y, groups):
    for i, (pipe, _) in enumerate(base_models):
        pipe.fit(X[train_idx], y[train_idx])
        meta_features[val_idx, i] = pipe.predict_proba(X[val_idx])[:, 1]
meta_learner.fit(meta_features, y)
```

**Detection:**
- Compare manual stacking (with groups) vs StackingClassifier — if StackingClassifier is >0.03 higher, leakage likely
- Monkey-patch `StratifiedGroupKFold.split()` with logging to verify `groups` is received

**Phase to address:** Stacking implementation (first phase of v2.0)
**Confidence:** HIGH — verified by reading sklearn 1.8.0 source code directly

---

### Pitfall 2: SMOTE Synthetic Samples Leak Between Stacking Layers

**What goes wrong:**
When generating out-of-fold predictions for stacking, if SMOTE is applied at the stacking level (before base model CV), synthetic samples from subject A's features can influence predictions on subject A's test fold via SMOTE's k-nearest-neighbors bridging across the CV boundary.

**Consequences:**
Meta-features optimistically biased for minority class (ADHD, n=200). Sensitivity inflated, specificity degraded.

**Prevention:**
SMOTE only inside each base model's `ImbPipeline`. Manual stacking loop handles this naturally — `.fit()` applies SMOTE to training data only, `.predict_proba()` does not.

**Detection:**
- Verify SMOTE is inside each base model's pipeline, not applied globally
- If ADHD meta-feature probabilities are suspiciously well-calibrated (mean ~0.5), SMOTE leakage likely

**Phase to address:** Stacking implementation
**Confidence:** MEDIUM

---

### Pitfall 3: Connectivity Feature Dimensionality Explosion (p >> n)

**What goes wrong:**
26 channels = C(26,2) = 325 pairs. With 5 bands x 2 metrics (coherence + PLV) = 3,250 new features. Combined with existing 992 = 4,242 total for ~493 subjects. Feature-to-sample ratio: 8.6:1.

**Consequences:**
- SelectKBest k=50 from 4,242 candidates unreliable — too many spurious correlations
- Permutation test p-values anti-conservative
- Runtime explosion

**Prevention:**
1. **ROI grouping:** 5-7 ROIs -> C(7,2) = 21 pairs x 3 bands x 1 metric = 63 features
2. **Band selection:** theta, alpha, beta only (3 bands most relevant to MDD/ADHD)
3. **Single metric:** wPLI only (see Pitfall 4)

**Detection:**
- Total features > 2x subjects -> mandatory reduction
- SelectKBest features differ >50% across outer folds -> space too large

**Phase to address:** Connectivity feature extraction (before stacking)
**Confidence:** HIGH — mathematical fact

---

### Pitfall 4: Volume Conduction Inflates Coherence and PLV

**What goes wrong:**
Scalp-level coherence and PLV are contaminated by volume conduction — same neural source projects to multiple electrodes with zero phase lag, creating spurious high coherence between nearby channels.

**Consequences:**
- Nearby channel pairs dominate with artifactually high values
- Classifier learns head geometry, not brain connectivity
- Results don't replicate across different montages

**Prevention:**
Use volume-conduction-robust metrics:
- **wPLI:** `mne_connectivity.spectral_connectivity_epochs(method='wpli')` — discards zero-phase-lag
- **Imaginary coherence:** Retains only non-zero phase lag component
- **Do NOT use:** Standard coherence, PLV, or amplitude envelope correlation at scalp level

**Detection:**
- Adjacent electrode connectivity (e.g., F3-Fz) > 0.8 -> volume conduction dominates
- Removing adjacent pairs drops AUC > 0.03 -> classifier relied on artifacts

**Phase to address:** Connectivity feature extraction
**Confidence:** HIGH — well-established (Stam 2007, Vinck 2011)

---

### Pitfall 5: Meta-Learner Overfitting on 3 Features x 493 Samples

**What goes wrong:**
Meta-learner receives only 3 features (one probability per base model). A complex meta-learner (RF, XGBoost) overfits to noise, memorizing which base model is "right" for specific subjects.

**Consequences:**
- Stacking AUC degrades on outer CV vs inner CV
- Performs worse than best single base model

**Prevention:**
- Use `LogisticRegression(C=1.0)` — only 4 parameters (3 weights + intercept)
- Do NOT use tree-based meta-learners for 3 input features
- If stacking AUC < best single model, revert to soft voting

**Detection:**
- Stacking AUC < best single model -> overfitting
- Meta-learner coefficients |w| > 10 -> insufficient regularization

**Phase to address:** Stacking implementation
**Confidence:** HIGH

---

## Moderate Pitfalls

### Pitfall 6: Permutation Test Pipeline Mismatch After Adding Stacking

**What goes wrong:**
Current `FIXED_PIPES` mirror individual models. Stacking adds degrees of freedom not captured in the null distribution, making p-values anti-conservative.

**Prevention:**
Permutation test must wrap the entire stacking procedure:
```python
for _ in range(n_permutations):
    y_perm = _group_permute()
    perm_auc = run_full_stacking_cv(X, y_perm, groups)
    perm_scores.append(perm_auc)
```

**Phase to address:** Permutation test update (after stacking works)
**Confidence:** HIGH

---

### Pitfall 7: Connectivity Computation Time Explosion

**What goes wrong:**
`mne_connectivity.spectral_connectivity_epochs()` computes all 325 pairs x n_freqs x n_epochs. For 493 subjects x 2 conditions, runtime is hours.

**Prevention:**
- Precompute connectivity once, cache to `.npz`
- Use ROI-averaged signals (7 ROIs = 21 pairs) before computing
- Connectivity is label-independent — only classifier re-runs during permutation

**Phase to address:** Connectivity feature extraction
**Confidence:** MEDIUM

---

### Pitfall 8: EC/EO Connectivity Difference Features Are Unreliable

**What goes wrong:**
Connectivity estimates from short recordings have high variance. Differencing (EO - EC) amplifies noise, yielding low SNR features.

**Prevention:**
- Concatenate EC/EO connectivity first, not difference
- Add difference only if concatenation shows clear benefit
- If difference features have lower F-scores than raw, they add noise

**Phase to address:** Connectivity feature engineering
**Confidence:** MEDIUM

---

### Pitfall 9: Stacking CV Folds Must Nest Inside Outer CV

**What goes wrong:**
If stacking internal CV operates on the full dataset rather than inside the outer training fold, subjects in the outer test set leak into stacking meta-feature generation.

**Prevention:**
```python
for train_idx, test_idx in outer_cv.split(X, y, groups):
    X_tr, y_tr, g_tr = X[train_idx], y[train_idx], groups[train_idx]
    meta_train = np.zeros((len(train_idx), n_models))
    for tr, va in stacking_cv.split(X_tr, y_tr, g_tr):
        for i, m in enumerate(base_models):
            m.fit(X_tr[tr], y_tr[tr])
            meta_train[va, i] = m.predict_proba(X_tr[va])[:, 1]
    meta_learner.fit(meta_train, y_tr)
```

**Phase to address:** Stacking implementation
**Confidence:** HIGH

---

## Minor Pitfalls

### Pitfall 10: mne-connectivity Not Installed

**What goes wrong:**
`mne.connectivity` split into separate `mne-connectivity` package since MNE 1.0. Import fails without it.

**Prevention:** `pip install mne-connectivity`

**Phase to address:** Environment setup
**Confidence:** HIGH

---

### Pitfall 11: Base Model Diversity Insufficient for Stacking Benefit

**What goes wrong:**
SVM-RBF, RF, XGBoost all use same SelectKBest features. If errors correlate, meta-learner has no complementary signal. Stacking degenerates to weighted averaging.

**Prevention:**
- Check pairwise error correlation > 0.8 -> diversity too low
- Consider adding LogisticRegression for different inductive bias

**Detection:** Stacking AUC equals soft-vote AUC -> stacking adds no value

**Phase to address:** Stacking implementation
**Confidence:** MEDIUM

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Connectivity extraction | Volume conduction (P4) | Use wPLI, not coherence/PLV |
| Connectivity extraction | Dimensionality explosion (P3) | ROI grouping: 7 ROIs, 3 bands, wPLI = 63 features |
| Connectivity extraction | Computation time (P7) | Cache to disk; ROI-averaged signals |
| Stacking implementation | Groups leak (P1) | Manual stacking loop with explicit groups |
| Stacking implementation | SMOTE leak (P2) | SMOTE inside base model pipeline only |
| Stacking implementation | Meta-learner overfit (P5) | LogisticRegression, not tree-based |
| Stacking implementation | Fold alignment (P9) | Stacking CV inside outer training fold |
| Permutation testing | Pipeline mismatch (P6) | Full stacking pipeline in permutation loop |
| EC/EO connectivity | Noisy differences (P8) | Concatenate first, difference only if beneficial |

## "Looks Done But Isn't" Checklist

- [ ] **No StackingClassifier used:** Manual stacking loop passes `groups` to every `.split()` call
- [ ] **SMOTE only in base pipelines:** No SMOTE at stacking level or on full dataset
- [ ] **Connectivity metric is wPLI:** Not standard coherence or PLV
- [ ] **ROI grouping applied:** Total connectivity features < 200 (not 3,250)
- [ ] **Meta-learner is LogisticRegression:** Not RF/XGBoost/SVM
- [ ] **Stacking CV inside outer fold:** `groups[test_idx]` zero overlap with stacking CV groups
- [ ] **Permutation test covers full stack:** Null distribution from full stacking pipeline
- [ ] **mne-connectivity installed:** Import works
- [ ] **Feature-to-sample ratio < 3:1:** Total features after connectivity < 1500
- [ ] **Stacking AUC >= best single model:** If not, revert to soft voting

## Recovery Strategies

| Pitfall | Cost | Recovery Steps |
|---------|------|----------------|
| StackingClassifier groups leak (P1) | HIGH | Rewrite as manual loop; re-run all experiments |
| SMOTE stacking leak (P2) | MEDIUM | Restructure pipeline; re-run stacking |
| Dimensionality explosion (P3) | MEDIUM | Add ROI grouping; re-extract connectivity |
| Volume conduction (P4) | MEDIUM | Switch to wPLI; re-extract connectivity |
| Meta-learner overfit (P5) | LOW | Swap to LogisticRegression; re-run |
| Permutation mismatch (P6) | MEDIUM | Update permutation loop; re-run 1000 perms |
| Computation time (P7) | LOW | Add caching; use ROI signals |

## Sources

- sklearn 1.8.0 source: `sklearn/ensemble/_stacking.py`, `_BaseStacking.fit()` — direct inspection (HIGH)
- [StratifiedGroupKFold in StackingClassifier — SO](https://stackoverflow.com/questions/78948520/using-stratifiedgroupkfold-in-stackingclassifier) (HIGH)
- [GroupKFold nested CV — sklearn #7646](https://github.com/scikit-learn/scikit-learn/issues/7646) (HIGH)
- [Volume Conduction Influences Connectivity — Frontiers 2016](https://www.frontiersin.org/articles/10.3389/fncom.2016.00121) (HIGH)
- [PLI vs PLV — Sapien Labs](https://sapienlabs.org/lab-talk/eeg-connectivity-using-phase-lag-index/) (MEDIUM)
- [wPLI and coherence — PubMed 2024](https://pubmed.ncbi.nlm.nih.gov/38825253) (MEDIUM)
- [Ensemble Learning Pitfalls — Moldstud](https://moldstud.com/articles/p-common-pitfalls-in-ensemble-learning-essential-mistakes-every-ml-developer-should-avoid) (MEDIUM)
- [Curse of dimensionality in EEG — Springer 2024](https://link.springer.com/article/10.1007/s10462-024-10711-8) (MEDIUM)
- [SMOTE leakage — imbalanced-learn docs](https://imbalanced-learn.org/stable/common_pitfalls.html) (HIGH)
- [Stacking with Dropout — Jason Brownlee](https://jasonbrownlee.me/blog/posts/stacking-dropout/) (MEDIUM)

---
*Pitfalls research for: EEG MDD/ADHD classification — v2.0 stacking + functional connectivity*
*Researched: 2026-02-24*

