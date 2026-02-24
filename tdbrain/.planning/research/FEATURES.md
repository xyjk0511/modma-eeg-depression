# Feature Landscape: v2.0 Stacking + Functional Connectivity

**Domain:** EEG-based psychiatric classification (MDD vs nonMDD), resting-state
**Researched:** 2026-02-24
**Confidence:** MEDIUM (stacking well-understood) / MEDIUM (connectivity needs dimensionality management)

## Context: What Already Exists (v1.3)

496-dim per condition: abs/rel band power (260), TBR+FAA (2), Hjorth (78), spectral entropy (156).
EC/EO concat = 992-dim. SelectKBest(k=50). SVM/RF/XGB soft-vote.
Best: ENS AUC=0.670, BA=0.603 (n=1138, MDD=305, nonMDD=833).

---

## Table Stakes

| Feature | Why Expected | Complexity | Dependencies |
|---------|--------------|------------|--------------|
| Stacking meta-learner (replace soft-vote) | Learns optimal weighting; standard in EEG ensemble papers (Frontiers 2025, MDPI 2022) | LOW | Existing SVM/RF/XGB |
| LogisticRegression as meta-learner | Low overfitting with 3 inputs; standard choice in sklearn docs | LOW | sklearn or manual |
| Out-of-fold meta-feature generation | Prevents leakage; MUST use StratifiedGroupKFold for subject grouping | LOW | Existing SGKF CV |
| Coherence (magnitude-squared) | Linear frequency-domain coupling; captures amplitude co-variation PSD misses | HIGH | mne-connectivity or manual CSD |
| PLV (Phase Locking Value) | Phase synchronization; amplitude-independent; ADHD studies (BMC Medicine 2017) | HIGH | mne-connectivity or Hilbert |
| Band-specific connectivity | Theta/alpha/beta carry different clinical meaning | MEDIUM | Band-averaged connectivity |

## Differentiators

| Feature | Value Proposition | Complexity | Dependencies |
|---------|-------------------|------------|--------------|
| ROI-averaged connectivity | 5 ROIs instead of 325 pairs; ~15 values per band per metric; interpretable | LOW | Channel-to-ROI mapping |
| EC/EO difference connectivity | Delta coherence/PLV captures network reactivity; alpha reactivity is MDD biomarker | MEDIUM | Connectivity for both EC and EO |
| Graph-theoretic summaries | Clustering coeff, path length, efficiency; 325 pairs to ~5 metrics | MEDIUM | Connectivity matrix; networkx |
| Meta-learner hyperparameter tuning | Regularization (C) of LR via nested CV | LOW | GridSearchCV |
## Anti-Features

| Anti-Feature | Why Avoid | Do Instead |
|--------------|-----------|------------|
| All-pairs connectivity (3250 features) | Curse of dimensionality; n=1138 insufficient | ROI-averaged (~90 features) |
| Deep learning on connectivity (GCN) | Incompatible with sklearn; needs larger data | Classical ML with flat vectors |
| Transfer entropy / Granger causality | Expensive; needs long epochs; unclear gain | Coherence + PLV |
| Imaginary coherence (ImCoh) | Discards real coupling; volume conduction consistent across subjects | Standard coherence |
| Neural network meta-learner | Only 3 inputs; will overfit | LogisticRegression |
| passthrough=True with 992-dim | 995 inputs to meta-learner; overfitting | passthrough=False first |
---

## Feature Dependencies

```
[Existing 992-dim spectral] --> [SVM/RF/XGB] --> [Stacking meta-learner (NEW)]
[Coherence/PLV (NEW)] --> [26x26 matrix per band] --> [ROI-averaged features]
[ROI-averaged + Spectral] --> [Updated classifiers] --> [Stacking]
```
### Critical: StackingClassifier groups= Limitation

sklearn StackingClassifier.fit() does NOT pass groups= to its internal cross_val_predict. Use manual stacking: call cross_val_predict per base model with groups=, then train LogisticRegression on stacked OOF probabilities.

---

## Dimensionality Analysis

| Strategy | Per band per metric | x3 bands x2 metrics | Total |
|----------|---------------------|----------------------|-------|
| All pairs | C(26,2) = 325 | x6 | 1950 |
| ROI-averaged (5 ROIs) | C(5,2)+5 = 15 | x6 | 90 |
| Graph summaries | ~5 metrics | x6 | 30 |

**Recommendation:** ROI-averaged (90 features). With spectral: 992+90=1082. SelectKBest(k=50-100) handles it.

### ROI Mapping for 26-Channel Montage

| ROI | Channels |
|-----|----------|
| Frontal | Fp1, Fp2, F7, F3, Fz, F4, F8 |
| Central | FC3, FCz, FC4, C3, Cz, C4 |
| Temporal | T7, T8 |
| Parietal | CP3, CPz, CP4, P7, P3, Pz, P4, P8 |
| Occipital | O1, Oz, O2 |

5 ROIs -> 10 inter-ROI + 5 intra-ROI = 15 per band per metric.

---

## MVP Recommendation

### Phase 1: Stacking Meta-Learner (LOW risk)

Replace soft-vote with manual stacking. Zero new features needed.

1. cross_val_predict per base model with groups=
2. Stack OOF probabilities -> LogisticRegression(C=1.0)
3. Compare vs soft-vote AUC (0.670 baseline)

Expected gain: 0.005-0.020 AUC.

### Phase 2: Functional Connectivity (MEDIUM risk, main effort)

1. Compute coherence + PLV per band (theta/alpha/beta) for EC and EO
2. ROI-average: 15 x 3 bands x 2 metrics x 2 conditions = 180 features
3. Append to 992-dim spectral = 1172 total
4. SelectKBest(k=50-100) handles reduction

Expected gain: 0.010-0.040 AUC.
### Phase 3: EC/EO Difference Connectivity (optional)

Only if Phase 2 shows connectivity signal in SelectKBest top features.
Delta connectivity (EO - EC) adds 90 features -> 1262 total.

### Defer

- Graph-theoretic summaries: only if ROI-averaged shows signal
- passthrough=True: test after baseline stacking validated
- All-pairs connectivity: avoid entirely

---

## Implementation Notes

### Manual Stacking (recommended over StackingClassifier)

```python
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression

sgkf = StratifiedGroupKFold(n_splits=5)
meta_features = np.column_stack([
    cross_val_predict(pipe, X, y, cv=sgkf, groups=groups,
                      method='predict_proba')[:, mdd_idx]
    for pipe in [svm_pipe, rf_pipe, xgb_pipe]
])  # (n_subjects, 3)
meta_clf = LogisticRegression(C=1.0, max_iter=1000)
meta_clf.fit(meta_features, y)
```
### ROI-Averaged Connectivity

```python
from mne_connectivity import spectral_connectivity_epochs

con = spectral_connectivity_epochs(
    epochs, method=['coh', 'plv'],
    fmin=(4, 8, 13), fmax=(8, 13, 30),
    faverage=True, indices=None,
)
# con.get_data() shape: (325, 3) for 3 bands
# Then average within/between ROI pairs
```

### Manual PLV (fallback if mne-connectivity unavailable)

```python
from scipy.signal import hilbert, butter, filtfilt

def compute_plv(data, sfreq, fmin, fmax):
    b, a = butter(4, [fmin, fmax], btype='band', fs=sfreq)
    filtered = filtfilt(b, a, data, axis=-1)
    phase = np.angle(hilbert(filtered, axis=-1))
    n_ch = data.shape[1]
    plv = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(i+1, n_ch):
            plv[i,j] = np.abs(np.mean(np.exp(1j*(phase[:,i,:]-phase[:,j,:]))))
            plv[j,i] = plv[i,j]
    return plv
```

---

## Sources

- [EEG-Based Emotion Classification Using Stacking (MDPI Sensors 2022)](https://mdpi.com/1424-8220/22/21/8550)
- [Schizophrenia EEG with stacking ensembles (Preprints 2025)](https://www.preprints.org/manuscript/202602.0513)
- [Pediatric schizophrenia EEG ensembles (Frontiers 2025)](https://doi.org/10.3389/fnhum.2025.1530291)
- [ADHD GCN with PLI+coherence (Frontiers Neurosci 2025)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1561994/full)
- [EEG coherence in ADHD (BMC Medicine 2017)](https://bmcmedicine.biomedcentral.com/articles/10.1186/s12916-017-0805-9)
- [MDD EEG oscillation and asymmetry (Frontiers 2024)](https://doi.org/10.3389/fnins.2024.1362111)
- [EC/EO resting-state in ADHD (Brain Sciences 2020)](https://www.mdpi.com/2076-3425/10/5/272/htm)
- [sklearn StackingClassifier docs](https://scikit-learn.org/1.0/modules/generated/sklearn.ensemble.StackingClassifier.html)
- [mne-connectivity spectral_connectivity_epochs](https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.spectral_connectivity_epochs.html)
- [Stacking overfitting (StackExchange)](https://stats.stackexchange.com/questions/440686/is-this-kind-of-stacked-ensemble-method-prone-to-over-fitting)
- [StackingClassifier CV leakage (StackExchange)](https://stats.stackexchange.com/questions/483888/cross-validation-in-stackingclassifier-scikit-learn)
- [Curse of dimensionality in EEG (Springer 2024)](https://link.springer.com/article/10.1007/s10462-024-10711-8)
- [Depression stacking ensemble (arXiv 2025)](https://arxiv.org/html/2506.14459v1)

---
*v2.0 milestone research | 2026-02-24*
