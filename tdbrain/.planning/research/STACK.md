# Technology Stack: v2.0 Additions

**Project:** TDBRAIN MDD vs ADHD Diagnostic
**Researched:** 2026-02-24
**Scope:** Only NEW dependencies for stacking ensemble + functional connectivity
**Confidence:** HIGH (verified against installed environment + official docs)

## Existing Stack (DO NOT CHANGE)

| Technology | Version | Purpose |
|------------|---------|---------|
| scikit-learn | 1.8.0 | ML pipeline, classifiers, CV |
| MNE-Python | 1.11.0 | EEG preprocessing, epochs, PSD |
| imbalanced-learn | 0.14.1 | SMOTE, ImbPipeline |
| XGBoost | 3.2.0 | Gradient boosting classifier |
| NumPy | 2.4.2 | Array operations |
| scipy | 1.17.0 | Signal processing |

## New Dependencies

### Required: mne-connectivity 0.7.0

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| mne-connectivity | 0.7.0 | Coherence + PLV | Only MNE-ecosystem lib for spectral_connectivity_epochs. Accepts MNE Epochs directly, supports per-band averaging. |

**Compatibility:** mne >= 1.6 (have 1.11.0), Python >= 3.9, numpy >= 1.21, scipy >= 1.4.0. All satisfied.

### NOT Required: StackingClassifier

sklearn.ensemble.StackingClassifier available since sklearn 0.22. Our 1.8.0 has it. Zero new packages.

## Installation

```bash
pip install mne-connectivity==0.7.0
```

## Integration Points

### 1. Functional Connectivity (mne-connectivity)

```python
from mne_connectivity import spectral_connectivity_epochs

con = spectral_connectivity_epochs(
    epochs,
    method=['coh', 'plv'],
    mode='multitaper',
    fmin=(4, 8, 13),
    fmax=(8, 13, 30),
    faverage=True,
    n_jobs=2,
)
# Returns: list[SpectralConnectivity], shape (n_connections, n_bands)
```

Use coh + plv first. Add imcoh/wpli only if volume conduction is confirmed.

### 2. Stacking Ensemble (sklearn -- already installed)

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

stacker = StackingClassifier(
    estimators=[
        ('svm', svm_pipeline),
        ('rf', rf_pipeline),
        ('xgb', xgb_pipeline),
    ],
    final_estimator=LogisticRegression(
        class_weight='balanced', random_state=42, max_iter=1000
    ),
    cv=5,
    stack_method='predict_proba',
    passthrough=False,
    n_jobs=2,
)
```

## Critical Integration Details

### SMOTE + StackingClassifier

SMOTE stays INSIDE each base ImbPipeline. StackingClassifier accepts any sklearn-API-compatible estimator, so ImbPipeline works directly as a base estimator. Internal CV generates OOF predictions -- SMOTE runs only on training folds. No data leakage.

Do NOT apply SMOTE at the stacking level.

### Nested CV Structure

```
Outer CV (StratifiedGroupKFold, 5-fold, group=subject)
  -> StackingClassifier.fit(X_train, y_train)
       -> Inner CV (StratifiedKFold, 5-fold) per base estimator
            -> ImbPipeline (Scaler -> SMOTE -> SelectKBest -> Clf)
       -> LogisticRegression meta-learner on OOF predictions
  -> StackingClassifier.predict_proba(X_test)
```

Inner CV uses StratifiedKFold (no group awareness). Acceptable because subject leakage is already prevented by the outer StratifiedGroupKFold.

### Connectivity Feature Dimensions

26 channels, faverage=True, 3 bands (theta/alpha/beta):
- Unique pairs: 26 * 25 / 2 = 325
- 2 methods x 3 bands = 1,950 features per condition
- EC/EO fusion: 3,900 connectivity features
- Combined with existing 992-dim spectral: ~4,892 total

Consider increasing SelectKBest k from 50 to 80-100.

### Computation Cost

~2-5 sec/subject for 26 channels. 477 subjects x 2 conditions = ~30-80 min total. Cache to disk using existing .npz pattern.

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Connectivity lib | mne-connectivity | scipy.signal.coherence | No MNE Epochs integration, no PLV, manual band handling |
| Connectivity lib | mne-connectivity | NeuroDSP / pactools | Not designed for channel-pair connectivity |
| Meta-learner | LogisticRegression | XGBoost meta-learner | Overfitting risk with only 3 probability inputs |
| Stacking impl | sklearn StackingClassifier | Manual OOF stacking | Reinventing the wheel; sklearn handles CV correctly |
| FC measures | coh + plv first | imcoh / wpli | Try standard measures first; switch only if volume conduction confirmed |

## What NOT to Add

| Library | Why Not |
|---------|---------|
| mlxtend | sklearn StackingClassifier already available |
| nilearn | Designed for fMRI, not EEG |
| tensorpac | Phase-amplitude coupling, not channel-pair FC |
| hyperopt / optuna | Nested CV sufficient at this scale |

## Sources

- [mne-connectivity 0.7.0 API](https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.spectral_connectivity_epochs.html) -- HIGH
- [mne-connectivity install docs](https://mne.tools/mne-connectivity/stable/install.html) -- HIGH
- [sklearn StackingClassifier docs](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html) -- HIGH
- [mne-connectivity PyPI](https://pypi.org/project/mne-connectivity/) -- HIGH
