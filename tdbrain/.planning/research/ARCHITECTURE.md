# Architecture Research: v2.0 Stacking + Functional Connectivity

**Domain:** EEG classification pipeline — stacking ensemble + connectivity features
**Researched:** 2026-02-24
**Confidence:** HIGH (code inspection) / MEDIUM (StackingClassifier groups handling)

## Critical Finding: Do NOT Use sklearn StackingClassifier

**Problem:** `StackingClassifier` internally calls `cross_val_predict` to generate
out-of-fold (OOF) meta-features. This internal CV does NOT propagate `groups` to
the splitter. `StratifiedGroupKFold.split()` requires `groups` but
`cross_val_predict` inside `StackingClassifier` does not forward them — causing
either a crash or silent group leakage.

**Workaround via metadata routing** (`sklearn.set_config(enable_metadata_routing=True)`)
exists in sklearn 1.8.0 but is experimental and poorly documented for stacking.

**Recommendation:** Manual stacking. The existing `classifier.py` already has a
manual outer CV loop — extending it for OOF meta-features is straightforward and
guarantees group-safe splits.

**Confidence:** MEDIUM — sklearn docs + StackOverflow reports confirm this issue.

## Existing Architecture (v1.3)

```
main.py (orchestrator)
  -> data_loader.py        : load_subjects(), load_raw()
  -> preprocessor.py       : preprocess() -> Epochs
  -> feature_extractor.py  : extract_features(epochs) -> np.ndarray (496-dim)
  -> classifier.py         : classify(X, y, groups) -> results dict
```

Current ensemble: soft-vote averaging of SVM/RF/XGB probabilities (lines 110-136
of classifier.py). No meta-learner — just `np.mean(fold_probas, axis=0)`.

## v2.0 New & Modified Components

| Component | File | Status | Purpose |
|-----------|------|--------|---------|
| Connectivity extractor | `connectivity_extractor.py` | NEW | Compute coherence/PLV from Epochs |
| Stacking meta-learner | `classifier.py` | MODIFIED | Replace soft-vote with trained LR meta-learner |
| Feature orchestration | `main.py` | MODIFIED | Call connectivity extractor, concat features |
| Connectivity config | `config.py` | MODIFIED | Add connectivity methods/pairs settings |
| data_loader.py | -- | UNCHANGED | Already parameterized on condition |
| preprocessor.py | -- | UNCHANGED | Epochs output suitable for connectivity |
| feature_extractor.py | -- | UNCHANGED | Spectral features stay 496-dim |

## Component 1: Functional Connectivity Extractor

### Design: Separate Module

Connectivity requires `mne.Epochs` objects (not PSD arrays), so it cannot live
inside `extract_features()`. New `connectivity_extractor.py` receives the same
Epochs object and returns a flat feature vector.

### API

```python
# connectivity_extractor.py
def extract_connectivity(epochs, methods=None, freq_bands=None):
    """Return 1D np.ndarray of shape (n_methods * n_pairs * n_bands,)."""
```

### mne-connectivity Usage

**Needs install:** `pip install mne-connectivity` (verified NOT installed)

```python
from mne_connectivity import spectral_connectivity_epochs

n_ch = len(epochs.ch_names)  # 26
seeds, targets = np.triu_indices(n_ch, k=1)  # upper triangle, 325 pairs

con = spectral_connectivity_epochs(
    epochs, method=["coh", "plv"],
    indices=(seeds, targets),
    fmin=[1, 4, 8, 13, 30], fmax=[4, 8, 13, 30, 40],
    faverage=True, verbose=False,
)
# Returns list of SpectralConnectivity; each .get_data() -> (325, 5)
```

### Feature Dimensionality

| Component | Count |
|-----------|-------|
| Channel pairs (26 choose 2) | 325 |
| Frequency bands | 5 |
| Methods (coh, PLV) | 2 |
| **Total connectivity features** | **3250** |
| Existing spectral (EO+EC) | 992 |
| **Combined total** | **4242** |

**Risk:** 4242 features / 477 subjects = 9:1 ratio. SelectKBest k=50 inside
pipeline handles this, but consider fewer pairs as fallback.

**Fallback:** Pre-select ~20 clinically relevant pairs (F3-F4, F7-F8, C3-C4,
P3-P4, O1-O2, frontal-parietal). Reduces to ~200 features.

## Component 2: Manual Stacking Meta-Learner

### Why Manual (Not StackingClassifier)

1. **Groups:** StratifiedGroupKFold requires `groups` at every split. StackingClassifier's internal `cross_val_predict` does not forward them.
2. **SMOTE:** Each base estimator uses ImbPipeline with SMOTE. Manual control ensures SMOTE runs inside each fold.
3. **Existing pattern:** classifier.py already has a manual outer CV loop — minimal code change.

### Stacking Data Flow

```
OUTER CV (5-fold StratifiedGroupKFold):
  For each outer fold (train_outer, test_outer):

    STACKING CV (3-fold on train_outer):
      For each stacking fold (train_stack, val_stack):
        For each base model (SVM, RF, XGB):
          GridSearchCV on train_stack -> predict_proba on val_stack
      -> OOF probabilities for all of train_outer

    REFIT on full train_outer:
      For each base model:
        GridSearchCV on train_outer -> predict_proba on test_outer

    META-LEARNER:
      fit on OOF probabilities (n_train, 3) + y[train_outer]
      predict on test probabilities (n_test, 3)

  -> Concatenate outer fold predictions -> final AUC
```

### Meta-Learner Choice

**Use LogisticRegression(C=1.0)** — only 3 input features (one prob per base
model), so no overfitting risk. Learns optimal weighting unlike equal-weight
soft vote. No SMOTE needed at meta level.

**passthrough=False:** Do NOT pass original features to meta-learner.

### CV Nesting Depth

```
Level 1: Outer CV (5-fold) -- unbiased performance estimate
  Level 2: Stacking CV (3-fold on train_outer) -- OOF meta-features
    Level 3: Inner CV (3-fold on train_stack) -- hyperparameter tuning
```

With 477 subjects: outer train ~382, stacking train ~254, inner train ~170.
170 subjects is adequate for GridSearchCV with current param grids.

### Runtime Impact

Current: 3 models x 5 outer folds x 1 GridSearchCV = 15 GridSearchCV calls
Stacking: 3 models x 5 outer folds x (3 stacking folds + 1 refit) = 60 calls

**4x increase** in classifier runtime. Permutation test multiplies further.
Mitigation: reduce permutations to 500, or run perm test only on stacking ensemble.

## Integration Points

### 1. preprocessor.py -> connectivity_extractor.py

`preprocess()` returns `(epochs, stats)`. The Epochs object is already suitable
for `spectral_connectivity_epochs` — same object, no conversion needed.

```python
epochs, stats = preprocess(raw)
spectral_feats = extract_features(epochs)        # existing
connect_feats = extract_connectivity(epochs)      # new
subject_vec = np.concatenate([spectral_feats, connect_feats])
```

### 2. main.py orchestration change

main.py currently stores `feats[cond][sid] = (extract_features(epochs), label)`.
Change to concatenate connectivity features:

```python
feats[cond][sid] = (
    np.concatenate([extract_features(epochs), extract_connectivity(epochs)]),
    row["indication"]
)
```

No other changes to fusion logic — it already concatenates EO+EC vectors.

### 3. classifier.py stacking integration

Replace soft-vote ensemble block (lines 110-136) with manual stacking loop.
Keep individual model results (lines 64-108) unchanged for comparison.
Function signature stays: `classify(X, y, groups) -> dict`.
Add `"stacking"` key to results dict alongside existing model keys.

### 4. config.py additions

```python
CONNECTIVITY_METHODS = ["coh", "plv"]
CONNECTIVITY_PAIRS = "all"  # or list of (seed, target) tuples
```

## Anti-Patterns to Avoid

### 1. Using StackingClassifier with StratifiedGroupKFold
Groups not propagated to internal cross_val_predict. Use manual stacking loop.

### 2. SMOTE at the meta-learner level
Meta-features are 3-dimensional. SMOTE on 3 features creates near-duplicate
synthetic points. Keep SMOTE only inside base model pipelines.

### 3. Connectivity computed on concatenated EO+EC epochs
EO and EC are different cognitive states. Compute connectivity separately per
condition, then concatenate feature vectors (same pattern as spectral features).

### 4. passthrough=True in stacking
Meta-learner receives 3 probs + ~7492 raw features, ignores base model
predictions and overfits to raw features. Meta-learner sees only probabilities.

### 5. Feature selection outside the pipeline
SelectKBest fitted on full X before CV -> label leakage. Keep inside ImbPipeline.

## Suggested Build Order

| Step | File | What | Dependency | Risk |
|------|------|------|------------|------|
| 1 | config.py | Add CONNECTIVITY_METHODS, CONNECTIVITY_PAIRS | None | Low |
| 2 | connectivity_extractor.py | NEW: extract_connectivity(epochs) | pip install mne-connectivity | Medium |
| 3 | main.py | Integrate connectivity into feature loop | Steps 1-2 | Low |
| 4 | (validate) | Run pipeline, verify feature vector shape | Step 3 | -- |
| 5 | classifier.py | Add manual stacking, replace soft-vote | None (parallel to 1-3) | Medium |
| 6 | (validate) | Run full pipeline, compare stacking vs soft-vote AUC | Steps 3-5 | -- |

Steps 2 and 5 are independent and can be developed in parallel.
Step 5 can be tested with existing 992-dim features before connectivity is ready.

## Scaling Considerations

| Concern | Current (v1.3) | v2.0 Impact | Mitigation |
|---------|----------------|-------------|------------|
| Feature dim | 992 | ~7492 | SelectKBest k=50 inside pipeline |
| Classifier runtime | 15 GridSearchCV | 60 GridSearchCV (4x) | Reduce permutations to 500 |
| Connectivity compute | N/A | ~30s per subject | Cache to disk; parallelize |
| Memory | ~4MB feature matrix | ~30MB feature matrix | Still fits in RAM |
| Permutation test | 1000 x 15 GS | 1000 x 60 GS | Run perm only on stacking |

## Sources

- Direct code inspection: classifier.py, main.py, feature_extractor.py, config.py, preprocessor.py (HIGH confidence)
- sklearn StackingClassifier docs: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html (HIGH)
- StackOverflow StratifiedGroupKFold + StackingClassifier: https://stackoverflow.com/questions/78948520 (MEDIUM)
- mne-connectivity spectral_connectivity_epochs: https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.spectral_connectivity_epochs.html (HIGH)
- sklearn common pitfalls: https://scikit-learn.org/stable/common_pitfalls.html (HIGH)

---
*Architecture research for: TDBRAIN EEG MDD vs ADHD pipeline (v2.0 stacking + connectivity)*
*Researched: 2026-02-24*
