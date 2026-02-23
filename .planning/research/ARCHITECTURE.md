# Architecture Research

**Domain:** ERP task-state MDD classification pipeline (MODMA dot-probe) — v3.0 ERP Deep Analysis
**Researched:** 2026-02-23
**Confidence:** HIGH

## Current State (v2.0 Baseline — Built and Validated)

`run_modma_erp.py` is the single-file pipeline. All v2.0 phases (7-9) are complete.

```
run_modma_erp.py
├── load_erp_features()          # 128-dim P300 mean per channel (hcue)
├── load_erp_features_dict(cond) # per-condition dict[sub_id → (feat, label)]
├── fuse_conditions(cond_dicts)  # intersect sub_ids, hstack features
├── _loso_ba(X, y)               # silent LOSO for permutation loop
├── permutation_test_loso(X, y)  # 1000-perm subject-level shuffle
└── run_loso(X, y, ids)          # LOSO CV: StandardScaler→PCA(20)→LogisticReg
```

Validated results:
- hcue single-condition: BA=0.670, p=0.021 (primary finding)
- 3-condition fusion: BA=0.521, p=0.412 (negative)
- scue-hcue contrast: BA=0.521, p=0.384 (negative)

## v3.0 Target Features

Four new analysis directions, all building on the existing `run_modma_erp.py` architecture:

1. LPP component extraction (500–800ms mean amplitude per channel)
2. Electrode selection — identify top-K channels by discriminative power
3. Time-window t-test — group-level MDD vs HC t-test across time points
4. Feature importance — SVM weights or permutation importance on LOSO folds

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      ERP Data Layer                         │
│  load_erp_features() — .raw → filter → epoch → avg_erp     │
│  UNCHANGED from v2.0                                        │
├─────────────────────────────────────────────────────────────┤
│                    Feature Extraction Layer                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  P300 feats  │  │  N200 feats  │  │  LPP feats   │       │
│  │  250–500ms   │  │  100–250ms   │  │  500–800ms   │       │
│  │  mean amp    │  │  mean amp    │  │  mean amp    │       │
│  │  (128,)      │  │  (128,)      │  │  (128,)      │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         └─────────────────┴─────────────────┘               │
│                    np.concatenate → (384,)                  │
├─────────────────────────────────────────────────────────────┤
│                  Electrode Selection Layer (NEW)            │
│  select_electrodes(X, y, k=10)                              │
│  rank channels by t-stat → return X[:, top_k_indices]       │
├─────────────────────────────────────────────────────────────┤
│                 Classification Layer (UNCHANGED)            │
│  run_loso(): StandardScaler → PCA(20) → LogisticRegression  │
├─────────────────────────────────────────────────────────────┤
│                  Analysis Layer (NEW)                       │
│  ┌────────────────────┐  ┌──────────────────────────────┐   │
│  │  time_window_ttest │  │  feature_importance()         │   │
│  │  MDD vs HC t-stat  │  │  SVM weights per feature dim  │   │
│  │  per time point    │  │                               │   │
│  └────────────────────┘  └──────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                  Evaluation Layer (UNCHANGED)               │
│  permutation_test_loso(): 1000-perm, BA, p-value            │
└─────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | New/Modified/Unchanged | Responsibility |
|-----------|------------------------|----------------|
| `load_erp_features()` | Modified | Add LPP_WIN; extend feat to 384-dim |
| `run_loso()` | Unchanged | LOSO CV — do not touch |
| `permutation_test_loso()` | Unchanged | Statistical validation |
| `select_electrodes(X, y, k)` | New | Rank 128 channels by t-stat, return top-k indices |
| `time_window_ttest(avg_erps, y, times)` | New | t-stat per time point, identify discriminative windows |
| `feature_importance(X, y, feat_names)` | New | SVM coef_ weights ranked by absolute value |

## Integration Points

### What Stays Unchanged

| Element | Why |
|---------|-----|
| `run_loso()` | Locked since phase 7. PCA inside pipeline prevents leakage. |
| `permutation_test_loso()` | Validated at p=0.021. No changes needed. |
| Config: TMIN, TMAX, BASELINE, FMIN, FMAX | Validated preprocessing parameters. |
| Epoch rejection: `eeg=150e-6`, min 10 trials | Validated QC thresholds. |

### What Changes (Modified)

| Element | Change | Reason |
|---------|--------|--------|
| `load_erp_features()` | Add LPP_WIN=(0.50,0.80); extend feat 128→384 dims | LPP is the third ERP component target |
| Config block | Add `LPP_WIN = (0.50, 0.80)` | New time window constant |
| `__main__` | Add calls to new analysis functions after LOSO | Drive electrode selection, t-test, importance |

### What Is New (Added)

| Element | Location | Purpose |
|---------|----------|---------|
| `LPP_WIN = (0.50, 0.80)` | config block | LPP time window |
| `select_electrodes(X, y, k)` | `run_modma_erp.py` | Rank channels by univariate t-stat |
| `time_window_ttest(avg_erps, y, times)` | `run_modma_erp.py` | MDD vs HC t-stat at each time point |
| `feature_importance(X, y, feat_names)` | `run_modma_erp.py` | LinearSVC coef_ weights, ranked |

## Data Flow

```
[ERP_DIR/*.raw]
    ↓
load_erp_features()  — unchanged I/O path
    per subject: filter → epoch(hcue) → avg_erp(128, n_times)
    P300 window (250-500ms): mean per channel → (128,)
    N200 window (100-250ms): mean per channel → (128,)
    LPP window (500-800ms):  mean per channel → (128,)  ← NEW
    feat = np.concatenate([p300_mean, n200_mean, lpp_mean])  → (384,)
    ↓ X: (52, 384), y: (52,), ids: (52,)
    also return avg_erps: (52, 128, n_times)  ← needed for t-test

[Branch A: Classification with electrode selection]
    select_electrodes(X, y, k=10)
        t-stat per channel → top-k indices
        X_reduced: (52, k*3)
    run_loso(X_reduced, y, ids)
    permutation_test_loso(X_reduced, y)
        ↓ BA, p-value

[Branch B: Time-Window Analysis]
    time_window_ttest(avg_erps, y, times)
        t-stat at each time point: (n_times,)
        identify peak discriminative window
        ↓ t_stats array, significant time points

[Branch C: Feature Importance]
    feature_importance(X, y, feat_names)
        fit LinearSVC on full X (post-hoc only)
        coef_ → (384,) weight vector
        top-20 features by |weight|
        ↓ ranked feature names + weights
```

## Architectural Patterns

### Pattern 1: Extend load_erp_features() for LPP

**What:** Add LPP_WIN constant; append lpp_mean to the concatenation. Minimal diff.
**When to use:** Always — LPP is a table-stakes ERP component for emotional processing tasks.
**Trade-offs:** 384-dim vs 128-dim. PCA(20) inside run_loso() absorbs the increase — no pipeline change needed.

```python
LPP_WIN = (0.50, 0.80)  # add to config block

# inside load_erp_features(), replace feat extraction:
p300_mask = (times >= P300_WIN[0]) & (times <= P300_WIN[1])
n200_mask = (times >= N200_WIN[0]) & (times <= N200_WIN[1])
lpp_mask  = (times >= LPP_WIN[0])  & (times <= LPP_WIN[1])
feat = np.concatenate([
    avg_erp[:, p300_mask].mean(axis=1),  # (128,)
    avg_erp[:, n200_mask].mean(axis=1),  # (128,)
    avg_erp[:, lpp_mask].mean(axis=1),   # (128,)
])  # (384,)
```

### Pattern 2: Electrode Selection via Univariate t-stat

**What:** For each of the 128 channels, compute independent-samples t-stat between MDD and HC. Select top-k by |t|.
**When to use:** Before LOSO when feature dimensionality is high relative to n_subjects (52 subjects, 384 dims).
**Trade-offs:** Univariate selection ignores channel interactions. For interpretability only — if used for BA claims, must be inside LOSO folds to avoid leakage.

```python
from scipy.stats import ttest_ind

def select_electrodes(X, y, k=10):
    # X: (n_subjects, n_features), features ordered as [p300_ch0..127, n200_ch0..127, lpp_ch0..127]
    t_stats = np.array([
        ttest_ind(X[y==1, j], X[y==0, j]).statistic
        for j in range(X.shape[1])
    ])
    top_idx = np.argsort(np.abs(t_stats))[-k:]
    return top_idx, t_stats
```

### Pattern 3: Time-Window t-test

**What:** Compute MDD vs HC t-stat at every time point. Identifies which milliseconds are most discriminative.
**When to use:** Exploratory analysis to validate P300/N200/LPP window choices.
**Trade-offs:** Multiple comparisons across ~150 time points. Report with FDR correction or explicit uncorrected caveat.

```python
def time_window_ttest(avg_erps, y, times):
    # avg_erps: (n_subjects, 128, n_times)
    # collapse to mean across all channels (or parietal subset)
    grand = avg_erps.mean(axis=1)  # (n_subjects, n_times)
    t_stats = np.array([
        ttest_ind(grand[y==1, t], grand[y==0, t]).statistic
        for t in range(len(times))
    ])
    return t_stats  # (n_times,)
```

### Pattern 4: Feature Importance via SVM Weights

**What:** Fit LinearSVC on full dataset, extract coef_ as feature weights. Rank by absolute value.
**When to use:** Post-hoc interpretation after LOSO confirms significance. Not for selecting features used in the reported BA.
**Trade-offs:** Fitted on full data — valid for interpretation, not for performance claims.

```python
from sklearn.svm import LinearSVC

def feature_importance(X, y, feat_names):
    X_s = StandardScaler().fit_transform(X)
    svm = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)
    svm.fit(X_s, y)
    ranked = sorted(zip(feat_names, svm.coef_[0]),
                    key=lambda x: abs(x[1]), reverse=True)
    return ranked
```

## Build Order

Dependencies dictate this sequence:

1. **Extend `load_erp_features()` with LPP** — add `LPP_WIN`, extend feat to 384-dim. Run hcue LOSO to confirm BA does not regress below 0.670. Foundation for all other v3.0 work.

2. **Add `time_window_ttest()`** — depends only on avg_erp arrays from step 1. Refactor load loop to also return avg_erps array. No classification dependency.

3. **Add `select_electrodes()`** — depends on X from step 1. Run LOSO on electrode-reduced X to compare BA vs full 384-dim.

4. **Add `feature_importance()`** — depends on X from step 1 and confirmed LOSO significance. Last because it is post-hoc interpretation, not a classification step.

## Anti-Patterns

### Anti-Pattern 1: Electrode Selection Outside LOSO Folds (for BA claims)

**What people do:** Select top-k electrodes on full dataset, then report LOSO BA on the reduced set.
**Why it's wrong:** Test subject's data influenced electrode selection — data leakage inflates BA.
**Do this instead:** For BA claims, wrap electrode selection inside LOSO folds. For interpretability only, full-dataset selection is acceptable with explicit caveat.

### Anti-Pattern 2: Re-epoching for Time-Window t-test

**What people do:** Call `load_erp_features()` again inside `time_window_ttest()`, re-reading all .raw files.
**Why it's wrong:** ~3-5s/subject × 52 subjects = unnecessary 3-4 minute wait.
**Do this instead:** Refactor `load_erp_features()` to optionally return `avg_erps` array (52, 128, n_times) alongside X. Pass it to `time_window_ttest()` directly.

### Anti-Pattern 3: Using 640-dim Features (Phase 7 Regression)

**What people do:** Re-apply the phase 7 640-dim extraction (P300 mean+peak+latency+AUC + N200 mean).
**Why it's wrong:** Phase 7 showed 640-dim regressed BA from 0.670 to 0.554. Peak, latency, and AUC features add noise, not signal.
**Do this instead:** Use mean amplitude only per component (128-dim each). 384-dim = P300_mean + N200_mean + LPP_mean.

### Anti-Pattern 4: Modifying run_loso() or permutation_test_loso()

**What people do:** Adjust the pipeline or permutation logic to accommodate new features.
**Why it's wrong:** These functions are validated and locked. Changes risk invalidating the p=0.021 baseline.
**Do this instead:** Pass different X matrices to the unchanged functions. All variation is in feature extraction.

## Sources

- `run_modma_erp.py` — direct code analysis (HIGH confidence)
- `.planning/phases/07-p300-feature-enrichment-n200/07-01-SUMMARY.md` — 640-dim regression BA=0.554 (HIGH confidence)
- `.planning/phases/08-permutation-test-validation/08-CONTEXT.md` — permutation test pattern (HIGH confidence)
- `.planning/phases/09-multi-condition-fusion-contrast-features/09-01-SUMMARY.md` — fusion negative result (HIGH confidence)
- `.planning/PROJECT.md` — v3.0 target features (HIGH confidence)

---
*Architecture research for: ERP task-state MDD classification pipeline — v3.0 ERP Deep Analysis*
*Researched: 2026-02-23*
