# Architecture Research

**Domain:** ERP task-state MDD classification pipeline (MODMA dot-probe)
**Researched:** 2026-02-23
**Confidence:** HIGH

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      ERP Data Layer                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  erp_io.py — load .raw, filter, epoch per condition  │   │
│  │  Input: ERP_DIR/*.raw  Output: dict[sub_id → epochs] │   │
│  └──────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                    ERP Feature Layer                        │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │  P300 feats  │  │  N200 feats  │                         │
│  │  250–500ms   │  │  100–250ms   │                         │
│  │  mean/peak/  │  │  mean amp    │                         │
│  │  latency     │  │  per channel │                         │
│  └──────┬───────┘  └──────┬───────┘                         │
│         └─────────────────┘                                 │
│                    ↓ per condition                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  fusion — np.hstack(hcue, fcue, scue feat vectors)   │   │
│  └──────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                 Classification Layer                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  LOSO CV: StandardScaler → PCA(20) → LogisticReg     │   │
│  └──────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                  Evaluation Layer                           │
│  ┌────────────────┐  ┌──────────────────────────────────┐   │
│  │  BA / AUC      │  │  Permutation test (n=1000)        │   │
│  └────────────────┘  └──────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| `run_modma_erp.py` (existing) | Entry point, orchestrates condition loop | erp_io, erp_features, classify, evaluate |
| `erp_io.py` (new) | Load .raw, filter, epoch, return avg ERP per subject | MNE, config |
| `erp_features.py` (new) | Extract P300 + N200 features from avg ERP | erp_io output |
| fusion (inline) | np.hstack per-condition feature vectors | erp_features output |
| `run_loso()` (existing) | LOSO CV with sklearn Pipeline | fusion output |
| `permutation_test_loso()` (new) | Permutation test on LOSO BA | run_loso output |

## Recommended Project Structure

```
D:/eeg/
├── run_modma_erp.py     # existing entry point — extend, do not replace
├── erp_io.py            # NEW: load_erp_epochs(condition) -> dict[sub_id -> (avg_erp, label)]
├── erp_features.py      # NEW: extract_p300_features(), extract_n200_features()
└── erp_evaluate.py      # NEW: permutation_test_loso(X, y, ids, pipe, n_perm)
```

Fusion and subject alignment stay inline in `run_modma_erp.py` — they are ~15 lines, no separate file needed.

### Structure Rationale

- `erp_io.py`: Isolates slow MNE I/O. Enables .npz caching and independent testing.
- `erp_features.py`: All ERP component extractors with standard signature `(avg_erp, times) -> (feat, names)`. Mirrors existing `features.py` pattern.
- `erp_evaluate.py`: Permutation test is expensive and reusable. Mirrors existing `evaluate.py` pattern from resting-state pipeline.
- `run_modma_erp.py`: Stays as thin orchestrator — reads epochs, calls extractors, fuses, classifies, evaluates.

## Architectural Patterns

### Pattern 1: Per-Component Feature Extractor

**What:** Each ERP component is a function `(avg_erp, times) -> (feat, names)`. Fusion is `np.concatenate`.
**When to use:** Always. Enables ablation (P300-only vs P300+N200 vs fused conditions).
**Trade-offs:** Slight indirection, but toggling components requires no logic changes.

```python
def extract_p300_features(avg_erp, times):
    # avg_erp: (128, n_times), times: (n_times,)
    mask = (times >= 0.25) & (times <= 0.50)
    win = avg_erp[:, mask]
    mean_amp = win.mean(axis=1)               # (128,)
    peak_amp = win.max(axis=1)                # (128,)
    peak_lat = times[mask][win.argmax(axis=1)]  # (128,)
    feat = np.concatenate([mean_amp, peak_amp, peak_lat])
    names = ([f"p300_mean_ch{i}" for i in range(128)] +
             [f"p300_peak_ch{i}" for i in range(128)] +
             [f"p300_lat_ch{i}"  for i in range(128)])
    return feat, names

def extract_n200_features(avg_erp, times):
    mask = (times >= 0.10) & (times <= 0.25)
    feat = avg_erp[:, mask].mean(axis=1)     # (128,)
    names = [f"n200_mean_ch{i}" for i in range(128)]
    return feat, names
```

### Pattern 2: Condition-Loop with Subject-Aligned Fusion

**What:** Run feature extraction per condition, intersect subject sets, then hstack.
**When to use:** Multi-condition paradigms (hcue/fcue/scue).
**Trade-offs:** Reduces n_subjects to those present in all conditions; wider feature vector captures condition-specific markers.

```python
cond_data = {}  # cond -> {sub_id: (feat_vec, label)}
for cond in ["hcue", "fcue", "scue"]:
    cond_data[cond] = load_erp_features(cond)  # returns dict keyed by sub_id

common_ids = sorted(
    set(cond_data["hcue"]) & set(cond_data["fcue"]) & set(cond_data["scue"])
)
X_fused = np.array([
    np.concatenate([cond_data[c][sid][0] for c in ["hcue", "fcue", "scue"]])
    for sid in common_ids
])
y = np.array([cond_data["hcue"][sid][1] for sid in common_ids])
```

### Pattern 3: Permutation Test Outside LOSO

**What:** Run LOSO once for observed BA. Shuffle labels 1000x, re-run LOSO each time, compute p-value.
**When to use:** Always for statistical validation.
**Trade-offs:** 1000x LOSO cost. Acceptable at n=52 with simple LR pipeline (~seconds per fold).

```python
def permutation_test_loso(X, y, ids, pipe, n_perm=1000, seed=42):
    rng = np.random.default_rng(seed)
    obs_ba, _ = run_loso(X, y, ids, pipe)
    null_bas = [run_loso(X, rng.permutation(y), ids, pipe)[0]
                for _ in range(n_perm)]
    p = (np.sum(np.array(null_bas) >= obs_ba) + 1) / (n_perm + 1)
    return obs_ba, p, np.array(null_bas)
```

## Data Flow

```
[ERP_DIR/*.raw]
    ↓
erp_io.py: load_erp_epochs(condition)
    per subject: filter(0.5-40Hz) → epoch(tmin=-0.1, tmax=0.5)
                 baseline(-0.1,0) → reject(eeg=150µV) → require ≥10 trials
    ↓ dict[sub_id → (avg_erp: 128×n_times, times, label)]
erp_features.py: extract_p300_features + extract_n200_features
    P300: mean(128) + peak(128) + latency(128) = 384 dims
    N200: mean(128) = 128 dims
    per-condition feat vector: 512 dims
    ↓ X_cond: (n_subjects, 512)
fusion (inline): intersect sub_ids, np.hstack([X_hcue, X_fcue, X_scue])
    ↓ X_fused: (n_subjects_common, 1536)
run_loso(): StandardScaler → PCA(20) → LogisticRegression LOSO
    ↓ (y_true, y_prob) per fold → BA, AUC
permutation_test_loso(): 1000x shuffle y → re-run LOSO → p-value
    ↓ BA, AUC, p-value
```

### Key Data Flows

1. **Load:** `.raw` → MNE EGI reader → filter → epoch per condition → avg across trials → (128, n_times) per subject
2. **Feature:** avg_erp → P300 window (250-500ms) → mean/peak/latency per channel; N200 window (100-250ms) → mean per channel
3. **Fusion:** align subjects across conditions by sub_id → hstack → (n_subjects_common, 1536)
4. **Classify:** LOSO with existing `run_loso()` — no changes needed
5. **Evaluate:** observed BA → 1000 label permutations → p-value

## Integration Points with Existing run_modma_erp.py

### What Stays (Unchanged)

| Element | Location | Notes |
|---------|----------|-------|
| `load_erp_features()` | lines 33-94 | Keep for single-condition baseline; extend to return richer features |
| `run_loso()` | lines 97-135 | Reuse directly — no changes needed |
| Config constants | top of file | Add N200_WIN; extend P300 to include peak/latency |
| `__main__` condition loop | lines 138-143 | Extend to collect per-condition data for fusion |

### What Changes (Modified)

| Element | Change | Reason |
|---------|--------|--------|
| `load_erp_features()` | Add P300 peak/latency + N200 mean; return sub_id dict | Richer features; fusion needs sub_id alignment |
| `__main__` | After per-condition loop, add fusion + fused LOSO + permutation test | Multi-condition experiment |
| `CONDITION` global | Pass as parameter to `load_erp_features(condition)` | Avoid fragile global mutation |

### What Is New (Added)

| Element | Where | Purpose |
|---------|-------|---------|
| `N200_WIN = (0.10, 0.25)` | config block | N200 time window |
| `extract_p300_features(avg_erp, times)` | `erp_features.py` | Modular P300 extractor |
| `extract_n200_features(avg_erp, times)` | `erp_features.py` | N200 extractor |
| `permutation_test_loso(X, y, ids, pipe, n_perm)` | `erp_evaluate.py` | Statistical validation |
| Fusion + alignment block | `run_modma_erp.py __main__` | Intersect subjects, hstack conditions |

## Build Order

Dependencies dictate this sequence:

1. **Extend `load_erp_features(condition)`** — add N200_WIN, return richer P300 features (peak, latency) + N200 mean. Return dict keyed by sub_id. Verify per-condition BA holds or improves.

2. **Add `permutation_test_loso()`** — depends only on existing `run_loso()`. Run on single-condition results first to validate p-value logic before fusion.

3. **Add fusion block in `__main__`** — collect per-condition dicts, intersect sub_ids, hstack, call `run_loso()` + `permutation_test_loso()`. Depends on steps 1 and 2.

4. **Extract to `erp_features.py` / `erp_evaluate.py`** — only if `run_modma_erp.py` exceeds ~300 lines. Not required for correctness.

## Anti-Patterns

### Anti-Pattern 1: Re-epoching Inside Permutation Loop

**What people do:** Call `load_erp_features()` (MNE I/O + epoching) inside each permutation iteration.
**Why it's wrong:** ~5s/subject × 52 subjects × 1000 permutations = ~72 hours.
**Do this instead:** Load features once. Permutation loop only shuffles `y` and re-runs `run_loso()`.

### Anti-Pattern 2: Subject Misalignment in Fusion

**What people do:** Assume all subjects appear in all three conditions; hstack without checking.
**Why it's wrong:** Some subjects may have <10 trials in one condition and get dropped. Misaligned rows corrupt labels silently.
**Do this instead:** Build a dict keyed by sub_id per condition, intersect keys, then align rows explicitly.

### Anti-Pattern 3: Global CONDITION Variable Mutation

**What people do:** Rely on `CONDITION = cond` module-level mutation in the loop (existing code pattern).
**Why it's wrong:** Fragile — any function reading the global gets wrong condition if called out of order.
**Do this instead:** Pass `condition` as a parameter to `load_erp_features(condition)`.

### Anti-Pattern 4: PCA Fitted on Full Dataset Before LOSO

**What people do:** Fit PCA on all subjects' features, then run LOSO on transformed data.
**Why it's wrong:** PCA sees test subject's data during fitting — data leakage inflates BA.
**Do this instead:** Keep PCA inside the sklearn Pipeline so it fits only on training folds (existing `run_loso()` already does this correctly).

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 52 subjects (current) | In-memory numpy, sequential LOSO, 1000 permutations in minutes |
| 200 subjects | Cache avg_erp arrays to .npz; joblib on permutation loop |
| 1000+ subjects | Out of scope for this dataset |

### Scaling Priorities

1. **First bottleneck:** MNE .raw loading (~3-5s/subject × 52 × 3 conditions). Fix with .npz cache of avg_erp arrays keyed by (sub_id, condition).
2. **Second bottleneck:** Permutation test (1000 × LOSO). Fix with `joblib.Parallel` on permutation loop.

## Sources

- `run_modma_erp.py` — direct code analysis (HIGH confidence)
- `.planning/codebase/ARCHITECTURE.md` — resting-state pipeline patterns (HIGH confidence)
- `.planning/research/ARCHITECTURE.md` (prior) — resting-state architecture patterns (HIGH confidence)
- MNE-Python Epochs API — epoching, baseline, rejection (HIGH confidence)
- Li et al. 2018 CMPB — P300 mean amplitude baseline referenced in existing script docstring (MEDIUM confidence)

---
*Architecture research for: ERP task-state MDD classification pipeline*
*Researched: 2026-02-23*
