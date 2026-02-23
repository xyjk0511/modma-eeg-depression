# Phase 11: Time-Window Analysis - Research

**Researched:** 2026-02-23
**Domain:** ERP time-series statistical analysis + ablation classification
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### t-test data source
- hcue condition only (consistent with Phase 8)
- t-test on raw ERP time series, per time point
- 5 parietal channels (Pz/P3/P4/Cz/CPz) averaged, then t-test
- FDR correction (BH), mark corrected p<0.05 significant time points

#### Visualization output
- t-stat vs time PNG saved to out_phase11/
- Annotate: FDR-significant time points + peak |t-stat| time point
- Single t-stat curve (5-ch parietal average)
- Overlay 5 x 50ms bin shading (gray semi-transparent)

#### Bins classifier config
- LR C=0.1 per bin (consistent with Phase 8/10)
- Features: mean ERP amplitude across bin time points, 128-dim full channels
- 1000 permutation test per bin, report p-value
- 128-dim full channels (not parietal subset)

#### Result reporting
- Ablation table printed to stdout AND saved to file
- Columns: Bin | BA | p-value | Sig | Note (best BA marked with star)
- Results saved to out_phase11/

### Claude Discretion
- Result file format (CSV or JSON)
- Permutation parallelism implementation details
- Plot color and style details

### Deferred Ideas (OUT OF SCOPE)
None
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TWIN-01 | Per-timepoint MDD vs HC t-test on avg_erp, output t-stat vs time plot | scipy.stats.ttest_ind + scipy.stats.false_discovery_control + matplotlib |
| TWIN-02 | 50ms bin ablation (5 windows: 250-300/300-350/350-400/400-450/450-500ms), 128-dim LOSO BA per bin | Reuse _loso_ba_subset + permutation_test_subset from Phase 10; LR C=0.1 |
</phase_requirements>

## Summary

Phase 11 adds two independent analyses on top of the existing run_modma_erp.py pipeline: (1) a descriptive t-test curve showing which time points within 250-500ms best separate MDD from HC, and (2) an ablation classification where each 50ms bin is used as a standalone feature set.

The critical implementation insight is that load_erp_features() currently discards avg_erp after computing the P300 mean. Phase 11 needs it retained. Add a new load_erp_timeseries() that returns (X_feat, avg_erps, times, y, ids).

FDR correction uses scipy.stats.false_discovery_control (scipy 1.17.0, confirmed installed). statsmodels is NOT installed. The bin classifier reuses _loso_ba_subset + permutation_test_subset from Phase 10, with C=0.1 instead of C=1.0.

**Primary recommendation:** Add load_erp_timeseries() returning avg_erp array, implement run_time_window_analysis() with plot_ttest_curve() and run_bin_ablation(), add to __main__ block, write results to out_phase11/.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scipy.stats.ttest_ind | 1.17.0 | Per-timepoint t-test | Welch (equal_var=False) handles unequal group sizes |
| scipy.stats.false_discovery_control | 1.17.0 | FDR correction (BH) | In scipy 1.17; statsmodels NOT installed |
| matplotlib.pyplot | 3.10.8 | PNG plot output | Already in project |
| numpy | 2.4.2 | Array slicing for time windows | Already in project |
| sklearn LogisticRegression C=0.1 | 1.8.0 | Bin classifier | Matches Phase 8/10 convention |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| joblib.Parallel | (via sklearn) | Permutation parallelism | Already used in permutation_test_subset |
| csv (stdlib) | stdlib | Save ablation table | Simpler than JSON for tabular data |

**No new installations required.**

## Architecture Patterns

### File Structure
```
run_modma_erp.py          # add load_erp_timeseries(), run_time_window_analysis()
out_phase11/
  ttest_curve.png         # t-stat vs time plot
  bin_ablation.csv        # ablation table
```

### Pattern 1: load_erp_timeseries()

New function — same loop as load_erp_features() but retains avg_erp per subject.

```python
def load_erp_timeseries(condition="hcue"):
    global CONDITION
    CONDITION = condition
    raw_files = sorted(ERP_DIR.glob("*.raw"))
    all_feat, all_erps, all_labels, all_ids = [], [], [], []
    times_ref = None
    for fpath in raw_files:
        # ... same label/sub_id/load/filter/epoch logic ...
        avg_erp = data.mean(axis=0)              # (128, n_times)
        if times_ref is None:
            times_ref = epochs.times.copy()
        p300_mask = (times_ref >= P300_WIN[0]) & (times_ref <= P300_WIN[1])
        feat = avg_erp[:, p300_mask].mean(axis=1)
        all_feat.append(feat)
        all_erps.append(avg_erp)                 # retain full time series
        # ...
    return (np.array(all_feat), np.array(all_erps),
            times_ref, np.array(all_labels), np.array(all_ids))
```

### Pattern 2: Per-Timepoint t-test with FDR

```python
from scipy.stats import ttest_ind, false_discovery_control

def plot_ttest_curve(avg_erps, times, y, out_dir):
    parietal_idx = list(get_parietal_indices().values())
    mask_time = (times >= 0.25) & (times <= 0.50)
    t_times = times[mask_time]
    erp_par = avg_erps[:, parietal_idx, :][:, :, mask_time].mean(axis=1)  # (N, n_t)
    mdd, hc = np.where(y == 1)[0], np.where(y == 0)[0]
    res = [ttest_ind(erp_par[mdd, t], erp_par[hc, t], equal_var=False)
           for t in range(erp_par.shape[1])]
    t_stats = np.array([r.statistic for r in res])
    p_vals  = np.array([r.pvalue   for r in res])
    sig_mask = false_discovery_control(p_vals) < 0.05   # adjusted p-values
    peak_t   = t_times[np.argmax(np.abs(t_stats))]
    # ... matplotlib plot + save to out_dir/ttest_curve.png ...
```

### Pattern 3: Bin Ablation with C=0.1

Add C parameter to _loso_ba_subset (default 1.0, backward compatible):

```python
def _loso_ba_subset(X, y, C=1.0):
    nc = min(X.shape[1], len(y) - 1)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=nc)),
        ("clf",    LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))
    ])
    preds = []
    for tr_idx, te_idx in LeaveOneOut().split(X):
        pipe.fit(X[tr_idx], y[tr_idx])
        preds.append(int(pipe.predict(X[te_idx])[0]))
    return balanced_accuracy_score(y, preds)
```

Also add C parameter to permutation_test_subset and forward it.

Bin loop:

```python
BINS = [
    ("250-300ms", 0.250, 0.300),
    ("300-350ms", 0.300, 0.350),
    ("350-400ms", 0.350, 0.400),
    ("400-450ms", 0.400, 0.450),
    ("450-500ms", 0.450, 0.500),
]

def run_bin_ablation(avg_erps, times, y, out_dir):
    results = []
    for i, (lbl, t0, t1) in enumerate(BINS):
        mask = (times >= t0) & (times < t1) if i < len(BINS)-1 else (times >= t0) & (times <= t1)
        X_bin = avg_erps[:, :, mask].mean(axis=2)   # (N, 128)
        ba, p = permutation_test_subset(X_bin, y, C=0.1)
        results.append((lbl, ba, p))
    _print_and_save_ablation(results, out_dir)
```

### Anti-Patterns to Avoid
- **Re-loading EDF files:** avg_erps collected in one pass via load_erp_timeseries(). Do not call load_erp_features() again.
- **Using statsmodels:** Not installed. Use scipy.stats.false_discovery_control only.
- **FDR on all time points:** Apply only to P300 window (250-500ms) time points.
- **Missing mkdir:** Path("out_phase11").mkdir(exist_ok=True) before any file write.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| FDR correction | Manual BH loop | scipy.stats.false_discovery_control | Correct, one-liner |
| t-test | Manual formula | scipy.stats.ttest_ind(equal_var=False) | Welch handles unequal group sizes |
| Permutation parallelism | Manual multiprocessing | joblib.Parallel | Already in codebase |

## Common Pitfalls

### Pitfall 1: false_discovery_control returns adjusted p-values, not booleans
**What goes wrong:** Treating return value as boolean mask.
**How to avoid:** sig_mask = false_discovery_control(p_vals) < 0.05

### Pitfall 2: times array not stored from load loop
**What goes wrong:** epochs.times discarded; Phase 11 has no time axis.
**How to avoid:** Store times_ref = epochs.times.copy() from first valid subject.

### Pitfall 3: Bin boundary double-counting
**What goes wrong:** Half-open bins overlap at boundary sample (e.g. 300ms in both 250-300 and 300-350).
**How to avoid:** Use (times >= t0) & (times < t1) for all bins except last.

### Pitfall 4: C=0.1 vs C=1.0 mismatch
**What goes wrong:** Copy-pasting _loso_ba_subset without changing C.
**How to avoid:** Add C parameter to _loso_ba_subset (default 1.0); call with C=0.1 for bins.

### Pitfall 5: permutation_test_subset not forwarding C
**What goes wrong:** permutation_test_subset calls _loso_ba_subset without C.
**How to avoid:** Add C=1.0 parameter to permutation_test_subset and forward it.

## Code Examples

### FDR correction (verified scipy 1.17.0)
```python
from scipy.stats import false_discovery_control
adj_pvals = false_discovery_control(p_vals)   # BH by default
sig_mask  = adj_pvals < 0.05
```

### Ablation table print + CSV save
```python
import csv
def _print_and_save_ablation(results, out_dir):
    best = max(range(len(results)), key=lambda i: results[i][1])
    rows = [["Bin", "BA", "p-value", "Sig", "Note"]]
    for i, (lbl, ba, p) in enumerate(results):
        sig  = "Yes" if p < 0.05 else "No"
        note = "star" if i == best else ""
        print(f"  {lbl}: BA={ba:.3f}  p={p:.4f}  {sig}  {note}", flush=True)
        rows.append([lbl, f"{ba:.3f}", f"{p:.4f}", sig, note])
    out = Path(out_dir) / "bin_ablation.csv"
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
```

### Matplotlib t-stat plot with bin shading
```python
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_times * 1000, t_stats, color="steelblue", lw=1.5)
ax.axhline(0, color="gray", lw=0.8, ls="--")
for lbl, t0, t1 in BINS:
    ax.axvspan(t0 * 1000, t1 * 1000, alpha=0.15, color="gray")
if sig_mask.any():
    ax.scatter(t_times[sig_mask] * 1000, t_stats[sig_mask],
               color="red", s=15, zorder=5, label="FDR p<0.05")
ax.axvline(peak_t * 1000, color="orange", lw=1.2, ls=":",
           label=f"Peak {peak_t*1000:.0f}ms")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("t-statistic (MDD vs HC)")
ax.set_title("P300 Window: Per-Timepoint t-test (parietal 5-ch avg)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(Path(out_dir) / "ttest_curve.png", dpi=150)
plt.close()
```

## Open Questions

1. **times array consistency across subjects**
   - What we know: All subjects use same epoch window (TMIN=-0.1, TMAX=0.5) and sfreq
   - Recommendation: Assert len(times_ref) == avg_erp.shape[1] for each subject

2. **Result file format (Claude discretion)**
   - Recommendation: CSV — simpler to inspect than JSON for tabular data

3. **Runtime for 5 bins x 1000 perms**
   - Estimate: ~2 min per bin (same cost as Phase 10 permutation_test_subset)
   - Total ~10 min — acceptable; no change to parallelism strategy

## Sources

### Primary (HIGH confidence)
- scipy 1.17.0 installed — false_discovery_control confirmed via python import
- run_modma_erp.py — existing LOSO, permutation, data loading patterns
- .planning/phases/10-electrode-selection/10-01-PLAN.md — _loso_ba_subset / permutation_test_subset patterns

### Secondary (MEDIUM confidence)
- scipy docs: false_discovery_control(ps, *, axis=0, method='bh') — BH is default

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries confirmed installed
- Architecture: HIGH — patterns derived directly from existing codebase (Phase 10)
- Pitfalls: HIGH — derived from code inspection and API verification

**Research date:** 2026-02-23
**Valid until:** 2026-03-25
