# Phase 15: Grand-Average ERP Plot - Research

**Researched:** 2026-02-24
**Domain:** ERP visualization (matplotlib), EGI HydroCel-128 channel mapping
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Electrode selection & layout
- Midline three channels: Fz, Cz, Pz (EGI equivalents)
- Three-row subplot layout, top to bottom: Fz -> Cz -> Pz
- Shared x-axis (time) and y-axis (amplitude) range for cross-channel comparison

#### Group visualization style
- MDD red solid line, HC blue dashed line
- Each mean line surrounded by +/-1 SEM translucent shading
- Academic style: linewidth 1.5pt, SEM shading alpha=0.2

#### P300 window annotation
- 250-500ms region covered by light gray translucent rectangle
- "P300" text label inside the shaded region
- Vertical dashed line at 0ms (stimulus onset)
- Y-axis positive up (standard Cartesian; P300 shows as upward deflection)

#### Additional elements
- No topomap
- Only grand-average mean + SEM, no individual waveforms overlaid
- Overall title at top (figure-level suptitle)
- Legend only in topmost subplot (Fz), upper right corner
- Output both PNG (300 DPI) and PDF vector

### Claude's Discretion
- Figure size (figsize)
- Font choice and size
- Gray shading rectangle alpha value
- Axis label wording

### Deferred Ideas (OUT OF SCOPE)
None
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ERP-04 | Grand-average ERP plot verifying P300 window (250-500ms positive deflection) | load_erp_timeseries() provides avg_erps (N,128,n_times) + times; get_parietal_indices() maps Fz/Cz/Pz to EGI indices; matplotlib subplot grid |
</phase_requirements>

## Summary

Phase 15 is a pure visualization task. All data infrastructure already exists: `load_erp_timeseries()` (Phase 11) returns per-subject average ERP arrays `(N, 128, n_times)` with a shared `times` vector, and `get_parietal_indices()` (Phase 10) maps standard 10-20 names to EGI HydroCel-128 indices via coordinate-distance fallback.

The implementation adds a single function `plot_grand_average_erp()` that: (1) calls `load_erp_timeseries("hcue")`, (2) extracts Fz/Cz/Pz channel indices, (3) splits by MDD/HC label, (4) computes group mean and SEM, (5) plots 3-row subplots with P300 window shading and stimulus onset line, (6) saves PNG (300 DPI) and PDF.

**Critical constraint:** The current epoch window is `TMIN=-0.1, TMAX=0.5` (i.e., -100ms to 500ms). The ROADMAP success criterion says "0-800ms" but the data only extends to 500ms. The plot should cover the full available epoch range (-100ms to 500ms), which is sufficient to show the P300 window (250-500ms) and baseline period. Do NOT change TMIN/TMAX — that would invalidate all prior classification results.

**Primary recommendation:** Add `plot_grand_average_erp()` to `run_modma_erp.py`, call from `__main__`, output to `outputs/out_phase15/`.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| matplotlib.pyplot | 3.10.x | Subplot figure, line plots, shading | Already in project |
| numpy | 2.4.x | Mean, SEM computation, boolean masking | Already in project |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy.stats.sem | 1.17.x | Standard error of mean | One-liner, avoids manual std/sqrt(n) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| scipy.stats.sem | Manual np.std/np.sqrt | sem() is cleaner; scipy already imported |
| mne.viz.plot_compare_evokeds | matplotlib manual | MNE function requires Evoked objects; we have raw numpy arrays; manual is simpler |

**No new installations required.**

## Architecture Patterns

### File Structure
```
scripts/modma/run_modma_erp.py    # add plot_grand_average_erp()
outputs/out_phase15/
  grand_avg_erp.png               # 300 DPI raster
  grand_avg_erp.pdf               # vector
```

### Pattern 1: Channel Index Lookup (existing)

`get_parietal_indices()` already supports arbitrary target names including Fz via coordinate-distance fallback. Already used at line 852:

```python
frontal_idx = list(dict.fromkeys(get_parietal_indices(("Fz",)).values()))
```

For Phase 15, call once with all three targets:
```python
ch_map = get_parietal_indices(("Fz", "Cz", "Pz"))
# Returns e.g. {"Fz": idx, "Cz": idx, "Pz": idx}
```

### Pattern 2: Grand-Average Computation

```python
_, avg_erps, times, y, _ = load_erp_timeseries("hcue")
# avg_erps: (N, 128, n_times), times: (n_times,)

mdd_mask = y == 1
hc_mask = y == 0

ch_erp = avg_erps[:, ch_idx, :] * 1e6  # to uV
mdd_mean = ch_erp[mdd_mask].mean(axis=0)
mdd_sem  = sem(ch_erp[mdd_mask], axis=0)
```

### Pattern 3: Matplotlib Subplot with Shared Axes

```python
fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True, sharey=True)
for i, (ax, (name, idx)) in enumerate(zip(axes, channels)):
    # plot mean +/- SEM for each group
    ax.plot(times_ms, mdd_mean, color="red", lw=1.5, label="MDD")
    ax.fill_between(times_ms, mdd_mean - se, mdd_mean + se,
                    color="red", alpha=0.2)
    # P300 window shading
    ax.axvspan(250, 500, color="gray", alpha=0.15)
    # stimulus onset
    ax.axvline(0, color="black", lw=0.8, ls="--")
```

### Pattern 4: Existing Matplotlib Convention in Codebase

From `plot_ttest_curve()` and `run_feature_importance()`:
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# ... plot ...
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
```

### Anti-Patterns to Avoid
- **Re-loading data:** Use `load_erp_timeseries()` once; do not call `load_erp_features()` separately.
- **Using mne.viz:** Data is already numpy arrays; MNE viz expects Evoked objects.
- **Hardcoding channel indices:** Always use `get_parietal_indices()` for EGI mapping.
- **Forgetting `matplotlib.use("Agg")`:** Required for headless environments.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SEM computation | `np.std(x)/np.sqrt(n)` | `scipy.stats.sem(x, axis=0)` | Handles ddof correctly, one-liner |
| Channel name -> index | Manual montage parsing | `get_parietal_indices(("Fz","Cz","Pz"))` | Already handles EGI coordinate fallback |
| ERP data loading | New file-reading loop | `load_erp_timeseries("hcue")` | Phase 11 infrastructure, tested |

## Common Pitfalls

### Pitfall 1: Epoch Time Range vs Success Criteria
**What goes wrong:** ROADMAP says "0-800ms" but TMAX=0.5 means data ends at 500ms.
**Why it happens:** Success criteria written before checking actual epoch parameters.
**How to avoid:** Plot the full available range (-100ms to 500ms). Do NOT change TMIN/TMAX — that would invalidate all prior classification results.
**Warning signs:** If times array max is ~0.5, the 800ms target is impossible with current data.

### Pitfall 2: ERP Amplitude Units
**What goes wrong:** MNE stores EEG in Volts (1e-6 scale). Plot y-axis shows tiny numbers.
**Why it happens:** No unit conversion applied.
**How to avoid:** Multiply by 1e6 to convert to microvolts (uV). Label y-axis accordingly.

### Pitfall 3: Y-axis Polarity Convention
**What goes wrong:** Some ERP literature uses "negative up" convention.
**Why it happens:** Historical EEG convention varies.
**How to avoid:** User locked: Y-axis positive up (standard Cartesian). Do NOT invert.

### Pitfall 4: SEM with Small Group Sizes
**What goes wrong:** With N_MDD=24, N_HC=28, SEM bands may be wide.
**How to avoid:** Expected and informative. No special handling needed.

### Pitfall 5: P300 Text Label Positioning
**What goes wrong:** Fixed y-coordinate for "P300" label falls outside visible range.
**Why it happens:** Y-axis range is data-dependent.
**How to avoid:** Place label using `ax.get_ylim()` or `ax.transAxes` coordinates after plotting.

## Code Examples

### Complete Plot Function Skeleton

```python
def plot_grand_average_erp(out_dir):
    """ERP-04: Grand-average ERP waveform plot (Fz, Cz, Pz)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import sem

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    _, avg_erps, times, y, _ = load_erp_timeseries("hcue")
    times_ms = times * 1000

    ch_map = get_parietal_indices(("Fz", "Cz", "Pz"))
    channels = [("Fz", ch_map["Fz"]),
                ("Cz", ch_map["Cz"]),
                ("Pz", ch_map["Pz"])]
    mdd_mask, hc_mask = y == 1, y == 0

    fig, axes = plt.subplots(3, 1, figsize=(8, 9),
                             sharex=True, sharey=True)
    for i, (ax, (name, idx)) in enumerate(zip(axes, channels)):
        ch_erp = avg_erps[:, idx, :] * 1e6  # V -> uV
        for mask, color, ls, label in [
            (mdd_mask, "red", "-", f"MDD (n={mdd_mask.sum()})"),
            (hc_mask, "blue", "--", f"HC (n={hc_mask.sum()})"),
        ]:
            m = ch_erp[mask].mean(axis=0)
            se = sem(ch_erp[mask], axis=0)
            ax.plot(times_ms, m, color=color, lw=1.5, ls=ls, label=label)
            ax.fill_between(times_ms, m - se, m + se,
                            color=color, alpha=0.2)
        ax.axvspan(250, 500, color="gray", alpha=0.15)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_ylabel(f"{name}\nAmplitude (uV)")
        # P300 label via axes transform for stable positioning
        ax.text(375, 0.85, "P300", ha="center", fontsize=9,
                color="gray", transform=ax.get_xaxis_transform())
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("Grand-Average ERP: hcue (MDD vs HC)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "grand_avg_erp.png", dpi=300,
                bbox_inches="tight")
    fig.savefig(out_dir / "grand_avg_erp.pdf", bbox_inches="tight")
    plt.close(fig)
```

## Open Questions

1. **Time range: -100ms to 500ms vs 0-800ms**
   - What we know: TMIN=-0.1, TMAX=0.5 means epochs span -100ms to 500ms
   - What's unclear: ROADMAP success criterion says "0-800ms" but data doesn't extend that far
   - Recommendation: Plot full available range (-100ms to 500ms). Changing TMAX would invalidate all prior phases. Update success criterion to match actual data.

2. **Output directory naming**
   - What we know: Prior phases use `outputs/out_phaseNN/` pattern
   - Recommendation: Use `outputs/out_phase15/` for consistency

## Sources

### Primary (HIGH confidence)
- `scripts/modma/run_modma_erp.py` — load_erp_timeseries(), get_parietal_indices(), matplotlib patterns
- `.planning/phases/11-time-window-analysis/11-RESEARCH.md` — load_erp_timeseries architecture
- `.planning/phases/15-grand-average-erp-plot/15-CONTEXT.md` — user decisions

### Secondary (MEDIUM confidence)
- scipy.stats.sem — standard library function, well-documented

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in project, no new dependencies
- Architecture: HIGH — reuses existing load_erp_timeseries() and get_parietal_indices()
- Pitfalls: HIGH — derived from code inspection (epoch range, units, polarity)

**Research date:** 2026-02-24
**Valid until:** 2026-03-24
