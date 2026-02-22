# Architecture Research

**Domain:** EEG depression classification research pipeline
**Researched:** 2026-02-22
**Confidence:** HIGH

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Configuration                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  config.py — all params in one place (dataclass)     │   │
│  └──────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                      Data Layer                             │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐               │
│  │  io.py    │→ │  qc.py    │→ │ (windows) │               │
│  │ BIDS/EDF  │  │ amp mask  │  │ in memory │               │
│  └───────────┘  └───────────┘  └───────────┘               │
├─────────────────────────────────────────────────────────────┤
│                    Feature Layer                            │
│  ┌────────┐ ┌──────┐ ┌────────┐ ┌──────┐ ┌───────────┐    │
│  │  PSD   │ │  DE  │ │ Hjorth │ │ Conn │ │ Riemannian│    │
│  │ /bands │ │      │ │        │ │ImCoh │ │ log-cov   │    │
│  └────┬───┘ └──┬───┘ └───┬────┘ └──┬───┘ └─────┬─────┘    │
│       └────────┴─────────┴─────────┴───────────┘           │
│                        ↓ concat                            │
├─────────────────────────────────────────────────────────────┤
│                   Classification Layer                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Nested CV (StratifiedGroupKFold)                    │   │
│  │  Inner: GridSearchCV over models + hyperparams       │   │
│  │  Outer: subject-level prediction aggregation         │   │
│  └──────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                    Evaluation Layer                         │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │  Metrics   │  │ Permutation  │  │  Report / Plots  │    │
│  │  (BA/AUC)  │  │  test        │  │  (JSON/CSV/PNG)  │    │
│  └────────────┘  └──────────────┘  └──────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| `config.py` | Central parameter store (frozen dataclass) | All components read from it |
| `regions.py` | Channel-to-region mapping (EGI-128, 10-20), L/R hemispheres | features, io |
| `io.py` | BIDS loading, EDF reading, filtering, resampling, windowing | config, regions |
| `qc.py` | Amplitude masking, bad-channel counting, subject dropping, QC report | io output, config |
| `features.py` | PSD, DE, Hjorth, asymmetry, connectivity, Riemannian extractors | qc output, regions |
| `classify.py` | Pipeline construction, nested CV, model selection, subject aggregation | features output |
| `evaluate.py` | Permutation test, bootstrap CI, stopping criteria, report/plots | classify output |
| `main.py` | CLI parsing, orchestrates full pipeline | all modules |

## Recommended Project Structure

```
modma_pipeline/
├── config.py           # Frozen dataclass with all parameters
├── regions.py          # EGI128/10-20 channel-region maps, L/R hemispheres
├── io.py               # BIDS loading, EDF reading, filtering, windowing
├── qc.py               # Quality masks, subject dropping, QC report
├── features.py         # All feature extractors (PSD, DE, Hjorth, conn, Riem)
├── classify.py         # Pipeline builder, nested CV, model selection
├── evaluate.py         # Permutation test, bootstrap CI, stopping criteria, plots
├── main.py             # CLI entry point, orchestrates full pipeline
└── tests/
    ├── test_io.py
    ├── test_qc.py
    ├── test_features.py
    ├── test_classify.py
    └── test_evaluate.py
```

### Structure Rationale

- **config.py:** Single source of truth. Frozen dataclass prevents mutation. No magic globals.
- **regions.py:** Pure data + one helper. Used by both io and features. Prevents circular imports.
- **io.py:** Slowest step (~2 min). Isolating enables .npz caching for instant reload.
- **qc.py:** Primary tuning knob. Iterate thresholds without touching features or classification.
- **features.py:** All extractors share input shape (n_win, n_ch, n_times) and output convention. ~300 lines.
- **classify.py:** Most complex logic. Independent testability prevents data leakage during refactoring.
- **evaluate.py:** Expensive permutation tests. Can run independently on saved CV results.

## Architectural Patterns

### Pattern 1: Config-as-Dataclass

**What:** All pipeline parameters in a single frozen dataclass. Functions receive config — never read globals.
**When to use:** Always. Foundation for reproducibility.
**Trade-offs:** More verbose signatures, but eliminates hidden state.

```python
@dataclass(frozen=True)
class PipelineConfig:
    bids_root: str
    bad_amp_uv: float = 200.0
    max_bad_channels: int = 15
    skip_first_window: bool = True
    n_permutations: int = 1000
    seed: int = 42
    n_jobs: int = 4
```

### Pattern 2: Cache-After-Load

**What:** Save windowed numpy arrays to .npz after EDF loading. Subsequent runs skip I/O.
**When to use:** Always. EDF loading ~2 min; .npz reload <1 sec.
**Trade-offs:** ~200MB disk. Invalidate cache when preprocessing params change.

```python
def _cache_key(cfg):
    params = f"{cfg.highpass_freq}_{cfg.resample_sfreq}_{cfg.crop_duration}"
    return hashlib.md5(params.encode()).hexdigest()[:8]
```

### Pattern 3: Feature Registry

**What:** Each extractor has signature `(X, sfreq, region_idx) -> (features, names)`. A list controls which run.
**When to use:** When iterating on feature sets — toggle extractors without editing logic.
**Trade-offs:** Slight indirection, but enables ablation studies.

## Data Flow

### Main Pipeline Flow

```
[CLI args] → parse → [PipelineConfig]
    ↓
[io.py] load_participants() → load_windows()
    ↓ (X: n_win x n_ch x n_times, y, groups, ch_names)
    ↓ optional: .npz cache
[qc.py] build_quality_mask() → filter → drop subjects
    ↓ (X_clean, y_clean, groups_clean + qc_report.csv)
[features.py] extract_features()
    ↓ (feature_matrix: n_win x n_features, feature_names)
[classify.py] run_full_model_selection()
    ↓ (cv_results: y_true, y_prob, fold_results)
[evaluate.py] permutation_test() → build_report() → conclusion
    ↓ (metrics.json, roc_curve.png, confusion_matrix.png)
```

### Key Data Flows

1. **Load:** BIDS participants.tsv → EDF discovery → MNE read_raw_edf → highpass → avg ref → resample → crop → window → numpy (n_win, n_ch, n_times)
2. **QC:** Raw windows → amplitude check → bad channel count → mask → drop subjects below threshold → QC report CSV
3. **Feature:** Clean windows → region aggregation → PSD + DE + Hjorth + asymmetry + connectivity + Riemannian → column-stack
4. **Classify:** Features → outer StratifiedGroupKFold → inner GridSearchCV → subject-level probability aggregation
5. **Evaluate:** Observed BA → permutation test (shuffle labels, rerun classification only) → p-value → bootstrap CI → stopping criteria

## Anti-Patterns

### Anti-Pattern 1: Monolithic Script

**What people do:** All 960 lines in one file — loading, QC, features, classification, reporting.
**Why it's wrong:** Can't iterate on QC without re-reading everything. Can't test components in isolation.
**Do this instead:** Split by responsibility with clear input/output contracts.

### Anti-Pattern 2: Global Mutable State

**What people do:** Module-level variables for thresholds, flags like `_lgbm_warned`.
**Why it's wrong:** Non-reproducible. Concurrent calls interfere. Hard to test.
**Do this instead:** Frozen dataclass config. Pass all state explicitly.

### Anti-Pattern 3: Re-extracting Features in Permutation Loop

**What people do:** Run full QC + feature extraction inside each of 1000 permutation iterations.
**Why it's wrong:** Features are label-independent. Re-running 1000x wastes ~95% of compute.
**Do this instead:** Extract features once, only permute labels and re-run classification.

## Integration Points

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| io → qc | numpy arrays (X, y, groups) | io returns raw windows; qc applies masks |
| qc → features | filtered numpy arrays | qc drops bad windows/subjects first |
| features → classify | 2D feature matrix + name list | (n_win, n_feat) array |
| classify → evaluate | cv_results dict (y_true, y_prob, fold_results) | evaluate never sees raw data |
| config → all | PipelineConfig dataclass | read-only, passed as argument |

## Build Order (Suggested Refactoring Sequence)

Dependencies between components dictate the order:

1. **config.py + regions.py** (no dependencies)
   - Extract PipelineConfig dataclass and region maps
   - Pure data, zero risk

2. **io.py** (depends on: config, regions)
   - Extract load_participants(), load_windows()
   - Add .npz caching and skip_first_window logic

3. **qc.py** (depends on: config)
   - Extract build_quality_mask(), validate_*, generate_qc_report()
   - Fix max_bad_channels from 3 to ~15 during extraction

4. **features.py** (depends on: regions)
   - Extract all feature extractors with standard signature
   - (X, sfreq, region_idx) -> (features, names)

5. **classify.py** (depends on: config, features, qc)
   - Extract pipeline builder, nested CV, model selection
   - Most complex module — test thoroughly

6. **evaluate.py** (depends on: config, classify)
   - Extract permutation test, report, conclusion, plots
   - Optimize: features once, only permute classification

7. **main.py** (depends on: all above)
   - Thin orchestrator: parse args -> config -> call modules

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 53 subjects (current) | In-memory numpy, joblib for permutations |
| 200 subjects | .npz caching mandatory, memory-mapped arrays |
| 1000+ subjects | Dask out-of-core, MNE-BIDS-Pipeline |

### Scaling Priorities

1. **First bottleneck:** EDF loading (~2 min). Fix with .npz caching.
2. **Second bottleneck:** Permutation test (1000x). Fix by extracting features once.

## Sources

- [MNE-BIDS-Pipeline](https://mne.tools/mne-bids-pipeline/dev/features/overview.html) — sequential step-based processing with caching (HIGH confidence)
- [MNE Design Philosophy](https://mne.tools/stable/documentation/design_philosophy.html) — scripts as analysis records (HIGH confidence)
- [Reproducible MEG/EEG Group Study](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2018.00530/full) — group study best practices (HIGH confidence)
- [EEG BCI Reproducibility Study](https://arxiv.org/html/2404.15319v1) — classification benchmarks (MEDIUM confidence)
- [EEGManyPipelines](https://www.researchgate.net/publication/367204012) — pipeline choice impact on results (MEDIUM confidence)

---
*Architecture research for: MODMA EEG depression classification pipeline*
*Researched: 2026-02-22*
