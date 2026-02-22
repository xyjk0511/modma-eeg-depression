# Phase 5: Confirmatory Replication with Pre-Registered Pipeline - Research

**Researched:** 2026-02-22
**Domain:** Cross-dataset EEG MDD classification, pre-registration, external validation
**Confidence:** MEDIUM

## Summary

This phase requires training the locked Phase 4 pipeline (5-dim parietal features, Logistic C=0.1) on MODMA and testing on an external public EEG dataset. The primary research questions are: (1) which external dataset meets the criteria, (2) how to map channels across different montages at region level, (3) what a proper pre-registration document looks like, and (4) how to implement cross-dataset permutation testing.

The strongest candidate external dataset is **TDBRAIN** (OpenNeuro ds003838): 1274 psychiatric patients including **426 MDD subjects**, 26-channel 10-20 montage EEG with eyes-closed resting state. This is the only publicly available dataset that clearly meets all criteria (>=50 subjects, >=25/group, eyes-closed resting, 10-20 montage, downloadable). The existing codebase already has a `STANDARD_1020_REGION_MAP` that covers the 19 standard 10-20 channels, making region-level channel mapping straightforward.

**Primary recommendation:** Use TDBRAIN (OpenNeuro ds003838) as the external test set. Filter to MDD vs healthy-control subjects. Implement a thin data adapter that loads BDF files and maps 10-20 channels to the 5 regions. Pre-register the full pipeline as a markdown document with git tag before any data touches the classifier.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
1. **Data Source:** External public EEG dataset; 10-20 montage OK; >=50 subjects, >=25/group; eyes-closed resting state only. Fallback: MODMA leave-N-out (labeled exploratory only).
2. **Pre-registration Scope:** End-to-end lockdown (pipeline + criteria + preprocessing params). Repo internal markdown, git tag locked. Adaptation whitelist: file format adapter, channel mapping, resampling only. Forbidden: changing QC thresholds, features, model/hyperparams, evaluation protocol. Tag BEFORE first run.
3. **Evaluation Protocol:** Same criteria as Phase 4 (BA>0.60, CI lower>0.50, p<0.05). Train-MODMA/Test-external. 1000 permutations on external test set. Complete report regardless of outcome.
4. **Data Compatibility:** Region-level channel mapping. Resample to 125Hz with spectrum sanity check. Re-reference to average. QC threshold strictly locked at 200uV.

### Claude's Discretion
- Specific external dataset selection (within criteria)
- Data adapter implementation details
- Pre-registration document structure
- Permutation test implementation for cross-dataset setting

### Deferred Ideas (OUT OF SCOPE)
- None explicitly listed
</user_constraints>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| mne | >=1.6 | EEG I/O (BDF/EDF), filtering, resampling, re-referencing | Already in use; handles BDF natively via `mne.io.read_raw_bdf` |
| scikit-learn | >=1.3 | LogisticRegression, StandardScaler, Pipeline | Already in use; locked model |
| numpy | >=1.24 | Array operations, permutation | Already in use |
| scipy | >=1.11 | Welch PSD | Already in use |
| pandas | >=2.0 | participants.tsv parsing, QC reports | Already in use |
| joblib | >=1.3 | Parallel permutation test | Already in use |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| matplotlib | >=3.7 | ROC/confusion matrix plots | Report generation |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| TDBRAIN | PRED+CT | PRED+CT has fewer MDD subjects, .mat format, less standardized |
| TDBRAIN | Mumtaz 2017 | Smaller sample, harder to access, not on OpenNeuro |

**Installation:** No new dependencies needed. All libraries already in the project.

## Architecture Patterns

### Recommended Project Structure
```
D:/eeg/
├── modma_mdd_real_experiment.py    # Existing — MODMA pipeline (DO NOT MODIFY)
├── run_phase5_replication.py       # NEW — main entry: train MODMA, test external
├── external_data_adapter.py        # NEW — TDBRAIN loader + channel mapping
└── docs/
    └── PRE_REGISTRATION.md         # NEW — pre-registration document (git-tagged)
```

### Pattern 1: Data Adapter Pattern
**What:** Thin adapter that loads external dataset and returns data in the same format as `load_windows()`.
**When to use:** When external dataset has different file format (BDF vs EDF), different channel names, different sampling rate.
**Example:**
```python
def load_tdbrain_windows(tdbrain_root, window_sec=10, resample_sfreq=125,
                         crop_duration=60, highpass_freq=1.0, bad_amp_uv=200):
    """Load TDBRAIN BDF files, filter to MDD+HC, return (X, y, groups, ch_names).
    Channel mapping: uses STANDARD_1020_REGION_MAP from main pipeline.
    Only keeps standard 10-20 channels present in TDBRAIN.
    """
    # 1. Parse participants.tsv -> filter MDD + HC
    # 2. For each subject: read_raw_bdf -> pick 10-20 channels
    # 3. Filter, resample, re-ref average, window, QC (200uV locked)
    # Returns same format as load_windows()
```

### Pattern 2: Train-A / Test-B with Frozen Model
**What:** Train on full MODMA dataset (no CV split), test on external dataset. No re-tuning.
**When to use:** Cross-dataset confirmatory validation.
**Example:**
```python
def train_modma_test_external(modma_features, modma_y, modma_groups,
                               ext_features, ext_y, ext_groups):
    pipe = build_feature_model_pipeline()  # Scaler + Logistic C=0.1
    sw = compute_subject_balanced_sample_weights(modma_groups)
    pipe.fit(modma_features, modma_y, clf__sample_weight=sw)
    ext_probs = pipe.predict_proba(ext_features)[:, 1]
    # Aggregate to subject level, compute BA, CI, permutation test
```

### Pattern 3: Cross-Dataset Permutation Test
**What:** Permute labels on external test set only. Trained model stays fixed.
**When to use:** Testing significance of cross-dataset generalization.
**Example:**
```python
def cross_dataset_permutation(pipe_fitted, ext_features, ext_y, ext_groups,
                               n_perms=1000, seed=42):
    rng = np.random.RandomState(seed)
    unique_subj, first_idx = np.unique(ext_groups, return_index=True)
    subj_labels = ext_y[first_idx]
    actual_ba = compute_subject_ba(pipe_fitted, ext_features, ext_y, ext_groups)
    perm_bas = []
    for _ in range(n_perms):
        perm_labels = rng.permutation(subj_labels)
        label_map = dict(zip(unique_subj, perm_labels))
        y_perm = np.array([label_map[g] for g in ext_groups])
        perm_bas.append(compute_subject_ba(pipe_fitted, ext_features, y_perm, ext_groups))
    p = (np.sum(np.array(perm_bas) >= actual_ba) + 1) / (len(perm_bas) + 1)
    return actual_ba, p, perm_bas
```

### Anti-Patterns to Avoid
- **Re-training on external data:** Model MUST be trained only on MODMA.
- **Adjusting QC thresholds:** 200uV is locked. Report as-is.
- **Cherry-picking external subjects:** All MDD+HC subjects that pass QC must be included.
- **Modifying features after seeing results:** 5-dim feature set is frozen.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| BDF file reading | Custom binary parser | `mne.io.read_raw_bdf()` | BDF format is complex; MNE handles it correctly |
| Channel name normalization | String matching heuristics | MNE montage + `STANDARD_1020_REGION_MAP` | Already implemented in codebase |
| Resampling | Manual decimation | `raw.resample()` | Anti-aliasing filter built in |
| Bootstrap CI | Manual loop | `get_ci_bootstrap()` | Already implemented in codebase |

## Common Pitfalls

### Pitfall 1: TDBRAIN May Lack Healthy Controls
**What goes wrong:** TDBRAIN is a clinical dataset (1274 psychiatric patients). There may not be a clean "HC" group.
**Why it happens:** Data collected at a clinical practice, not a case-control study.
**How to avoid:** Download participants.tsv first and check for HC/neurotypical label. If absent, fall back to MODMA leave-N-out (exploratory only).
**Warning signs:** All subjects have a psychiatric diagnosis.
**Confidence: LOW — this is the single biggest risk for this phase.**

### Pitfall 2: TDBRAIN Extra Channels Beyond 10-20
**What goes wrong:** 26 channels > 19 standard 10-20. Extra channels (e.g., A1, A2 mastoids, extended positions) break region mapping if not filtered.
**How to avoid:** After loading BDF, explicitly pick only channels in `STANDARD_1020_REGION_MAP`. Drop extras. Log kept/dropped.
**Warning signs:** Feature extraction returns zeros or NaN for a region.

### Pitfall 3: Permuting at Wrong Level
**What goes wrong:** Permuting window-level labels instead of subject-level inflates significance.
**How to avoid:** Always permute subject->label mapping, then propagate to windows.
**Warning signs:** Suspiciously low p-values with modest BA.

### Pitfall 4: Data Leakage via Scaler
**What goes wrong:** Fitting StandardScaler on combined MODMA+external data leaks external distribution info.
**How to avoid:** Fit entire pipeline on MODMA only. Transform external via MODMA-fitted scaler. sklearn Pipeline handles this if you call `pipe.fit(modma)` then `pipe.predict(external)`.

### Pitfall 5: Spectrum Sanity Check After Resampling
**What goes wrong:** TDBRAIN native rate may differ (500-2048Hz). Resampling to 125Hz without checking can introduce artifacts.
**How to avoid:** After resampling, compute PSD on a few subjects: verify alpha peak ~10Hz, no Nyquist spikes, smooth rolloff above 40Hz.

### Pitfall 6: Pre-Registration Timing
**What goes wrong:** Running pipeline "just to check" before tagging invalidates pre-registration.
**How to avoid:** Test code on synthetic/dummy data only. Tag commit. Then run on real external data.

## Code Examples

### Loading TDBRAIN BDF Files
```python
import mne
raw = mne.io.read_raw_bdf("sub-001_eeg.bdf", preload=True, verbose=False)
keep_chs = [ch for ch in raw.ch_names if ch in STANDARD_1020_REGION_MAP]
raw.pick_channels(keep_chs)
montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage, on_missing="warn", verbose=False)
```

### Region Mapping (Already in Codebase)
```python
# modma_mdd_real_experiment.py lines 43-52, 65-76
# build_region_indices() auto-detects EGI vs 10-20 — works for both datasets
```

### Pre-Registration Document Template
```markdown
# Pre-Registration: Confirmatory Replication of MODMA MDD Classifier

## Locked Pipeline
- Features: parietal_theta_rel, parietal_alpha_rel, alpha_asym_frontal,
            parietal_TBR, riem_central_temporal (5 dimensions)
- Model: LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000)
- Scaler: StandardScaler (fit on MODMA training data only)

## Locked Preprocessing
- Highpass: 1.0 Hz, Lowpass: 45.0 Hz, Resample: 125 Hz
- Reference: average, QC threshold: 200 uV (no adjustment)
- Window: 10 sec (skip window 0), Crop: 60 sec
- Bad channel interpolation: spherical spline if <= 25% bad

## Locked Evaluation
- Train: Full MODMA (43 subjects), Test: External dataset
- Metric: Subject-level balanced accuracy
- Success: BA > 0.60, CI lower > 0.50, p < 0.05
- Permutations: 1000 (label permutation on external test set)

## Adaptation Whitelist
- File format adapter (BDF -> same internal format)
- Channel mapping (10-20 region-level)
- Resampling from native rate to 125 Hz

## Explicitly Forbidden
- Changing QC thresholds, features, model, hyperparams, evaluation criteria
- Excluding subjects based on results
```

## Open Questions

1. **Does TDBRAIN have a healthy control group?**
   - What we know: 1274 participants, 426 MDD. Paper says "psychiatric patients."
   - What's unclear: Whether participants.tsv includes HC/neurotypical category.
   - Recommendation: Download participants.tsv first. If no HC, fall back to MODMA leave-N-out.
   - **Confidence: LOW — single biggest risk for this phase.**

2. **TDBRAIN exact channel list**
   - What we know: 26 channels, likely 19 standard 10-20 + 7 extra.
   - Recommendation: Download one BDF, inspect `raw.ch_names`.
   - **Confidence: MEDIUM — 10-20 subset almost certainly present.**

3. **TDBRAIN eyes-closed resting state duration**
   - What we know: Paper mentions resting-state EEG. Clinical EEG typically >= 2 min.
   - Recommendation: Check one recording. Need >= 60s for crop_duration.
   - **Confidence: MEDIUM.**

4. **Fallback: MODMA leave-N-out**
   - If external data fails, use leave-5-out repeated 20 times. Label as exploratory.
   - **Confidence: HIGH — straightforward but scientifically weaker.**

## Sources

### Primary (HIGH confidence)
- Existing codebase: `modma_mdd_real_experiment.py` — full pipeline with region maps, feature extraction, permutation test
- Phase 4 verification: `04-VERIFICATION.md` — BA=0.613, p=0.135, 5-dim features, Logistic C=0.1
- MNE documentation — BDF reading, resampling, montage handling

### Secondary (MEDIUM confidence)
- TDBRAIN paper (Nature Scientific Data 2022) — 1274 participants, 426 MDD, 26-channel resting-state EEG
- TDBRAIN on OpenNeuro (ds003838) — download location
- Transformers paper using TDBRAIN — confirms MDD classification use case

### Tertiary (LOW confidence)
- TDBRAIN healthy control availability — NOT verified, critical risk
- TDBRAIN exact channel names — NOT verified from actual data files
- TDBRAIN eyes-closed segment duration — NOT verified from actual recordings

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new libraries needed
- Architecture: HIGH — straightforward extensions of existing code
- External dataset selection: MEDIUM — TDBRAIN best candidate but HC availability unverified
- Pitfalls: HIGH — well-understood from existing pipeline experience

**Research date:** 2026-02-22
**Valid until:** 2026-03-22 (30 days — main risk is dataset availability check)
