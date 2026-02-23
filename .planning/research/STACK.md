# Stack Research

**Domain:** EEG preprocessing, QC, and MDD biomarker extraction
**Researched:** 2026-02-22
**Confidence:** MEDIUM (versions verified via PyPI; some library interactions unverified in practice)

## Current Stack (Already Installed — Do Not Replace)

| Technology | Version | Purpose | Status |
|------------|---------|---------|--------|
| MNE-Python | 1.11.0 | Core EEG I/O, filtering, referencing, epoching | Keep |
| NumPy | 2.4.2 | Array operations | Keep |
| SciPy | 1.17.0 | Signal processing (welch, csd) | Keep |
| scikit-learn | 1.8.0 | Classification pipelines, CV, metrics | Keep |
| pyriemann | 0.10 | Riemannian geometry features (covariance matrices) | Keep |
| pandas | 3.0.1 | Tabular data, participants.tsv | Keep |
| joblib | 1.5.3 | Parallel permutation testing | Keep |
| matplotlib | 3.10.8 | Plotting, QC reports | Keep |
| mne-bids | 0.18.0 | BIDS dataset handling | Keep |

## Recommended Additions

### Tier 1: Critical — Directly Solves the 80% Dropout Problem

| Library | Version | Purpose | Why Recommended | Confidence |
|---------|---------|---------|-----------------|------------|
| autoreject | 0.4.3 | Data-driven epoch rejection + per-channel repair | Replaces hardcoded `bad_amp_uv=200` and `max_bad_channels=3` with statistically learned thresholds. "Local" mode **interpolates** bad channels within individual epochs instead of dropping them — critical for retaining subjects. For 128-channel data, dense spatial sampling makes interpolation highly reliable. | HIGH |
| pyprep | 0.5.0 | Automated bad channel detection (PREP pipeline) | Implements PREP's `NoisyChannels` detection: correlation, deviation, RANSAC, high-frequency noise. Replaces naive amplitude-only check with multi-criteria detection that adapts to each recording's noise profile. Directly addresses the EDF physical-scaling anomaly. | HIGH |

**Why these two together:** pyprep identifies globally bad channels on continuous data (before epoching). autoreject then handles epoch-level bad channels and repairs them. This two-stage approach maximizes data retention.

### Tier 2: Important — Removes Artifacts That Contaminate Features

| Library | Version | Purpose | Why Recommended | Confidence |
|---------|---------|---------|-----------------|------------|
| mne-icalabel | 0.8.1 | Automated ICA component classification (ICLabel) | Classifies ICA components as brain/eye/muscle/heart/line-noise/channel-noise/other. Removes ocular and muscular artifacts that inflate PSD and corrupt connectivity — both key MDD biomarker features. Eye blinks in frontal channels directly corrupt FAA (frontal alpha asymmetry), the primary depression biomarker. | HIGH |

**ICLabel requirements:** ICA must use extended infomax, data must be common-average referenced and filtered [1, 100] Hz. Current pipeline already does CAR and highpass 0.5 Hz — raise lowpass from 45 to 100 Hz for ICLabel fitting, then lowpass to 45 Hz after artifact removal.

### Tier 3: Beneficial — Better Connectivity Features

| Library | Version | Purpose | Why Recommended | Confidence |
|---------|---------|---------|-----------------|------------|
| mne-connectivity | 0.7.0 | Spectral connectivity (coherence, PLV, ImCoh, wPLI) | Replaces hand-rolled `scipy.signal.csd`-based ImCoh with validated implementations. Adds wPLI (weighted phase lag index) which is more robust to volume conduction than ImCoh — a known confound in depression connectivity studies. | MEDIUM |

## Recommended Preprocessing Pipeline Order

```
1. Load raw EDF (MNE)
2. Highpass filter 1 Hz (MNE) — required for ICLabel; also removes DC offset
3. Detect + interpolate globally bad channels (pyprep NoisyChannels + RANSAC)
4. Common average re-reference (MNE)
5. ICA decomposition — extended infomax (MNE)
6. Classify + remove artifact components (mne-icalabel ICLabel)
7. Bandpass filter [1, 45] Hz (MNE) — for feature extraction
8. Epoch into windows, skip window 0 (filter transient)
9. autoreject local — interpolate per-epoch bad channels, drop unsalvageable epochs
10. Feature extraction (PSD, FAA, connectivity, Riemannian)
```

**Key change from current pipeline:** Steps 3, 5-6, and 9 are new. Current pipeline goes directly from filtering to epoching with a fixed amplitude threshold.

## Installation

```bash
# Tier 1 — Critical (install first)
pip install autoreject==0.4.3 pyprep==0.5.0

# Tier 2 — Important
pip install mne-icalabel==0.8.1

# Tier 3 — Beneficial
pip install mne-connectivity==0.7.0
```

## Alternatives Considered

| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| autoreject (local) | Fixed amplitude threshold (current) | Hardcoded 200uV drops 80% of subjects; autoreject learns optimal thresholds per channel |
| pyprep NoisyChannels | MNE built-in `find_bad_channels_maxwell()` | Maxwell filtering is for MEG, not EEG. pyprep is purpose-built for EEG PREP pipeline |
| mne-icalabel (ICLabel) | Manual ICA component inspection | Not reproducible, not scalable to 53 subjects, requires EEG expertise |
| mne-connectivity | Hand-rolled scipy.signal.csd ImCoh | Current ImCoh had a formula bug (fixed in ea78c01). Using validated library avoids future bugs |
| autoreject (local) | autoreject (global) | Global finds one threshold per channel type; local finds per-channel per-epoch thresholds and can repair |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| DISCOVER-EEG | MATLAB-based (EEGLab + FieldTrip), not Python | MNE + autoreject + pyprep + mne-icalabel |
| eegprep 0.2.23 | Young project, limited community adoption, overlaps with pyprep + autoreject | pyprep + autoreject |
| mne-bids-pipeline | Full pipeline orchestrator — too opinionated, hard to integrate into existing custom pipeline | Use individual libraries (autoreject, pyprep, mne-icalabel) directly |
| Deep learning artifact removal (e.g., EEGDenoiseNet) | Requires large training data, overkill for 53 subjects, adds GPU dependency | ICLabel via mne-icalabel |
| Manual ICA inspection | Not reproducible, not scalable | mne-icalabel automated classification |
| Fixed amplitude thresholds | Dataset-dependent, caused 80% dropout on MODMA | autoreject data-driven thresholds |

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| autoreject 0.4.3 | MNE >= 1.0, NumPy >= 1.20, scikit-learn >= 0.24 | All satisfied by current stack |
| pyprep 0.5.0 | Python >= 3.9, MNE (via dependency) | Requires montage set on Raw object |
| mne-icalabel 0.8.1 | Python >= 3.10, MNE >= 1.6 | Current MNE 1.11.0 is compatible |
| mne-connectivity 0.7.0 | Python >= 3.9, MNE >= 1.3 | Current MNE 1.11.0 is compatible |

## Stack Patterns by Variant

**If subject retention is still < 35 after Tier 1:**
- Add ICA artifact removal (Tier 2) before autoreject
- ICA removes systematic artifacts (blinks, muscle) that cause channels to exceed thresholds
- This reduces the number of bad channels per epoch, improving autoreject retention

**If classification BA is still < 0.6 after Tier 1+2:**
- Add mne-connectivity (Tier 3) for wPLI features
- wPLI is more robust to volume conduction than ImCoh
- Depression literature shows wPLI in theta/alpha bands discriminates MDD vs HC

**If RAM is a constraint (32GB limit):**
- Run pyprep per-subject (already sequential in current pipeline)
- For autoreject local: set `n_interpolate` max to ~16 (not default 32) for 128 channels
- For ICA: use `n_components=0.99` (variance explained) instead of fixed count

## Caveats and Risk Areas

### autoreject local on windowed continuous data
- autoreject is designed for event-related epochs, not arbitrary time windows
- For resting-state 10s windows, it still works but `consensus` parameter may need tuning
- Risk: LOW — the algorithm is agnostic to epoch content, only cares about channel statistics
- Confidence: MEDIUM

### pyprep RANSAC on MODMA EDF data
- MODMA EDF files have unusual physical ranges (DC offsets in header)
- pyprep's NoisyChannels expects data in Volts after MNE loading
- Must apply highpass filter BEFORE running pyprep to remove DC offset
- Risk: MEDIUM — test on 2-3 subjects before full run
- Confidence: MEDIUM

### ICLabel on EGI 128-channel montage
- ICLabel was trained on standard 10-20 montage data
- EGI HydroCel 128 uses non-standard channel names (E1, E2, ...)
- mne-icalabel may need montage mapping to standard positions
- Risk: MEDIUM — verify ICLabel predictions make sense on first few subjects
- Confidence: LOW

### mne-connectivity replacing hand-rolled ImCoh
- Current pipeline computes ImCoh from scipy.signal.csd directly
- mne-connectivity uses epoch-based spectral_connectivity_epochs()
- Data must be in MNE Epochs format (not raw numpy windows)
- Migration requires refactoring the feature extraction to use Epochs objects
- Risk: LOW — straightforward API change
- Confidence: HIGH

## Sources

- [autoreject PyPI](https://pypi.org/project/autoreject/) — version 0.4.3 verified, HIGH confidence
- [autoreject docs](https://autoreject.github.io/stable/index.html) — local/global mode details, HIGH confidence
- [pyprep GitHub](https://github.com/sappelhoff/pyprep) — version 0.5.0 verified via PyPI, HIGH confidence
- [pyprep docs](https://pyprep.readthedocs.io/en/stable/) — NoisyChannels API, RANSAC, HIGH confidence
- [mne-icalabel PyPI](https://pypi.org/project/mne-icalabel/) — version 0.8.1 verified, HIGH confidence
- [mne-icalabel docs](https://mne.tools/mne-icalabel/stable/) — ICLabel requirements (infomax, CAR, [1,100] Hz), HIGH confidence
- [mne-connectivity PyPI](https://pypi.org/project/mne-connectivity/) — version 0.7.0 verified, HIGH confidence
- [MNE-BIDS-Pipeline](https://mne.tools/mne-bids-pipeline/stable/) — autoreject integration reference, MEDIUM confidence
- [EEGLab list: interpolation on 128-ch](https://sccn.ucsd.edu/pipermail/eeglablist/2025/018756.html) — 128-ch interpolation reliability, MEDIUM confidence
- [Nature: EEG is better left alone](https://www.nature.com/articles/s41598-023-27528-0) — minimal preprocessing evidence, MEDIUM confidence
- [Nature: Technical considerations for EEG biomarkers in MDD](https://www.nature.com/articles/s44184-023-00038-7) — depression biomarker review, MEDIUM confidence

---
*Stack research for: EEG MDD classification preprocessing improvement*
*Researched: 2026-02-22*

---

## ERP Task-State Milestone — Stack Additions

**Domain:** ERP task-state MDD classification (P300/N200, multi-condition fusion)
**Researched:** 2026-02-22
**Confidence:** HIGH

### Verdict: Zero New Library Installs Required

Every capability needed for P300/N200 extraction, multi-condition fusion, and permutation significance testing is already present in the installed stack.

| Capability | How | Library |
|------------|-----|---------|
| ERP epoching + baseline correction | `mne.Epochs(tmin, tmax, baseline)` | MNE 1.11.0 (installed) |
| Evoked averaging | `epochs.average()` → `mne.Evoked` | MNE 1.11.0 (installed) |
| P300 mean amplitude (250–500ms) | `data[:, mask].mean(axis=1)` | NumPy 2.4.2 (installed) |
| P300 peak amplitude per channel | `data[:, mask].max(axis=1)` | NumPy 2.4.2 (installed) |
| P300 peak latency per channel | `times[mask][data[:, mask].argmax(axis=1)]` | NumPy 2.4.2 (installed) |
| P300 area under curve | `np.trapezoid(data[:, mask], times[mask], axis=1)` | NumPy 2.4.2 (installed) |
| N200 amplitude (100–250ms, negative) | `data[:, mask].min(axis=1)` | NumPy 2.4.2 (installed) |
| Multi-condition fusion | `np.concatenate([feat_hcue, feat_fcue, feat_scue])` | NumPy 2.4.2 (installed) |
| Missing-condition imputation | `SimpleImputer(strategy='mean')` inside Pipeline | sklearn 1.8.0 (installed) |
| Permutation significance test | `joblib.Parallel` + label shuffle + LOSO rerun | joblib 1.5.3 (installed) |

### Key Implementation Notes

**np.trapz renamed in NumPy 2.x:** Use `np.trapezoid` (not `np.trapz`) to avoid DeprecationWarning in NumPy >= 2.0.

**get_peak is not suitable for per-channel feature extraction:** `mne.Evoked.get_peak()` returns one (channel, time) pair across all channels. For 128-channel feature vectors, use NumPy windowed `argmax`/`argmin` directly on `evoked.get_data()`.

**Keep manual permutation loop, not sklearn.permutation_test_score:** `permutation_test_score` with `LeaveOneOut` does not support subject-level BA aggregation — it would compute window-level BA in the null distribution, biasing p-values. The existing `joblib.Parallel` pattern from the resting-state pipeline is correct and should be reused.

**Multi-condition missing subjects:** Some subjects may have trials for only 1–2 of the 3 conditions (hcue/fcue/scue). Use `SimpleImputer(strategy='mean')` inside the sklearn Pipeline to handle NaN blocks rather than dropping subjects.

### What NOT to Add for ERP Milestone

| Avoid | Why |
|-------|-----|
| autoreject for ERP epochs | Existing `reject=dict(eeg=150e-6)` in `mne.Epochs` is appropriate for ERP; autoreject is calibrated for resting-state windows |
| pyriemann | Riemannian covariance features are for resting-state PSD; ERP features are time-domain amplitude/latency |
| neurokit2 | Adds large dependency for peak detection that NumPy already handles with `argmax` |
| mne-bids | ERP .raw files are EGI format, not BIDS — no value here |
| Deep learning (EEGNet, etc.) | 52 subjects is insufficient; ruled out in PROJECT.md |

### Integration with Existing run_modma_erp.py

The existing skeleton already has the correct structure. Changes are purely additive:
- Feature extraction: extend from `mean_amp` only → add `peak_amp`, `peak_lat`, `area`, `n200_amp`, `n200_lat`
- Multi-condition: run `load_erp_features()` per condition, `np.concatenate` per subject before LOSO
- Permutation test: copy pattern from `modma_mdd_real_experiment.py` — permute subject labels, rerun LOSO, compute p-value

### Sources

- [MNE Evoked.get_data() docs](https://mne.tools/stable/generated/mne.Evoked.html) — confirmed tmin/tmax/picks params, HIGH confidence
- [MNE Evoked.get_peak() discourse](https://mne.discourse.group/t/how-to-get-the-peak-value-of-each-channel-in-mne/2653) — per-channel limitation confirmed, HIGH confidence
- [scipy.signal.find_peaks v1.17.0](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html) — available but not needed, HIGH confidence
- [sklearn permutation_test_score](https://scikit-learn.org/1.0/auto_examples/model_selection/plot_permutation_tests_for_classification.html) — groups param behavior, MEDIUM confidence

---
*ERP stack addendum researched: 2026-02-22*
