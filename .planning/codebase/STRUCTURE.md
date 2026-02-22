# Codebase Structure

**Analysis Date:** 2026-02-22

## Directory Layout

```
/d/eeg/
├── modma_mdd_classification.py      # Demo: synthetic data + Alpha Asymmetry + SVM
├── modma_mdd_real_experiment.py     # Main: real BIDS data, 144-dim features, full pipeline
├── run_modma_real.py                # Entry point: simplified real data runner
├── run_real_debug.py                # Debug script (minimal)
├── run_test_local.py                # Local test runner
├── tests/                           # Test suite
│   ├── test_modma_mdd_real_experiment.py
│   └── test_run_modma_real.py
├── docs/                            # Documentation
│   ├── modma-experiment.md          # Experiment guidelines
│   └── plans/                       # Planning documents
├── MODMA_EEG_BIDS_format/           # BIDS dataset (resting state + dot probe)
│   ├── EEG_LZU_2015_2_resting state/
│   │   ├── participants.tsv         # Subject metadata (participant_id, group)
│   │   └── sub-*/
│   │       ├── eeg/                 # EDF files
│   │       └── behavioral data/
│   └── EEG_LZU_2015_2_dot probe/
├── 854301_EEG_*/                    # Raw EEG data directories
├── results/                         # Output: metrics, plots, reports
├── .planning/                       # GSD planning documents
└── .git/                            # Version control
```

## Directory Purposes

**Root Python Scripts:**
- Purpose: Entry points and utilities for running experiments
- Contains: Main pipeline runners, debug scripts, test runners
- Key files: `modma_mdd_real_experiment.py` (961 lines, core logic), `run_modma_real.py` (249 lines, simplified runner)

**`tests/`:**
- Purpose: Unit and integration tests
- Contains: Test fixtures, mocking, validation tests
- Key files: `test_modma_mdd_real_experiment.py` (594 lines), `test_run_modma_real.py` (115 lines)

**`docs/`:**
- Purpose: Documentation and planning
- Contains: Experiment guidelines, implementation plans
- Key files: `modma-experiment.md` (validation guidelines)

**`MODMA_EEG_BIDS_format/`:**
- Purpose: BIDS-formatted EEG dataset
- Contains: Resting-state and dot-probe EEG recordings in EDF format
- Structure: `sub-NNN/eeg/*.EDF` + `participants.tsv` with group labels (MDD/HC)

**`results/`:**
- Purpose: Output directory for metrics, plots, classification reports
- Contains: Generated during pipeline execution
- Generated: Yes
- Committed: No

## Key File Locations

**Entry Points:**
- `modma_mdd_real_experiment.py`: Full pipeline with argparse CLI, permutation testing, grid search
- `run_modma_real.py`: Simplified runner (hardcoded BIDS path, max_subjects=20)
- `modma_mdd_classification.py`: Synthetic data demo

**Configuration:**
- Environment variables: `MODMA_BIDS_ROOT` (default: `d:\eeg\MODMA_EEG_BIDS_format\EEG_LZU_2015_2_resting state`)
- Command-line args: `--bids-root`, `--max-subjects`, `--resample-sfreq`, `--n-permutations`, `--window-sec`, `--output-dir`, etc.

**Core Logic:**
- Data loading: `load_windows()` (lines 160-213), `load_real_modma_data()` (lines 50-137 in run_modma_real.py)
- Feature extraction: `extract_features()` (lines 215-284)
- Feature families: `compute_de_features()`, `compute_hjorth_features()`, `compute_connectivity_features()`, `compute_riemannian_features()`
- Classification: `build_feature_model_pipeline()`, cross-validation logic
- Evaluation: `run_evaluation()`, permutation testing loop

**Testing:**
- `tests/test_modma_mdd_real_experiment.py`: Tests for feature extraction, QC, validation
- `tests/test_run_modma_real.py`: Tests for simplified runner

## Naming Conventions

**Files:**
- Pattern: `modma_mdd_*.py` for main modules, `run_*.py` for runners, `test_*.py` for tests
- Example: `modma_mdd_real_experiment.py`, `run_modma_real.py`, `test_modma_mdd_real_experiment.py`

**Directories:**
- Pattern: Lowercase with underscores for code dirs (`tests/`, `docs/`), BIDS-compliant for data (`sub-NNN/eeg/`)
- Example: `MODMA_EEG_BIDS_format/`, `854301_EEG_128Channels_Resting_Lanzhou_2015/`

**Functions:**
- Pattern: snake_case, descriptive names with domain context
- Examples: `load_windows()`, `extract_features()`, `compute_connectivity_features()`, `build_region_indices()`

**Variables:**
- Pattern: snake_case for local/module vars, UPPERCASE for constants
- Examples: `REGIONS = ["frontal", "central", ...]`, `n_win`, `ch_names`, `band_power`

**Constants:**
- Pattern: UPPERCASE, module-level
- Examples: `SFREQ`, `N_FFT`, `ALPHA_FMIN`, `ALPHA_FMAX`, `CV_N_SPLITS`, `EGI128_REGION_MAP`

## Where to Add New Code

**New Feature Family:**
- Implementation: Add function `compute_<feature_name>_features(X, sfreq, region_idx)` in `/d/eeg/modma_mdd_real_experiment.py` (after line 415)
- Return: `(feature_array, feature_names)` tuple
- Integration: Call in `extract_features()` (line 256+), append to `features` and `feature_names` lists

**New Classifier:**
- Implementation: Add case to `build_feature_model_pipeline()` (line 418+)
- Pattern: `elif model_name == "new_model": steps.append(("clf", NewClassifier(...)))`
- Grid search: Add tuple to `_get_model_candidates()` (line 444+)

**New Validation Check:**
- Implementation: Add function `validate_<check_name>()` in `/d/eeg/modma_mdd_real_experiment.py`
- Pattern: Raise ValueError with descriptive message if check fails
- Integration: Call in `load_windows()` or `run_evaluation()` as appropriate

**New Test:**
- Location: `/d/eeg/tests/test_modma_mdd_real_experiment.py`
- Pattern: Use pytest, mock MNE objects, test with synthetic data
- Example: `def test_extract_features_shape()`, `def test_qc_mask_filtering()`

**Utilities/Helpers:**
- Shared helpers: Add to `/d/eeg/modma_mdd_real_experiment.py` module level
- Pattern: Prefix with `_` if internal (e.g., `_alpha_asymmetry()`)

## Special Directories

**`.planning/codebase/`:**
- Purpose: GSD-generated architecture and structure documents
- Generated: Yes (by GSD mapper)
- Committed: Yes

**`results/`:**
- Purpose: Pipeline output (metrics, plots, reports)
- Generated: Yes (during pipeline execution)
- Committed: No (in .gitignore)

**`MODMA_EEG_BIDS_format/`:**
- Purpose: BIDS dataset (read-only reference)
- Generated: No (external data)
- Committed: No (too large, in .gitignore)

---

*Structure analysis: 2026-02-22*
