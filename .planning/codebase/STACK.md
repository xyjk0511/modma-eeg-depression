# Technology Stack

**Analysis Date:** 2026-02-22

## Languages

**Primary:**
- Python 3.x - All application code, data processing, and machine learning

## Runtime

**Environment:**
- Python 3.x (CPython)

**Package Manager:**
- pip
- Lockfile: Not detected

## Frameworks

**Core:**
- MNE-Python - EEG signal processing and analysis (`modma_mdd_real_experiment.py`, `run_modma_real.py`)
- scikit-learn - Machine learning pipelines and classifiers (`modma_mdd_real_experiment.py`, `run_modma_real.py`)

**Testing:**
- pytest - Test framework and runner (`tests/test_modma_mdd_real_experiment.py`, `tests/test_run_modma_real.py`)

**Build/Dev:**
- joblib - Parallel processing for permutation loops (`modma_mdd_real_experiment.py` line 23)

## Key Dependencies

**Critical:**
- numpy - Numerical computing and array operations (imported in all main modules)
- pandas - Data manipulation and BIDS participant file parsing (`modma_mdd_real_experiment.py` line 2, `run_modma_real.py` line 12)
- scipy - Signal processing (welch, csd functions from `scipy.signal`)
- matplotlib - Visualization and plotting (`modma_mdd_real_experiment.py` line 9-11)
- mne - EEG data I/O and preprocessing (MNE-Python core)

**Machine Learning:**
- scikit-learn - Pipeline, preprocessing, feature selection, model selection, metrics
  - StandardScaler - Feature normalization
  - SVC - Support Vector Classifier
  - LogisticRegression - Alternative classifier
  - SelectKBest, f_classif - Feature selection
  - PCA - Dimensionality reduction
  - StratifiedGroupKFold, GridSearchCV - Cross-validation strategies
  - roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score - Evaluation metrics

**Optional:**
- lightgbm - LGBMClassifier (optional, gracefully handled if missing)
  - Conditional import with HAS_LGBM flag (`modma_mdd_real_experiment.py` lines 25-29)

## Configuration

**Environment:**
- MODMA_BIDS_ROOT - Environment variable for BIDS dataset root path
  - Default fallback: `d:\eeg\MODMA_EEG_BIDS_format\EEG_LZU_2015_2_resting state` (`run_modma_real.py` lines 22-25)

**Command-line Arguments:**
- `--bids-root` - Path to MODMA BIDS dataset (required)
- `--max-subjects` - Maximum subjects to process (optional, default: None)
- `--resample-sfreq` - Resampling frequency in Hz (default: 125.0)
- `--n-permutations` - Number of permutations for significance testing (default: 1000)
- `--seed` - Random seed for reproducibility (default: 42)
- `--min-windows-per-subject` - Minimum windows per subject after QC (default: 3)
- `--crop-duration` - Crop duration in seconds (default: 60.0)
- `--window-sec` - Window size in seconds (default: 10.0)
- `--output-dir` - Output directory for results (default: "results")
- `--bad-amp-uv` - Amplitude threshold in microvolts (default: 200.0)
- `--max-bad-channels` - Maximum bad channels per window (default: 3)

## Platform Requirements

**Development:**
- Windows 11 (project developed on Windows, uses forward slashes in paths)
- Python 3.x with pip

**Production:**
- Any platform supporting Python 3.x and MNE-Python
- Requires BIDS-formatted EEG dataset in EDF format
- Minimum 32GB RAM (based on joblib n_jobs=4 configuration to avoid OOM)

---

*Stack analysis: 2026-02-22*
