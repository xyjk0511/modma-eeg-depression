# External Integrations

**Analysis Date:** 2026-02-22

## APIs & External Services

**None detected** - This is a standalone data analysis pipeline with no external API integrations.

## Data Storage

**Databases:**
- None - Uses local filesystem only

**File Storage:**
- Local filesystem only
  - Input: BIDS-formatted EEG dataset in EDF format
  - Location: Specified via `--bids-root` argument or `MODMA_BIDS_ROOT` environment variable
  - Format: BIDS (Brain Imaging Data Structure) with EDF files
  - Participant metadata: `participants.tsv` file in BIDS root (`run_modma_real.py` line 56)

**Caching:**
- None

## Data Input Formats

**EEG Data:**
- EDF (European Data Format) files
  - Read via MNE-Python: `mne.io.read_raw_edf()` (`run_modma_real.py` line 36)
  - Channels: EGI-128 or Standard 10-20 electrode systems
  - Sampling rates: Variable (resampled to specified frequency, default 125 Hz)

**Participant Metadata:**
- TSV (Tab-Separated Values) format
  - File: `participants.tsv` in BIDS root
  - Columns: `participant_id`, `group` (MDD or HC)
  - Parsed with pandas: `pd.read_csv(participants_file, sep=r'\s+')` (`run_modma_real.py` line 60)

## Authentication & Identity

**Auth Provider:**
- None - No authentication required

## Monitoring & Observability

**Error Tracking:**
- None

**Logs:**
- Python logging module
  - Logger: `logging.getLogger(__name__)` (all main modules)
  - Format: Configurable via `logging.basicConfig()`
  - Output: Console and optional log files

## Output & Results

**Results Storage:**
- Local filesystem
  - Location: Specified via `--output-dir` argument (default: `results/`)
  - Format: JSON metrics file (`metrics.json`)
  - Contents: Classification metrics, permutation test results, bootstrap confidence intervals
  - Example: `results_test40/metrics.json` (from git status)

**Visualization:**
- Matplotlib figures saved to output directory
  - Backend: Agg (non-interactive, suitable for headless environments)
  - Format: PNG/PDF (matplotlib default)

## CI/CD & Deployment

**Hosting:**
- Local execution only - No cloud deployment

**CI Pipeline:**
- None detected - Manual test execution via pytest

## Environment Configuration

**Required env vars:**
- `MODMA_BIDS_ROOT` - Optional, specifies BIDS dataset root (has default fallback)

**Secrets location:**
- None - No secrets required

## Data Flow

**Input Pipeline:**
1. Read BIDS `participants.tsv` to identify subjects and groups
2. Locate EDF files for each participant
3. Load EDF via MNE-Python
4. Preprocess: filter (0.5-45 Hz), resample to target frequency
5. Segment into windows (default 10 seconds)
6. Quality control: reject windows with saturated amplitude

**Processing Pipeline:**
1. Extract features: Power Spectral Density (PSD), frontal alpha asymmetry
2. Normalize features with StandardScaler
3. Optional feature selection with SelectKBest
4. Train SVM classifier with StratifiedGroupKFold cross-validation
5. Run permutation testing for significance (default 1000 permutations)
6. Calculate bootstrap confidence intervals

**Output Pipeline:**
1. Write metrics.json with classification results
2. Generate ROC curves and confusion matrices
3. Save plots to output directory

---

*Integration audit: 2026-02-22*
