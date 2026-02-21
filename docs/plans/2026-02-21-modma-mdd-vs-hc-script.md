# MODMA MDD vs HC Classification Script Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a reproducible Python script on MODMA EEG that objectively answers whether MDD and HC can be separated under strict cross-subject, leakage-free evaluation.

**Architecture:** Keep one end-to-end pipeline: `load (BIDS) -> quality control -> preprocess/window -> feature extraction -> grouped nested CV -> subject-level aggregation -> report`. All split operations must use `participant_id` as group. Normalization and dimensionality reduction must be fit only on training folds via sklearn `Pipeline`.

**Tech Stack:** Python, MNE, NumPy, Pandas, SciPy, scikit-learn, matplotlib, pytest, pytest-cov.

---

## Approaches (Choose One Before Coding)

1. **Quick baseline only (fastest)**
   - FAA + alpha PSD + RBF-SVM with grouped CV.
   - Pros: Lowest effort, immediate answer.
   - Cons: High risk of underfitting and unstable conclusions.

2. **Balanced feature set + nested grouped CV (Recommended)**
   - Add multiband PSD, relative power, asymmetry, and simple nonlinear stats.
   - Use nested `StratifiedGroupKFold` for tuning + unbiased outer score.
   - Pros: Strong methodology, still feasible in current repo.
   - Cons: More engineering and runtime.

3. **Deep learning prototype**
   - CNN/Transformer on windows with subject grouping.
   - Pros: Potential upside if enough data.
   - Cons: MODMA sample size likely too small; reproducibility weaker.

**Recommendation:** Option 2. It is the best tradeoff between scientific rigor and engineering cost in this codebase.

## Success Criteria (This Plan's "Done" Definition)

- Primary metric: **Subject-level Balanced Accuracy** (subject-grouped outer CV mean and 95% CI).
- Secondary metrics (subject level): ROC-AUC, F1, sensitivity, specificity.
- Statistical check: grouped permutation test (`p < 0.05`) where each permutation re-runs the full nested CV pipeline.
- Engineering check: new module test coverage `>= 80%`.
- Practical conclusion rule:
  - `BA >= 0.70` and CI lower bound `>= 0.60`: promising separability.
  - Otherwise: currently not reliable for clinical use; continue feature/model iteration.

---

### Task 1: Add New Real-Data Experiment Entry Script

**Files:**
- Create: `modma_mdd_real_experiment.py`
- Test: `tests/test_modma_mdd_real_experiment.py`

**Step 1: Write the failing test**

```python
def test_cli_parses_required_arguments():
    args = parse_args([
        "--bids-root", "D:/eeg/MODMA_EEG_BIDS_format",
        "--max-subjects", "20",
        "--resample-sfreq", "125",
        "--n-permutations", "1000",
        "--seed", "42",
        "--min-windows-per-subject", "4",
    ])
    assert args.bids_root.endswith("MODMA_EEG_BIDS_format")
    assert args.max_subjects == 20
    assert args.resample_sfreq == 125.0
    assert args.n_permutations == 1000
    assert args.seed == 42
    assert args.min_windows_per_subject == 4
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_modma_mdd_real_experiment.py::test_cli_parses_required_arguments -v`
Expected: FAIL (`parse_args` not defined).

**Step 3: Write minimal implementation**

```python
def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids-root", required=True)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--resample-sfreq", type=float, default=125.0)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-windows-per-subject", type=int, default=3)
    return parser.parse_args(argv)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_modma_mdd_real_experiment.py::test_cli_parses_required_arguments -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add modma_mdd_real_experiment.py tests/test_modma_mdd_real_experiment.py
git commit -m "feat: add modma real experiment CLI skeleton"
```

### Task 2: Robust Participant Metadata Parsing

**Files:**
- Modify: `modma_mdd_real_experiment.py`
- Test: `tests/test_modma_mdd_real_experiment.py`

**Step 1: Write the failing test**

```python
def test_load_participants_handles_irregular_whitespace(tmp_path):
    tsv = tmp_path / "participants.tsv"
    tsv.write_text("participant_id   group\nsub-001   MDD\nsub-025      HC\n", encoding="utf-8")
    df = load_participants(tsv)
    assert set(df["group"]) == {"MDD", "HC"}
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_modma_mdd_real_experiment.py::test_load_participants_handles_irregular_whitespace -v`
Expected: FAIL (`load_participants` missing).

**Step 3: Write minimal implementation**

```python
def load_participants(path):
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    df["participant_id"] = df["participant_id"].astype(str).str.strip()
    df["group"] = df["group"].astype(str).str.strip()
    return df[df["group"].isin(["MDD", "HC"])][["participant_id", "group"]]
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_modma_mdd_real_experiment.py::test_load_participants_handles_irregular_whitespace -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add modma_mdd_real_experiment.py tests/test_modma_mdd_real_experiment.py
git commit -m "feat: robust participants parser for MODMA"
```

### Task 2.5: Signal Quality Filtering Before Feature Extraction

**Files:**
- Modify: `modma_mdd_real_experiment.py`
- Test: `tests/test_modma_mdd_real_experiment.py`

**Step 1: Write the failing test**

```python
def test_rejects_windows_with_saturated_amplitude():
    # MNE unit is Volt. 200 uV threshold must be 200e-6.
    X = np.zeros((2, 128, 1250))
    X[1, 0, 10] = 300e-6
    keep_mask = build_quality_mask(X, bad_amp_uv=200.0, max_bad_channels=0)
    assert keep_mask.tolist() == [True, False]
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_modma_mdd_real_experiment.py::test_rejects_windows_with_saturated_amplitude -v`
Expected: FAIL (`build_quality_mask` missing).

**Step 3: Write minimal implementation**

```python
def build_quality_mask(X, bad_amp_uv=200.0, max_bad_channels=3):
    thr_v = bad_amp_uv * 1e-6
    bad_ch = np.any(np.abs(X) > thr_v, axis=2)
    bad_ch_count = bad_ch.sum(axis=1)
    return bad_ch_count <= max_bad_channels
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_modma_mdd_real_experiment.py::test_rejects_windows_with_saturated_amplitude -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add modma_mdd_real_experiment.py tests/test_modma_mdd_real_experiment.py
git commit -m "feat: add pre-feature signal quality filtering for bad channels"
```

### Task 2.6: Post-QC Data Availability Guards

**Files:**
- Modify: `modma_mdd_real_experiment.py`
- Test: `tests/test_modma_mdd_real_experiment.py`

**Step 1: Write the failing test**

```python
def test_raises_when_subject_has_too_few_windows_after_qc():
    groups = np.array(["s1", "s1", "s2", "s3"])
    y = np.array([1, 1, 0, 0])
    keep_mask = np.array([True, False, True, True])
    with pytest.raises(ValueError, match="min_windows_per_subject"):
        validate_post_qc_availability(groups, y, keep_mask, min_windows_per_subject=2)

def test_raises_when_class_has_too_few_subjects_after_qc():
    subject_labels = {"s1": 1, "s2": 0}
    with pytest.raises(ValueError, match="at least 2 subjects per class"):
        validate_subject_class_counts(subject_labels)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_modma_mdd_real_experiment.py -k "too_few_windows_after_qc or too_few_subjects_after_qc" -v`
Expected: FAIL.

**Step 3: Write minimal implementation**

```python
# enforce:
# 1) each subject keeps >= min_windows_per_subject windows
# 2) each class has >= 2 subjects after QC
# otherwise raise ValueError with explicit counts
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_modma_mdd_real_experiment.py -k "too_few_windows_after_qc or too_few_subjects_after_qc" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add modma_mdd_real_experiment.py tests/test_modma_mdd_real_experiment.py
git commit -m "feat: enforce post-qc minimum windows and class subject counts"
```

### Task 3: Subject-Grouped Window Extraction

**Files:**
- Modify: `modma_mdd_real_experiment.py`
- Test: `tests/test_modma_mdd_real_experiment.py`

**Step 1: Write the failing test**

```python
def test_window_extraction_returns_labels_and_groups(monkeypatch):
    # mock strategy: monkeypatch mne.io.read_raw_edf to return FakeRaw
    # FakeRaw provides filter/resample/get_data/ch_names/info/times
    participants_df = pd.DataFrame(
        {"participant_id": ["sub-001", "sub-025"], "group": ["MDD", "HC"]}
    )
    X, y, groups = load_windows(participants_df, bids_root="D:/fake", window_sec=10, resample_sfreq=125.0)
    assert len(X) == len(y) == len(groups)
    assert len(set(groups)) > 1
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_modma_mdd_real_experiment.py::test_window_extraction_returns_labels_and_groups -v`
Expected: FAIL.

**Step 3: Write minimal implementation**

```python
# read EDF -> filter/resample -> non-overlap windows
# use CLI param `resample_sfreq` (default 125.0)
# append one label and one participant_id per window
return X_windows, y_windows, group_ids
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_modma_mdd_real_experiment.py::test_window_extraction_returns_labels_and_groups -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add modma_mdd_real_experiment.py tests/test_modma_mdd_real_experiment.py
git commit -m "feat: grouped window extraction for MODMA EDF data"
```

### Task 4: Implement Feature Blocks (Baseline + Upgraded) + Optional Dimensionality Reduction

**Files:**
- Modify: `modma_mdd_real_experiment.py`
- Test: `tests/test_modma_mdd_real_experiment.py`

**Step 1: Write the failing test**

```python
def test_feature_matrix_has_expected_blocks():
    feats = extract_features(X_windows, sfreq=125.0)
    assert "alpha_asymmetry" in feats.feature_names
    assert feats.X.shape[0] == len(X_windows)

def test_pipeline_contains_scaler_and_optional_reducer():
    pipe = build_feature_model_pipeline(model_name="svm", reducer="pca")
    assert "scaler" in pipe.named_steps
    assert "pca" in pipe.named_steps
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_modma_mdd_real_experiment.py -k "feature_matrix_has_expected_blocks or pipeline_contains_scaler_and_optional_reducer" -v`
Expected: FAIL.

**Step 3: Write minimal implementation**

```python
# blocks:
# 1) band power (delta/theta/alpha/beta)
# 2) relative band power
# 3) frontal asymmetry (mapped EGI channels)
# 4) simple nonlinear features (std, hjorth mobility/complexity)
# optional reducers:
# - PCA(n_components=0.95)
# - SelectKBest(f_classif, k=64)
# NOTE: scaler/reducer are inside sklearn Pipeline, never fit globally.
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_modma_mdd_real_experiment.py -k "feature_matrix_has_expected_blocks or pipeline_contains_scaler_and_optional_reducer" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add modma_mdd_real_experiment.py tests/test_modma_mdd_real_experiment.py
git commit -m "feat: add multi-block EEG features for MODMA classification"
```

### Task 5: Leakage-Free Nested Cross-Subject Evaluation

**Files:**
- Modify: `modma_mdd_real_experiment.py`
- Test: `tests/test_modma_mdd_real_experiment.py`

**Step 1: Write the failing test**

```python
def test_evaluation_never_mixes_subjects_between_train_test():
    out = run_nested_group_cv(X, y, groups)
    assert out["leakage_detected"] is False
    assert "subject_level_metrics" in out

def test_classifier_uses_balanced_class_weights():
    pipe = build_feature_model_pipeline(model_name="svm", reducer="none")
    assert pipe.named_steps["clf"].class_weight == "balanced"

def test_subject_level_aggregation_is_used_for_primary_metrics():
    out = run_nested_group_cv(X, y, groups)
    assert out["primary_metric_level"] == "subject"

def test_training_uses_inverse_window_count_subject_weights():
    w = compute_subject_balanced_sample_weights(np.array(["s1", "s1", "s2"]))
    assert np.isclose(w[0], w[1])
    assert np.isclose(w[0] + w[1], w[2])
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_modma_mdd_real_experiment.py -k "evaluation_never_mixes_subjects_between_train_test or classifier_uses_balanced_class_weights or subject_level_aggregation_is_used_for_primary_metrics or inverse_window_count_subject_weights" -v`
Expected: FAIL.

**Step 3: Write minimal implementation**

```python
# outer: StratifiedGroupKFold
# inner: StratifiedGroupKFold + GridSearchCV
# model pipeline:
# - StandardScaler()
# - optional PCA/SelectKBest
# - classifier
# classifier candidates:
# - LogisticRegression(class_weight="balanced")
# - SVC(RBF, class_weight="balanced")
# in each outer fold:
# - compute sample_weight per training window as inverse subject window count
# - predict window probabilities on outer test set
# - aggregate to subject-level probability (mean over windows per subject)
# - compute primary metrics on aggregated subject predictions
# return window-level and subject-level metrics, with subject-level as primary
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_modma_mdd_real_experiment.py -k "evaluation_never_mixes_subjects_between_train_test or classifier_uses_balanced_class_weights or subject_level_aggregation_is_used_for_primary_metrics or inverse_window_count_subject_weights" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add modma_mdd_real_experiment.py tests/test_modma_mdd_real_experiment.py
git commit -m "feat: add nested grouped CV for unbiased MODMA evaluation"
```

### Task 6: Add Significance + Uncertainty Reporting

**Files:**
- Modify: `modma_mdd_real_experiment.py`
- Test: `tests/test_modma_mdd_real_experiment.py`

**Step 1: Write the failing test**

```python
def test_report_contains_ci_and_permutation_pvalue():
    report = build_report(..., n_permutations=200, seed=42)
    assert "balanced_accuracy_ci95" in report
    assert "permutation_pvalue" in report
    assert report["n_permutations"] == 200
    assert report["permutation_strategy"] == "full_nested_cv_rerun"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_modma_mdd_real_experiment.py::test_report_contains_ci_and_permutation_pvalue -v`
Expected: FAIL.

**Step 3: Write minimal implementation**

```python
# bootstrap by participant_id for CI
# grouped permutation test for chance-level check
# each permutation shuffles subject-level labels and reruns full nested CV
# n_permutations comes from CLI --n-permutations (default 1000)
# random state comes from CLI --seed
# return JSON-serializable dict
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_modma_mdd_real_experiment.py::test_report_contains_ci_and_permutation_pvalue -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add modma_mdd_real_experiment.py tests/test_modma_mdd_real_experiment.py
git commit -m "feat: add confidence intervals and permutation significance reporting"
```

### Task 7: Add Artifact Export and Reproducible Run Command

**Files:**
- Modify: `modma_mdd_real_experiment.py`
- Create: `results/.gitkeep`
- Test: `tests/test_modma_mdd_real_experiment.py`

**Step 1: Write the failing test**

```python
def test_main_writes_metrics_json(tmp_path):
    run_main_with_output_dir(tmp_path)
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "feature_importance.csv").exists()
    assert (tmp_path / "roc_curve.png").exists()
    assert (tmp_path / "confusion_matrix.png").exists()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_modma_mdd_real_experiment.py::test_main_writes_metrics_json -v`
Expected: FAIL.

**Step 3: Write minimal implementation**

```python
# write:
# - metrics.json
# - fold_scores.csv
# - predictions.csv
# - confusion_matrix.csv
# - feature_importance.csv (permutation importance on each outer test fold, then mean/std across folds)
# - roc_curve.png
# - confusion_matrix.png
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_modma_mdd_real_experiment.py::test_main_writes_metrics_json -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add modma_mdd_real_experiment.py tests/test_modma_mdd_real_experiment.py results/.gitkeep
git commit -m "feat: export reproducible MODMA experiment artifacts"
```

### Task 8: Full Verification and Real-Data Trial Run

**Files:**
- Modify: `README.md` (if absent, create `docs/modma-experiment.md`)

**Step 1: Run full test suite and coverage gate**

Run: `python -m pip install matplotlib pytest-cov`
Expected: dependencies available for plotting and coverage checks.

Run: `python -m pytest -q`
Expected: all PASS.

Run: `python -m pytest --cov=modma_mdd_real_experiment --cov-report=term-missing --cov-fail-under=80`
Expected: PASS with coverage >= 80%.

**Step 2: Run real MODMA experiment**

Run:

```bash
python modma_mdd_real_experiment.py --bids-root "d:\eeg\MODMA_EEG_BIDS_format\EEG_LZU_2015_2_resting state" --max-subjects 40 --crop-duration 60 --window-sec 10 --resample-sfreq 125 --min-windows-per-subject 3 --n-permutations 1000 --seed 42 --output-dir results/modma_run_01
```

Expected:
- Script exits 0.
- `results/modma_run_01/metrics.json` exists.
- `results/modma_run_01/feature_importance.csv` exists.
- `results/modma_run_01/roc_curve.png` exists.
- `results/modma_run_01/confusion_matrix.png` exists.
- Console prints grouped CV metrics, subject-level primary metrics, and permutation significance.

**Step 3: Verify deterministic reproducibility with fixed seed**

Run:

```bash
python modma_mdd_real_experiment.py --bids-root "d:\eeg\MODMA_EEG_BIDS_format\EEG_LZU_2015_2_resting state" --max-subjects 40 --crop-duration 60 --window-sec 10 --resample-sfreq 125 --min-windows-per-subject 3 --n-permutations 200 --seed 42 --output-dir results/modma_run_02
```

Expected:
- `results/modma_run_01/metrics.json` and `results/modma_run_02/metrics.json` have identical primary metrics.
- If not identical, log nondeterministic component and fix random_state plumbing.

**Step 4: Document interpretation rules**

Add runbook notes:
- If BA is near 0.5 and `p >= 0.05`, current feature set cannot reliably separate MDD vs HC.
- If BA improves but CI is wide, increase subjects and stabilize features.
- Never claim clinical usability from single-center retrospective results.

**Step 5: Commit**

```bash
git add docs/modma-experiment.md README.md
git commit -m "docs: add MODMA experiment runbook and interpretation rules"
```

---

## Execution Notes

- Keep `participant_id` grouping in every split path.
- Keep scaler/reducer/model in one sklearn `Pipeline` to avoid fold leakage.
- Primary metrics must be computed at subject level (aggregate window probabilities per subject first).
- Do not report only accuracy; always include balanced metrics and uncertainty.
- Default classifier settings must use `class_weight="balanced"`.
- Bad-channel threshold uses microvolt input but convert to Volt internally (`uV * 1e-6`).
- Grouped permutation test must rerun the full nested CV for every permutation.
- Use fixed random seeds and print them in output report for reproducibility.
- Preserve existing `run_modma_real.py` as baseline; new script is the experiment track.
