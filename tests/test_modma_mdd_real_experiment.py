import pytest
import numpy as np
import os
from modma_mdd_real_experiment import parse_args

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

def test_load_participants_handles_irregular_whitespace(tmp_path):
    # We will import it locally so the first test still passes if load_participants is missing, 
    # but this test will fail during execution.
    from modma_mdd_real_experiment import load_participants
    tsv = tmp_path / "participants.tsv"
    tsv.write_text("participant_id   group\nsub-001   MDD\nsub-025      HC\n", encoding="utf-8")
    df = load_participants(tsv)
    assert set(df["group"]) == {"MDD", "HC"}

def test_rejects_windows_with_saturated_amplitude():
    from modma_mdd_real_experiment import build_quality_mask
    # MNE unit is Volt. 200 uV threshold must be 200e-6.
    X = np.zeros((2, 128, 1250))
    X[1, 0, 10] = 300e-6
    keep_mask = build_quality_mask(X, bad_amp_uv=200.0, max_bad_channels=0)
    assert keep_mask.tolist() == [True, False]

def test_raises_when_subject_has_too_few_windows_after_qc():
    from modma_mdd_real_experiment import validate_post_qc_availability
    groups = np.array(["s1", "s1", "s2", "s3"])
    y = np.array([1, 1, 0, 0])
    keep_mask = np.array([True, False, True, True])
    with pytest.raises(ValueError, match="min_windows_per_subject"):
        validate_post_qc_availability(groups, y, keep_mask, min_windows_per_subject=2)

def test_raises_when_class_has_too_few_subjects_after_qc():
    from modma_mdd_real_experiment import validate_subject_class_counts
    subject_labels = {"s1": 1, "s2": 0}
    with pytest.raises(ValueError, match="at least 2 subjects per class"):
        validate_subject_class_counts(subject_labels)

def test_window_extraction_returns_labels_and_groups(monkeypatch):
    from modma_mdd_real_experiment import load_windows
    import mne
    import pandas as pd
    
    class FakeRaw:
        def __init__(self, *args, **kwargs):
            self.info = {"sfreq": 250.0, "ch_names": ["Fp1", "Fp2"]}
            self.times = np.arange(0, 30, 1/250.0)
        def load_data(self):
            return self
        def copy(self):
            return self
        def crop(self, *args, **kwargs):
            return self
        def filter(self, *args, **kwargs):
            return self
        def resample(self, sfreq, *args, **kwargs):
            self.info["sfreq"] = sfreq
            return self
        def get_data(self, units="uV"):
            return np.ones((2, int(30 * self.info["sfreq"])))
    
    monkeypatch.setattr(mne.io, "read_raw_edf", FakeRaw)
    import glob
    monkeypatch.setattr(glob, "glob", lambda x: ["fake.edf"])
    
    participants_df = pd.DataFrame(
        {"participant_id": ["sub-001", "sub-025"], "group": ["MDD", "HC"]}
    )
    X, y, groups, no_edf, ch_names = load_windows(participants_df, bids_root="D:/fake", window_sec=10, resample_sfreq=125.0)
    assert len(X) == len(y) == len(groups)
    assert len(set(groups)) > 1

def test_feature_matrix_has_expected_blocks():
    from modma_mdd_real_experiment import extract_features
    X_windows = np.random.randn(2, 128, 1250)
    feats_X, feat_names = extract_features(X_windows, sfreq=125.0)
    assert any("alpha_asymmetry" in name for name in feat_names)
    assert feats_X.shape[0] == len(X_windows)

def test_pipeline_contains_scaler_and_optional_reducer():
    from modma_mdd_real_experiment import build_feature_model_pipeline
    pipe = build_feature_model_pipeline(model_name="svm", reducer="pca")
    assert "scaler" in pipe.named_steps
    assert "pca" in pipe.named_steps

def test_evaluation_never_mixes_subjects_between_train_test():
    from modma_mdd_real_experiment import run_nested_group_cv
    X = np.random.randn(10, 5)
    y = np.array([0, 1] * 5)
    groups = np.array(["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4", "s5", "s5"])
    out = run_nested_group_cv(X, y, groups, n_splits=5)
    assert out["leakage_detected"] is False
    assert "subject_level_metrics" in out

def test_classifier_uses_balanced_class_weights():
    from modma_mdd_real_experiment import build_feature_model_pipeline
    pipe = build_feature_model_pipeline(model_name="svm", reducer="none")
    assert pipe.named_steps["clf"].class_weight == "balanced"

def test_subject_level_aggregation_is_used_for_primary_metrics():
    from modma_mdd_real_experiment import run_nested_group_cv
    X = np.random.randn(10, 5)
    y = np.array([0, 1] * 5)
    groups = np.array(["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4", "s5", "s5"])
    out = run_nested_group_cv(X, y, groups, n_splits=5)
    assert out["primary_metric_level"] == "subject"

def test_training_uses_inverse_window_count_subject_weights():
    from modma_mdd_real_experiment import compute_subject_balanced_sample_weights
    w = compute_subject_balanced_sample_weights(np.array(["s1", "s1", "s2"]))
    assert np.isclose(w[0], w[1])
    assert np.isclose(w[0] + w[1], w[2])

def test_report_contains_ci_and_permutation_pvalue():
    from modma_mdd_real_experiment import build_report
    cv_results = {
        "subject_level_metrics": {
            "balanced_accuracy": 0.75,
            "roc_auc": 0.80,
            "f1": 0.70,
        },
        "y_subj_true": np.array([0, 1, 0, 1, 0, 1]),
        "y_subj_prob": np.array([0.2, 0.8, 0.4, 0.6, 0.3, 0.7]),
        "subj_list": ["s1", "s2", "s3", "s4", "s5", "s6"]
    }
    X = np.random.randn(6, 4, 500)
    y = np.array([0, 1, 0, 1, 0, 1])
    groups = np.array(["s1", "s2", "s3", "s4", "s5", "s6"])
    ch_names = [f"ch{i}" for i in range(4)]

    report = build_report(X, y, groups, cv_results, ch_names=ch_names, sfreq=125.0,
                          n_permutations=2, seed=42)
    assert "balanced_accuracy_ci95" in report
    assert "permutation_pvalue" in report
    assert report["n_permutations"] == 2
    assert report["permutation_strategy"] == "full_model_selection_rerun"

def test_main_writes_metrics_json(tmp_path, monkeypatch):
    from modma_mdd_real_experiment import run_main_with_output_dir
    import mne, glob, pandas as pd
    
    class FakeRaw:
        def __init__(self, *args, **kwargs):
            self.info = {"sfreq": 250.0, "ch_names": ["Fp1", "Fp2"]}
            self.times = np.arange(0, 30, 1/250.0)
        def load_data(self): return self
        def copy(self): return self
        def crop(self, *args, **kwargs): return self
        def filter(self, *args, **kwargs): return self
        def resample(self, sfreq, *args, **kwargs):
            self.info["sfreq"] = sfreq
            return self
        def get_data(self, units="uV"):
            return np.ones((2, int(30 * self.info["sfreq"])))
            
    monkeypatch.setattr(mne.io, "read_raw_edf", FakeRaw)
    monkeypatch.setattr(glob, "glob", lambda x: ["fake.edf"] if "eeg" in x else [x])
    
    def fake_load_participants(path):
        return pd.DataFrame({
            "participant_id": ["s1", "s2", "s3", "s4"],
            "group": ["MDD", "HC", "MDD", "HC"]
        })
    import modma_mdd_real_experiment
    monkeypatch.setattr(modma_mdd_real_experiment, "load_participants", fake_load_participants)
    _real_exists = os.path.exists
    monkeypatch.setattr(os.path, "exists", lambda p: True if "participants" in p else _real_exists(p))

    run_main_with_output_dir(
        bids_root="D:/fake",
        output_dir=str(tmp_path),
        max_subjects=4,
        resample_sfreq=125.0,
        n_permutations=2,
        seed=42,
        min_windows_per_subject=2,
        window_sec=10,
        crop_duration=60
    )
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "feature_importance.csv").exists()
    assert (tmp_path / "roc_curve.png").exists()
    assert (tmp_path / "confusion_matrix.png").exists()

    # Verify feature_importance.csv contains real f_classif scores, not random
    imp_df = pd.read_csv(tmp_path / "feature_importance.csv")
    assert len(imp_df) > 0
    assert not imp_df["importance"].between(0, 1).all(), "Looks like random [0,1) values, not F-scores"


# --- Regression tests for 6 code-review findings ---

def test_stratified_sampling_ensures_both_classes():
    """Critical #1: head(max_subjects) could yield single-class subset."""
    from modma_mdd_real_experiment import run_main_with_output_dir
    import pandas as pd
    import modma_mdd_real_experiment as mod

    # Simulate a participants list where all MDD come first
    orig_load = mod.load_participants
    def fake_load(path):
        return pd.DataFrame({
            "participant_id": [f"mdd{i}" for i in range(10)] + [f"hc{i}" for i in range(10)],
            "group": ["MDD"] * 10 + ["HC"] * 10,
        })

    mod.load_participants = fake_load
    try:
        # Patch os.path.exists so participants.tsv check passes
        _real = os.path.exists
        os.path.exists = lambda p: True if "participants" in p else _real(p)

        # We only need to test the selection logic, so patch load_windows to inspect the df
        captured = {}
        orig_load_windows = mod.load_windows
        def spy_load_windows(df, **kw):
            captured["groups"] = set(df["group"])
            raise StopIteration("spy done")
        mod.load_windows = spy_load_windows

        with pytest.raises(StopIteration):
            run_main_with_output_dir("fake", "out", max_subjects=4,
                resample_sfreq=125, n_permutations=0, seed=42,
                min_windows_per_subject=1, window_sec=10, crop_duration=60)
        assert captured["groups"] == {"MDD", "HC"}
    finally:
        mod.load_participants = orig_load
        mod.load_windows = orig_load_windows
        os.path.exists = _real


def test_feature_importance_not_random():
    """Critical #2: importance must come from f_classif, not np.random."""
    from modma_mdd_real_experiment import extract_features
    from sklearn.feature_selection import f_classif as real_f_classif
    import inspect, modma_mdd_real_experiment as mod

    # Verify f_classif is called in run_main_with_output_dir (not np.random.rand)
    src = inspect.getsource(mod.run_main_with_output_dir)
    assert "f_classif(features" in src
    assert "np.random.rand" not in src

    # Verify f_classif produces deterministic, data-dependent scores
    X = np.random.RandomState(0).randn(20, 4, 500)
    y = np.array([0]*10 + [1]*10)
    feats, names = extract_features(X, sfreq=125.0)
    f1, _ = real_f_classif(feats, y)
    f2, _ = real_f_classif(feats, y)
    np.testing.assert_array_almost_equal(f1, f2)


def test_missing_participants_tsv_raises():
    """Important #5: must raise FileNotFoundError, not fall back to fake.tsv."""
    from modma_mdd_real_experiment import run_main_with_output_dir
    with pytest.raises(FileNotFoundError, match="participants.tsv not found"):
        run_main_with_output_dir("nonexistent_dir", "out", max_subjects=None,
            resample_sfreq=125, n_permutations=0, seed=42,
            min_windows_per_subject=1, window_sec=10, crop_duration=60)


def test_validate_subject_class_counts_is_called(monkeypatch):
    """Important #4: validate_subject_class_counts must be wired into the pipeline."""
    import modma_mdd_real_experiment as mod
    import inspect
    # Verify the function is called inside run_main_with_output_dir source
    src = inspect.getsource(mod.run_main_with_output_dir)
    assert "validate_subject_class_counts" in src


def test_roc_curve_uses_real_predictions(tmp_path, monkeypatch):
    """Important #3: ROC plot must use actual model predictions."""
    import modma_mdd_real_experiment as mod
    import mne, glob, pandas as pd, json

    class FakeRaw:
        def __init__(self, *a, **kw):
            self.info = {"sfreq": 250.0, "ch_names": ["Fp1", "Fp2"]}
            self.times = np.arange(0, 30, 1/250.0)
        def load_data(self): return self
        def copy(self): return self
        def crop(self, *a, **kw): return self
        def filter(self, *a, **kw): return self
        def resample(self, sfreq, *a, **kw):
            self.info["sfreq"] = sfreq
            return self
        def get_data(self, units="uV"):
            return np.ones((2, int(30 * self.info["sfreq"])))

    monkeypatch.setattr(mne.io, "read_raw_edf", FakeRaw)
    monkeypatch.setattr(glob, "glob", lambda x: ["fake.edf"] if "eeg" in x else [x])
    monkeypatch.setattr(mod, "load_participants",
        lambda p: pd.DataFrame({"participant_id": ["s1","s2","s3","s4"], "group": ["MDD","HC","MDD","HC"]}))
    _real = os.path.exists
    monkeypatch.setattr(os.path, "exists", lambda p: True if "participants" in p else _real(p))

    mod.run_main_with_output_dir("fake", str(tmp_path), max_subjects=4,
        resample_sfreq=125, n_permutations=0, seed=42,
        min_windows_per_subject=2, window_sec=10, crop_duration=60)

    report = json.loads((tmp_path / "metrics.json").read_text())
    # If ROC used real data, roc_auc should be a finite number (not necessarily 0.5)
    assert isinstance(report["roc_auc"], float)
    assert 0.0 <= report["roc_auc"] <= 1.0


def test_inner_cv_tunes_hyperparameters():
    """Important #6: run_nested_group_cv must contain inner CV (GridSearchCV)."""
    import inspect
    from modma_mdd_real_experiment import run_nested_group_cv
    src = inspect.getsource(run_nested_group_cv)
    assert "GridSearchCV" in src


def test_no_edf_files_raises_clear_error(tmp_path, monkeypatch):
    """No valid EDF → clear ValueError, not AxisError."""
    from modma_mdd_real_experiment import run_main_with_output_dir
    import modma_mdd_real_experiment as mod
    import pandas as pd

    monkeypatch.setattr(mod, "load_participants",
        lambda p: pd.DataFrame({"participant_id": ["s1", "s2"], "group": ["MDD", "HC"]}))
    _real = os.path.exists
    monkeypatch.setattr(os.path, "exists", lambda p: True if "participants" in p else _real(p))

    with pytest.raises(ValueError, match="No valid EDF windows loaded"):
        run_main_with_output_dir(str(tmp_path), str(tmp_path / "out"), max_subjects=None,
            resample_sfreq=125, n_permutations=0, seed=42,
            min_windows_per_subject=1, window_sec=10, crop_duration=60)


def test_single_class_all_subjects_raises():
    """validate_subject_class_counts must reject single-class data."""
    from modma_mdd_real_experiment import validate_subject_class_counts
    with pytest.raises(ValueError, match="need at least 2 distinct classes"):
        validate_subject_class_counts({"s1": 0, "s2": 0, "s3": 0})


def test_qc_report_has_detailed_drop_reasons():
    from modma_mdd_real_experiment import generate_qc_report
    import pandas as pd
    participants_df = pd.DataFrame({
        "participant_id": ["s1", "s2", "s3", "s4"],
        "group": ["MDD", "HC", "MDD", "HC"]
    })
    groups = np.array(["s1", "s1", "s1", "s2", "s2", "s2", "s3"])
    # s1: 3 windows, all kept; s2: 3 windows, none kept; s3: 1 window, kept but < min
    keep_mask = np.array([True, True, True, False, False, False, True])
    no_edf = [("s4", "HC")]
    report = generate_qc_report(participants_df, groups, keep_mask, no_edf, min_windows_per_subject=3)
    reasons = dict(zip(report["subject"], report["drop_reason"]))
    assert reasons["s4"] == "no_edf"
    assert reasons["s2"] == "dropped_by_amp"
    assert reasons["s3"] == "dropped_by_min_windows"
    assert reasons["s1"] == "kept"
    assert set(report.columns) == {"subject", "label", "raw_windows", "kept_windows", "drop_reason"}


def test_cli_accepts_bad_amp_uv_and_max_bad_channels():
    args = parse_args(["--bids-root", "fake", "--bad-amp-uv", "300", "--max-bad-channels", "5"])
    assert args.bad_amp_uv == 300.0
    assert args.max_bad_channels == 5


def test_region_mapping_invariant_to_channel_order():
    from modma_mdd_real_experiment import extract_features, build_region_indices
    rng = np.random.RandomState(0)
    ch_names = [f"E{i}" for i in range(1, 129)]
    X = rng.randn(3, 128, 500)
    feats1, names1 = extract_features(X, sfreq=125.0, ch_names=ch_names)
    # Shuffle channel order
    perm = rng.permutation(128)
    X2 = X[:, perm, :]
    ch_names2 = [ch_names[i] for i in perm]
    feats2, names2 = extract_features(X2, sfreq=125.0, ch_names=ch_names2)
    assert names1 == names2
    np.testing.assert_allclose(feats1, feats2, rtol=1e-5)


def test_region_aggregation_reduces_dimensionality():
    from modma_mdd_real_experiment import extract_features
    ch_names = [f"E{i}" for i in range(1, 129)]
    X = np.random.RandomState(1).randn(2, 128, 500)
    feats, names = extract_features(X, sfreq=125.0, ch_names=ch_names)
    # Should be ~46 features (5 regions * 9 + 1 asymmetry), not 1153
    assert feats.shape[1] < 150
    assert feats.shape[1] == len(names)


def test_de_features_shape():
    from modma_mdd_real_experiment import compute_de_features, build_region_indices
    ch_names = [f"E{i}" for i in range(1, 129)]
    region_idx = build_region_indices(ch_names)
    X = np.random.RandomState(0).randn(3, 128, 500)
    feats, names = compute_de_features(X, sfreq=125.0, region_idx=region_idx)
    assert feats.shape == (3, 20)  # 5 regions * 4 bands
    assert len(names) == 20


def test_hjorth_features_shape():
    from modma_mdd_real_experiment import compute_hjorth_features, build_region_indices
    ch_names = [f"E{i}" for i in range(1, 129)]
    region_idx = build_region_indices(ch_names)
    X = np.random.RandomState(0).randn(3, 128, 500)
    feats, names = compute_hjorth_features(X, region_idx=region_idx)
    assert feats.shape == (3, 15)  # 5 regions * 3 (activity, mobility, complexity)
    assert len(names) == 15


def test_permutation_reruns_full_selection():
    """Permutation must rerun full model selection, not just nested CV."""
    import inspect
    from modma_mdd_real_experiment import build_report
    src = inspect.getsource(build_report)
    assert "run_full_model_selection" in src
    assert "run_nested_group_cv" not in src


def test_full_model_selection_returns_fold_results():
    from modma_mdd_real_experiment import run_full_model_selection
    rng = np.random.RandomState(0)
    X = rng.randn(20, 4, 500)
    y = np.array([0]*10 + [1]*10)
    groups = np.array([f"s{i//2}" for i in range(20)])
    ch = [f"ch{i}" for i in range(4)]
    out = run_full_model_selection(X, y, groups, ch, 125.0,
                                   bad_amp_candidates=(9999,),
                                   max_bad_channels=999,
                                   min_windows_per_subject=2,
                                   n_splits=2, seed=42)
    assert "fold_results" in out
    assert "subject_level_metrics" in out
    assert len(out["fold_results"]) > 0


def test_negative_conclusion_when_ci_lower_at_chance():
    from modma_mdd_real_experiment import determine_conclusion
    report = {
        "balanced_accuracy": 0.55,
        "balanced_accuracy_ci95": (0.40, 0.70),
        "permutation_pvalue": 0.30,
    }
    subject_labels = {f"s{i}": i % 2 for i in range(10)}
    result = determine_conclusion(report, subject_labels)
    assert result["conclusion"] == "negative"
    assert result["criteria"]["ci_lower_gt_0.5"] is False


def test_positive_conclusion_when_all_criteria_met():
    from modma_mdd_real_experiment import determine_conclusion
    report = {
        "balanced_accuracy": 0.75,
        "balanced_accuracy_ci95": (0.60, 0.90),
        "permutation_pvalue": 0.01,
    }
    subject_labels = {f"s{i}": i % 2 for i in range(12)}
    result = determine_conclusion(report, subject_labels)
    assert result["conclusion"] == "positive"
