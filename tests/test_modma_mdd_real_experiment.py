import pytest
import numpy as np
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
    X, y, groups = load_windows(participants_df, bids_root="D:/fake", window_sec=10, resample_sfreq=125.0)
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
    X = np.random.randn(6, 5)
    y = np.array([0, 1, 0, 1, 0, 1])
    groups = np.array(["s1", "s2", "s3", "s4", "s5", "s6"])
    
    report = build_report(X, y, groups, cv_results, n_permutations=2, seed=42)
    assert "balanced_accuracy_ci95" in report
    assert "permutation_pvalue" in report
    assert report["n_permutations"] == 2
    assert report["permutation_strategy"] == "full_nested_cv_rerun"
