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
