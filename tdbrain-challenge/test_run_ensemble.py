import numpy as np

from run_ensemble import _fold_ba, _proba_to_encoded_labels


def test_proba_to_encoded_labels_handles_mdd_index_zero():
    """Thresholded MDD probability should map back to encoded class labels."""
    proba_mdd = np.array([0.9, 0.2, 0.5, 0.49])
    preds = _proba_to_encoded_labels(proba_mdd, mdd_idx=0, threshold=0.5)
    assert preds.tolist() == [0, 1, 0, 1]


def test_proba_to_encoded_labels_handles_mdd_index_one():
    """Mapping should also work when MDD is encoded as class 1."""
    proba_mdd = np.array([0.9, 0.2, 0.5, 0.49])
    preds = _proba_to_encoded_labels(proba_mdd, mdd_idx=1, threshold=0.5)
    assert preds.tolist() == [1, 0, 1, 0]


def test_fold_ba_with_encoded_predictions_is_consistent():
    """Fold BA should be computed from encoded labels, not raw binary flags."""
    y_enc = np.array([0, 0, 1, 1])  # mdd_idx=0
    preds_enc = np.array([0, 1, 1, 1])
    ba = _fold_ba(y_enc, preds_enc, mdd_idx=0, test_idx=np.arange(4))
    assert ba == 0.75
