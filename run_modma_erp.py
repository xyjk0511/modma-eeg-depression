"""MODMA ERP (Dot-probe hcue) subject-level classification.

P300 features: mean amplitude in 250-500ms window per channel, computed from
the average ERP (mean across trials per subject). One vector per subject.
Reference: Li et al. 2018, CMPB — ERP waveform shape features for MDD classification.

Usage: python run_modma_erp.py
"""
import os, gc, warnings, numpy as np
from pathlib import Path

import mne
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")

# ── Config ──────────────────────────────────────────────────────
ERP_DIR = Path("854301_EEG_128Channels_ERP_Lanzhou_2015/EEG_128channels_ERP_lanzhou_2015")
CONDITION = "hcue"          # overridden in __main__ loop
TMIN, TMAX = -0.1, 0.5     # epoch window (sec)
BASELINE = (-0.1, 0.0)     # baseline correction
FMIN, FMAX = 0.5, 40.0     # bandpass filter
N_EEG_CH = 128
P300_WIN = (0.25, 0.50)    # P300 time window (sec)


def load_erp_features():
    """One feature vector per subject: mean P300 amplitude per channel."""
    raw_files = sorted(ERP_DIR.glob("*.raw"))
    print(f"Found {len(raw_files)} .raw files", flush=True)

    all_feat, all_labels, all_ids = [], [], []

    for fpath in raw_files:
        fname = fpath.name
        if fname.startswith("0201"):
            label = 1  # MDD
        elif fname.startswith("0202") or fname.startswith("0203"):
            label = 0  # HC
        else:
            continue

        sub_id = fname.split("erp")[0].strip().replace("_", "")

        try:
            raw = mne.io.read_raw_egi(str(fpath), preload=True, verbose=False)
            raw.pick(raw.ch_names[:N_EEG_CH])
            raw.filter(FMIN, FMAX, verbose=False)

            events, event_id = mne.events_from_annotations(raw, verbose=False)
            if CONDITION not in event_id:
                del raw; gc.collect(); continue

            cond_events = events[events[:, 2] == event_id[CONDITION]]
            epochs = mne.Epochs(
                raw, cond_events, tmin=TMIN, tmax=TMAX,
                baseline=BASELINE, preload=True, verbose=False,
                reject=dict(eeg=150e-6)
            )
            data = epochs.get_data()   # (n_trials, 128, n_times)
            times = epochs.times
            del raw, epochs; gc.collect()

            if data.shape[0] < 10:
                del data; continue

            n_trials = data.shape[0]
            # Average ERP across trials, extract mean amplitude in P300 window
            avg_erp = data.mean(axis=0)                          # (128, n_times)
            p300_mask = (times >= P300_WIN[0]) & (times <= P300_WIN[1])
            feat = avg_erp[:, p300_mask].mean(axis=1)            # (128,)
            del data, avg_erp

            all_feat.append(feat)
            all_labels.append(label)
            all_ids.append(sub_id)
            print(f"  {sub_id}: label={label}, trials={n_trials}", flush=True)

        except Exception as e:
            print(f"  Skip {fname}: {e}", flush=True)
            gc.collect()

    X = np.array(all_feat)    # (n_subjects, 128)
    y = np.array(all_labels)
    ids = np.array(all_ids)
    n_mdd = sum(y == 1); n_hc = sum(y == 0)
    print(f"\nTotal: {len(y)} subjects (MDD={n_mdd}, HC={n_hc})", flush=True)
    return X, y, ids


def run_loso(X, y, ids):
    """Leave-one-subject-out on subject-level features."""
    loo = LeaveOneOut()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=20)),
        ("clf",    LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
    ])

    true_labels, probs = [], []
    n = len(y)
    print(f"\nRunning LOSO: {n} folds...", flush=True)

    for i, (tr_idx, te_idx) in enumerate(loo.split(X)):
        pipe.fit(X[tr_idx], y[tr_idx])
        prob = pipe.predict_proba(X[te_idx])[:, 1][0]
        true_labels.append(y[te_idx][0])
        probs.append(prob)

        pred = int(prob >= 0.5)
        status = "OK" if pred == y[te_idx][0] else "MISS"
        print(f"  {i+1}/{n}: {ids[te_idx][0]} true={y[te_idx][0]} "
              f"prob={prob:.3f} [{status}]", flush=True)

    true_labels = np.array(true_labels)
    probs = np.array(probs)
    preds = (probs >= 0.5).astype(int)

    ba  = balanced_accuracy_score(true_labels, preds)
    auc = roc_auc_score(true_labels, probs)

    print(f"\n{'='*50}")
    print(f"Subject-level results ({CONDITION} condition, avg-ERP P300):")
    print(f"  BA  = {ba:.3f}")
    print(f"  AUC = {auc:.3f}")
    print(f"  Acc = {np.mean(preds == true_labels):.3f}")
    print(f"  N   = {len(true_labels)} (MDD={sum(true_labels==1)}, HC={sum(true_labels==0)})")
    print(f"{'='*50}")
    return ba, auc


if __name__ == "__main__":
    for cond in ["hcue", "fcue", "scue"]:
        CONDITION = cond
        X, y, ids = load_erp_features()
        print(f"Feature matrix: {X.shape}", flush=True)
        run_loso(X, y, ids)
