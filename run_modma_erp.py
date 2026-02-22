"""MODMA ERP (Dot-probe hcue) subject-level classification.

Strict LOSO CV on 128ch ERP data, replicating the protocol from
'An End-to-End Depression Recognition Method Based on EEGNet' (2022).

Usage: python run_modma_erp.py
"""
import os, gc, warnings, numpy as np
from pathlib import Path

import mne
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")

# ── Config ──────────────────────────────────────────────────────
ERP_DIR = Path("854301_EEG_128Channels_ERP_Lanzhou_2015/EEG_128channels_ERP_lanzhou_2015")
CONDITION = "hcue"          # happy-neutral cue (best in paper)
TMIN, TMAX = -0.1, 0.5     # epoch window (sec)
BASELINE = (-0.1, 0.0)     # baseline correction interval
FMIN, FMAX = 0.5, 40.0     # bandpass filter
N_EEG_CH = 128             # first 128 channels are EEG


def load_erp_features():
    """Load subjects one at a time, extract features, free raw immediately."""
    raw_files = sorted(ERP_DIR.glob("*.raw"))
    print(f"Found {len(raw_files)} .raw files", flush=True)

    all_feat, all_labels, all_groups = [], [], []

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
                del raw; gc.collect()
                continue

            cond_events = events[events[:, 2] == event_id[CONDITION]]
            epochs = mne.Epochs(
                raw, cond_events, tmin=TMIN, tmax=TMAX,
                baseline=BASELINE, preload=True, verbose=False,
                reject=dict(eeg=150e-6)
            )
            data = epochs.get_data()  # (n_trials, 128, n_times)
            del raw, epochs; gc.collect()

            if data.shape[0] < 10:
                continue

            # Flatten downsampled epoch -> feature vector per trial
            feat = data[:, :, ::2].reshape(data.shape[0], -1)
            del data

            all_feat.append(feat)
            all_labels.extend([label] * feat.shape[0])
            all_groups.extend([sub_id] * feat.shape[0])
            print(f"  {sub_id}: label={label}, trials={feat.shape[0]}", flush=True)

        except Exception as e:
            print(f"  Skip {fname}: {e}", flush=True)
            gc.collect()

    X = np.concatenate(all_feat, axis=0)
    y = np.array(all_labels)
    groups = np.array(all_groups)
    print(f"\nTotal: {X.shape[0]} trials, {len(np.unique(groups))} subjects "
          f"(MDD={sum(y==1)}, HC={sum(y==0)})", flush=True)
    return X, y, groups


def run_loso(X, y, groups):
    """Strict LOSO, fixed hyperparams (no grid search to save memory)."""
    logo = LeaveOneGroupOut()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=30)),
        ("clf", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
    ])

    subj_ids, subj_true, subj_prob = [], [], []
    n_folds = len(np.unique(groups))
    print(f"\nRunning LOSO: {n_folds} folds...", flush=True)

    for i, (tr_idx, te_idx) in enumerate(logo.split(X, y, groups)):
        pipe.fit(X[tr_idx], y[tr_idx])
        prob = pipe.predict_proba(X[te_idx])[:, 1]

        sid = groups[te_idx][0]
        subj_prob_mean = prob.mean()
        subj_label = y[te_idx][0]

        subj_ids.append(sid)
        subj_true.append(subj_label)
        subj_prob.append(subj_prob_mean)

        pred = int(subj_prob_mean >= 0.5)
        status = "OK" if pred == subj_label else "MISS"
        print(f"  {i+1}/{n_folds}: {sid} true={subj_label} "
              f"prob={subj_prob_mean:.3f} [{status}]", flush=True)

    subj_true = np.array(subj_true)
    subj_prob = np.array(subj_prob)
    subj_pred = (subj_prob >= 0.5).astype(int)

    ba = balanced_accuracy_score(subj_true, subj_pred)
    auc = roc_auc_score(subj_true, subj_prob)

    print(f"\n{'='*50}")
    print(f"Subject-level results ({CONDITION} condition):")
    print(f"  BA  = {ba:.3f}")
    print(f"  AUC = {auc:.3f}")
    print(f"  Acc = {np.mean(subj_pred == subj_true):.3f}")
    print(f"  N   = {len(subj_true)} (MDD={sum(subj_true==1)}, HC={sum(subj_true==0)})")
    print(f"{'='*50}")
    return ba, auc


if __name__ == "__main__":
    X, y, groups = load_erp_features()
    print(f"Feature matrix: {X.shape}", flush=True)
    run_loso(X, y, groups)
