"""MODMA ERP (Dot-probe hcue) subject-level classification.

P300 features: mean amplitude in 250-500ms window per channel, computed from
the average ERP (mean across trials per subject). One vector per subject.
Reference: Li et al. 2018, CMPB — ERP waveform shape features for MDD classification.

Usage: python run_modma_erp.py
"""
import os, gc, re, warnings, numpy as np
from pathlib import Path
from joblib import Parallel, delayed

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
N200_WIN = (0.10, 0.25)    # N200 time window (sec)


def get_parietal_indices(targets=("Pz", "P3", "P4", "Cz", "CPz")):
    """Return {name: channel_index} for EGI HydroCel-128 montage."""
    montage = mne.channels.make_standard_montage("GSN-HydroCel-128")
    name_to_idx = {n: i for i, n in enumerate(montage.ch_names)}
    result = {}
    for t in targets:
        if t in name_to_idx:
            result[t] = name_to_idx[t]
            print(f"  {t}: index={result[t]}", flush=True)
        else:
            ref = mne.channels.make_standard_montage("standard_1020")
            ref_pos = dict(zip(ref.ch_names, ref.get_positions()["ch_pos"].values()))
            gsn_pos = montage.get_positions()["ch_pos"]
            if t not in ref_pos:
                raise ValueError(f"{t} not in standard_1020")
            tp = ref_pos[t]
            nearest = min(gsn_pos, key=lambda n: np.linalg.norm(gsn_pos[n] - tp))
            result[t] = name_to_idx[nearest]
            print(f"  {t} -> nearest GSN: {nearest} (idx={result[t]})", flush=True)
    return result


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

        m = re.match(r'(\d{8})', fname)
        sub_id = m.group(1) if m else fname.split("erp")[0].strip().replace("_", "")

        try:
            raw = mne.io.read_raw_egi(str(fpath), preload=True, verbose=False)
            raw.pick(raw.ch_names[:N_EEG_CH])
            raw.filter(FMIN, FMAX, verbose=False)

            events, event_id = mne.events_from_annotations(raw, verbose=False)
            if CONDITION not in event_id:
                print(f"  Skip {fname}: condition '{CONDITION}' not found", flush=True)
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
                print(f"  Skip {fname}: only {data.shape[0]} trials (<10)", flush=True)
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


def load_erp_features_dict(condition):
    """Load features as dict[sub_id -> (feat, label)] for multi-condition fusion."""
    global CONDITION
    CONDITION = condition
    X, y, ids = load_erp_features()
    return {sid: (X[i], y[i]) for i, sid in enumerate(ids)}


def fuse_conditions(cond_dicts):
    """Intersect sub_ids across conditions and hstack features."""
    common = sorted(set.intersection(*[set(d.keys()) for d in cond_dicts]))
    X = np.hstack([np.array([d[s][0] for s in common]) for d in cond_dicts])
    y = np.array([cond_dicts[0][s][1] for s in common])
    ids = np.array(common)
    return X, y, ids


def _loso_ba_subset(X, y):
    """LOSO BA for low-dim subsets; PCA n_components capped at feature count."""
    nc = min(X.shape[1], len(y) - 1)
    loo = LeaveOneOut()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=nc)),
        ("clf",    LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
    ])
    preds = []
    for tr_idx, te_idx in loo.split(X):
        pipe.fit(X[tr_idx], y[tr_idx])
        preds.append(int(pipe.predict(X[te_idx])[0]))
    return balanced_accuracy_score(y, preds)


def permutation_test_subset(X, y, n_perm=1000, seed=42):
    """Permutation test using _loso_ba_subset."""
    assert len(y) == X.shape[0]
    rng = np.random.default_rng(seed)
    obs_ba = _loso_ba_subset(X, y)
    seeds = rng.integers(0, 2**31, size=n_perm)

    def one_perm(s):
        return _loso_ba_subset(X, np.random.default_rng(s).permutation(y))

    print(f"Running {n_perm} permutations (n_jobs=4)...", flush=True)
    perm_bas = Parallel(n_jobs=4)(delayed(one_perm)(s) for s in seeds)
    perm_arr = np.array(perm_bas)
    p_val = float((np.sum(perm_arr >= obs_ba) + 1) / (n_perm + 1))
    print(f"  Observed BA={obs_ba:.3f}  p={p_val:.4f}  "
          f"(perm mean={np.mean(perm_arr):.3f})", flush=True)
    return obs_ba, p_val


def run_electrode_selection(X, y, ids):
    """Phase 10: compare 3-dim and 5-dim parietal subsets vs 128-dim baseline."""
    print("\n" + "="*50)
    print("Phase 10: Electrode Selection")
    print("="*50)
    print("\nEGI HydroCel-128 parietal channel indices:")
    idx = get_parietal_indices()

    ch3 = ["Pz", "P3", "P4"]
    X_3 = X[:, [idx[c] for c in ch3]]
    print(f"\n3-dim {ch3}  shape={X_3.shape}")
    ba3, p3 = permutation_test_subset(X_3, y, n_perm=1000)

    ch5 = ["Pz", "P3", "P4", "Cz", "CPz"]
    X_5 = X[:, [idx[c] for c in ch5]]
    print(f"\n5-dim {ch5}  shape={X_5.shape}")
    ba5, p5 = permutation_test_subset(X_5, y, n_perm=1000)

    print("\n" + "="*50)
    print("Electrode Selection Results vs Baseline:")
    print(f"  128-dim (baseline): BA=0.670  p=0.021")
    print(f"  3-dim  (Pz/P3/P4): BA={ba3:.3f}  p={p3:.4f}")
    print(f"  5-dim  (+Cz/CPz):  BA={ba5:.3f}  p={p5:.4f}")
    print("="*50)


def _loso_ba(X, y):
    """Silent LOSO — returns BA only (for permutation loop)."""
    loo = LeaveOneOut()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=20)),
        ("clf",    LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
    ])
    preds = []
    for tr_idx, te_idx in loo.split(X):
        pipe.fit(X[tr_idx], y[tr_idx])
        preds.append(int(pipe.predict(X[te_idx])[0]))
    return balanced_accuracy_score(y, preds)


def permutation_test_loso(X, y, n_perm=1000, seed=42):
    """Subject-level permutation test. Features extracted once outside loop."""
    assert len(y) == X.shape[0], "y must be subject-level"
    rng = np.random.default_rng(seed)
    obs_ba = _loso_ba(X, y)
    seeds = rng.integers(0, 2**31, size=n_perm)

    def one_perm(s):
        return _loso_ba(X, np.random.default_rng(s).permutation(y))

    print(f"Running {n_perm} permutations (n_jobs=4)...", flush=True)
    perm_bas = Parallel(n_jobs=4)(delayed(one_perm)(s) for s in seeds)
    perm_arr = np.array(perm_bas)
    p_val = float((np.sum(perm_arr >= obs_ba) + 1) / (n_perm + 1))
    print(f"  Observed BA={obs_ba:.3f}  p={p_val:.4f}  "
          f"(perm mean={np.mean(perm_arr):.3f})", flush=True)
    return obs_ba, p_val


def run_loso(X, y, ids, label=None):
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
    print(f"Subject-level results ({label or CONDITION} condition, avg-ERP P300):")
    print(f"  BA  = {ba:.3f}")
    print(f"  AUC = {auc:.3f}")
    print(f"  Acc = {np.mean(preds == true_labels):.3f}")
    print(f"  N   = {len(true_labels)} (MDD={sum(true_labels==1)}, HC={sum(true_labels==0)})")
    print(f"{'='*50}")
    return ba, auc


if __name__ == "__main__":
    # Phase 8: hcue single-condition + permutation test
    CONDITION = "hcue"
    X, y, ids = load_erp_features()
    print(f"Feature matrix: {X.shape}", flush=True)
    run_loso(X, y, ids, label="hcue")
    permutation_test_loso(X, y, n_perm=1000)

    # Phase 9: multi-condition fusion (hcue + fcue + scue)
    print("\n" + "="*50)
    print("Multi-condition fusion: hcue + fcue + scue")
    dicts = [load_erp_features_dict(c) for c in ["hcue", "fcue", "scue"]]
    X_fused, y_fused, ids_fused = fuse_conditions(dicts)
    print(f"Fused matrix: {X_fused.shape}  (n={len(y_fused)})", flush=True)
    run_loso(X_fused, y_fused, ids_fused, label="fusion(hcue+fcue+scue)")
    permutation_test_loso(X_fused, y_fused, n_perm=1000)

    # Contrast: scue - hcue
    print("\n" + "="*50)
    print("Contrast: scue - hcue")
    common = sorted(set(dicts[0]) & set(dicts[2]))
    X_contrast = np.array([dicts[2][s][0] - dicts[0][s][0] for s in common])
    y_contrast = np.array([dicts[0][s][1] for s in common])
    ids_contrast = np.array(common)
    print(f"Contrast matrix: {X_contrast.shape}", flush=True)
    run_loso(X_contrast, y_contrast, ids_contrast, label="contrast(scue-hcue)")
    permutation_test_loso(X_contrast, y_contrast, n_perm=1000)

    # Phase 10: Electrode Selection (reuse hcue X, y, ids from top of __main__)
    run_electrode_selection(X, y, ids)
