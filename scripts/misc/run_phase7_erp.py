"""Phase 7: ERP Spatial & Temporal Delineation.

1. Spatial Selection (3-dim, 5-dim, 128-dim)
2. Temporal Window Analysis (50ms bins, 250-500ms) with t-test & LOSO
3. Spatial Feature Importance Mapping via PCA inversion.
"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _root import ROOT

import os, gc, re, warnings, numpy as np
import scipy.stats as stats
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
ERP_DIR = ROOT / "data/modma_erp/EEG_128channels_ERP_lanzhou_2015"
CONDITION = "hcue"
TMIN, TMAX = -0.1, 0.5     # epoch window (sec)
BASELINE = (-0.1, 0.0)     # baseline correction
FMIN, FMAX = 0.5, 40.0     # bandpass filter
N_EEG_CH = 128
P300_WIN = (0.25, 0.50)    # P300 overall time window (sec)

# 10-20 to EGI-128 approximation (HydroCel Geodesic Sensor Net)
# References for typical mappings: Pz~62, P3~52/53, P4~92, Cz~129(Ref)/VREF, CPz~54/55(approx)
# Let's extract the actual channel names from a raw file first to map accurately.
EGI_CH_MAP = {
    'Pz': 'E62',
    'P3': 'E52',
    'P4': 'E92',
    'Cz': 'Cz',    # Reference channel often not in 1-128 data, or E129. We'll use approx.
    'CPz': 'E55',
}

def get_channel_indices(ch_names, roi_type):
    """Return indices for spatial ROIs."""
    if roi_type == "128-dim":
        return list(range(128))
    
    # Try exact match or fallback to close EGI-128 equivalents
    # Standard mapping for HydroCel GSN 128:
    # Pz -> E62, P3 -> E52, P4 -> E92, Cz -> E129 (if avail) or VREF, CPz -> E55
    target_names = []
    if roi_type == "3-dim":
        target_names = ['E62', 'E52', 'E92']
    elif roi_type == "5-dim":
        # Using E11 (Fz), E55 (CPz), E62 (Pz), E52 (P3), E92 (P4) if Cz is not available
        target_names = ['E62', 'E52', 'E92', 'E55', 'E11'] # Substituting Cz with Fz/CPz just for 5-dim spread

    idx = []
    for t in target_names:
        if t in ch_names:
            idx.append(ch_names.index(t))
    return idx


def load_raw_data():
    """Load and cache the raw averaged ERPs (mean across trials)."""
    raw_files = sorted(ERP_DIR.glob("*.raw"))
    print(f"Loading {len(raw_files)} .raw files...", flush=True)

    all_erp = []     # (n_subj, 128, n_times)
    all_labels = []
    all_ids = []
    times = None
    ch_names = None

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
            raw.filter(FMIN, FMAX, verbose=False)
            
            if ch_names is None:
                ch_names = raw.ch_names[:N_EEG_CH]

            events, event_id = mne.events_from_annotations(raw, verbose=False)
            if CONDITION not in event_id:
                del raw; gc.collect(); continue

            cond_events = events[events[:, 2] == event_id[CONDITION]]
            epochs = mne.Epochs(
                raw, cond_events, tmin=TMIN, tmax=TMAX,
                baseline=BASELINE, preload=True, verbose=False,
                reject=dict(eeg=150e-6)
            )
            data = epochs.get_data()[:, :N_EEG_CH, :]  # (n_trials, 128, n_times)
            if times is None:
                times = epochs.times
                
            del raw, epochs; gc.collect()

            if data.shape[0] < 10:
                del data; continue

            avg_erp = data.mean(axis=0)  # (128, n_times)
            all_erp.append(avg_erp)
            all_labels.append(label)
            all_ids.append(sub_id)

        except Exception as e:
            gc.collect()

    return np.array(all_erp), np.array(all_labels), np.array(all_ids), times, ch_names


def _loso_ba(X, y, use_pca=True):
    """Silent LOSO for permutation test."""
    loo = LeaveOneOut()
    steps = [("scaler", StandardScaler())]
    if use_pca and X.shape[1] > 20:
        steps.append(("pca", PCA(n_components=20)))
    steps.append(("clf", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")))
    
    pipe = Pipeline(steps)
    preds = []
    for tr_idx, te_idx in loo.split(X):
        pipe.fit(X[tr_idx], y[tr_idx])
        preds.append(int(pipe.predict(X[te_idx])[0]))
    return balanced_accuracy_score(y, preds)


def permutation_test_loso(X, y, use_pca=True, n_perm=1000, seed=42):
    """Subject-level permutation test."""
    rng = np.random.default_rng(seed)
    obs_ba = _loso_ba(X, y, use_pca)
    seeds = rng.integers(0, 2**31, size=n_perm)

    def one_perm(s):
        return _loso_ba(X, np.random.default_rng(s).permutation(y), use_pca)

    perm_bas = Parallel(n_jobs=4)(delayed(one_perm)(s) for s in seeds)
    perm_arr = np.array(perm_bas)
    p_val = float((np.sum(perm_arr >= obs_ba) + 1) / (n_perm + 1))
    return obs_ba, p_val


def run_loso(X, y, ids, label, use_pca=True):
    """Leave-one-subject-out on subject-level features."""
    loo = LeaveOneOut()
    steps = [("scaler", StandardScaler())]
    if use_pca and X.shape[1] > 20:
        steps.append(("pca", PCA(n_components=20)))
    steps.append(("clf", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")))
    
    pipe = Pipeline(steps)
    true_labels, probs = [], []

    for tr_idx, te_idx in loo.split(X):
        pipe.fit(X[tr_idx], y[tr_idx])
        prob = pipe.predict_proba(X[te_idx])[:, 1][0]
        true_labels.append(y[te_idx][0])
        probs.append(prob)

    true_labels = np.array(true_labels)
    probs = np.array(probs)
    preds = (probs >= 0.5).astype(int)

    ba = balanced_accuracy_score(true_labels, preds)
    return ba


def format_res(label, ba, p_val):
    sig = "*" if p_val < 0.05 else " "
    print(f"[{label:^10s}] BA: {ba:.3f} | p-value: {p_val:.4f} {sig}")


# =====================================================================
# 1. Electrode Selection (Spatial Delineation)
# =====================================================================
def experiment_spatial(erps, y, ids, times, ch_names):
    print("\n" + "="*60)
    print("1. ELECTRODE SELECTION (Spatial Delineation, 250-500ms)")
    print("="*60)
    
    # Extract overall P300 window
    p300_mask = (times >= P300_WIN[0]) & (times <= P300_WIN[1])
    win_erps = erps[:, :, p300_mask].mean(axis=2)  # (n_subj, 128)

    rois = ["128-dim", "5-dim", "3-dim"]
    
    for roi in rois:
        idx = get_channel_indices(ch_names, roi)
        if len(idx) == 0:
            print(f"[{roi:^10s}] Error: Could not map channel indices.")
            continue
            
        X = win_erps[:, idx]
        use_pca = True if roi == "128-dim" else False
        ba, p_val = permutation_test_loso(X, y, use_pca=use_pca, n_perm=1000)
        format_res(roi, ba, p_val)


# =====================================================================
# 2. Time-Window Analysis (Temporal Delineation)
# =====================================================================
def experiment_temporal(erps, y, ids, times, ch_names):
    print("\n" + "="*60)
    print("2. TIME-WINDOW ANALYSIS (Temporal Delineation, 128-dim)")
    print("="*60)
    print(f"{'Bin (ms)':<15s} {'t-stat':<10s} {'t-test_p':<10s} | {'LOSO_BA':<10s} {'LOSO_p':<10s}")
    print("-" * 60)
    
    bins = [
        (0.25, 0.30),
        (0.30, 0.35),
        (0.35, 0.40),
        (0.40, 0.45),
        (0.45, 0.50),
    ]

    for t_start, t_end in bins:
        mask = (times >= t_start) & (times <= t_end)
        X = erps[:, :, mask].mean(axis=2)  # (n_subj, 128)
        
        # 1. Independent t-test across all 128 channels (average of channel p-values for display, or min p-val)
        # To simplify the display, let's test the Global Field Power (std across channels) or Grand Mean
        # We will test the grand mean amplitude across all 128 channels for a simple 1D physiological index.
        X_mean = X.mean(axis=1) # (n_subj,)
        mdd_vals = X_mean[y == 1]
        hc_vals  = X_mean[y == 0]
        t_stat, t_p = stats.ttest_ind(mdd_vals, hc_vals, equal_var=False)
        
        # 2. LOSO Classification on 128-dim
        ba, p_val = permutation_test_loso(X, y, use_pca=True, n_perm=1000)
        
        label = f"{int(t_start*1000)}-{int(t_end*1000)}"
        
        sig_t = "*" if t_p < 0.05 else " "
        sig_loso = "*" if p_val < 0.05 else " "
        
        print(f"[{label:<13s}] {t_stat:>7.3f}    {t_p:^8.4f}{sig_t}  | {ba:^8.3f}   {p_val:^8.4f}{sig_loso}")


# =====================================================================
# 3. Feature Importance (Spatial Mapping)
# =====================================================================
def experiment_importance(erps, y, times, ch_names):
    print("\n" + "="*60)
    print("3. FEATURE IMPORTANCE (Spatial Mapping, 250-500ms, 128-dim)")
    print("="*60)
    
    p300_mask = (times >= P300_WIN[0]) & (times <= P300_WIN[1])
    X = erps[:, :, p300_mask].mean(axis=2)  # (n_subj, 128)
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=20)),
        ("clf",    LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
    ])
    
    pipe.fit(X, y)
    
    # Extract components
    pca = pipe.named_steps["pca"]
    clf = pipe.named_steps["clf"]
    
    # pca.components_ shape is (n_components, n_features) -> (20, 128)
    # clf.coef_ shape is (1, n_components) -> (1, 20)
    
    # Project weights back to sensor space
    # W_sensor = clf.coef_ @ pca.components_
    # Shape: (1, 20) @ (20, 128) = (1, 128)
    sensor_weights = clf.coef_ @ pca.components_
    weights_1d = sensor_weights.flatten()
    
    # Identify top 10 channels with the largest absolute weights
    abs_weights = np.abs(weights_1d)
    top_indices = np.argsort(abs_weights)[::-1][:15]
    
    print("Top 15 channels contributing to MDD vs HC separation:")
    print(f"{'Rank':<5s} | {'Channel':<10s} | {'Weight (sign matters)':<20s}")
    print("-" * 50)
    
    for i, idx in enumerate(top_indices):
        ch = ch_names[idx]
        w = weights_1d[idx]
        print(f"{i+1:<5d} | {ch:<10s} | {w:>10.4f}")


if __name__ == "__main__":
    print("Loading ERP data...", flush=True)
    erps, y, ids, times, ch_names = load_raw_data()
    print(f"Loaded: N={len(y)} subjects, Shape={erps.shape}", flush=True)
    
    # Run the three ablation/interpretation experiments
    experiment_spatial(erps, y, ids, times, ch_names)
    experiment_temporal(erps, y, ids, times, ch_names)
    experiment_importance(erps, y, times, ch_names)
    
    print("\n=== Phase 7 Execution Complete ===")
