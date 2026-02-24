import argparse
import hashlib
import pandas as pd
import numpy as np
import os
import glob
import logging
import warnings
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from scipy.signal import welch, hilbert as sig_hilbert, sosfilt, butter, coherence as sig_coherence
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, confusion_matrix, roc_curve
import time
from joblib import Parallel, delayed
from pyprep import NoisyChannels

logger = logging.getLogger(__name__)

REGIONS = ["frontal", "central", "temporal", "parietal", "occipital"]

def _egi_range(start, end):
    return [f"E{i}" for i in range(start, end + 1)]

EGI128_REGION_MAP = {}
for ch in _egi_range(1,4) + _egi_range(8,14) + _egi_range(19,24) + _egi_range(26,28) + ["E33","E34"] + _egi_range(117,128):
    EGI128_REGION_MAP[ch] = "frontal"
for ch in _egi_range(5,7) + _egi_range(29,32) + _egi_range(35,37) + ["E54","E55","E79","E80"] + _egi_range(104,106):
    EGI128_REGION_MAP[ch] = "central"
for ch in _egi_range(38,46) + _egi_range(96,103) + _egi_range(107,116):
    EGI128_REGION_MAP[ch] = "temporal"
for ch in _egi_range(47,53) + _egi_range(56,62) + ["E77","E78"] + _egi_range(85,95):
    EGI128_REGION_MAP[ch] = "parietal"
for ch in _egi_range(63,76) + _egi_range(81,84):
    EGI128_REGION_MAP[ch] = "occipital"

STANDARD_1020_REGION_MAP = {
    "Fp1": "frontal", "Fp2": "frontal", "F3": "frontal", "F4": "frontal",
    "F7": "frontal", "F8": "frontal", "Fz": "frontal",
    "C3": "central", "C4": "central", "Cz": "central",
    "T3": "temporal", "T4": "temporal", "T5": "temporal", "T6": "temporal",
    "T7": "temporal", "T8": "temporal",
    "P3": "parietal", "P4": "parietal", "Pz": "parietal",
    "P7": "parietal", "P8": "parietal",
    "O1": "occipital", "O2": "occipital", "Oz": "occipital",
}

# EGI-128 left/right hemisphere channels for asymmetry
EGI128_LEFT = {
    "frontal": _egi_range(19,24) + ["E26","E27","E28","E33"],
    "temporal": _egi_range(38,46),
}
EGI128_RIGHT = {
    "frontal": _egi_range(117,124) + ["E1","E2","E3","E4"],
    "temporal": _egi_range(96,103),
}


def build_region_indices(ch_names):
    """Map channel names to region indices. Auto-detects EGI vs 10-20."""
    egi_hits = sum(1 for ch in ch_names if ch in EGI128_REGION_MAP)
    std_hits = sum(1 for ch in ch_names if ch in STANDARD_1020_REGION_MAP)
    region_map = EGI128_REGION_MAP if egi_hits >= std_hits else STANDARD_1020_REGION_MAP

    region_indices = {r: [] for r in REGIONS}
    for i, ch in enumerate(ch_names):
        region = region_map.get(ch, "other")
        if region in region_indices:
            region_indices[region].append(i)
    return region_indices


_CACHE_VERSION = "3"  # bump when preprocessing logic changes


def _compute_cache_key(participants_df, bids_root, window_sec, resample_sfreq,
                       crop_duration, highpass_freq, bad_amp_uv,
                       use_pyprep=True, max_bad_pct=0.20):
    """SHA-256 of params + id→group mapping + per-EDF mtime → 16-char hex key."""
    h = hashlib.sha256()
    h.update(_CACHE_VERSION.encode())
    id_group = sorted(zip(participants_df["participant_id"],
                          participants_df["group"]))
    h.update(json.dumps(id_group).encode())
    for v in (window_sec, resample_sfreq, crop_duration, highpass_freq, bad_amp_uv,
              use_pyprep, max_bad_pct):
        h.update(str(v).encode())
    for sid, _ in id_group:
        safe_id = os.path.basename(sid)
        for ext in (".EDF", ".edf"):
            p = os.path.join(bids_root, safe_id, "eeg",
                             f"{safe_id}_task-Resting-state_eeg{ext}")
            if os.path.exists(p):
                h.update(str(os.path.getmtime(p)).encode())
                break
    return h.hexdigest()[:16]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MODMA MDD vs HC Classification")
    parser.add_argument("--bids-root", required=True, help="Path to MODMA BIDS dataset")
    parser.add_argument("--max-subjects", type=int, default=None, help="Max subjects to process")
    parser.add_argument("--resample-sfreq", type=float, default=125.0, help="Resampling frequency")
    parser.add_argument("--n-permutations", type=int, default=1000, help="Number of permutations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min-windows-per-subject", type=int, default=3, help="Min windows per subject")
    parser.add_argument("--crop-duration", type=float, default=60.0, help="Crop duration in seconds")
    parser.add_argument("--window-sec", type=float, default=10.0, help="Window size in seconds")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--bad-amp-uv", type=float, default=200.0, help="Amplitude threshold in uV")
    parser.add_argument("--highpass-freq", type=float, default=1.0, help="High-pass filter frequency in Hz")
    parser.add_argument("--n-jobs", type=int, default=4, help="Parallel jobs for permutation test")
    parser.add_argument("--no-cache", action="store_true", help="Disable .npz window cache")
    parser.add_argument("--use-pyprep", action=argparse.BooleanOptionalAction, default=True,
                        help="Use pyprep NoisyChannels for bad channel detection")
    parser.add_argument("--max-bad-pct", type=float, default=0.20,
                        help="Max fraction of bad channels before subject exclusion")
    parser.add_argument("--ab-compare", action="store_true",
                        help="Run A/B comparison: pyprep vs amplitude detection")

    return parser.parse_args(argv)

def load_participants(path):
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    df["participant_id"] = df["participant_id"].astype(str).str.strip()
    df["group"] = df["group"].astype(str).str.strip()
    return df[df["group"].isin(["MDD", "HC"])][["participant_id", "group"]]

def build_quality_mask(X, bad_amp_uv=200.0, max_bad_channels=3):
    thr_v = bad_amp_uv * 1e-6
    bad_ch = np.any(np.abs(X) > thr_v, axis=2)
    bad_ch_count = bad_ch.sum(axis=1)
    return bad_ch_count <= max_bad_channels

def validate_post_qc_availability(groups, y, keep_mask, min_windows_per_subject=3):
    kept_groups = groups[keep_mask]
    unique_groups, counts = np.unique(kept_groups, return_counts=True)
    for g, c in zip(unique_groups, counts):
        if c < min_windows_per_subject:
            raise ValueError(f"Subject {g} has only {c} windows after QC (min is {min_windows_per_subject} min_windows_per_subject)")

def validate_subject_class_counts(subject_labels):
    classes, counts = np.unique(list(subject_labels.values()), return_counts=True)
    if len(classes) < 2:
        raise ValueError(f"Only {len(classes)} class(es) found after QC (need at least 2 distinct classes)")
    for c, count in zip(classes, counts):
        if count < 2:
            raise ValueError(f"Class {c} has only {count} subjects after QC (need at least 2 subjects per class)")

def generate_qc_report(participants_df, groups, keep_mask, no_edf_subjects, min_windows_per_subject, interp_info=None):
    if interp_info is None:
        interp_info = {}
    rows = []
    # Subjects with no EDF
    for sub_id, group_label in no_edf_subjects:
        rows.append({"subject": sub_id, "label": group_label, "raw_windows": 0,
                      "kept_windows": 0, "drop_reason": "no_edf",
                      "n_interpolated": interp_info.get(sub_id, 0)})
    # Subjects that had windows loaded
    if len(groups) > 0:
        unique_subs = np.unique(groups)
        for sub in unique_subs:
            sub_mask = groups == sub
            raw_count = int(sub_mask.sum())
            kept_count = int((sub_mask & keep_mask).sum())
            if kept_count == 0:
                reason = "dropped_by_amp"
            elif kept_count < min_windows_per_subject:
                reason = "dropped_by_min_windows"
            else:
                reason = "kept"
            match = participants_df[participants_df["participant_id"] == sub]
            label = match["group"].iloc[0] if len(match) > 0 else "unknown"
            rows.append({"subject": sub, "label": label, "raw_windows": raw_count,
                          "kept_windows": kept_count, "drop_reason": reason,
                          "n_interpolated": interp_info.get(sub, 0)})
    df = pd.DataFrame(rows)

    # Group retention stats
    n_mdd_total = len(df[df["label"] == "MDD"])
    n_hc_total = len(df[df["label"] == "HC"])
    n_mdd_kept = len(df[(df["label"] == "MDD") & (df["drop_reason"] == "kept")])
    n_hc_kept = len(df[(df["label"] == "HC") & (df["drop_reason"] == "kept")])
    mdd_rate = n_mdd_kept / n_mdd_total if n_mdd_total > 0 else 0.0
    hc_rate = n_hc_kept / n_hc_total if n_hc_total > 0 else 0.0
    balance_diff = abs(mdd_rate - hc_rate)
    logger.info(f"Group retention - MDD: {mdd_rate:.0%}, HC: {hc_rate:.0%}, diff: {balance_diff:.0%}")

    retention_stats = {"mdd_rate": mdd_rate, "hc_rate": hc_rate, "diff": balance_diff}
    return df, retention_stats


def load_windows(participants_df, bids_root, window_sec, resample_sfreq, crop_duration=60.0, highpass_freq=1.0, bad_amp_uv=200.0, cache_dir=None, use_pyprep=True, max_bad_pct=0.20):
    # --- cache hit ---
    if cache_dir:
        key = _compute_cache_key(participants_df, bids_root, window_sec,
                                 resample_sfreq, crop_duration, highpass_freq, bad_amp_uv,
                                 use_pyprep=use_pyprep, max_bad_pct=max_bad_pct)
        cache_path = os.path.join(cache_dir, f"{key}.npz")
        if os.path.exists(cache_path):
            try:
                d = np.load(cache_path, allow_pickle=True)
                logger.info("Cache hit: %s", cache_path)
                return (d["X"], d["y"], d["groups"],
                        [tuple(x) for x in d["no_edf"]],
                        list(d["ch_names"]),
                        d["interp_info"].item())
            except Exception as e:
                logger.warning("Corrupt cache %s, deleting: %s", cache_path, e)
                os.remove(cache_path)

    all_epochs = []
    labels = []
    groups = []
    no_edf_subjects = []
    ch_names = None
    interp_info = {}

    label_map = {'MDD': 1, 'HC': 0}
    thr_v = bad_amp_uv * 1e-6

    for _, row in participants_df.iterrows():
        sub_id = row['participant_id']
        group_label = row['group']

        safe_id = os.path.basename(sub_id)
        edf_pattern = os.path.join(bids_root, safe_id, 'eeg', f'{safe_id}_task-Resting-state_eeg.EDF')
        edf_files = glob.glob(edf_pattern)
        if not edf_files:
            edf_pattern = os.path.join(bids_root, safe_id, 'eeg', f'{safe_id}_task-Resting-state_eeg.edf')
            edf_files = glob.glob(edf_pattern)

        if not edf_files:
            logger.warning(f"找不到 {safe_id} 的 EDF 文件，跳过此人。")
            no_edf_subjects.append((sub_id, group_label))
            continue

        edf_path = edf_files[0]

        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            if ch_names is None:
                ch_names = raw.info['ch_names']

            # Set montage for 3D coordinates (required for interpolation)
            montage = mne.channels.make_standard_montage("GSN-HydroCel-128")
            raw.set_montage(montage, verbose=False)

            if raw.times[-1] > crop_duration:
                raw.crop(tmin=0, tmax=crop_duration)

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                raw.filter(l_freq=highpass_freq, h_freq=45.0, verbose=False)

                # Detect bad channels
                n_ch = len(raw.ch_names)
                if use_pyprep:
                    nc = NoisyChannels(raw, do_detrend=False, random_state=42)
                    nc.find_all_bads(ransac=True, channel_wise=False)
                    bad_chs = nc.get_bads()
                    method = "pyprep"
                else:
                    data_raw = raw.get_data()
                    bad_mask = np.any(np.abs(data_raw) > thr_v, axis=1)
                    bad_chs = [raw.ch_names[i] for i in range(n_ch) if bad_mask[i]]
                    method = "amplitude"

                n_bad = len(bad_chs)
                logger.info(f"  {sub_id}: {n_bad}/{n_ch} bad channels ({method}) {bad_chs}")

                n_interpolated = 0
                if use_pyprep:
                    # pyprep path: exclude if > max_bad_pct
                    max_bad = int(n_ch * max_bad_pct)
                    if n_bad > max_bad:
                        logger.info(f"  {sub_id}: EXCLUDED ({n_bad} > {max_bad})")
                        no_edf_subjects.append((sub_id, group_label))
                        continue
                    if bad_chs:
                        raw.info['bads'] = bad_chs
                        raw.interpolate_bads(reset_bads=True, verbose=False)
                        n_interpolated = n_bad
                else:
                    # amplitude path: interpolate if <=25% bad (original behavior)
                    if 0 < n_bad <= int(n_ch * 0.25):
                        raw.info['bads'] = bad_chs
                        raw.interpolate_bads(reset_bads=True, verbose=False)
                        n_interpolated = n_bad
                interp_info[sub_id] = n_interpolated

                # Average reference AFTER interpolation
                raw.set_eeg_reference('average', verbose=False)

                if raw.info['sfreq'] != resample_sfreq:
                    raw.resample(resample_sfreq, verbose=False)

            data = raw.get_data()
            n_channels = data.shape[0]
            window_size = int(window_sec * raw.info['sfreq'])

            # Skip Window 0 (filter transient); per-window QC with dynamic threshold
            max_bad_win = max(1, int(n_channels * 0.12))
            for s in range(window_size, data.shape[1] - window_size + 1, window_size):
                segment = data[:, s:s + window_size]
                bad_ch_count = np.sum(np.any(np.abs(segment) > thr_v, axis=1))
                if bad_ch_count > max_bad_win:
                    continue  # exceeds 12% bad channel threshold, discard
                all_epochs.append(segment)
                labels.append(label_map[group_label])
                groups.append(sub_id)

        except Exception as e:
            logger.error(f"处理 {sub_id} 时发生错误: {e}")

    result = (np.array(all_epochs), np.array(labels), np.array(groups),
              no_edf_subjects, ch_names, interp_info)

    # --- cache save ---
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache_path,
                 X=result[0], y=result[1], groups=result[2],
                 no_edf=np.array(no_edf_subjects, dtype=object),
                 ch_names=np.array(ch_names if ch_names else [], dtype=object),
                 interp_info=np.array(interp_info))

    return result

def _compute_plv(X, region_idx, sfreq, region_a, region_b, fmin, fmax):
    """Phase locking value between two regions in a frequency band."""
    sig_a = np.mean(X[:, region_idx[region_a], :], axis=1)
    sig_b = np.mean(X[:, region_idx[region_b], :], axis=1)
    sos = butter(4, [fmin, fmax], btype='band', fs=sfreq, output='sos')
    fa = sosfilt(sos, sig_a, axis=1)
    fb = sosfilt(sos, sig_b, axis=1)
    pa = np.angle(sig_hilbert(fa, axis=1))
    pb = np.angle(sig_hilbert(fb, axis=1))
    plv = np.abs(np.mean(np.exp(1j * (pa - pb)), axis=1))
    return plv[:, np.newaxis]


def _compute_coherence(X, region_idx, sfreq, region_a, region_b, fmin, fmax):
    """Mean coherence between two regions in a frequency band."""
    sig_a = np.mean(X[:, region_idx[region_a], :], axis=1)
    sig_b = np.mean(X[:, region_idx[region_b], :], axis=1)
    n_win = X.shape[0]
    nperseg = min(X.shape[2], int(sfreq * 2))
    result = np.zeros(n_win)
    for i in range(n_win):
        freqs_c, cxy = sig_coherence(sig_a[i], sig_b[i], fs=sfreq, nperseg=nperseg)
        band_mask = (freqs_c >= fmin) & (freqs_c <= fmax)
        result[i] = np.nan_to_num(np.mean(cxy[band_mask])) if band_mask.any() else 0.0
    return result[:, np.newaxis]


def extract_features(X, sfreq, ch_names=None):
    """Extract 15-dim features: 5 existing + 4 beta + 3 connectivity + 3 temporal/central."""
    n_win, n_ch, n_times = X.shape
    if ch_names is None:
        ch_names = [f"ch{i}" for i in range(n_ch)]
    region_idx = build_region_indices(ch_names)

    nperseg = min(n_times, int(sfreq * 2))
    freqs, psd = welch(X, fs=sfreq, axis=2, nperseg=nperseg)
    total_power = np.sum(psd, axis=2)
    pidx = region_idx["parietal"]
    tidx = region_idx["temporal"]
    fidx = region_idx["frontal"]

    def _band_rel(region_indices, fmin, fmax):
        if not region_indices:
            return np.zeros((n_win, 1))
        mask = (freqs >= fmin) & (freqs <= fmax)
        bp = np.mean(np.mean(psd[:, region_indices][:, :, mask], axis=2), axis=1)
        tp = np.mean(total_power[:, region_indices], axis=1)
        return (bp / (tp + 1e-10))[:, np.newaxis]

    # --- Original 5 features ---
    features = [
        _band_rel(pidx, 4, 8),   # parietal theta rel
        _band_rel(pidx, 8, 13),  # parietal alpha rel
    ]

    alpha_mask = (freqs >= 8) & (freqs <= 13)
    alpha_power = np.mean(psd[:, :, alpha_mask], axis=2)
    features.append(_alpha_asymmetry(alpha_power, ch_names, n_win, "frontal"))

    theta_rel = _band_rel(pidx, 4, 8)
    beta_rel = _band_rel(pidx, 13, 30)
    features.append(theta_rel / (beta_rel + 1e-10))

    ct = ["central", "temporal"]
    sigs = [np.mean(X[:, region_idx[r], :], axis=1) if region_idx[r]
            else np.zeros((n_win, n_times)) for r in ct]
    sigs = np.stack(sigs, axis=1)
    covs = np.array([np.cov(sigs[i]) for i in range(n_win)])
    covs += 1e-6 * np.eye(2)[np.newaxis]
    features.append(np.log(np.abs(covs[:, 0, 1:2]) + 1e-10))

    # --- A. Beta power (4 features) ---
    features.append(_band_rel(tidx, 16, 24))   # temporal_beta2_rel
    features.append(_band_rel(fidx, 16, 24))   # frontal_beta2_rel
    features.append(_band_rel(tidx, 24, 40))   # temporal_beta3_rel
    beta2_mask = (freqs >= 16) & (freqs <= 24)
    beta2_power = np.mean(psd[:, :, beta2_mask], axis=2)
    features.append(_alpha_asymmetry(beta2_power, ch_names, n_win, "temporal"))  # beta_asym_temporal

    # --- B. Connectivity (3 features) ---
    has_frontal = len(region_idx["frontal"]) > 0
    has_parietal = len(pidx) > 0
    has_temporal = len(tidx) > 0
    has_central = len(region_idx["central"]) > 0

    if has_frontal and has_parietal:
        features.append(_compute_plv(X, region_idx, sfreq, "frontal", "parietal", 8, 13))
    else:
        features.append(np.zeros((n_win, 1)))
    if has_frontal and has_temporal:
        features.append(_compute_plv(X, region_idx, sfreq, "frontal", "temporal", 4, 8))
    else:
        features.append(np.zeros((n_win, 1)))
    if has_central and has_temporal:
        features.append(_compute_coherence(X, region_idx, sfreq, "central", "temporal", 13, 30))
    else:
        features.append(np.zeros((n_win, 1)))

    # --- C. Temporal/central (3 features) ---
    features.append(_band_rel(tidx, 4, 8))    # temporal_theta_rel
    features.append(_band_rel(tidx, 8, 13))   # temporal_alpha_rel
    features.append(_alpha_asymmetry(alpha_power, ch_names, n_win, "temporal"))  # alpha_asym_temporal

    names = [
        "parietal_theta_rel", "parietal_alpha_rel", "alpha_asym_frontal",
        "parietal_TBR", "riem_central_temporal",
        "temporal_beta2_rel", "frontal_beta2_rel", "temporal_beta3_rel", "beta_asym_temporal",
        "plv_frontal_parietal_alpha", "plv_frontal_temporal_theta", "coh_central_temporal_beta",
        "temporal_theta_rel", "temporal_alpha_rel", "alpha_asym_temporal",
    ]
    return np.column_stack(features), names


def _alpha_asymmetry(alpha_power, ch_names, n_win, region="frontal"):
    """Compute alpha asymmetry using region-aware L/R channels."""
    ch_set = set(ch_names)
    left = [ch_names.index(c) for c in EGI128_LEFT.get(region, []) if c in ch_set]
    right = [ch_names.index(c) for c in EGI128_RIGHT.get(region, []) if c in ch_set]
    if not left or not right:
        if region == "frontal":
            left = [ch_names.index(c) for c in ["F3", "Fp1", "F7"] if c in ch_set]
            right = [ch_names.index(c) for c in ["F4", "Fp2", "F8"] if c in ch_set]
        elif region == "temporal":
            left = [ch_names.index(c) for c in ["T3", "T5", "T7"] if c in ch_set]
            right = [ch_names.index(c) for c in ["T4", "T6", "T8"] if c in ch_set]
    if not left or not right:
        return np.zeros((n_win, 1))
    l_alpha = np.mean(alpha_power[:, left], axis=1)
    r_alpha = np.mean(alpha_power[:, right], axis=1)
    result = np.log(r_alpha + 1e-10) - np.log(l_alpha + 1e-10)
    return result[:, np.newaxis]

def build_feature_model_pipeline(use_l1=False):
    """Scaler + Logistic C=0.1 (fixed, no tuning). L1/saga when use_l1=True."""
    penalty, solver = ("l1", "saga") if use_l1 else ("l2", "lbfgs")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, penalty=penalty, solver=solver,
                                   max_iter=1000, random_state=42,
                                   class_weight="balanced")),
    ])

def compute_subject_balanced_sample_weights(groups):
    unique_groups, counts = np.unique(groups, return_counts=True)
    weight_map = {g: 1.0 / c for g, c in zip(unique_groups, counts)}
    w = np.array([weight_map[g] for g in groups])
    return w


def run_simplified_cv(features, y, groups, n_splits=5, seed=42):
    """CV on pre-extracted 5-dim features. Fixed Logistic C=0.1, no tuning."""
    cv_outer = StratifiedGroupKFold(n_splits=n_splits)
    y_true_subject = {}
    y_pred_subject_probs = {}
    fold_results = []

    for fold_i, (train_ix, test_ix) in enumerate(cv_outer.split(features, y, groups=groups)):
        X_tr, y_tr, g_tr = features[train_ix], y[train_ix], groups[train_ix]
        X_te, y_te, g_te = features[test_ix], y[test_ix], groups[test_ix]

        pipe = build_feature_model_pipeline()
        sw = compute_subject_balanced_sample_weights(g_tr)
        pipe.fit(X_tr, y_tr, clf__sample_weight=sw)
        preds = pipe.predict_proba(X_te)[:, 1]

        for i, g in enumerate(g_te):
            if g not in y_true_subject:
                y_true_subject[g] = y_te[i]
                y_pred_subject_probs[g] = []
            y_pred_subject_probs[g].append(preds[i])

        fold_results.append({"fold": fold_i})

    if not y_true_subject:
        raise ValueError("No valid predictions from simplified CV")

    subj_list = list(y_true_subject.keys())
    y_subj_true = np.array([y_true_subject[g] for g in subj_list])
    y_subj_prob = np.array([np.mean(y_pred_subject_probs[g]) for g in subj_list])
    y_subj_pred = (y_subj_prob >= 0.5).astype(int)

    return {
        "primary_metric_level": "subject",
        "subject_level_metrics": {
            "balanced_accuracy": balanced_accuracy_score(y_subj_true, y_subj_pred),
            "roc_auc": roc_auc_score(y_subj_true, y_subj_prob) if len(np.unique(y_subj_true)) > 1 else 0.5,
            "f1": f1_score(y_subj_true, y_subj_pred, zero_division=0),
        },
        "y_subj_true": y_subj_true,
        "y_subj_prob": y_subj_prob,
        "subj_list": subj_list,
        "fold_results": fold_results,
    }


def get_ci_bootstrap(y_true, y_pred, metric_fn, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2 and metric_fn in [roc_auc_score]:
            continue
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    if not scores:
        return (0.0, 0.0)
    return (float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5)))


def _run_one_perm_cv(features, y_perm, groups, n_splits, seed):
    """Run simplified CV on permuted labels, return BA or None on failure."""
    try:
        res = run_simplified_cv(features, y_perm, groups, n_splits, seed)
        return res["subject_level_metrics"]["balanced_accuracy"]
    except Exception:
        return None


def build_report(features, y, groups, cv_results, n_permutations=1000,
                 seed=42, n_jobs=4, n_splits=5):
    y_subj_true = cv_results["y_subj_true"]
    y_subj_prob = cv_results["y_subj_prob"]
    y_subj_pred = (y_subj_prob >= 0.5).astype(int)

    ba_ci = get_ci_bootstrap(y_subj_true, y_subj_pred, balanced_accuracy_score, seed=seed)

    report = {
        "experiment_type": "feature_reduced_v5",
        "primary_metric": "subject_level_balanced_accuracy",
        "balanced_accuracy": cv_results["subject_level_metrics"]["balanced_accuracy"],
        "balanced_accuracy_ci95": ba_ci,
        "roc_auc": cv_results["subject_level_metrics"]["roc_auc"],
        "f1": cv_results["subject_level_metrics"]["f1"],
        "n_permutations": n_permutations,
        "permutation_strategy": "label_permutation_pre_extracted",
        "seed": seed
    }

    if n_permutations > 0:
        rng = np.random.RandomState(seed)
        unique_groups, first_idx = np.unique(groups, return_index=True)
        group_labels = y[first_idx]

        # Pre-generate permuted labels sequentially for deterministic seeding
        jobs = []
        for _ in range(n_permutations):
            shuffled_labels = rng.permutation(group_labels)
            if len(np.unique(shuffled_labels)) < 2:
                continue
            label_map = dict(zip(unique_groups, shuffled_labels))
            y_perm = np.array([label_map[g] for g in groups])
            jobs.append(y_perm)

        logger.info(f"Running {len(jobs)} permutations with joblib (n_jobs={n_jobs})...")
        results = Parallel(n_jobs=n_jobs)(
            delayed(_run_one_perm_cv)(features, y_p, groups, n_splits, seed)
            for y_p in jobs)
        permuted_bas = [r for r in results if r is not None]

        report["effective_permutations"] = len(permuted_bas)
        if permuted_bas:
            permuted_bas = np.array(permuted_bas)
            actual_ba = report["balanced_accuracy"]
            p_value = (np.sum(permuted_bas >= actual_ba) + 1.0) / (len(permuted_bas) + 1.0)
            report["permutation_pvalue"] = p_value
            report["permuted_bas_mean"] = float(np.mean(permuted_bas))
            report["permuted_bas_std"] = float(np.std(permuted_bas))
            report["permutation_scores"] = permuted_bas.tolist()
        else:
            report["permutation_pvalue"] = None
    else:
        report["effective_permutations"] = 0
        report["permutation_pvalue"] = None

    return report


def determine_conclusion(report, subject_labels):
    """Strict stopping criteria: all must hold for positive conclusion."""
    ba = report.get("balanced_accuracy", 0)
    ci = report.get("balanced_accuracy_ci95", (0, 0))
    p = report.get("permutation_pvalue")
    p_passes = (p is not None and p < 0.05)
    classes, counts = np.unique(list(subject_labels.values()), return_counts=True)
    min_per_class = int(min(counts)) if len(counts) > 0 else 0

    positive = (ba > 0.6 and ci[0] > 0.5 and p_passes and min_per_class >= 5)
    return {
        "conclusion": "positive" if positive else "negative",
        "criteria": {
            "ba_gt_0.6": bool(ba > 0.6),
            "ci_lower_gt_0.5": bool(ci[0] > 0.5),
            "p_lt_0.05": bool(p_passes),
            "min_per_class_gte_5": bool(min_per_class >= 5),
        },
        "values": {
            "balanced_accuracy": float(ba),
            "ci_lower": float(ci[0]),
            "permutation_pvalue": p,
            "min_subjects_per_class": min_per_class,
        }
    }


def run_main_with_output_dir(bids_root, output_dir, max_subjects, resample_sfreq, n_permutations, seed, min_windows_per_subject, window_sec, crop_duration, bad_amp_uv=200.0, highpass_freq=1.0, n_jobs=4, use_cache=True, use_pyprep=True, max_bad_pct=0.20):
    os.makedirs(output_dir, exist_ok=True)
    participants_path = os.path.join(bids_root, "participants.tsv")

    if not os.path.exists(participants_path):
        raise FileNotFoundError(f"participants.tsv not found at {participants_path}")
    participants_df = load_participants(participants_path)

    if max_subjects is not None:
        half = max_subjects // 2
        mdd = participants_df[participants_df["group"] == "MDD"].head(half)
        hc = participants_df[participants_df["group"] == "HC"].head(max_subjects - len(mdd))
        participants_df = pd.concat([mdd, hc], ignore_index=True)

    cache_dir = os.path.join(output_dir, ".cache") if use_cache else None
    X_windows, y_windows, groups, no_edf_subjects, ch_names, interp_info = load_windows(
        participants_df,
        bids_root=bids_root,
        window_sec=window_sec,
        resample_sfreq=resample_sfreq,
        crop_duration=crop_duration,
        highpass_freq=highpass_freq,
        bad_amp_uv=bad_amp_uv,
        cache_dir=cache_dir,
        use_pyprep=use_pyprep,
        max_bad_pct=max_bad_pct,
    )

    if X_windows.ndim != 3 or len(X_windows) == 0:
        raise ValueError(f"No valid EDF windows loaded (got shape {X_windows.shape}). Check that EDF files exist under bids_root.")

    # Dynamic max_bad_channels: 12% of channel count
    n_channels = X_windows.shape[1]
    dynamic_max_bad = max(1, int(n_channels * 0.12))
    logger.info(f"Dynamic max_bad_channels: {dynamic_max_bad} (12% of {n_channels})")

    # QC mask with fixed 200uV + dynamic_max_bad
    keep_mask = build_quality_mask(X_windows, bad_amp_uv=bad_amp_uv, max_bad_channels=dynamic_max_bad)
    qc_report, retention_stats = generate_qc_report(participants_df, groups, keep_mask, no_edf_subjects, min_windows_per_subject, interp_info=interp_info)
    qc_report.to_csv(os.path.join(output_dir, "qc_report.csv"), index=False)

    # Filter by QC mask + min windows per subject
    kept_groups = groups[keep_mask]
    if len(kept_groups) == 0:
        raise ValueError("No windows survive QC")
    unique_groups, counts = np.unique(kept_groups, return_counts=True)
    valid_groups = unique_groups[counts >= min_windows_per_subject]
    final_mask = keep_mask & np.isin(groups, list(valid_groups))

    validate_post_qc_availability(groups, y_windows, final_mask, min_windows_per_subject)

    X_filtered = X_windows[final_mask]
    y_filtered = y_windows[final_mask]
    groups_filtered = groups[final_mask]

    subject_labels = {}
    for g, label in zip(groups_filtered, y_filtered):
        if g not in subject_labels:
            subject_labels[g] = label
    validate_subject_class_counts(subject_labels)

    # Determine n_splits from filtered data
    unique_groups, first_idx = np.unique(groups_filtered, return_index=True)
    group_labels = y_filtered[first_idx]
    if len(np.unique(group_labels)) > 1:
        n_splits = min(5, np.min(np.unique(group_labels, return_counts=True)[1]))
    else:
        n_splits = 2
    if n_splits < 2:
        n_splits = 2

    # Extract features ONCE
    features, feature_names = extract_features(X_filtered, sfreq=resample_sfreq, ch_names=ch_names)

    t0 = time.time()
    cv_results = run_simplified_cv(features, y_filtered, groups_filtered, n_splits, seed)

    report = build_report(features, y_filtered, groups_filtered, cv_results,
                          n_permutations=n_permutations, seed=seed,
                          n_jobs=n_jobs, n_splits=n_splits)
    report["elapsed_seconds"] = round(time.time() - t0, 1)

    # Save model comparison
    if cv_results.get("fold_results"):
        pd.DataFrame(cv_results["fold_results"]).to_csv(
            os.path.join(output_dir, "model_comparison.csv"), index=False)

    # Save permutation scores as separate CSV
    perm_scores = report.pop("permutation_scores", None)
    if perm_scores is not None and len(perm_scores) > 0:
        pd.DataFrame({"balanced_accuracy": perm_scores}).to_csv(
            os.path.join(output_dir, "permutation_scores.csv"), index=False)

    # Stopping criteria — use CV-evaluated subjects, not lenient-filtered set
    cv_subject_labels = dict(zip(cv_results["subj_list"],
                                  cv_results["y_subj_true"].tolist()))
    conclusion = determine_conclusion(report, cv_subject_labels)
    report["conclusion"] = conclusion
    report["group_retention"] = retention_stats

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    f_scores, _ = f_classif(features, y_filtered)
    feature_imp_df = pd.DataFrame({"feature": feature_names, "importance": f_scores})
    feature_imp_df.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
    
    y_subj_true = cv_results["y_subj_true"]
    y_subj_prob = cv_results["y_subj_prob"]
    fpr, tpr, _ = roc_curve(y_subj_true, y_subj_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC={report['roc_auc']:.2f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    fig.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close(fig)
    
    y_subj_pred = (y_subj_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_subj_true, y_subj_pred)
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap="Blues")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close(fig)
    
    return report


def _run_with_adaptive_threshold(args, use_pyprep):
    """Run pipeline with adaptive threshold relaxation to retain >= 35 subjects."""
    for threshold in [0.20, 0.25, 0.30]:
        logger.info(f"Trying max_bad_pct={threshold} (use_pyprep={use_pyprep})")
        report = run_main_with_output_dir(
            bids_root=args.bids_root,
            output_dir=args.output_dir,
            max_subjects=args.max_subjects,
            resample_sfreq=args.resample_sfreq,
            n_permutations=0,  # skip permutations for threshold search
            seed=args.seed,
            min_windows_per_subject=args.min_windows_per_subject,
            window_sec=args.window_sec,
            crop_duration=args.crop_duration,
            bad_amp_uv=args.bad_amp_uv,
            highpass_freq=args.highpass_freq,
            n_jobs=args.n_jobs,
            use_cache=False,
            use_pyprep=use_pyprep,
            max_bad_pct=threshold,
        )
        # Count retained subjects from the CV results
        n_subjects = len(report.get("conclusion", {}).get("values", {}).get("min_subjects_per_class", 0) and
                         report.get("conclusion", {}).get("values", {}))
        # Read qc_report to count retained subjects
        import pandas as _pd
        qc_path = os.path.join(args.output_dir, "qc_report.csv")
        if os.path.exists(qc_path):
            qc = _pd.read_csv(qc_path)
            n_retained = len(qc[qc["drop_reason"] == "kept"])
        else:
            n_retained = 0
        logger.info(f"  Threshold {threshold}: {n_retained} subjects retained")
        if n_retained >= 35:
            return report, threshold, n_retained
    return report, 0.30, n_retained


def run_ab_comparison(args):
    """Run A/B comparison: pyprep vs amplitude detection."""
    results = {}
    for method, use_pp in [("pyprep", True), ("amplitude", False)]:
        logger.info(f"\n{'='*60}\nRunning {method} detection\n{'='*60}")
        report, threshold, n_retained = _run_with_adaptive_threshold(args, use_pp)
        ba = report.get("balanced_accuracy", 0)
        auc = report.get("roc_auc", 0)
        retention = report.get("group_retention", {})
        results[method] = {
            "BA": ba, "AUC": auc,
            "n_subjects": n_retained,
            "n_mdd": round(retention.get("mdd_rate", 0) * n_retained),
            "n_hc": round(retention.get("hc_rate", 0) * n_retained),
            "threshold_used": threshold,
        }

    # Print comparison table
    logger.info(f"\n{'='*60}\nA/B Comparison: pyprep vs amplitude\n{'='*60}")
    logger.info(f"{'Metric':<20} {'pyprep':>10} {'amplitude':>10}")
    logger.info("-" * 42)
    for key in ["BA", "AUC", "n_subjects", "n_mdd", "n_hc", "threshold_used"]:
        v1 = results["pyprep"].get(key, "N/A")
        v2 = results["amplitude"].get(key, "N/A")
        fmt = f".3f" if isinstance(v1, float) else ""
        logger.info(f"{key:<20} {v1:>10{fmt}} {v2:>10{fmt}}")

    # Regression gate (resting-state baseline BA=0.613)
    pyprep_ba = results["pyprep"]["BA"]
    if pyprep_ba >= 0.613:
        logger.info("\nRegression gate: PASSED (pyprep BA >= 0.613)")
    elif 0.580 <= pyprep_ba < 0.613:
        logger.info("\nRegression gate: CONDITIONAL (0.580 <= BA < 0.613, check retention)")
    else:
        logger.info(f"\nRegression gate: FAILED (pyprep BA={pyprep_ba:.3f} < 0.580)")
        logger.info("Recommendation: use --no-use-pyprep")

    # Save JSON
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "pyprep_ab_comparison.json")
    with open(out_path, "w", newline='\n') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved: {out_path}")


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    try:
        if args.ab_compare:
            run_ab_comparison(args)
        else:
            run_main_with_output_dir(
                bids_root=args.bids_root,
                output_dir=args.output_dir,
                max_subjects=args.max_subjects,
                resample_sfreq=args.resample_sfreq,
                n_permutations=args.n_permutations,
                seed=args.seed,
                min_windows_per_subject=args.min_windows_per_subject,
                window_sec=args.window_sec,
                crop_duration=args.crop_duration,
                bad_amp_uv=args.bad_amp_uv,
                highpass_freq=args.highpass_freq,
                n_jobs=args.n_jobs,
                use_cache=not args.no_cache,
                use_pyprep=args.use_pyprep,
                max_bad_pct=args.max_bad_pct,
            )
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Error: {e}")
        raise SystemExit(1)
