import argparse
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
from scipy.signal import welch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix, roc_curve
import time

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

_lgbm_warned = False
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
    parser.add_argument("--max-bad-channels", type=int, default=3, help="Max bad channels per window")

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

def generate_qc_report(participants_df, groups, keep_mask, no_edf_subjects, min_windows_per_subject):
    label_map_inv = {0: 'HC', 1: 'MDD'}
    rows = []
    # Subjects with no EDF
    for sub_id, group_label in no_edf_subjects:
        rows.append({"subject": sub_id, "label": group_label, "raw_windows": 0,
                      "kept_windows": 0, "drop_reason": "no_edf"})
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
            # Find label from participants_df
            match = participants_df[participants_df["participant_id"] == sub]
            label = match["group"].iloc[0] if len(match) > 0 else "unknown"
            rows.append({"subject": sub, "label": label, "raw_windows": raw_count,
                          "kept_windows": kept_count, "drop_reason": reason})
    return pd.DataFrame(rows)


def load_windows(participants_df, bids_root, window_sec, resample_sfreq, crop_duration=60.0):
    all_epochs = []
    labels = []
    groups = []
    no_edf_subjects = []
    ch_names = None

    label_map = {'MDD': 1, 'HC': 0}

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
            if raw.times[-1] > crop_duration:
                raw.crop(tmin=0, tmax=crop_duration)

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                raw.filter(l_freq=0.5, h_freq=45.0, verbose=False)
                if raw.info['sfreq'] != resample_sfreq:
                    raw.resample(resample_sfreq, verbose=False)

            data = raw.get_data()
            window_size = int(window_sec * raw.info['sfreq'])

            for s in range(0, data.shape[1] - window_size + 1, window_size):
                segment = data[:, s:s + window_size]
                all_epochs.append(segment)
                labels.append(label_map[group_label])
                groups.append(sub_id)

        except Exception as e:
            logger.error(f"处理 {sub_id} 时发生错误: {e}")

    return np.array(all_epochs), np.array(labels), np.array(groups), no_edf_subjects, ch_names

def extract_features(X, sfreq, ch_names=None):
    n_win, n_ch, n_times = X.shape
    features = []
    feature_names = []

    if ch_names is None:
        ch_names = [f"ch{i}" for i in range(n_ch)]
    region_idx = build_region_indices(ch_names)

    # Per-channel PSD computation
    nperseg = min(n_times, int(sfreq * 2))
    freqs, psd = welch(X, fs=sfreq, axis=2, nperseg=nperseg)
    bands = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}

    # Per-channel band power
    band_power = {}
    for band_name, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        band_power[band_name] = np.mean(psd[:, :, mask], axis=2)
    total_power = np.sum(psd, axis=2)

    # Region-aggregated features: std, abs band power, rel band power
    for region in REGIONS:
        idx = region_idx[region]
        if not idx:
            features.append(np.zeros((n_win, 1 + 2 * len(bands))))
            feature_names.append(f"{region}_std")
            for b in bands:
                feature_names.extend([f"{region}_{b}", f"{region}_{b}_rel"])
            continue
        # Std
        features.append(np.mean(np.std(X[:, idx, :], axis=2), axis=1, keepdims=True))
        feature_names.append(f"{region}_std")
        for b in bands:
            bp = np.mean(band_power[b][:, idx], axis=1, keepdims=True)
            tp = np.mean(total_power[:, idx], axis=1, keepdims=True)
            features.append(bp)
            features.append(bp / (tp + 1e-10))
            feature_names.extend([f"{region}_{b}", f"{region}_{b}_rel"])

    # DE features
    de_feats, de_names = compute_de_features(X, sfreq, region_idx)
    features.append(de_feats)
    feature_names.extend(de_names)

    # Hjorth features
    hjorth_feats, hjorth_names = compute_hjorth_features(X, region_idx)
    features.append(hjorth_feats)
    feature_names.extend(hjorth_names)

    # Lateral asymmetry (frontal + temporal, all bands)
    asym_feats, asym_names = compute_lateral_asymmetry(band_power, ch_names, n_win)
    features.append(asym_feats)
    feature_names.extend(asym_names)

    # Alpha asymmetry (frontal)
    features.append(_alpha_asymmetry(band_power["alpha"], ch_names, n_win))
    feature_names.append("alpha_asymmetry")

    return np.column_stack(features), feature_names


def _alpha_asymmetry(alpha_power, ch_names, n_win):
    """Compute frontal alpha asymmetry using region-aware L/R channels."""
    ch_set = set(ch_names)
    # Try EGI first
    left = [ch_names.index(c) for c in EGI128_LEFT.get("frontal", []) if c in ch_set]
    right = [ch_names.index(c) for c in EGI128_RIGHT.get("frontal", []) if c in ch_set]
    if not left or not right:
        # Fallback to 10-20
        left = [ch_names.index(c) for c in ["F3", "Fp1", "F7"] if c in ch_set]
        right = [ch_names.index(c) for c in ["F4", "Fp2", "F8"] if c in ch_set]
    if not left or not right:
        return np.zeros((n_win, 1))
    l_alpha = np.mean(alpha_power[:, left], axis=1)
    r_alpha = np.mean(alpha_power[:, right], axis=1)
    denom = r_alpha + l_alpha
    return np.where(denom != 0, (r_alpha - l_alpha) / denom, 0.0)[:, np.newaxis]

def compute_de_features(X, sfreq, region_idx):
    """Differential entropy per band per region: 0.5 * log(2*pi*e*var)."""
    n_win = X.shape[0]
    bands = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
    nperseg = min(X.shape[2], int(sfreq * 2))
    freqs, psd = welch(X, fs=sfreq, axis=2, nperseg=nperseg)
    features, names = [], []
    for region in REGIONS:
        idx = region_idx[region]
        for band_name, (fmin, fmax) in bands.items():
            if not idx:
                features.append(np.zeros((n_win, 1)))
            else:
                mask = (freqs >= fmin) & (freqs <= fmax)
                bp = np.mean(psd[:, idx][:, :, mask], axis=(1, 2))
                de = 0.5 * np.log(2 * np.pi * np.e * (bp + 1e-10))
                features.append(de[:, np.newaxis])
            names.append(f"{region}_{band_name}_de")
    return np.column_stack(features), names


def compute_hjorth_features(X, region_idx):
    """Hjorth activity, mobility, complexity per region."""
    n_win = X.shape[0]
    features, names = [], []
    for region in REGIONS:
        idx = region_idx[region]
        if not idx:
            features.append(np.zeros((n_win, 3)))
            names.extend([f"{region}_activity", f"{region}_mobility", f"{region}_complexity"])
            continue
        sig = np.mean(X[:, idx, :], axis=1)  # (n_win, n_times)
        d1 = np.diff(sig, axis=1)
        d2 = np.diff(d1, axis=1)
        activity = np.var(sig, axis=1, keepdims=True)
        m0 = np.std(sig, axis=1)
        m1 = np.std(d1, axis=1)
        m2 = np.std(d2, axis=1)
        mobility = (m1 / (m0 + 1e-10))[:, np.newaxis]
        complexity = ((m2 / (m1 + 1e-10)) / (m1 / (m0 + 1e-10) + 1e-10))[:, np.newaxis]
        features.append(np.hstack([activity, mobility, complexity]))
        names.extend([f"{region}_activity", f"{region}_mobility", f"{region}_complexity"])
    return np.column_stack(features), names


def compute_lateral_asymmetry(band_power, ch_names, n_win):
    """L-R asymmetry for frontal and temporal regions."""
    ch_set = set(ch_names)
    features, names = [], []
    for region in ["frontal", "temporal"]:
        left = [ch_names.index(c) for c in EGI128_LEFT.get(region, []) if c in ch_set]
        right = [ch_names.index(c) for c in EGI128_RIGHT.get(region, []) if c in ch_set]
        for band_name, bp in band_power.items():
            if not left or not right:
                features.append(np.zeros((n_win, 1)))
            else:
                l_pow = np.mean(bp[:, left], axis=1)
                r_pow = np.mean(bp[:, right], axis=1)
                denom = r_pow + l_pow
                asym = np.where(denom != 0, (r_pow - l_pow) / denom, 0.0)
                features.append(asym[:, np.newaxis])
            names.append(f"{region}_{band_name}_asym")
    return np.column_stack(features), names


def build_feature_model_pipeline(model_name="svm", reducer="none"):
    steps = [("scaler", StandardScaler())]
    if reducer == "pca":
        steps.append(("pca", PCA(n_components=0.95)))
    elif reducer == "selectkbest":
        steps.append(("selectkbest", SelectKBest(f_classif, k=64)))
        
    if model_name == "svm":
        steps.append(("clf", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)))
    elif model_name == "logistic":
        steps.append(("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)))
    elif model_name == "lgbm":
        if not HAS_LGBM:
            raise ValueError("LightGBM not installed")
        steps.append(("clf", LGBMClassifier(class_weight="balanced", n_estimators=100, random_state=42, verbose=-1)))
    else:
        raise ValueError(f"Unknown classifier {model_name}")
        
    return Pipeline(steps)

def compute_subject_balanced_sample_weights(groups):
    unique_groups, counts = np.unique(groups, return_counts=True)
    weight_map = {g: 1.0 / c for g, c in zip(unique_groups, counts)}
    w = np.array([weight_map[g] for g in groups])
    return w

def _get_model_candidates():
    """Return list of (model_name, param_grid) tuples."""
    candidates = [
        ("svm", {"clf__C": [0.1, 1.0, 10.0]}),
        ("logistic", {"clf__C": [0.01, 0.1, 1.0]}),
    ]
    if HAS_LGBM:
        candidates.append(("lgbm", {"clf__n_estimators": [50, 100], "clf__max_depth": [3, 5]}))
    return candidates


def _apply_qc_and_extract(X_raw, y, groups, bad_amp_uv, max_bad_channels,
                           min_windows_per_subject, ch_names, sfreq):
    """Apply QC filtering and extract features. Returns (features, y, groups) or None."""
    mask = build_quality_mask(X_raw, bad_amp_uv=bad_amp_uv, max_bad_channels=max_bad_channels)
    # Drop subjects with too few windows
    kept_groups = groups[mask]
    if len(kept_groups) == 0:
        return None
    unique_g, counts = np.unique(kept_groups, return_counts=True)
    valid_g = set(unique_g[counts >= min_windows_per_subject])
    final_mask = mask & np.isin(groups, list(valid_g))
    if final_mask.sum() == 0:
        return None
    X_f = X_raw[final_mask]
    y_f = y[final_mask]
    g_f = groups[final_mask]
    # Need at least 2 classes
    subj_labels = {}
    for g, lab in zip(g_f, y_f):
        if g not in subj_labels:
            subj_labels[g] = lab
    classes = set(subj_labels.values())
    if len(classes) < 2:
        return None
    feats, names = extract_features(X_f, sfreq=sfreq, ch_names=ch_names)
    return feats, y_f, g_f, names


def run_full_model_selection(X_raw, y, groups, ch_names, sfreq,
                              bad_amp_candidates=(200, 300, 400),
                              max_bad_channels=3,
                              min_windows_per_subject=3,
                              n_splits=5, seed=42):
    """Outer CV with inner search over QC thresholds + models + hyperparams."""
    global _lgbm_warned
    if not HAS_LGBM and not _lgbm_warned:
        logger.warning("LightGBM not installed, using SVM and logistic regression only")
        _lgbm_warned = True
    # Try each QC candidate to find one that works for outer CV splitting
    result = None
    for qc_thr in bad_amp_candidates:
        result = _apply_qc_and_extract(X_raw, y, groups, qc_thr, max_bad_channels,
                                        min_windows_per_subject, ch_names, sfreq)
        if result is not None:
            break
    if result is None:
        raise ValueError("No valid data after QC with any candidate threshold")
    feats_default, y_default, g_default, feat_names = result

    cv_outer = StratifiedGroupKFold(n_splits=n_splits)
    y_true_subject = {}
    y_pred_subject_probs = {}
    fold_results = []

    for fold_i, (train_ix, test_ix) in enumerate(cv_outer.split(feats_default, y_default, groups=g_default)):
        train_subjects = set(g_default[train_ix])
        test_subjects = set(g_default[test_ix])

        # Get raw windows for train/test subjects
        train_raw_mask = np.isin(groups, list(train_subjects))
        test_raw_mask = np.isin(groups, list(test_subjects))

        best_score, best_cfg = -1, None

        for qc_thr in bad_amp_candidates:
            res = _apply_qc_and_extract(X_raw[train_raw_mask], y[train_raw_mask],
                                         groups[train_raw_mask], qc_thr, max_bad_channels,
                                         min_windows_per_subject, ch_names, sfreq)
            if res is None:
                continue
            X_tr, y_tr, g_tr, _ = res

            ug, fi = np.unique(g_tr, return_index=True)
            gl = y_tr[fi]
            min_cls = np.min(np.bincount(gl)) if len(np.unique(gl)) > 1 else 0

            for model_name, param_grid in _get_model_candidates():
                pipe = build_feature_model_pipeline(model_name=model_name, reducer="none")
                if min_cls >= 3:
                    inner_cv = StratifiedGroupKFold(n_splits=min(3, min_cls))
                    grid = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring="balanced_accuracy")
                    try:
                        grid.fit(X_tr, y_tr, groups=g_tr)
                        score = grid.best_score_
                        best_params = grid.best_params_
                    except Exception:
                        continue
                else:
                    try:
                        sw = compute_subject_balanced_sample_weights(g_tr)
                        pipe.fit(X_tr, y_tr, clf__sample_weight=sw)
                        score = 0.5  # no inner CV score available
                        best_params = {}
                    except Exception:
                        continue

                if score > best_score:
                    best_score = score
                    best_cfg = {"qc_thr": qc_thr, "model": model_name, "params": best_params}

        # Fallback
        if best_cfg is None:
            best_cfg = {"qc_thr": bad_amp_candidates[0], "model": "svm", "params": {}}

        # Retrain on full train with best config, predict on test
        res_tr = _apply_qc_and_extract(X_raw[train_raw_mask], y[train_raw_mask],
                                        groups[train_raw_mask], best_cfg["qc_thr"],
                                        max_bad_channels, min_windows_per_subject, ch_names, sfreq)
        res_te = _apply_qc_and_extract(X_raw[test_raw_mask], y[test_raw_mask],
                                        groups[test_raw_mask], best_cfg["qc_thr"],
                                        max_bad_channels, 1, ch_names, sfreq)

        if res_tr is None or res_te is None:
            continue

        X_tr, y_tr, g_tr, _ = res_tr
        X_te, y_te, g_te, _ = res_te

        pipe = build_feature_model_pipeline(model_name=best_cfg["model"], reducer="none")
        for k, v in best_cfg["params"].items():
            part, param = k.split("__", 1)
            setattr(pipe.named_steps[part], param, v)
        sw = compute_subject_balanced_sample_weights(g_tr)
        pipe.fit(X_tr, y_tr, clf__sample_weight=sw)
        preds = pipe.predict_proba(X_te)[:, 1]

        for i, g in enumerate(g_te):
            if g not in y_true_subject:
                y_true_subject[g] = y_te[i]
                y_pred_subject_probs[g] = []
            y_pred_subject_probs[g].append(preds[i])

        fold_results.append({"fold": fold_i, **best_cfg, "inner_score": best_score})

    subj_list = list(y_true_subject.keys())
    y_subj_true = np.array([y_true_subject[g] for g in subj_list])
    y_subj_prob = np.array([np.mean(y_pred_subject_probs[g]) for g in subj_list])
    y_subj_pred = (y_subj_prob >= 0.5).astype(int)

    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_subj_true, y_subj_pred),
        "roc_auc": roc_auc_score(y_subj_true, y_subj_prob) if len(np.unique(y_subj_true)) > 1 else 0.5,
        "f1": f1_score(y_subj_true, y_subj_pred, zero_division=0),
    }

    return {
        "primary_metric_level": "subject",
        "subject_level_metrics": metrics,
        "y_subj_true": y_subj_true,
        "y_subj_prob": y_subj_prob,
        "subj_list": subj_list,
        "fold_results": fold_results,
    }


def run_nested_group_cv(X, y, groups, n_splits=5):
    cv_outer = StratifiedGroupKFold(n_splits=n_splits)
    
    y_pred_probs = np.zeros(len(y))
    y_true_subject = {}
    y_pred_subject_probs = {}
    
    leakage_detected = False
    
    for train_ix, test_ix in cv_outer.split(X, y, groups=groups):
        train_groups = set(groups[train_ix])
        test_groups = set(groups[test_ix])
        if not train_groups.isdisjoint(test_groups):
            leakage_detected = True
            
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        g_train = groups[train_ix]
        
        sample_weights = compute_subject_balanced_sample_weights(g_train)

        # Inner CV for hyperparameter tuning
        train_unique_groups, train_first_idx = np.unique(g_train, return_index=True)
        train_group_labels = y_train[train_first_idx]
        min_class_in_train = np.min(np.bincount(train_group_labels)) if len(np.unique(train_group_labels)) > 1 else 0

        base_pipe = build_feature_model_pipeline(model_name="svm", reducer="none")
        if min_class_in_train >= 3:
            inner_cv = StratifiedGroupKFold(n_splits=min(3, min_class_in_train))
            param_grid = {"clf__C": [0.1, 1.0, 10.0]}
            grid = GridSearchCV(base_pipe, param_grid, cv=inner_cv, scoring="balanced_accuracy")
            grid.fit(X_train, y_train, groups=g_train)
            best_C = grid.best_params_["clf__C"]
            pipe = build_feature_model_pipeline(model_name="svm", reducer="none")
            pipe.named_steps["clf"].C = best_C
        else:
            pipe = base_pipe
        pipe.fit(X_train, y_train, clf__sample_weight=sample_weights)
        
        preds = pipe.predict_proba(X_test)[:, 1]
        y_pred_probs[test_ix] = preds
        
        for i, g in enumerate(groups[test_ix]):
            if g not in y_true_subject:
                y_true_subject[g] = y_test[i]
                y_pred_subject_probs[g] = []
            y_pred_subject_probs[g].append(preds[i])
            
    subj_list = list(y_true_subject.keys())
    y_subj_true = np.array([y_true_subject[g] for g in subj_list])
    y_subj_prob = np.array([np.mean(y_pred_subject_probs[g]) for g in subj_list])
    y_subj_pred = (y_subj_prob >= 0.5).astype(int)
    
    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_subj_true, y_subj_pred),
        "roc_auc": roc_auc_score(y_subj_true, y_subj_prob) if len(np.unique(y_subj_true)) > 1 else 0.5,
        "f1": f1_score(y_subj_true, y_subj_pred, zero_division=0),
    }
    
    return {
        "leakage_detected": leakage_detected,
        "primary_metric_level": "subject",
        "subject_level_metrics": metrics,
        "y_subj_true": y_subj_true,
        "y_subj_prob": y_subj_prob,
        "subj_list": subj_list
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

def build_report(X_raw, y, groups, cv_results, ch_names, sfreq,
                  bad_amp_candidates=(200, 300, 400), max_bad_channels=3,
                  min_windows_per_subject=3, n_permutations=1000, seed=42):
    y_subj_true = cv_results["y_subj_true"]
    y_subj_prob = cv_results["y_subj_prob"]
    y_subj_pred = (y_subj_prob >= 0.5).astype(int)

    ba_ci = get_ci_bootstrap(y_subj_true, y_subj_pred, balanced_accuracy_score, seed=seed)

    report = {
        "primary_metric": "subject_level_balanced_accuracy",
        "balanced_accuracy": cv_results["subject_level_metrics"]["balanced_accuracy"],
        "balanced_accuracy_ci95": ba_ci,
        "roc_auc": cv_results["subject_level_metrics"]["roc_auc"],
        "f1": cv_results["subject_level_metrics"]["f1"],
        "n_permutations": n_permutations,
        "permutation_strategy": "full_model_selection_rerun",
        "seed": seed
    }

    if n_permutations > 0:
        rng = np.random.RandomState(seed)
        unique_groups, first_idx = np.unique(groups, return_index=True)
        group_labels = y[first_idx]

        permuted_bas = []
        for i in range(n_permutations):
            if (i+1) % 10 == 0:
                logger.info(f"Permutation {i+1}/{n_permutations}...")
            shuffled_labels = rng.permutation(group_labels)
            label_map = {g: l for g, l in zip(unique_groups, shuffled_labels)}
            y_permuted = np.array([label_map[g] for g in groups])

            permuted_group_labels = y_permuted[first_idx]
            if len(np.unique(permuted_group_labels)) < 2:
                continue
            n_splits = min(5, np.min(np.unique(permuted_group_labels, return_counts=True)[1]))
            if n_splits < 2:
                n_splits = 2

            try:
                out = run_full_model_selection(
                    X_raw, y_permuted, groups, ch_names, sfreq,
                    bad_amp_candidates=bad_amp_candidates,
                    max_bad_channels=max_bad_channels,
                    min_windows_per_subject=min_windows_per_subject,
                    n_splits=n_splits, seed=seed)
                permuted_bas.append(out["subject_level_metrics"]["balanced_accuracy"])
            except (ValueError, Exception):
                continue

        report["effective_permutations"] = len(permuted_bas)
        if permuted_bas:
            permuted_bas = np.array(permuted_bas)
            actual_ba = report["balanced_accuracy"]
            p_value = (np.sum(permuted_bas >= actual_ba) + 1.0) / (len(permuted_bas) + 1.0)
            report["permutation_pvalue"] = p_value
            report["permuted_bas_mean"] = float(np.mean(permuted_bas))
            report["permuted_bas_std"] = float(np.std(permuted_bas))
        else:
            logger.warning("All permutations failed; p-value is NaN")
            report["permutation_pvalue"] = float('nan')
    else:
        report["effective_permutations"] = 0
        report["permutation_pvalue"] = float('nan')

    return report


def determine_conclusion(report, subject_labels):
    """Strict stopping criteria: all must hold for positive conclusion."""
    ba = report.get("balanced_accuracy", 0)
    ci = report.get("balanced_accuracy_ci95", (0, 0))
    p = report.get("permutation_pvalue", 1.0)
    classes, counts = np.unique(list(subject_labels.values()), return_counts=True)
    min_per_class = int(min(counts)) if len(counts) > 0 else 0

    positive = (ba > 0.6 and ci[0] > 0.5 and p < 0.05 and min_per_class >= 5)
    return {
        "conclusion": "positive" if positive else "negative",
        "criteria": {
            "ba_gt_0.6": bool(ba > 0.6),
            "ci_lower_gt_0.5": bool(ci[0] > 0.5),
            "p_lt_0.05": bool(p < 0.05),
            "min_per_class_gte_5": bool(min_per_class >= 5),
        },
        "values": {
            "balanced_accuracy": float(ba),
            "ci_lower": float(ci[0]),
            "permutation_pvalue": float(p),
            "min_subjects_per_class": min_per_class,
        }
    }


def run_main_with_output_dir(bids_root, output_dir, max_subjects, resample_sfreq, n_permutations, seed, min_windows_per_subject, window_sec, crop_duration, bad_amp_uv=200.0, max_bad_channels=3):
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

    X_windows, y_windows, groups, no_edf_subjects, ch_names = load_windows(
        participants_df,
        bids_root=bids_root,
        window_sec=window_sec,
        resample_sfreq=resample_sfreq,
        crop_duration=crop_duration
    )

    if X_windows.ndim != 3 or len(X_windows) == 0:
        raise ValueError(f"No valid EDF windows loaded (got shape {X_windows.shape}). Check that EDF files exist under bids_root.")

    bad_amp_candidates = [bad_amp_uv] if bad_amp_uv not in [200, 300, 400] else [200, 300, 400]

    # QC report uses user-specified threshold
    keep_mask = build_quality_mask(X_windows, bad_amp_uv=bad_amp_uv, max_bad_channels=max_bad_channels)
    qc_report = generate_qc_report(participants_df, groups, keep_mask, no_edf_subjects, min_windows_per_subject)
    qc_report.to_csv(os.path.join(output_dir, "qc_report.csv"), index=False)

    # Validation uses most lenient candidate so stricter thresholds don't block the pipeline
    lenient_mask = build_quality_mask(X_windows, bad_amp_uv=max(bad_amp_candidates), max_bad_channels=max_bad_channels)
    kept_groups = groups[lenient_mask]
    if len(kept_groups) == 0:
        raise ValueError("No windows survive QC even at most lenient threshold")
    unique_groups, counts = np.unique(kept_groups, return_counts=True)
    valid_groups = unique_groups[counts >= min_windows_per_subject]
    final_mask = lenient_mask & np.isin(groups, list(valid_groups))

    validate_post_qc_availability(groups, y_windows, final_mask, min_windows_per_subject)

    X_filtered = X_windows[final_mask]
    y_filtered = y_windows[final_mask]
    groups_filtered = groups[final_mask]

    subject_labels = {}
    for g, label in zip(groups_filtered, y_filtered):
        if g not in subject_labels:
            subject_labels[g] = label
    validate_subject_class_counts(subject_labels)

    # Determine n_splits from lenient-filtered data
    unique_groups, first_idx = np.unique(groups_filtered, return_index=True)
    group_labels = y_filtered[first_idx]
    if len(np.unique(group_labels)) > 1:
        n_splits = min(5, np.min(np.unique(group_labels, return_counts=True)[1]))
    else:
        n_splits = 2
    if n_splits < 2:
        n_splits = 2

    t0 = time.time()
    cv_results = run_full_model_selection(
        X_windows, y_windows, groups, ch_names, resample_sfreq,
        bad_amp_candidates=bad_amp_candidates,
        max_bad_channels=max_bad_channels,
        min_windows_per_subject=min_windows_per_subject,
        n_splits=n_splits, seed=seed)

    report = build_report(
        X_windows, y_windows, groups, cv_results, ch_names, resample_sfreq,
        bad_amp_candidates=bad_amp_candidates,
        max_bad_channels=max_bad_channels,
        min_windows_per_subject=min_windows_per_subject,
        n_permutations=n_permutations, seed=seed)
    report["elapsed_seconds"] = round(time.time() - t0, 1)

    # Save model comparison
    if cv_results.get("fold_results"):
        pd.DataFrame(cv_results["fold_results"]).to_csv(
            os.path.join(output_dir, "model_comparison.csv"), index=False)

    # Feature importance on lenient-QC filtered data
    features, feature_names = extract_features(X_filtered, sfreq=resample_sfreq, ch_names=ch_names)

    # Stopping criteria
    conclusion = determine_conclusion(report, subject_labels)
    report["conclusion"] = conclusion

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

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    try:
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
            max_bad_channels=args.max_bad_channels
        )
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Error: {e}")
        raise SystemExit(1)
