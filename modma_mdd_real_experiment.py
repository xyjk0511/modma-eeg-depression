import argparse
import pandas as pd
import numpy as np
import os
import glob
import logging
import warnings
import mne
from scipy.signal import welch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix

logger = logging.getLogger(__name__)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MODMA MDD vs HC Classification")
    parser.add_argument("--bids-root", required=True, help="Path to MODMA BIDS dataset")
    parser.add_argument("--max-subjects", type=int, default=None, help="Max subjects to process")
    parser.add_argument("--resample-sfreq", type=float, default=125.0, help="Resampling frequency")
    parser.add_argument("--n-permutations", type=int, default=1000, help="Number of permutations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min-windows-per-subject", type=int, default=3, help="Min windows per subject")
    
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
    for c, count in zip(classes, counts):
        if count < 2:
            raise ValueError(f"Class {c} has only {count} subjects after QC (need at least 2 subjects per class)")

def load_windows(participants_df, bids_root, window_sec, resample_sfreq, crop_duration=60.0):
    all_epochs = []
    labels = []
    groups = []
    
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
            continue
            
        edf_path = edf_files[0]
        
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
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
            
    return np.array(all_epochs), np.array(labels), np.array(groups)

def extract_features(X, sfreq):
    n_win, n_ch, n_times = X.shape
    features = []
    feature_names = []
    
    std_feat = np.std(X, axis=2)
    features.append(std_feat)
    feature_names.extend([f"ch{ch}_std" for ch in range(n_ch)])
    
    nperseg = min(X.shape[2], int(sfreq * 2))
    freqs, psd = welch(X, fs=sfreq, axis=2, nperseg=nperseg)
    
    bands = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
    psd_feats = []
    for band_name, (fmin, fmax) in bands.items():
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        band_power = np.mean(psd[:, :, band_mask], axis=2)
        psd_feats.append(band_power)
        feature_names.extend([f"ch{ch}_{band_name}" for ch in range(n_ch)])
        
    total_power = np.sum(psd, axis=2)
    for i, band_name in enumerate(bands.keys()):
        rel_power = psd_feats[i] / (total_power + 1e-10)
        features.append(rel_power)
        feature_names.extend([f"ch{ch}_{band_name}_rel" for ch in range(n_ch)])
        
    features.extend(psd_feats)
        
    alpha_power = psd_feats[2]
    if n_ch >= 124:
        f3_idx, f4_idx = 23, 123
        f3_alpha = alpha_power[:, f3_idx]
        f4_alpha = alpha_power[:, f4_idx]
        denom = f4_alpha + f3_alpha
        alpha_asymmetry = np.where(denom != 0, (f4_alpha - f3_alpha)/denom, 0.0)
        features.append(alpha_asymmetry[:, np.newaxis])
    else:
        features.append(np.zeros((n_win, 1)))
        
    feature_names.append("alpha_asymmetry")
    X_features = np.column_stack(features)
    return X_features, feature_names

def build_feature_model_pipeline(model_name="svm", reducer="none"):
    steps = [("scaler", StandardScaler())]
    if reducer == "pca":
        steps.append(("pca", PCA(n_components=0.95)))
    elif reducer == "selectkbest":
        steps.append(("selectkbest", SelectKBest(f_classif, k=64)))
        
    if model_name == "svm":
        steps.append(("clf", SVC(kernel="rbf", class_weight="balanced", probability=True)))
    elif model_name == "logistic":
        steps.append(("clf", LogisticRegression(class_weight="balanced", max_iter=1000)))
    else:
        raise ValueError(f"Unknown classifier {model_name}")
        
    return Pipeline(steps)

def compute_subject_balanced_sample_weights(groups):
    unique_groups, counts = np.unique(groups, return_counts=True)
    weight_map = {g: 1.0 / c for g, c in zip(unique_groups, counts)}
    w = np.array([weight_map[g] for g in groups])
    return w

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
        
        pipe = build_feature_model_pipeline(model_name="svm", reducer="none")
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

if __name__ == "__main__":
    args = parse_args()
    print(args)
