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

logger = logging.getLogger(__name__)

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
        steps.append(("clf", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)))
    elif model_name == "logistic":
        steps.append(("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)))
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

def build_report(X, y, groups, cv_results, n_permutations=1000, seed=42):
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
        "permutation_strategy": "full_nested_cv_rerun",
        "seed": seed
    }
    
    if n_permutations > 0:
        rng = np.random.RandomState(seed)
        unique_groups, first_idx = np.unique(groups, return_index=True)
        group_labels = y[first_idx]
        
        permuted_bas = []
        for i in range(n_permutations):
            if (i+1) % 10 == 0:
                logger.info(f"Running permutation {i+1}/{n_permutations}...")
            shuffled_labels = rng.permutation(group_labels)
            label_map = {g: l for g, l in zip(unique_groups, shuffled_labels)}
            y_permuted = np.array([label_map[g] for g in groups])
            
            permuted_group_labels = y_permuted[first_idx]
            if len(np.unique(permuted_group_labels)) > 1:
                n_splits = min(5, np.min(np.unique(permuted_group_labels, return_counts=True)[1]))
            else:
                n_splits = 2
            if n_splits < 2: n_splits = 2
            
            out = run_nested_group_cv(X, y_permuted, groups, n_splits=n_splits)
            permuted_bas.append(out["subject_level_metrics"]["balanced_accuracy"])
            
        permuted_bas = np.array(permuted_bas)
        actual_ba = report["balanced_accuracy"]
        p_value = (np.sum(permuted_bas >= actual_ba) + 1.0) / (n_permutations + 1.0)
        report["permutation_pvalue"] = p_value
        report["permuted_bas_mean"] = float(np.mean(permuted_bas))
        report["permuted_bas_std"] = float(np.std(permuted_bas))
    else:
        report["permutation_pvalue"] = float('nan')
        
    return report

def run_main_with_output_dir(bids_root, output_dir, max_subjects, resample_sfreq, n_permutations, seed, min_windows_per_subject, window_sec, crop_duration):
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
        
    X_windows, y_windows, groups = load_windows(
        participants_df,
        bids_root=bids_root,
        window_sec=window_sec,
        resample_sfreq=resample_sfreq,
        crop_duration=crop_duration
    )

    if X_windows.ndim != 3 or len(X_windows) == 0:
        raise ValueError(f"No valid EDF windows loaded (got shape {X_windows.shape}). Check that EDF files exist under bids_root.")

    keep_mask = build_quality_mask(X_windows)
    
    kept_groups = groups[keep_mask]
    unique_groups, counts = np.unique(kept_groups, return_counts=True)
    valid_groups = unique_groups[counts >= min_windows_per_subject]
    
    if len(valid_groups) < len(unique_groups):
        dropped = set(unique_groups) - set(valid_groups)
        logger.warning(f"Dropping subjects with too few windows after QC: {dropped}")
        keep_mask = keep_mask & np.isin(groups, list(valid_groups))
        
    validate_post_qc_availability(groups, y_windows, keep_mask, min_windows_per_subject)
    
    X_windows = X_windows[keep_mask]
    y_windows = y_windows[keep_mask]
    groups = groups[keep_mask]

    subject_labels = {}
    for g, label in zip(groups, y_windows):
        if g not in subject_labels:
            subject_labels[g] = label
    validate_subject_class_counts(subject_labels)

    features, feature_names = extract_features(X_windows, sfreq=resample_sfreq)
    
    unique_groups, first_idx = np.unique(groups, return_index=True)
    group_labels = y_windows[first_idx]
    if len(np.unique(group_labels)) > 1:
        n_splits = min(5, np.min(np.unique(group_labels, return_counts=True)[1]))
    else:
        n_splits = 2
    if n_splits < 2: n_splits = 2
        
    cv_results = run_nested_group_cv(features, y_windows, groups, n_splits=n_splits)
    
    report = build_report(features, y_windows, groups, cv_results, n_permutations=n_permutations, seed=seed)
    
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    f_scores, _ = f_classif(features, y_windows)
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
            crop_duration=args.crop_duration
        )
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Error: {e}")
        raise SystemExit(1)
