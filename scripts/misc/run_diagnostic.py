"""Quick diagnostic: compare feature/tuning strategies on cached data."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _root import ROOT

import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import balanced_accuracy_score
from modma_mdd_real_experiment import (
    load_participants, load_windows, build_quality_mask,
    extract_features, compute_subject_balanced_sample_weights,
    build_region_indices, REGIONS,
)
from scipy.signal import welch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BIDS = str(ROOT / "data/modma_bids/EEG_LZU_2015_2_resting state")
CACHE = str(ROOT / "outputs/results_phase03_verify/.cache")
SFREQ = 125.0


def load_filtered_data():
    """Load and QC-filter data, return (X_raw, y, groups, ch_names)."""
    import os, pandas as pd
    df = load_participants(os.path.join(BIDS, "participants.tsv"))
    X, y, groups, _, ch_names, _ = load_windows(
        df, BIDS, window_sec=10.0, resample_sfreq=SFREQ,
        crop_duration=60.0, highpass_freq=1.0, bad_amp_uv=200.0,
        cache_dir=CACHE)
    mask = build_quality_mask(X, bad_amp_uv=200.0,
                              max_bad_channels=max(1, int(X.shape[1] * 0.12)))
    kept = groups[mask]
    ug, cnt = np.unique(kept, return_counts=True)
    valid = set(ug[cnt >= 3])
    final = mask & np.isin(groups, list(valid))
    return X[final], y[final], groups[final], ch_names


def extract_minimal_features(X, sfreq, ch_names):
    """5-dim: frontal_theta_rel, frontal_alpha_rel, alpha_asym_frontal, frontal_TBR, riem_ct."""
    n_win, n_ch, n_times = X.shape
    region_idx = build_region_indices(ch_names)
    nperseg = min(n_times, int(sfreq * 2))
    freqs, psd = welch(X, fs=sfreq, axis=2, nperseg=nperseg)
    total_power = np.sum(psd, axis=2)

    def band_rel(region, fmin, fmax):
        idx = region_idx[region]
        mask = (freqs >= fmin) & (freqs <= fmax)
        bp = np.mean(np.mean(psd[:, idx][:, :, mask], axis=2), axis=1)
        tp = np.mean(total_power[:, idx], axis=1)
        return (bp / (tp + 1e-10))[:, np.newaxis]

    feats = [
        band_rel("frontal", 4, 8),    # theta_rel
        band_rel("frontal", 8, 13),   # alpha_rel
    ]
    # alpha asymmetry frontal
    from modma_mdd_real_experiment import _alpha_asymmetry, EGI128_LEFT, EGI128_RIGHT
    bands = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
    alpha_mask = (freqs >= 8) & (freqs <= 13)
    alpha_power = np.mean(psd[:, :, alpha_mask], axis=2)
    feats.append(_alpha_asymmetry(alpha_power, ch_names, n_win, "frontal"))
    # TBR frontal
    fidx = region_idx["frontal"]
    theta_mask = (freqs >= 4) & (freqs <= 8)
    beta_mask = (freqs >= 13) & (freqs <= 30)
    theta_p = np.mean(np.mean(psd[:, fidx][:, :, theta_mask], axis=2), axis=1, keepdims=True)
    beta_p = np.mean(np.mean(psd[:, fidx][:, :, beta_mask], axis=2), axis=1, keepdims=True)
    feats.append(theta_p / (beta_p + 1e-10))
    # riem central-temporal
    ct_regions = ["central", "temporal"]
    sigs = []
    for r in ct_regions:
        idx = region_idx[r]
        sigs.append(np.mean(X[:, idx, :], axis=1) if idx else np.zeros((n_win, n_times)))
    sigs = np.stack(sigs, axis=1)
    covs = np.array([np.cov(sigs[i]) for i in range(n_win)])
    covs += 1e-6 * np.eye(2)[np.newaxis]
    feats.append(np.log(np.abs(covs[:, 0, 1:2]) + 1e-10))

    names = ["frontal_theta_rel", "frontal_alpha_rel", "alpha_asym_frontal",
             "frontal_TBR", "riem_central_temporal"]
    return np.column_stack(feats), names


def run_cv(features, y, groups, tune=True, label=""):
    """Run 5-fold CV, return subject-level BA."""
    ug, fi = np.unique(groups, return_index=True)
    gl = y[fi]
    n_splits = min(5, np.min(np.unique(gl, return_counts=True)[1]))
    n_splits = max(2, n_splits)
    cv = StratifiedGroupKFold(n_splits=n_splits)

    y_true_s, y_pred_s = {}, {}
    for train_ix, test_ix in cv.split(features, y, groups=groups):
        X_tr, y_tr, g_tr = features[train_ix], y[train_ix], groups[train_ix]
        X_te, y_te, g_te = features[test_ix], y[test_ix], groups[test_ix]

        if tune:
            pipe = Pipeline([("scaler", StandardScaler()),
                             ("clf", SVC(kernel="rbf", class_weight="balanced",
                                         probability=True, random_state=42))])
            ug2, fi2 = np.unique(g_tr, return_index=True)
            min_cls = np.min(np.bincount(y_tr[fi2])) if len(np.unique(y_tr[fi2])) > 1 else 0
            if min_cls >= 3:
                grid = GridSearchCV(pipe, {"clf__C": [0.1, 1.0, 10.0]},
                                    cv=StratifiedGroupKFold(n_splits=min(3, min_cls)),
                                    scoring="balanced_accuracy")
                grid.fit(X_tr, y_tr, groups=g_tr)
                pipe = grid.best_estimator_
            sw = compute_subject_balanced_sample_weights(g_tr)
            pipe.fit(X_tr, y_tr, clf__sample_weight=sw)
        else:
            pipe = Pipeline([("scaler", StandardScaler()),
                             ("clf", SVC(kernel="rbf", C=1.0, class_weight="balanced",
                                         probability=True, random_state=42))])
            sw = compute_subject_balanced_sample_weights(g_tr)
            pipe.fit(X_tr, y_tr, clf__sample_weight=sw)

        preds = pipe.predict_proba(X_te)[:, 1]
        for i, g in enumerate(g_te):
            if g not in y_true_s:
                y_true_s[g] = y_te[i]
                y_pred_s[g] = []
            y_pred_s[g].append(preds[i])

    subjs = list(y_true_s.keys())
    yt = np.array([y_true_s[s] for s in subjs])
    yp = (np.array([np.mean(y_pred_s[s]) for s in subjs]) >= 0.5).astype(int)
    ba = balanced_accuracy_score(yt, yp)
    logger.info(f"  {label}: BA={ba:.3f} ({len(subjs)} subjects, {features.shape[1]} features)")
    return ba


if __name__ == "__main__":
    X_raw, y, groups, ch_names = load_filtered_data()
    logger.info(f"Data: {X_raw.shape[0]} windows, {len(np.unique(groups))} subjects")

    # 29-dim features
    feats_29, _ = extract_features(X_raw, sfreq=SFREQ, ch_names=ch_names)
    # 5-dim features
    feats_5, _ = extract_minimal_features(X_raw, SFREQ, ch_names)

    logger.info("\n=== Experiment A: 29-dim + GridSearchCV ===")
    run_cv(feats_29, y, groups, tune=True, label="29dim+tune")

    logger.info("\n=== Experiment B: 29-dim + fixed C=1.0 ===")
    run_cv(feats_29, y, groups, tune=False, label="29dim+fixed")

    logger.info("\n=== Experiment C: 5-dim + GridSearchCV ===")
    run_cv(feats_5, y, groups, tune=True, label="5dim+tune")

    logger.info("\n=== Experiment D: 5-dim + fixed C=1.0 ===")
    run_cv(feats_5, y, groups, tune=False, label="5dim+fixed")

    # Bonus: Logistic regression (linear, less overfitting)
    logger.info("\n=== Experiment E: 5-dim + Logistic C=0.1 ===")
    # Quick inline
    from sklearn.metrics import balanced_accuracy_score as bas
    ug, fi = np.unique(groups, return_index=True)
    gl = y[fi]
    ns = max(2, min(5, np.min(np.unique(gl, return_counts=True)[1])))
    cv = StratifiedGroupKFold(n_splits=ns)
    yt_s, yp_s = {}, {}
    for tr, te in cv.split(feats_5, y, groups=groups):
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(C=0.1, class_weight="balanced",
                                                    max_iter=1000, random_state=42))])
        sw = compute_subject_balanced_sample_weights(groups[tr])
        pipe.fit(feats_5[tr], y[tr], clf__sample_weight=sw)
        preds = pipe.predict_proba(feats_5[te])[:, 1]
        for i, g in enumerate(groups[te]):
            if g not in yt_s:
                yt_s[g] = y[te][i]
                yp_s[g] = []
            yp_s[g].append(preds[i])
    subjs = list(yt_s.keys())
    yt = np.array([yt_s[s] for s in subjs])
    yp = (np.array([np.mean(yp_s[s]) for s in subjs]) >= 0.5).astype(int)
    logger.info(f"  5dim+logistic: BA={bas(yt, yp):.3f} ({len(subjs)} subjects)")
