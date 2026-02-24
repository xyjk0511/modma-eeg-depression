"""Phase 6: Internal MODMA validation — 5-dim vs 15-dim feature comparison.

Runs StratifiedGroupKFold CV on MODMA with both feature sets,
plus permutation test on 15-dim to assess significance.
"""
import argparse
import json
import logging
import os
import time

import numpy as np
from joblib import Parallel, delayed

import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _root import ROOT

from modma_mdd_real_experiment import (
    build_feature_model_pipeline,
    compute_subject_balanced_sample_weights,
    extract_features,
    load_participants,
    load_windows,
    run_simplified_cv,
    _run_one_perm_cv,
)

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Phase 6: 5-dim vs 15-dim MODMA validation")
    p.add_argument("--bids-root", required=True)
    p.add_argument("--output-dir", default=str(ROOT / "outputs/results_phase6"))
    p.add_argument("--n-permutations", type=int, default=1000)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resample-sfreq", type=float, default=125.0)
    p.add_argument("--crop-duration", type=float, default=60.0)
    p.add_argument("--window-sec", type=float, default=10.0)
    p.add_argument("--bad-amp-uv", type=float, default=200.0)
    p.add_argument("--highpass-freq", type=float, default=1.0)
    p.add_argument("--min-windows-per-subject", type=int, default=3)
    return p.parse_args(argv)


def _run_cv_with_pipeline(features, y, groups, n_splits, seed, use_l1=False):
    """CV using specified pipeline (l1 or l2)."""
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score

    cv = StratifiedGroupKFold(n_splits=n_splits)
    y_true_subj, y_prob_subj = {}, {}

    for train_ix, test_ix in cv.split(features, y, groups=groups):
        pipe = build_feature_model_pipeline(use_l1=use_l1)
        sw = compute_subject_balanced_sample_weights(groups[train_ix])
        pipe.fit(features[train_ix], y[train_ix], clf__sample_weight=sw)
        probs = pipe.predict_proba(features[test_ix])[:, 1]
        for i, g in enumerate(groups[test_ix]):
            y_true_subj.setdefault(g, y[test_ix][i])
            y_prob_subj.setdefault(g, []).append(probs[i])

    subjs = sorted(y_true_subj.keys())
    yt = np.array([y_true_subj[s] for s in subjs])
    yp = np.array([np.mean(y_prob_subj[s]) for s in subjs])
    ypred = (yp >= 0.5).astype(int)

    return {
        "balanced_accuracy": float(balanced_accuracy_score(yt, ypred)),
        "roc_auc": float(roc_auc_score(yt, yp)) if len(np.unique(yt)) > 1 else 0.5,
        "f1": float(f1_score(yt, ypred, zero_division=0)),
    }


def run_validation(args):
    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    # Load MODMA
    logger.info("Loading MODMA data...")
    participants_df = load_participants(os.path.join(args.bids_root, "participants.tsv"))
    X_win, y_win, groups, _, ch_names, _ = load_windows(
        participants_df, bids_root=args.bids_root,
        window_sec=args.window_sec, resample_sfreq=args.resample_sfreq,
        crop_duration=args.crop_duration, highpass_freq=args.highpass_freq,
        bad_amp_uv=args.bad_amp_uv,
    )
    if len(X_win) == 0:
        raise ValueError("No MODMA windows loaded")

    # QC: 200uV, 12% dynamic bad, skip W0, min 3 windows/subject
    n_ch = X_win.shape[1]
    thr_v = args.bad_amp_uv * 1e-6
    max_bad = max(1, int(n_ch * 0.12))
    bad_counts = np.sum(np.any(np.abs(X_win) > thr_v, axis=2), axis=1)
    qc_mask = bad_counts <= max_bad

    kept_g = groups[qc_mask]
    u_g, c_g = np.unique(kept_g, return_counts=True)
    valid_g = set(u_g[c_g >= args.min_windows_per_subject])
    final_mask = qc_mask & np.array([g in valid_g for g in groups])

    X_f = X_win[final_mask]
    y_f = y_win[final_mask]
    g_f = groups[final_mask]
    logger.info(f"MODMA: {len(set(g_f))} subjects, {len(X_f)} windows after QC")

    # Extract 15-dim features
    features_15, feat_names = extract_features(X_f, sfreq=args.resample_sfreq, ch_names=ch_names)
    features_5 = features_15[:, :5]

    # Determine n_splits
    u_groups, first_idx = np.unique(g_f, return_index=True)
    group_labels = y_f[first_idx]
    n_splits = min(5, int(np.min(np.unique(group_labels, return_counts=True)[1])))
    n_splits = max(2, n_splits)

    # CV: 5-dim (L2) vs 15-dim (L1)
    logger.info("Running 5-dim CV (L2)...")
    res_5 = _run_cv_with_pipeline(features_5, y_f, g_f, n_splits, args.seed, use_l1=False)
    logger.info(f"5-dim BA={res_5['balanced_accuracy']:.3f}")

    logger.info("Running 15-dim CV (L1)...")
    res_15 = _run_cv_with_pipeline(features_15, y_f, g_f, n_splits, args.seed, use_l1=True)
    logger.info(f"15-dim BA={res_15['balanced_accuracy']:.3f}")

    # Permutation test on 15-dim
    p_value = None
    if args.n_permutations > 0:
        rng = np.random.RandomState(args.seed)
        unique_g, first_i = np.unique(g_f, return_index=True)
        g_labels = y_f[first_i]

        jobs = []
        for _ in range(args.n_permutations):
            sl = rng.permutation(g_labels)
            if len(np.unique(sl)) < 2:
                continue
            lm = dict(zip(unique_g, sl))
            jobs.append(np.array([lm[g] for g in g_f]))

        logger.info(f"Running {len(jobs)} permutations on 15-dim...")
        perm_bas = Parallel(n_jobs=args.n_jobs)(
            delayed(_run_one_perm_cv)(features_15, yp, g_f, n_splits, args.seed)
            for yp in jobs
        )
        perm_bas = np.array([b for b in perm_bas if b is not None])
        if len(perm_bas) > 0:
            actual_ba = res_15["balanced_accuracy"]
            p_value = float((np.sum(perm_bas >= actual_ba) + 1) / (len(perm_bas) + 1))
        logger.info(f"Permutation p={p_value}")

    # Save metrics
    report = {
        "ba_5dim": res_5["balanced_accuracy"],
        "ba_15dim": res_15["balanced_accuracy"],
        "auc_5dim": res_5["roc_auc"],
        "auc_15dim": res_15["roc_auc"],
        "p_value": p_value,
        "n_permutations": args.n_permutations,
        "feature_names": feat_names,
        "n_subjects": len(set(g_f)),
        "n_windows": len(X_f),
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Results saved to {args.output_dir}/metrics.json")
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    try:
        run_validation(args)
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Error: {e}")
        raise SystemExit(1)
