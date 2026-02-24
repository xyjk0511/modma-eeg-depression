"""Phase 6: Cross-dataset replication with 15-dim literature-informed features.

Train 15-dim L1 model on full MODMA, test on TDBRAIN, compare with Phase 5 baseline.
"""
import argparse
import json
import logging
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from modma_mdd_real_experiment import (
    build_feature_model_pipeline,
    compute_subject_balanced_sample_weights,
    determine_conclusion,
    extract_features,
    get_ci_bootstrap,
    load_participants,
    load_windows,
)
from external_data_adapter import load_tdbrain_features

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Phase 6: 15-dim cross-dataset replication (MODMA->TDBRAIN)"
    )
    p.add_argument("--bids-root", required=True)
    p.add_argument("--tdbrain-root", required=True)
    p.add_argument("--output-dir", default="results_phase6_replication")
    p.add_argument("--n-permutations", type=int, default=1000)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resample-sfreq", type=float, default=125.0)
    p.add_argument("--crop-duration", type=float, default=60.0)
    p.add_argument("--window-sec", type=float, default=10.0)
    p.add_argument("--bad-amp-uv", type=float, default=200.0)
    p.add_argument("--highpass-freq", type=float, default=1.0)
    p.add_argument("--min-windows-per-subject", type=int, default=3)
    p.add_argument("--use-formal-status", action="store_true")
    return p.parse_args(argv)


def _aggregate_to_subject(probs, y, groups):
    """Aggregate window-level predictions to subject-level (mean prob)."""
    subj_probs, subj_true = {}, {}
    for prob, label, g in zip(probs, y, groups):
        subj_probs.setdefault(g, []).append(prob)
        subj_true[g] = label
    subjects = sorted(subj_probs.keys())
    y_true = np.array([subj_true[s] for s in subjects])
    y_prob = np.array([np.mean(subj_probs[s]) for s in subjects])
    return subjects, y_true, y_prob


def _one_perm_ba(pipe, ext_features, ext_groups, perm_labels, unique_subj):
    """Compute subject-level BA for one permutation."""
    label_map = dict(zip(unique_subj, perm_labels))
    y_perm = np.array([label_map[g] for g in ext_groups])
    probs = pipe.predict_proba(ext_features)[:, 1]
    _, y_true_s, y_prob_s = _aggregate_to_subject(probs, y_perm, ext_groups)
    y_pred_s = (y_prob_s >= 0.5).astype(int)
    return balanced_accuracy_score(y_true_s, y_pred_s)


def _compute_feature_distributions(modma_feat, tdbrain_feat, feat_names):
    """Per-feature mean/std for both datasets."""
    dists = {}
    for i, name in enumerate(feat_names):
        dists[name] = {
            "modma_mean": float(np.mean(modma_feat[:, i])),
            "modma_std": float(np.std(modma_feat[:, i])),
            "tdbrain_mean": float(np.mean(tdbrain_feat[:, i])),
            "tdbrain_std": float(np.std(tdbrain_feat[:, i])),
        }
    return dists


def run_replication(args):
    """Train 15-dim L1 on MODMA, test on TDBRAIN, permutation test, report."""
    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    # --- 1. Load MODMA ---
    logger.info("Loading MODMA training data...")
    participants_path = os.path.join(args.bids_root, "participants.tsv")
    if not os.path.exists(participants_path):
        raise FileNotFoundError(f"MODMA participants.tsv not found: {participants_path}")

    participants_df = load_participants(participants_path)
    X_win, y_win, groups, _, ch_names, _ = load_windows(
        participants_df, bids_root=args.bids_root,
        window_sec=args.window_sec, resample_sfreq=args.resample_sfreq,
        crop_duration=args.crop_duration, highpass_freq=args.highpass_freq,
        bad_amp_uv=args.bad_amp_uv,
    )
    if len(X_win) == 0:
        raise ValueError("No MODMA windows loaded")

    # QC
    n_ch = X_win.shape[1]
    thr_v = args.bad_amp_uv * 1e-6
    max_bad = max(1, int(n_ch * 0.12))
    bad_counts = np.sum(np.any(np.abs(X_win) > thr_v, axis=2), axis=1)
    qc_mask = bad_counts <= max_bad
    kept_g = groups[qc_mask]
    u_g, c_g = np.unique(kept_g, return_counts=True)
    valid_g = set(u_g[c_g >= args.min_windows_per_subject])
    final_mask = qc_mask & np.array([g in valid_g for g in groups])

    X_modma = X_win[final_mask]
    y_modma = y_win[final_mask]
    g_modma = groups[final_mask]
    logger.info(f"MODMA: {len(set(g_modma))} subjects, {len(X_modma)} windows")

    modma_features, feat_names = extract_features(
        X_modma, sfreq=args.resample_sfreq, ch_names=ch_names
    )

    # --- 2. Train on full MODMA with L1 (15-dim) ---
    logger.info("Training 15-dim L1 pipeline on full MODMA...")
    pipe = build_feature_model_pipeline(use_l1=True)
    sw = compute_subject_balanced_sample_weights(g_modma)
    pipe.fit(modma_features, y_modma, clf__sample_weight=sw)

    # --- 3. Load TDBRAIN ---
    logger.info("Loading TDBRAIN external test data...")
    ext_features, ext_y, ext_groups, _, ext_qc = load_tdbrain_features(
        args.tdbrain_root, resample_sfreq=args.resample_sfreq,
        crop_duration=args.crop_duration, highpass_freq=args.highpass_freq,
        bad_amp_uv=args.bad_amp_uv, window_sec=args.window_sec,
        min_windows_per_subject=args.min_windows_per_subject,
        use_formal_status=args.use_formal_status,
    )
    logger.info(
        f"TDBRAIN: {ext_qc.get('subjects_final', '?')} subjects, "
        f"{len(ext_features)} windows, {ext_features.shape[1]} features"
    )

    # --- 4. Predict on TDBRAIN ---
    ext_probs = pipe.predict_proba(ext_features)[:, 1]
    subjects, y_subj_true, y_subj_prob = _aggregate_to_subject(
        ext_probs, ext_y, ext_groups
    )
    y_subj_pred = (y_subj_prob >= 0.5).astype(int)

    ba = balanced_accuracy_score(y_subj_true, y_subj_pred)
    auc = roc_auc_score(y_subj_true, y_subj_prob) if len(np.unique(y_subj_true)) > 1 else 0.5
    f1 = f1_score(y_subj_true, y_subj_pred, zero_division=0)
    ba_ci = get_ci_bootstrap(y_subj_true, y_subj_pred, balanced_accuracy_score, seed=args.seed)
    logger.info(f"Subject-level BA={ba:.3f} CI=[{ba_ci[0]:.3f}, {ba_ci[1]:.3f}]")

    # --- 5. Cross-dataset permutation test ---
    unique_subj, first_idx = np.unique(ext_groups, return_index=True)
    subj_labels = ext_y[first_idx]

    rng = np.random.RandomState(args.seed)
    perm_label_sets = []
    for _ in range(args.n_permutations):
        pl = rng.permutation(subj_labels)
        if len(np.unique(pl)) >= 2:
            perm_label_sets.append(pl)

    logger.info(f"Running {len(perm_label_sets)} permutations (n_jobs={args.n_jobs})...")
    perm_bas = Parallel(n_jobs=args.n_jobs)(
        delayed(_one_perm_ba)(pipe, ext_features, ext_groups, pl, unique_subj)
        for pl in perm_label_sets
    )
    perm_bas = np.array([b for b in perm_bas if b is not None])
    p_value = (np.sum(perm_bas >= ba) + 1) / (len(perm_bas) + 1) if len(perm_bas) > 0 else None
    logger.info(f"Permutation p={p_value}")

    # --- 6. Feature distribution analysis ---
    feat_dists = _compute_feature_distributions(modma_features, ext_features, feat_names)

    # --- 7. Build report ---
    report = {
        "experiment_type": "cross_dataset_replication_15dim",
        "primary_metric": "subject_level_balanced_accuracy",
        "train_dataset": "MODMA",
        "test_dataset": "TDBRAIN",
        "n_features": len(feat_names),
        "feature_names": feat_names,
        "model": "LogisticRegression_L1_C0.1",
        "modma_subjects": len(set(g_modma)),
        "modma_windows": len(X_modma),
        "tdbrain_subjects": ext_qc.get("subjects_final", 0),
        "tdbrain_windows": len(ext_features),
        "tdbrain_qc": ext_qc,
        "balanced_accuracy": ba,
        "balanced_accuracy_ci95": ba_ci,
        "roc_auc": auc,
        "f1": f1,
        "n_permutations": args.n_permutations,
        "effective_permutations": len(perm_bas),
        "permutation_pvalue": p_value,
        "permuted_bas_mean": float(np.mean(perm_bas)) if len(perm_bas) > 0 else None,
        "permuted_bas_std": float(np.std(perm_bas)) if len(perm_bas) > 0 else None,
        "feature_distributions": feat_dists,
        "phase5_baseline": {"ba": 0.500, "p_value": 1.0, "conclusion": "negative"},
        "seed": args.seed,
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    # Conclusion
    subj_labels_dict = dict(zip(subjects, y_subj_true.tolist()))
    conclusion = determine_conclusion(report, subj_labels_dict)
    report["conclusion"] = conclusion
    logger.info(f"Conclusion: {conclusion['conclusion']}")

    # --- 8. Save outputs ---
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_subj_true, y_subj_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC={auc:.2f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Phase 6: 15-dim Cross-Dataset ROC (MODMA->TDBRAIN)")
    ax.legend()
    fig.savefig(os.path.join(args.output_dir, "roc_curve.png"))
    plt.close(fig)

    # Confusion matrix
    cm = confusion_matrix(y_subj_true, y_subj_pred)
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap="Blues")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Phase 6: Subject-Level Confusion Matrix")
    fig.savefig(os.path.join(args.output_dir, "confusion_matrix.png"))
    plt.close(fig)

    # Permutation scores CSV
    if len(perm_bas) > 0:
        import pandas as pd
        pd.DataFrame({"balanced_accuracy": perm_bas}).to_csv(
            os.path.join(args.output_dir, "permutation_scores.csv"), index=False
        )

    logger.info(f"Results saved to {args.output_dir}/")
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    if not os.path.isdir(args.tdbrain_root):
        logger.error(
            f"TDBRAIN root not found: {args.tdbrain_root}\n"
            "Download TDBRAIN from https://brainclinics.com/resources/\n"
            "Then re-run with --tdbrain-root <local-path>"
        )
        sys.exit(1)

    try:
        run_replication(args)
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
