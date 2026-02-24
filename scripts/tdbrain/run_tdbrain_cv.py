"""TDBRAIN internal subject-level CV: can EEG features separate MDD from HC?"""
import argparse
import logging
import json
import os
import time
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from external_data_adapter import load_tdbrain_features
from modma_mdd_real_experiment import (
    build_feature_model_pipeline,
    compute_subject_balanced_sample_weights,
)

logger = logging.getLogger(__name__)


def subject_level_cv(features, y, groups, n_splits=5, seed=42, use_l1=False):
    """StratifiedGroupKFold CV with subject-level aggregation."""
    cv = StratifiedGroupKFold(n_splits=n_splits)
    y_true_subj, y_prob_subj = {}, {}

    for train_ix, test_ix in cv.split(features, y, groups=groups):
        X_tr, y_tr, g_tr = features[train_ix], y[train_ix], groups[train_ix]
        X_te, y_te, g_te = features[test_ix], y[test_ix], groups[test_ix]

        pipe = build_feature_model_pipeline(use_l1=use_l1)
        sw = compute_subject_balanced_sample_weights(g_tr)
        pipe.fit(X_tr, y_tr, clf__sample_weight=sw)
        probs = pipe.predict_proba(X_te)[:, 1]

        for i, g in enumerate(g_te):
            if g not in y_true_subj:
                y_true_subj[g] = y_te[i]
                y_prob_subj[g] = []
            y_prob_subj[g].append(probs[i])

    subjs = list(y_true_subj.keys())
    yt = np.array([y_true_subj[s] for s in subjs])
    yp = np.array([np.mean(y_prob_subj[s]) for s in subjs])
    ypred = (yp >= 0.5).astype(int)

    ba = balanced_accuracy_score(yt, ypred)
    auc = roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else 0.5
    return ba, auc, yt, yp


def permutation_test(features, y, groups, observed_ba, n_perm=200, seed=42, n_splits=5, use_l1=False):
    rng = np.random.RandomState(seed)
    unique_g, first_idx = np.unique(groups, return_index=True)
    group_labels = y[first_idx]

    count_ge = 0
    for i in range(n_perm):
        perm_labels = rng.permutation(group_labels)
        if len(np.unique(perm_labels)) < 2:
            continue
        label_map = dict(zip(unique_g, perm_labels))
        y_perm = np.array([label_map[g] for g in groups])
        try:
            ba, _, _, _ = subject_level_cv(features, y_perm, groups, n_splits, seed, use_l1)
            if ba >= observed_ba:
                count_ge += 1
        except Exception:
            pass
    p = (count_ge + 1) / (n_perm + 1)
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tdbrain-root", required=True)
    parser.add_argument("--output-dir", default="results_tdbrain_cv")
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--use-formal-status", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.time()
    logger.info("Loading TDBRAIN data...")
    features, y, groups, ch_names, qc = load_tdbrain_features(
        args.tdbrain_root, use_formal_status=args.use_formal_status)

    unique_g, first_idx = np.unique(groups, return_index=True)
    g_labels = y[first_idx]
    n_mdd = np.sum(g_labels == 1)
    n_hc = np.sum(g_labels == 0)
    n_splits = min(5, min(n_mdd, n_hc))
    if n_splits < 2:
        n_splits = 2

    logger.info(f"Subjects: {n_mdd} MDD, {n_hc} HC | Windows: {len(y)} | Splits: {n_splits}")

    # 5-dim CV
    logger.info("Running 5-dim CV...")
    ba5, auc5, _, _ = subject_level_cv(features[:, :5], y, groups, n_splits, args.seed)
    logger.info(f"5-dim: BA={ba5:.3f}, AUC={auc5:.3f}")

    # 15-dim CV (L1)
    logger.info("Running 15-dim CV...")
    ba15, auc15, yt, yp = subject_level_cv(features, y, groups, n_splits, args.seed, use_l1=True)
    logger.info(f"15-dim: BA={ba15:.3f}, AUC={auc15:.3f}")

    # Pick best for permutation test
    best_dim = "15-dim" if ba15 > ba5 else "5-dim"
    best_ba = max(ba5, ba15)
    best_feat = features if ba15 > ba5 else features[:, :5]
    best_l1 = ba15 > ba5

    logger.info(f"Permutation test on {best_dim} (BA={best_ba:.3f})...")
    p_val = permutation_test(best_feat, y, groups, best_ba,
                             args.n_permutations, args.seed, n_splits, best_l1)
    logger.info(f"p-value: {p_val:.4f}")

    elapsed = round(time.time() - t0, 1)
    result = {
        "dataset": "TDBRAIN",
        "use_formal_status": args.use_formal_status,
        "n_mdd": int(n_mdd), "n_hc": int(n_hc),
        "n_windows": int(len(y)),
        "ba_5dim": round(ba5, 4), "auc_5dim": round(auc5, 4),
        "ba_15dim": round(ba15, 4), "auc_15dim": round(auc15, 4),
        "best_model": best_dim, "best_ba": round(best_ba, 4),
        "p_value": round(p_val, 4),
        "n_permutations": args.n_permutations,
        "n_splits": n_splits,
        "elapsed_seconds": elapsed,
    }

    path = os.path.join(args.output_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results saved to {path}")
    logger.info(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
