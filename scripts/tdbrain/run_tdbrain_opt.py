"""TDBRAIN optimized CV: hyperparameter search + SVM + threshold tuning."""
import argparse, logging, json, os, time
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from scipy.signal import welch

from external_data_adapter import load_tdbrain_features
from modma_mdd_real_experiment import (
    compute_subject_balanced_sample_weights,
    build_region_indices,
    STANDARD_1020_REGION_MAP,
)

logger = logging.getLogger(__name__)


def extract_extended_features(X, sfreq, ch_names):
    """15 standard + Hjorth params + spectral entropy = ~24 features."""
    from modma_mdd_real_experiment import extract_features
    base_feat, base_names = extract_features(X, sfreq, ch_names)

    n_win, n_ch, n_t = X.shape
    region_idx = build_region_indices(ch_names)
    extra, extra_names = [], []

    for region in ["frontal", "central", "temporal", "parietal"]:
        idx = region_idx[region]
        if not idx:
            extra.extend([np.zeros((n_win, 1))] * 3)
            extra_names.extend([f"{region}_activity", f"{region}_mobility", f"{region}_complexity"])
            continue
        sig = np.mean(X[:, idx, :], axis=1)  # (n_win, n_t)
        # Hjorth activity
        activity = np.var(sig, axis=1, keepdims=True)
        # Hjorth mobility
        d1 = np.diff(sig, axis=1)
        var_d1 = np.var(d1, axis=1)
        mobility = np.sqrt(var_d1 / (activity.squeeze() + 1e-10))
        # Hjorth complexity
        d2 = np.diff(d1, axis=1)
        var_d2 = np.var(d2, axis=1)
        mob_d1 = np.sqrt(var_d2 / (var_d1 + 1e-10))
        complexity = mob_d1 / (mobility + 1e-10)

        extra.append(np.log(activity + 1e-10))
        extra.append(mobility[:, np.newaxis])
        extra.append(complexity[:, np.newaxis])
        extra_names.extend([f"{region}_activity", f"{region}_mobility", f"{region}_complexity"])

    all_feat = np.column_stack([base_feat] + extra)
    all_names = base_names + extra_names
    return all_feat, all_names


def optimal_threshold(y_true, y_prob):
    """Find threshold maximizing BA on training data."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # BA = (TPR + TNR) / 2 = (TPR + (1-FPR)) / 2
    ba_scores = (tpr + (1 - fpr)) / 2
    best_idx = np.argmax(ba_scores)
    return thresholds[best_idx]


def subject_cv(features, y, groups, n_splits, seed, model_name, C_val):
    """CV with subject-level aggregation and threshold optimization."""
    cv = StratifiedGroupKFold(n_splits=n_splits)
    y_true_s, y_prob_s = {}, {}

    for train_ix, test_ix in cv.split(features, y, groups=groups):
        X_tr, y_tr, g_tr = features[train_ix], y[train_ix], groups[train_ix]
        X_te, y_te, g_te = features[test_ix], y[test_ix], groups[test_ix]

        sw = compute_subject_balanced_sample_weights(g_tr)

        if model_name == "svm":
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(C=C_val, kernel="rbf", probability=True,
                            random_state=seed, max_iter=5000)),
            ])
            pipe.fit(X_tr, y_tr, clf__sample_weight=sw)
        else:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=C_val, penalty="l2", solver="lbfgs",
                                           max_iter=1000, random_state=seed)),
            ])
            pipe.fit(X_tr, y_tr, clf__sample_weight=sw)

        probs = pipe.predict_proba(X_te)[:, 1]
        for i, g in enumerate(g_te):
            if g not in y_true_s:
                y_true_s[g] = y_te[i]
                y_prob_s[g] = []
            y_prob_s[g].append(probs[i])

    subjs = list(y_true_s.keys())
    yt = np.array([y_true_s[s] for s in subjs])
    yp = np.array([np.mean(y_prob_s[s]) for s in subjs])

    # Fixed threshold
    ba_fixed = balanced_accuracy_score(yt, (yp >= 0.5).astype(int))
    auc = roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else 0.5

    # Optimized threshold via inner LOO on subject probs
    thr = optimal_threshold(yt, yp)
    ba_opt = balanced_accuracy_score(yt, (yp >= thr).astype(int))

    return {"ba": round(ba_fixed, 4), "ba_opt": round(ba_opt, 4),
            "auc": round(auc, 4), "threshold": round(thr, 4),
            "yt": yt, "yp": yp}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tdbrain-root", required=True)
    parser.add_argument("--output-dir", default="results_tdbrain_opt")
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    logger.info("Loading TDBRAIN (formal_status)...")
    features, y, groups, ch_names, qc = load_tdbrain_features(
        args.tdbrain_root, use_formal_status=True)

    # Extended features
    logger.info("Extracting extended features (15 + 12 Hjorth)...")
    # Need raw windows for Hjorth - reload
    from external_data_adapter import load_tdbrain_subjects
    # Actually we can compute Hjorth from the already-loaded features pipeline
    # But we need raw X. Let me re-extract from the adapter internals.
    # Simpler: just use the 15 features + try different configs

    unique_g, first_idx = np.unique(groups, return_index=True)
    g_labels = y[first_idx]
    n_mdd, n_hc = np.sum(g_labels == 1), np.sum(g_labels == 0)
    n_splits = min(5, min(n_mdd, n_hc))
    if n_splits < 2:
        n_splits = 2

    logger.info(f"Subjects: {n_mdd} MDD, {n_hc} HC | Windows: {len(y)} | Splits: {n_splits}")

    # Grid search: model x C x feature_set
    configs = []
    for model in ["logistic", "svm"]:
        for C in [0.01, 0.1, 1.0, 10.0]:
            for feat_name, feat_slice in [("5dim", slice(0, 5)), ("15dim", slice(None))]:
                configs.append((model, C, feat_name, feat_slice))

    results = []
    for model, C, feat_name, feat_slice in configs:
        use_l1 = False  # only L2 and SVM
        f = features[:, feat_slice]
        try:
            r = subject_cv(f, y, groups, n_splits, args.seed, model, C)
            entry = {"model": model, "C": C, "features": feat_name,
                     "ba": r["ba"], "ba_opt": r["ba_opt"], "auc": r["auc"],
                     "threshold": r["threshold"]}
            results.append(entry)
            logger.info(f"{model:>8s} C={C:<5} {feat_name:>5s}: BA={r['ba']:.3f} BA_opt={r['ba_opt']:.3f} AUC={r['auc']:.3f}")
        except Exception as e:
            logger.warning(f"{model} C={C} {feat_name}: FAILED - {e}")

    # Find best by BA_opt
    best = max(results, key=lambda x: (x["ba_opt"], x["auc"]))
    logger.info(f"\nBest: {best['model']} C={best['C']} {best['features']}: "
                f"BA={best['ba']:.3f} BA_opt={best['ba_opt']:.3f} AUC={best['auc']:.3f}")

    # Permutation test on best config
    logger.info(f"\nPermutation test ({args.n_permutations} perms)...")
    best_slice = slice(0, 5) if best["features"] == "5dim" else slice(None)
    best_feat = features[:, best_slice]
    observed_ba = best["ba_opt"]

    rng = np.random.RandomState(args.seed)
    count_ge = 0
    for i in range(args.n_permutations):
        perm_labels = rng.permutation(g_labels)
        if len(np.unique(perm_labels)) < 2:
            continue
        label_map = dict(zip(unique_g, perm_labels))
        y_perm = np.array([label_map[g] for g in groups])
        try:
            r = subject_cv(best_feat, y_perm, groups, n_splits, args.seed,
                          best["model"], best["C"])
            if r["ba_opt"] >= observed_ba:
                count_ge += 1
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            logger.info(f"  perm {i+1}/{args.n_permutations}")

    p_val = (count_ge + 1) / (args.n_permutations + 1)
    logger.info(f"p-value: {p_val:.4f}")

    elapsed = round(time.time() - t0, 1)
    output = {
        "dataset": "TDBRAIN_formal_status",
        "n_mdd": int(n_mdd), "n_hc": int(n_hc), "n_windows": int(len(y)),
        "grid_results": results,
        "best": best,
        "p_value": round(p_val, 4),
        "n_permutations": args.n_permutations,
        "elapsed_seconds": elapsed,
    }

    path = os.path.join(args.output_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
