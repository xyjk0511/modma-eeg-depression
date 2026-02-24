"""TDBRAIN opt v3: longer crop + XGBoost + feature selection."""
import argparse, logging, json, os, time
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _root import ROOT

from external_data_adapter import load_tdbrain_features
from modma_mdd_real_experiment import (
    compute_subject_balanced_sample_weights,
    build_region_indices,
)

logger = logging.getLogger(__name__)


def compute_hjorth(X, ch_names):
    n_win, n_ch, n_t = X.shape
    region_idx = build_region_indices(ch_names)
    feats, names = [], []
    for region in ["frontal", "central", "temporal", "parietal"]:
        idx = region_idx[region]
        if not idx:
            feats.extend([np.zeros((n_win, 1))] * 3)
        else:
            sig = np.mean(X[:, idx, :], axis=1)
            var0 = np.var(sig, axis=1, keepdims=True)
            d1 = np.diff(sig, axis=1)
            var1 = np.var(d1, axis=1, keepdims=True)
            d2 = np.diff(d1, axis=1)
            var2 = np.var(d2, axis=1, keepdims=True)
            mob = np.sqrt(var1 / (var0 + 1e-10))
            mob_d1 = np.sqrt(var2 / (var1 + 1e-10))
            feats.extend([np.log(var0 + 1e-10), mob, mob_d1 / (mob + 1e-10)])
        names.extend([f"{region}_activity", f"{region}_mobility", f"{region}_complexity"])
    return np.column_stack(feats), names


def optimal_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return thr[np.argmax((tpr + (1 - fpr)) / 2)]


def subject_cv(features, y, groups, n_splits, seed, model_type, **model_kw):
    cv = StratifiedGroupKFold(n_splits=n_splits)
    y_true_s, y_prob_s = {}, {}

    for train_ix, test_ix in cv.split(features, y, groups=groups):
        X_tr, y_tr, g_tr = features[train_ix], y[train_ix], groups[train_ix]
        X_te, y_te, g_te = features[test_ix], y[test_ix], groups[test_ix]
        sw = compute_subject_balanced_sample_weights(g_tr)

        if model_type == "gb":
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GradientBoostingClassifier(
                    n_estimators=model_kw.get("n_est", 100),
                    max_depth=model_kw.get("depth", 3),
                    learning_rate=model_kw.get("lr", 0.1),
                    subsample=0.8, random_state=seed)),
            ])
            pipe.fit(X_tr, y_tr, clf__sample_weight=sw)
        else:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    C=model_kw.get("C", 1.0),
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

    ba = balanced_accuracy_score(yt, (yp >= 0.5).astype(int))
    auc = roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else 0.5

    # Nested LOO threshold
    n = len(yt)
    preds = np.zeros(n)
    for i in range(n):
        m = np.ones(n, dtype=bool); m[i] = False
        preds[i] = 1 if yp[i] >= optimal_threshold(yt[m], yp[m]) else 0
    ba_nested = balanced_accuracy_score(yt, preds)

    return {"ba": round(ba, 4), "ba_nested": round(ba_nested, 4),
            "auc": round(auc, 4), "yt": yt, "yp": yp}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tdbrain-root", required=True)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/results_tdbrain_opt3"))
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--crop-duration", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    logger.info(f"Loading TDBRAIN (crop={args.crop_duration}s)...")
    features, y, groups, ch_names, qc, X_raw = load_tdbrain_features(
        args.tdbrain_root, use_formal_status=True,
        crop_duration=args.crop_duration, return_windows=True)

    hjorth_feat, hjorth_names = compute_hjorth(X_raw, ch_names)
    feat_27 = np.column_stack([features, hjorth_feat])

    unique_g, first_idx = np.unique(groups, return_index=True)
    g_labels = y[first_idx]
    n_mdd, n_hc = np.sum(g_labels == 1), np.sum(g_labels == 0)
    n_splits = min(5, min(n_mdd, n_hc))
    if n_splits < 2:
        n_splits = 2
    logger.info(f"Subjects: {n_mdd} MDD, {n_hc} HC | Windows: {len(y)} | Splits: {n_splits}")

    # Configs: Logistic + GradientBoosting
    configs = [
        ("logistic", "27dim", feat_27, {"C": 10.0}),
        ("logistic", "15dim", features, {"C": 1.0}),
        ("logistic", "15dim", features, {"C": 10.0}),
        ("gb", "27dim", feat_27, {"n_est": 50, "depth": 2, "lr": 0.05}),
        ("gb", "27dim", feat_27, {"n_est": 100, "depth": 2, "lr": 0.05}),
        ("gb", "27dim", feat_27, {"n_est": 100, "depth": 3, "lr": 0.05}),
        ("gb", "27dim", feat_27, {"n_est": 100, "depth": 3, "lr": 0.1}),
        ("gb", "15dim", features, {"n_est": 100, "depth": 3, "lr": 0.05}),
        ("gb", "15dim", features, {"n_est": 50, "depth": 2, "lr": 0.1}),
    ]

    results = []
    for model_type, fname, fdata, kw in configs:
        try:
            r = subject_cv(fdata, y, groups, n_splits, args.seed, model_type, **kw)
            kw_str = " ".join(f"{k}={v}" for k, v in kw.items())
            entry = {"model": model_type, "features": fname, "params": kw_str,
                     "ba": r["ba"], "ba_nested": r["ba_nested"], "auc": r["auc"]}
            results.append(entry)
            logger.info(f"{model_type:>8s} {fname:>5s} {kw_str:>25s}: "
                       f"BA={r['ba']:.3f} BA_n={r['ba_nested']:.3f} AUC={r['auc']:.3f}")
        except Exception as e:
            logger.warning(f"{model_type} {fname}: FAILED - {e}")

    best = max(results, key=lambda x: (x["ba_nested"], x["auc"]))
    logger.info(f"\nBest: {best['model']} {best['features']} {best['params']}")
    logger.info(f"  BA={best['ba']:.3f} BA_nested={best['ba_nested']:.3f} AUC={best['auc']:.3f}")

    # Permutation test
    best_cfg = next(c for c in configs
                    if c[0] == best["model"] and c[1] == best["features"]
                    and " ".join(f"{k}={v}" for k, v in c[3].items()) == best["params"])
    best_feat = best_cfg[2]
    best_kw = best_cfg[3]
    observed = best["ba_nested"]

    logger.info(f"Permutation test ({args.n_permutations} perms)...")
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
                          best["model"], **best_kw)
            if r["ba_nested"] >= observed:
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
        "crop_duration": args.crop_duration,
        "n_mdd": int(n_mdd), "n_hc": int(n_hc), "n_windows": int(len(y)),
        "grid_results": results, "best": best,
        "p_value": round(p_val, 4), "n_permutations": args.n_permutations,
        "elapsed_seconds": elapsed,
    }
    path = os.path.join(args.output_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved to {path}")


if __name__ == "__main__":
    main()
