"""TDBRAIN opt v5: ensemble + spectral entropy."""
import argparse, logging, json, os, time
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.signal import welch

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


def compute_spectral_entropy(X, sfreq, ch_names):
    """Spectral entropy per region — measures spectral flatness."""
    n_win, n_ch, n_t = X.shape
    region_idx = build_region_indices(ch_names)
    nperseg = min(n_t, int(sfreq * 2))
    feats, names = [], []
    for region in ["frontal", "central", "temporal", "parietal"]:
        idx = region_idx[region]
        if not idx:
            feats.append(np.zeros((n_win, 1)))
        else:
            sig = np.mean(X[:, idx, :], axis=1)
            _, psd = welch(sig, fs=sfreq, nperseg=nperseg, axis=1)
            # Normalize PSD to probability distribution
            psd_norm = psd / (np.sum(psd, axis=1, keepdims=True) + 1e-10)
            # Shannon entropy
            se = -np.sum(psd_norm * np.log2(psd_norm + 1e-10), axis=1)
            feats.append(se[:, np.newaxis])
        names.append(f"{region}_spectral_entropy")
    return np.column_stack(feats), names


def optimal_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return thr[np.argmax((tpr + (1 - fpr)) / 2)]


def ensemble_cv(features, y, groups, n_splits, seed, C_list):
    """Ensemble: average predictions from multiple C values."""
    cv = StratifiedGroupKFold(n_splits=n_splits)
    y_true_s, y_prob_s = {}, {}

    for train_ix, test_ix in cv.split(features, y, groups=groups):
        X_tr, y_tr, g_tr = features[train_ix], y[train_ix], groups[train_ix]
        X_te, y_te, g_te = features[test_ix], y[test_ix], groups[test_ix]
        sw = compute_subject_balanced_sample_weights(g_tr)

        # Train multiple models, average predictions
        all_probs = []
        for C in C_list:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=C, max_iter=1000, random_state=seed)),
            ])
            pipe.fit(X_tr, y_tr, clf__sample_weight=sw)
            all_probs.append(pipe.predict_proba(X_te)[:, 1])
        probs = np.mean(all_probs, axis=0)

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
    parser.add_argument("--output-dir", default="results_tdbrain_opt5")
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    logger.info("Loading TDBRAIN...")
    features, y, groups, ch_names, qc, X_raw = load_tdbrain_features(
        args.tdbrain_root, use_formal_status=True, return_windows=True)

    hjorth_feat, hjorth_names = compute_hjorth(X_raw, ch_names)
    se_feat, se_names = compute_spectral_entropy(X_raw, 125.0, ch_names)

    feat_27 = np.column_stack([features, hjorth_feat])
    feat_31 = np.column_stack([features, hjorth_feat, se_feat])
    all_names = list(range(15)) + hjorth_names + se_names

    unique_g, first_idx = np.unique(groups, return_index=True)
    g_labels = y[first_idx]
    n_mdd, n_hc = np.sum(g_labels == 1), np.sum(g_labels == 0)
    n_splits = min(5, min(n_mdd, n_hc))
    if n_splits < 2:
        n_splits = 2
    logger.info(f"Subjects: {n_mdd} MDD, {n_hc} HC | Windows: {len(y)} | Features: {feat_31.shape[1]}")

    results = []
    configs = [
        ("single_C10", feat_27, "27dim", [10.0]),
        ("single_C10", feat_31, "31dim", [10.0]),
        ("ens_1_10", feat_27, "27dim", [1.0, 10.0]),
        ("ens_1_10", feat_31, "31dim", [1.0, 10.0]),
        ("ens_01_1_10", feat_27, "27dim", [0.1, 1.0, 10.0]),
        ("ens_01_1_10", feat_31, "31dim", [0.1, 1.0, 10.0]),
        ("ens_1_5_10_20", feat_31, "31dim", [1.0, 5.0, 10.0, 20.0]),
    ]

    for name, fdata, fdim, C_list in configs:
        try:
            r = ensemble_cv(fdata, y, groups, n_splits, args.seed, C_list)
            entry = {"config": name, "features": fdim, "C_list": C_list,
                     "ba": r["ba"], "ba_nested": r["ba_nested"], "auc": r["auc"]}
            results.append(entry)
            logger.info(f"{name:>15s} {fdim:>5s}: BA={r['ba']:.3f} BA_n={r['ba_nested']:.3f} AUC={r['auc']:.3f}")
        except Exception as e:
            logger.warning(f"{name} {fdim}: {e}")

    best = max(results, key=lambda x: (x["ba_nested"], x["auc"]))
    logger.info(f"\nBest: {best['config']} {best['features']} C={best['C_list']}")
    logger.info(f"  BA={best['ba']:.3f} BA_nested={best['ba_nested']:.3f} AUC={best['auc']:.3f}")

    # Permutation test
    best_cfg = next(c for c in configs if c[0] == best["config"] and c[2] == best["features"])
    best_feat, best_Cs = best_cfg[1], best_cfg[3]
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
            r = ensemble_cv(best_feat, y_perm, groups, n_splits, args.seed, best_Cs)
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
        "n_mdd": int(n_mdd), "n_hc": int(n_hc), "n_windows": int(len(y)),
        "feature_dims": feat_31.shape[1],
        "grid_results": [{k: v for k, v in r.items()} for r in results],
        "best": best, "p_value": round(p_val, 4),
        "n_permutations": args.n_permutations, "elapsed_seconds": elapsed,
    }
    path = os.path.join(args.output_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved to {path}")


if __name__ == "__main__":
    main()
