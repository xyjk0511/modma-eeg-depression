"""TDBRAIN opt v7: band-specific Riemannian + tuning."""
import argparse, logging, json, os, time
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.signal import butter, sosfilt
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from external_data_adapter import load_tdbrain_features
from modma_mdd_real_experiment import (
    compute_subject_balanced_sample_weights,
    build_region_indices,
)

logger = logging.getLogger(__name__)


def compute_hjorth(X, ch_names):
    n_win, n_ch, n_t = X.shape
    region_idx = build_region_indices(ch_names)
    feats = []
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
    return np.column_stack(feats)


def bandpass(X, fmin, fmax, sfreq):
    sos = butter(4, [fmin, fmax], btype='band', fs=sfreq, output='sos')
    return sosfilt(sos, X, axis=2)


def optimal_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return thr[np.argmax((tpr + (1 - fpr)) / 2)]


def subject_aggregate(g_arr, y_arr, prob_arr):
    y_true_s, y_prob_s = {}, {}
    for i, g in enumerate(g_arr):
        if g not in y_true_s:
            y_true_s[g] = y_arr[i]
            y_prob_s[g] = []
        y_prob_s[g].append(prob_arr[i])
    subjs = list(y_true_s.keys())
    yt = np.array([y_true_s[s] for s in subjs])
    yp = np.array([np.mean(y_prob_s[s]) for s in subjs])
    return yt, yp


def eval_metrics(yt, yp):
    ba = balanced_accuracy_score(yt, (yp >= 0.5).astype(int))
    auc = roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else 0.5
    n = len(yt)
    preds = np.zeros(n)
    for i in range(n):
        m = np.ones(n, dtype=bool); m[i] = False
        preds[i] = 1 if yp[i] >= optimal_threshold(yt[m], yp[m]) else 0
    ba_n = balanced_accuracy_score(yt, preds)
    return {"ba": round(ba, 4), "ba_nested": round(ba_n, 4), "auc": round(auc, 4)}


def extract_riemann_features(X_train, X_test, estimator="lwf"):
    """Tangent space features from covariance matrices."""
    cov = Covariances(estimator=estimator)
    ts = TangentSpace()
    C_tr = cov.fit_transform(X_train)
    F_tr = ts.fit_transform(C_tr)
    C_te = cov.transform(X_test)
    F_te = ts.transform(C_te)
    return F_tr, F_te


def extract_band_riemann(X_train, X_test, sfreq, bands, estimator="lwf"):
    """Band-specific Riemannian: filter → cov → tangent space per band, concat."""
    all_tr, all_te = [], []
    for fmin, fmax in bands:
        X_tr_b = bandpass(X_train, fmin, fmax, sfreq)
        X_te_b = bandpass(X_test, fmin, fmax, sfreq)
        F_tr, F_te = extract_riemann_features(X_tr_b, X_te_b, estimator)
        all_tr.append(F_tr)
        all_te.append(F_te)
    return np.column_stack(all_tr), np.column_stack(all_te)


def run_cv(X_raw, feat_27, y, groups, n_splits, seed, method, C=10.0, sfreq=125.0):
    cv = StratifiedGroupKFold(n_splits=n_splits)
    all_g, all_y, all_p = [], [], []

    bands = [(4, 8), (8, 13), (13, 30)]  # theta, alpha, beta

    for train_ix, test_ix in cv.split(X_raw, y, groups=groups):
        X_tr, y_tr, g_tr = X_raw[train_ix], y[train_ix], groups[train_ix]
        X_te, y_te, g_te = X_raw[test_ix], y[test_ix], groups[test_ix]
        sw = compute_subject_balanced_sample_weights(g_tr)

        if method == "riemann+hc":
            R_tr, R_te = extract_riemann_features(X_tr, X_te)
            X_tr_f = np.column_stack([R_tr, feat_27[train_ix]])
            X_te_f = np.column_stack([R_te, feat_27[test_ix]])

        elif method == "band_riemann":
            X_tr_f, X_te_f = extract_band_riemann(X_tr, X_te, sfreq, bands)

        elif method == "band_riemann+hc":
            R_tr, R_te = extract_band_riemann(X_tr, X_te, sfreq, bands)
            X_tr_f = np.column_stack([R_tr, feat_27[train_ix]])
            X_te_f = np.column_stack([R_te, feat_27[test_ix]])

        elif method == "riemann+band_riemann+hc":
            R_tr, R_te = extract_riemann_features(X_tr, X_te)
            B_tr, B_te = extract_band_riemann(X_tr, X_te, sfreq, bands)
            X_tr_f = np.column_stack([R_tr, B_tr, feat_27[train_ix]])
            X_te_f = np.column_stack([R_te, B_te, feat_27[test_ix]])

        else:  # handcraft baseline
            X_tr_f = feat_27[train_ix]
            X_te_f = feat_27[test_ix]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=2000, random_state=seed)),
        ])
        pipe.fit(X_tr_f, y_tr, clf__sample_weight=sw)
        probs = pipe.predict_proba(X_te_f)[:, 1]

        all_g.extend(g_te)
        all_y.extend(y_te)
        all_p.extend(probs)

    yt, yp = subject_aggregate(np.array(all_g), np.array(all_y), np.array(all_p))
    return eval_metrics(yt, yp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tdbrain-root", required=True)
    parser.add_argument("--output-dir", default="results_tdbrain_opt7")
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    logger.info("Loading TDBRAIN...")
    features, y, groups, ch_names, qc, X_raw = load_tdbrain_features(
        args.tdbrain_root, use_formal_status=True, return_windows=True)
    hjorth = compute_hjorth(X_raw, ch_names)
    feat_27 = np.column_stack([features, hjorth])

    unique_g, first_idx = np.unique(groups, return_index=True)
    g_labels = y[first_idx]
    n_mdd, n_hc = np.sum(g_labels == 1), np.sum(g_labels == 0)
    n_splits = min(5, min(n_mdd, n_hc))
    if n_splits < 2:
        n_splits = 2
    logger.info(f"Subjects: {n_mdd} MDD, {n_hc} HC | Windows: {len(y)}")

    methods = ["handcraft", "riemann+hc", "band_riemann", "band_riemann+hc",
               "riemann+band_riemann+hc"]
    results = []

    for method in methods:
        for C in [1.0, 10.0]:
            try:
                r = run_cv(X_raw, feat_27, y, groups, n_splits, args.seed, method, C)
                r["method"] = method
                r["C"] = C
                results.append(r)
                logger.info(f"{method:>30s} C={C:<5}: "
                           f"BA={r['ba']:.3f} BA_n={r['ba_nested']:.3f} AUC={r['auc']:.3f}")
            except Exception as e:
                logger.warning(f"{method} C={C}: FAILED - {e}")

    best = max(results, key=lambda x: (x["ba_nested"], x["auc"]))
    logger.info(f"\nBest: {best['method']} C={best['C']}")
    logger.info(f"  BA={best['ba']:.3f} BA_n={best['ba_nested']:.3f} AUC={best['auc']:.3f}")

    # Permutation test
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
            r = run_cv(X_raw, feat_27, y_perm, groups, n_splits, args.seed,
                      best["method"], best["C"])
            if r["ba_nested"] >= observed:
                count_ge += 1
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            logger.info(f"  perm {i+1}/{args.n_permutations}")

    p_val = (count_ge + 1) / (args.n_permutations + 1)
    logger.info(f"p-value: {p_val:.4f}")

    output = {
        "dataset": "TDBRAIN_formal_status",
        "n_mdd": int(n_mdd), "n_hc": int(n_hc), "n_windows": int(len(y)),
        "results": results, "best": best,
        "p_value": round(p_val, 4), "n_permutations": args.n_permutations,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    path = os.path.join(args.output_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved to {path}")


if __name__ == "__main__":
    main()
