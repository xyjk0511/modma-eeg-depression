"""TDBRAIN v9: fix statistical methodology — permutation wraps full grid search."""
import argparse, logging, json, os, time
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
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


def extract_riemann(X_tr, X_te, est="lwf"):
    cov = Covariances(estimator=est)
    ts = TangentSpace()
    F_tr = ts.fit_transform(cov.fit_transform(X_tr))
    F_te = ts.transform(cov.transform(X_te))
    return F_tr, F_te


def extract_band_riemann(X_tr, X_te, sfreq, bands, est="lwf"):
    all_tr, all_te = [], []
    for fmin, fmax in bands:
        Xb_tr = bandpass(X_tr, fmin, fmax, sfreq)
        Xb_te = bandpass(X_te, fmin, fmax, sfreq)
        F_tr, F_te = extract_riemann(Xb_tr, Xb_te, est)
        all_tr.append(F_tr)
        all_te.append(F_te)
    return np.column_stack(all_tr), np.column_stack(all_te)


BANDS = [(4, 8), (8, 13), (13, 30)]

# Fixed config grid — no post-hoc selection
CONFIGS = [
    ("band", 10.0, None),
    ("band+hc", 10.0, 30),
    ("band+broad", 1.0, 30),
    ("band+broad+hc", 1.0, 30),
]


def single_cv(X_raw, feat_27, y, groups, n_splits, seed, method, C, pca_n, sfreq=125.0):
    """One CV run. Returns fixed-threshold BA and AUC (no post-hoc threshold)."""
    cv = StratifiedGroupKFold(n_splits=n_splits)
    all_g, all_y, all_p = [], [], []

    for train_ix, test_ix in cv.split(X_raw, y, groups=groups):
        X_tr, y_tr, g_tr = X_raw[train_ix], y[train_ix], groups[train_ix]
        X_te, y_te, g_te = X_raw[test_ix], y[test_ix], groups[test_ix]
        sw = compute_subject_balanced_sample_weights(g_tr)

        parts_tr, parts_te = [], []
        if "band" in method:
            B_tr, B_te = extract_band_riemann(X_tr, X_te, sfreq, BANDS)
            parts_tr.append(B_tr); parts_te.append(B_te)
        if "broad" in method:
            R_tr, R_te = extract_riemann(X_tr, X_te)
            parts_tr.append(R_tr); parts_te.append(R_te)
        if "hc" in method:
            parts_tr.append(feat_27[train_ix]); parts_te.append(feat_27[test_ix])

        X_tr_f = np.column_stack(parts_tr)
        X_te_f = np.column_stack(parts_te)

        steps = [("scaler", StandardScaler())]
        if pca_n and pca_n < X_tr_f.shape[1]:
            steps.append(("pca", PCA(n_components=pca_n, random_state=seed)))
        steps.append(("clf", LogisticRegression(C=C, max_iter=2000, random_state=seed)))

        pipe = Pipeline(steps)
        pipe.fit(X_tr_f, y_tr, clf__sample_weight=sw)
        probs = pipe.predict_proba(X_te_f)[:, 1]
        all_g.extend(g_te); all_y.extend(y_te); all_p.extend(probs)

    yt, yp = subject_aggregate(np.array(all_g), np.array(all_y), np.array(all_p))
    # FIXED threshold only — no post-hoc optimization
    ba = balanced_accuracy_score(yt, (yp >= 0.5).astype(int))
    auc = roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else 0.5
    return ba, auc


def full_grid_search(X_raw, feat_27, y, groups, n_splits, seed):
    """Run all configs, return best BA (fixed threshold). This is the test statistic."""
    best_ba, best_auc, best_cfg = 0, 0, None
    for method, C, pca_n in CONFIGS:
        try:
            ba, auc = single_cv(X_raw, feat_27, y, groups, n_splits, seed, method, C, pca_n)
            if ba > best_ba or (ba == best_ba and auc > best_auc):
                best_ba, best_auc = ba, auc
                best_cfg = (method, C, pca_n)
        except Exception:
            pass
    if best_cfg is None:
        raise RuntimeError("All grid configs failed")
    return best_ba, best_auc, best_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tdbrain-root", required=True)
    parser.add_argument("--output-dir", default="results_tdbrain_v9")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    logger.info("Loading TDBRAIN (formal_status)...")
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

    # Step 1: Run full grid on real labels
    logger.info("Running grid search on real labels...")
    all_results = []
    for method, C, pca_n in CONFIGS:
        try:
            ba, auc = single_cv(X_raw, feat_27, y, groups, n_splits, args.seed, method, C, pca_n)
            pca_str = f"pca={pca_n}" if pca_n else "noPCA"
            all_results.append({"method": method, "C": C, "pca": pca_str,
                                "ba": round(ba, 4), "auc": round(auc, 4)})
            logger.info(f"  {method:>20s} C={C:<5} {pca_str:>7s}: BA={ba:.3f} AUC={auc:.3f}")
        except Exception as e:
            logger.warning(f"  {method} C={C}: FAILED - {e}")

    observed_ba, observed_auc, best_cfg = full_grid_search(
        X_raw, feat_27, y, groups, n_splits, args.seed)
    logger.info(f"\nBest: {best_cfg} BA={observed_ba:.3f} AUC={observed_auc:.3f}")

    # Step 2: Permutation test — re-run FULL grid search per permutation
    logger.info(f"\nPermutation test ({args.n_permutations} perms, full grid each)...")
    rng = np.random.RandomState(args.seed)
    count_ge = 0
    n_valid = 0

    for i in range(args.n_permutations):
        perm_labels = rng.permutation(g_labels)
        if len(np.unique(perm_labels)) < 2:
            logger.warning(f"  perm {i+1}: skipped (single class)")
            continue

        label_map = dict(zip(unique_g, perm_labels))
        y_perm = np.array([label_map[g] for g in groups])

        try:
            perm_ba, _, _ = full_grid_search(
                X_raw, feat_27, y_perm, groups, n_splits, args.seed)
            n_valid += 1
            if perm_ba >= observed_ba:
                count_ge += 1
        except Exception as e:
            logger.warning(f"  perm {i+1}: FAILED - {e}")
            # Do NOT count failed permutations in denominator

        if (i + 1) % 25 == 0:
            logger.info(f"  perm {i+1}/{args.n_permutations} "
                       f"(valid={n_valid}, count_ge={count_ge})")

    if n_valid == 0:
        p_val = None
        logger.warning("No valid permutations completed!")
    else:
        p_val = (count_ge + 1) / (n_valid + 1)
    logger.info(f"p-value: {p_val} (valid={n_valid}, count_ge={count_ge})")

    elapsed = round(time.time() - t0, 1)
    output = {
        "dataset": "TDBRAIN_formal_status",
        "n_mdd": int(n_mdd), "n_hc": int(n_hc), "n_windows": int(len(y)),
        "methodology": {
            "evidence_level": "exploratory",
            "note": "CONFIGS derived from prior same-dataset optimization",
            "threshold": "fixed_0.5",
            "model_selection": "grid_search_inside_permutation",
            "permutation_denominator": "n_valid_only",
        },
        "grid_results": all_results,
        "best_config": {"method": best_cfg[0], "C": best_cfg[1],
                        "pca": best_cfg[2]} if best_cfg else None,
        "observed_ba": round(observed_ba, 4),
        "observed_auc": round(observed_auc, 4),
        "p_value": round(p_val, 4) if p_val is not None else None,
        "n_permutations_requested": args.n_permutations,
        "n_permutations_valid": n_valid,
        "count_ge": count_ge,
        "elapsed_seconds": elapsed,
    }
    path = os.path.join(args.output_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved to {path}")


if __name__ == "__main__":
    main()
