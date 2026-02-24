"""Full permutation test: 5-dim + Logistic C=0.1."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _root import ROOT

import numpy as np, json, time, logging, os
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
from joblib import Parallel, delayed
from run_diagnostic import load_filtered_data, extract_minimal_features
from modma_mdd_real_experiment import compute_subject_balanced_sample_weights, get_ci_bootstrap

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SFREQ = 125.0
N_PERM = 1000
N_JOBS = 4
SEED = 42
OUT_DIR = str(ROOT / "outputs/results_5dim_logistic")


def run_cv_logistic(features, y, groups, n_splits):
    cv = StratifiedGroupKFold(n_splits=n_splits)
    yt_s, yp_s = {}, {}
    for tr, te in cv.split(features, y, groups=groups):
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(C=0.1, class_weight="balanced",
                                                    max_iter=1000, random_state=42))])
        sw = compute_subject_balanced_sample_weights(groups[tr])
        pipe.fit(features[tr], y[tr], clf__sample_weight=sw)
        preds = pipe.predict_proba(features[te])[:, 1]
        for i, g in enumerate(groups[te]):
            if g not in yt_s:
                yt_s[g] = y[te][i]
                yp_s[g] = []
            yp_s[g].append(preds[i])
    subjs = list(yt_s.keys())
    yt = np.array([yt_s[s] for s in subjs])
    yprob = np.array([np.mean(yp_s[s]) for s in subjs])
    ypred = (yprob >= 0.5).astype(int)
    return {
        "ba": balanced_accuracy_score(yt, ypred),
        "auc": roc_auc_score(yt, yprob) if len(np.unique(yt)) > 1 else 0.5,
        "f1": f1_score(yt, ypred, zero_division=0),
        "y_true": yt, "y_prob": yprob, "subjs": subjs,
    }


def _one_perm(features, y_perm, groups, n_splits):
    try:
        r = run_cv_logistic(features, y_perm, groups, n_splits)
        return r["ba"]
    except Exception:
        return None


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    X_raw, y, groups, ch_names = load_filtered_data()
    feats, feat_names = extract_minimal_features(X_raw, SFREQ, ch_names)
    logger.info(f"Features: {feats.shape}, subjects: {len(np.unique(groups))}")

    ug, fi = np.unique(groups, return_index=True)
    gl = y[fi]
    n_splits = max(2, min(5, np.min(np.unique(gl, return_counts=True)[1])))

    t0 = time.time()
    cv = run_cv_logistic(feats, y, groups, n_splits)
    logger.info(f"Actual BA={cv['ba']:.3f}, AUC={cv['auc']:.3f}, F1={cv['f1']:.3f}")

    # Permutations
    rng = np.random.RandomState(SEED)
    jobs = []
    for _ in range(N_PERM):
        sl = rng.permutation(gl)
        if len(np.unique(sl)) < 2:
            continue
        lm = dict(zip(ug, sl))
        jobs.append(np.array([lm[g] for g in groups]))

    logger.info(f"Running {len(jobs)} permutations (n_jobs={N_JOBS})...")
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_one_perm)(feats, yp, groups, n_splits) for yp in jobs)
    perm_bas = np.array([r for r in results if r is not None])

    p_val = (np.sum(perm_bas >= cv["ba"]) + 1) / (len(perm_bas) + 1)
    ba_ci = get_ci_bootstrap(cv["y_true"], (cv["y_prob"] >= 0.5).astype(int),
                             balanced_accuracy_score, seed=SEED)
    elapsed = round(time.time() - t0, 1)

    report = {
        "features": feat_names,
        "n_features": len(feat_names),
        "model": "LogisticRegression(C=0.1)",
        "balanced_accuracy": cv["ba"],
        "balanced_accuracy_ci95": ba_ci,
        "roc_auc": cv["auc"], "f1": cv["f1"],
        "n_permutations": len(perm_bas),
        "permutation_pvalue": float(p_val),
        "permuted_bas_mean": float(np.mean(perm_bas)),
        "permuted_bas_std": float(np.std(perm_bas)),
        "elapsed_seconds": elapsed,
    }
    logger.info(f"\np-value={p_val:.4f}, perm_mean={np.mean(perm_bas):.3f}, elapsed={elapsed}s")

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    np.savetxt(os.path.join(OUT_DIR, "permutation_scores.csv"),
               perm_bas, header="balanced_accuracy", comments="")
    logger.info(f"Results saved to {OUT_DIR}/")
