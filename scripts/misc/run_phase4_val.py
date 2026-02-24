"""Full 1000-perm validation for Config B (parietal focus)."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _root import ROOT

import numpy as np, json, time, logging, os
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
from joblib import Parallel, delayed
from run_diagnostic import load_filtered_data
from run_phase4_diag import make_features
from modma_mdd_real_experiment import compute_subject_balanced_sample_weights, get_ci_bootstrap

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

CFG_B = [
    ("brel", "parietal", 4, 8), ("brel", "parietal", 8, 13),
    ("asym", "frontal"), ("tbr", "parietal"), ("riem", "central", "temporal"),
]
OUT = str(ROOT / "outputs/results_phase4_parietal")
N_PERM, N_JOBS, SEED = 1000, 4, 42


def run_cv(feats, y, groups, n_splits):
    cv = StratifiedGroupKFold(n_splits=n_splits)
    yt_s, yp_s = {}, {}
    for tr, te in cv.split(feats, y, groups=groups):
        pipe = Pipeline([("s", StandardScaler()),
                         ("c", LogisticRegression(C=0.1, class_weight="balanced",
                                                  max_iter=1000, random_state=42))])
        sw = compute_subject_balanced_sample_weights(groups[tr])
        pipe.fit(feats[tr], y[tr], c__sample_weight=sw)
        p = pipe.predict_proba(feats[te])[:, 1]
        for i, g in enumerate(groups[te]):
            if g not in yt_s: yt_s[g] = y[te][i]; yp_s[g] = []
            yp_s[g].append(p[i])
    subjs = list(yt_s.keys())
    yt = np.array([yt_s[s] for s in subjs])
    yprob = np.array([np.mean(yp_s[s]) for s in subjs])
    ypred = (yprob >= 0.5).astype(int)
    return {"ba": balanced_accuracy_score(yt, ypred),
            "auc": roc_auc_score(yt, yprob) if len(np.unique(yt)) > 1 else 0.5,
            "f1": f1_score(yt, ypred, zero_division=0),
            "y_true": yt, "y_prob": yprob, "subjs": subjs}


def _one(feats, yp, groups, ns):
    try: return run_cv(feats, yp, groups, ns)["ba"]
    except: return None


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    X, y, groups, ch_names = load_filtered_data()
    feats, fnames = make_features(X, ch_names, CFG_B)
    log.info(f"Features: {fnames}, shape={feats.shape}")

    ug, fi = np.unique(groups, return_index=True)
    gl = y[fi]
    ns = max(2, min(5, np.min(np.unique(gl, return_counts=True)[1])))

    t0 = time.time()
    cv = run_cv(feats, y, groups, ns)
    log.info(f"BA={cv['ba']:.3f}, AUC={cv['auc']:.3f}, F1={cv['f1']:.3f}")

    rng = np.random.RandomState(SEED)
    jobs = []
    for _ in range(N_PERM):
        sl = rng.permutation(gl)
        if len(np.unique(sl)) < 2: continue
        jobs.append(np.array([dict(zip(ug, sl))[g] for g in groups]))

    log.info(f"Running {len(jobs)} permutations...")
    res = Parallel(n_jobs=N_JOBS)(delayed(_one)(feats, yp, groups, ns) for yp in jobs)
    perm = np.array([r for r in res if r is not None])

    pv = (np.sum(perm >= cv["ba"]) + 1) / (len(perm) + 1)
    ci = get_ci_bootstrap(cv["y_true"], (cv["y_prob"] >= 0.5).astype(int),
                          balanced_accuracy_score, seed=SEED)
    elapsed = round(time.time() - t0, 1)

    report = {"features": fnames, "n_features": len(fnames),
              "model": "LogisticRegression(C=0.1)",
              "balanced_accuracy": cv["ba"], "balanced_accuracy_ci95": ci,
              "roc_auc": cv["auc"], "f1": cv["f1"],
              "n_permutations": len(perm), "permutation_pvalue": float(pv),
              "permuted_bas_mean": float(np.mean(perm)),
              "permuted_bas_std": float(np.std(perm)),
              "elapsed_seconds": elapsed}

    log.info(f"\np={pv:.4f}, perm_mean={np.mean(perm):.3f}, elapsed={elapsed}s")
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    np.savetxt(os.path.join(OUT, "perm_scores.csv"), perm, header="ba", comments="")
