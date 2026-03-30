"""SVM + XGBoost weighted soft voting ensemble experiment.
Compares 3 strategies using exclude_high_fp filtering (Phase 7 baseline BA=0.6913):
  1. svm_only: SVC(rbf, class_weight=balanced) — Phase 7 baseline
  2. xgb_only: XGBClassifier with nested CV tuning (inner 3-fold, outer 5-fold)
  3. ensemble: Weighted soft voting (SVM + XGB) with weight search

Usage:
  python run_ensemble.py              # run full experiment
  python run_ensemble.py --report-only  # regenerate charts/summary from JSON
"""
import argparse
import datetime
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import permutation_test
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
from xgboost import XGBClassifier

from config import PARTICIPANTS_TSV
from run_ablation import load_all, _indices_for_condition

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
V3_CATEGORIES = ["abs_bp", "rel_bp", "Hjorth"]
N_SPLITS = 5
N_INNER_SPLITS = 3
RANDOM_STATE = 42

HIGH_FP_EXCLUSIONS = {
    "BURNOUT", "CHRONIC PAIN", "INSOMNIA", "OCD",
    "PARKINSON", "SMC", "TINNITUS",
}

WEIGHT_CANDIDATES = [
    (0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3),
]

XGB_PARAM_DIST = {
    "clf__n_estimators": [100, 200, 300, 500],
    "clf__max_depth": [3, 4, 5, 6, 8],
    "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "clf__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "clf__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "clf__min_child_weight": [1, 3, 5, 7],
}
N_RANDOM_ITER = 50

RESULTS_PATH = Path("ensemble_results.json")
SUMMARY_PATH = Path("ensemble_summary.md")
FIGURE_DIR = Path("figures")
MODEL_DIR = Path("models")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_svm_pipeline(rs=RANDOM_STATE):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True,
                     class_weight="balanced", random_state=rs)),
    ])


def _make_xgb_pipeline(scale_pos_weight, rs=RANDOM_STATE):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=rs,
            n_jobs=1,
        )),
    ])


def _load_indication_map(subject_ids):
    df = pd.read_csv(PARTICIPANTS_TSV, sep="\t")
    df = df[df["participants_ID"].isin(subject_ids)].drop_duplicates(
        subset=["participants_ID"])
    df["indication_clean"] = df["indication"].fillna("UNKNOWN").astype(str).str.strip()
    df.loc[df["indication_clean"].isin(["nan", ""]), "indication_clean"] = "UNKNOWN"
    return dict(zip(df["participants_ID"], df["indication_clean"]))


def _compute_metrics(y_bin, preds_bin, probas, mdd_idx):
    """Compute BA, AUC, SEN, SPE from binary labels and predictions."""
    auc = float(roc_auc_score(y_bin, probas))
    tn, fp, fn, tp = confusion_matrix(y_bin, preds_bin).ravel()
    sen = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spe = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    ba = (sen + spe) / 2
    return {"ba": round(ba, 4), "auc": round(auc, 4),
            "sen": round(sen, 4), "spe": round(spe, 4)}


def _permutation_test(fold_ba_exp, fold_ba_base):
    a = np.array(fold_ba_exp)
    b = np.array(fold_ba_base)

    def stat_fn(x, y, axis):
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    result = permutation_test(
        (a, b), statistic=stat_fn,
        permutation_type="independent", n_resamples=9999,
        alternative="greater", random_state=RANDOM_STATE)
    return round(float(result.pvalue), 6)


def _fold_ba(y_enc, preds, mdd_idx, test_idx):
    """Compute BA for a single fold."""
    y_bin = (y_enc[test_idx] == mdd_idx).astype(int)
    p_bin = (preds == mdd_idx).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_bin, p_bin).ravel()
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return (sen + spe) / 2


def _proba_to_encoded_labels(proba_mdd, mdd_idx, threshold=0.5):
    """Convert P(MDD) to encoded class labels using the active class index."""
    non_mdd_idx = 1 - int(mdd_idx)
    is_mdd = np.asarray(proba_mdd) >= threshold
    return np.where(is_mdd, int(mdd_idx), non_mdd_idx).astype(int)


# ---------------------------------------------------------------------------
# CV evaluation: SVM-only
# ---------------------------------------------------------------------------
def _cv_svm_only(X, y_enc, groups, mdd_idx):
    cv = StratifiedGroupKFold(n_splits=N_SPLITS)
    fold_bas = []
    all_probas = np.full(len(y_enc), np.nan)
    all_preds = np.full(len(y_enc), -1, dtype=int)

    for fold_i, (train_idx, test_idx) in enumerate(
            tqdm(cv.split(X, y_enc, groups), total=N_SPLITS, desc="svm_only")):
        pipe = _make_svm_pipeline()
        pipe.fit(X[train_idx], y_enc[train_idx])
        all_probas[test_idx] = pipe.predict_proba(X[test_idx])[:, mdd_idx]
        all_preds[test_idx] = pipe.predict(X[test_idx])
        fold_bas.append(_fold_ba(y_enc, all_preds[test_idx], mdd_idx, test_idx))

    y_bin = (y_enc == mdd_idx).astype(int)
    pred_bin = (all_preds == mdd_idx).astype(int)
    metrics = _compute_metrics(y_bin, pred_bin, all_probas, mdd_idx)
    metrics["per_fold_ba"] = [round(b, 4) for b in fold_bas]
    return metrics, all_probas, y_bin


# ---------------------------------------------------------------------------
# CV evaluation: XGB-only with nested CV tuning
# ---------------------------------------------------------------------------
def _cv_xgb_only(X, y_enc, groups, mdd_idx, scale_pos_weight):
    outer_cv = StratifiedGroupKFold(n_splits=N_SPLITS)
    inner_cv = StratifiedGroupKFold(n_splits=N_INNER_SPLITS)
    fold_bas = []
    all_probas = np.full(len(y_enc), np.nan)
    all_preds = np.full(len(y_enc), -1, dtype=int)
    best_params_per_fold = []

    for fold_i, (train_idx, test_idx) in enumerate(
            tqdm(outer_cv.split(X, y_enc, groups), total=N_SPLITS, desc="xgb_only")):
        pipe = _make_xgb_pipeline(scale_pos_weight)
        search = RandomizedSearchCV(
            pipe, XGB_PARAM_DIST, n_iter=N_RANDOM_ITER,
            scoring="balanced_accuracy", cv=inner_cv, refit=True,
            random_state=RANDOM_STATE, n_jobs=1,
        )
        search.fit(X[train_idx], y_enc[train_idx],
                    groups=groups[train_idx])
        best_params_per_fold.append(search.best_params_)

        all_probas[test_idx] = search.predict_proba(X[test_idx])[:, mdd_idx]
        all_preds[test_idx] = search.predict(X[test_idx])
        fold_bas.append(_fold_ba(y_enc, all_preds[test_idx], mdd_idx, test_idx))

    y_bin = (y_enc == mdd_idx).astype(int)
    pred_bin = (all_preds == mdd_idx).astype(int)
    metrics = _compute_metrics(y_bin, pred_bin, all_probas, mdd_idx)
    metrics["per_fold_ba"] = [round(b, 4) for b in fold_bas]
    metrics["best_params_per_fold"] = best_params_per_fold
    return metrics, all_probas, y_bin


# ---------------------------------------------------------------------------
# CV evaluation: Weighted soft voting ensemble
# ---------------------------------------------------------------------------
def _cv_ensemble(X, y_enc, groups, mdd_idx, scale_pos_weight):
    outer_cv = StratifiedGroupKFold(n_splits=N_SPLITS)
    inner_cv = StratifiedGroupKFold(n_splits=N_INNER_SPLITS)
    fold_bas = []
    all_probas = np.full(len(y_enc), np.nan)
    all_preds = np.full(len(y_enc), -1, dtype=int)
    best_weights_per_fold = []

    for fold_i, (train_idx, test_idx) in enumerate(
            tqdm(outer_cv.split(X, y_enc, groups), total=N_SPLITS, desc="ensemble")):
        X_tr, y_tr, g_tr = X[train_idx], y_enc[train_idx], groups[train_idx]
        X_te = X[test_idx]

        # Train SVM (fixed config)
        svm_pipe = _make_svm_pipeline()
        svm_pipe.fit(X_tr, y_tr)
        svm_proba_te = svm_pipe.predict_proba(X_te)[:, mdd_idx]

        # Train XGB (nested CV tuning on train split)
        xgb_pipe = _make_xgb_pipeline(scale_pos_weight)
        search = RandomizedSearchCV(
            xgb_pipe, XGB_PARAM_DIST, n_iter=N_RANDOM_ITER,
            scoring="balanced_accuracy", cv=inner_cv, refit=True,
            random_state=RANDOM_STATE, n_jobs=1,
        )
        search.fit(X_tr, y_tr, groups=g_tr)
        xgb_proba_te = search.predict_proba(X_te)[:, mdd_idx]

        # Search best weight on inner CV
        best_w, best_inner_ba = (0.5, 0.5), 0.0
        for w_svm, w_xgb in WEIGHT_CANDIDATES:
            inner_bas = []
            for i_tr, i_val in inner_cv.split(X_tr, y_tr, g_tr):
                svm_inner = _make_svm_pipeline()
                svm_inner.fit(X_tr[i_tr], y_tr[i_tr])
                svm_p = svm_inner.predict_proba(X_tr[i_val])[:, mdd_idx]

                xgb_inner = _make_xgb_pipeline(scale_pos_weight)
                xgb_inner.set_params(**search.best_params_)
                xgb_inner.fit(X_tr[i_tr], y_tr[i_tr])
                xgb_p = xgb_inner.predict_proba(X_tr[i_val])[:, mdd_idx]

                ens_p = w_svm * svm_p + w_xgb * xgb_p
                ens_pred = (ens_p >= 0.5).astype(int)
                y_val_bin = (y_tr[i_val] == mdd_idx).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_val_bin, ens_pred).ravel()
                sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                inner_bas.append((sen + spe) / 2)
            mean_ba = np.mean(inner_bas)
            if mean_ba > best_inner_ba:
                best_inner_ba = mean_ba
                best_w = (w_svm, w_xgb)

        best_weights_per_fold.append(best_w)
        w_svm, w_xgb = best_w

        # Combine on test fold
        ens_proba = w_svm * svm_proba_te + w_xgb * xgb_proba_te
        ens_pred = _proba_to_encoded_labels(ens_proba, mdd_idx, threshold=0.5)
        all_probas[test_idx] = ens_proba
        all_preds[test_idx] = ens_pred
        fold_bas.append(_fold_ba(y_enc, ens_pred, mdd_idx, test_idx))

    y_bin = (y_enc == mdd_idx).astype(int)
    pred_bin = (all_preds == mdd_idx).astype(int)
    metrics = _compute_metrics(y_bin, pred_bin, all_probas, mdd_idx)
    metrics["per_fold_ba"] = [round(b, 4) for b in fold_bas]
    metrics["best_weights_per_fold"] = [list(w) for w in best_weights_per_fold]
    return metrics, all_probas, y_bin


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
def run_experiments():
    """Run 3-strategy ensemble comparison with exclude_high_fp filtering."""
    print("Loading cached features...")
    datasets = load_all()
    X_full, y_raw, groups_full = datasets["combined"]

    # v3_subset features
    v3_idx = _indices_for_condition(V3_CATEGORIES, "combined")
    X_v3 = X_full[:, v3_idx]
    print(f"v3_subset: {X_v3.shape[1]}-dim")

    # Apply exclude_high_fp filtering
    indication_map = _load_indication_map(list(groups_full))
    indications = np.array([indication_map.get(g, "UNKNOWN") for g in groups_full])

    def _should_exclude(ind):
        primary = ind.split("/")[0].strip()
        return primary in HIGH_FP_EXCLUSIONS

    mask = np.array([
        not (_should_exclude(indications[i]) and y_raw[i] == "nonMDD")
        for i in range(len(groups_full))
    ])

    X = X_v3[mask]
    y_f = y_raw[mask]
    g_f = groups_full[mask]

    le = LabelEncoder()
    le.fit(["MDD", "nonMDD"])
    y_enc = le.transform(y_f)
    mdd_idx = list(le.classes_).index("MDD")

    n_mdd = int((y_f == "MDD").sum())
    n_nonmdd = int((y_f == "nonMDD").sum())
    scale_pos_weight = n_nonmdd / n_mdd if n_mdd > 0 else 1.0
    print(f"\nexclude_high_fp: n={len(g_f)} (MDD={n_mdd}, nonMDD={n_nonmdd})")
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")

    # Check for existing results (checkpoint/resume)
    results = _load_results()
    if "experiments" not in results:
        results["experiments"] = {}
    if "roc_data" not in results:
        results["roc_data"] = {}

    # --- Strategy 1: SVM-only ---
    if "svm_only" not in results["experiments"]:
        print("\n[svm_only] SVC(rbf, class_weight=balanced)")
        metrics, probas, y_bin = _cv_svm_only(X, y_enc, g_f, mdd_idx)
        metrics["n_subjects"] = len(g_f)
        metrics["n_mdd"] = n_mdd
        metrics["n_nonmdd"] = n_nonmdd
        metrics["n_features"] = int(X.shape[1])
        results["experiments"]["svm_only"] = metrics
        fpr, tpr, _ = roc_curve(y_bin, probas)
        results["roc_data"]["svm_only"] = {
            "fpr": [round(float(v), 6) for v in fpr],
            "tpr": [round(float(v), 6) for v in tpr],
        }
        print(f"  BA={metrics['ba']:.4f}, AUC={metrics['auc']:.4f}")
        _save_results(results)
    else:
        print("\n[svm_only] already done, skipping")

    # --- Strategy 2: XGB-only ---
    if "xgb_only" not in results["experiments"]:
        print("\n[xgb_only] XGBClassifier with nested CV tuning")
        metrics, probas, y_bin = _cv_xgb_only(X, y_enc, g_f, mdd_idx, scale_pos_weight)
        metrics["n_subjects"] = len(g_f)
        metrics["n_mdd"] = n_mdd
        metrics["n_nonmdd"] = n_nonmdd
        metrics["n_features"] = int(X.shape[1])
        results["experiments"]["xgb_only"] = metrics
        fpr, tpr, _ = roc_curve(y_bin, probas)
        results["roc_data"]["xgb_only"] = {
            "fpr": [round(float(v), 6) for v in fpr],
            "tpr": [round(float(v), 6) for v in tpr],
        }
        print(f"  BA={metrics['ba']:.4f}, AUC={metrics['auc']:.4f}")
        _save_results(results)
    else:
        print("\n[xgb_only] already done, skipping")

    # --- Strategy 3: Ensemble ---
    if "ensemble" not in results["experiments"]:
        print("\n[ensemble] Weighted soft voting (SVM + XGB)")
        metrics, probas, y_bin = _cv_ensemble(X, y_enc, g_f, mdd_idx, scale_pos_weight)
        metrics["n_subjects"] = len(g_f)
        metrics["n_mdd"] = n_mdd
        metrics["n_nonmdd"] = n_nonmdd
        metrics["n_features"] = int(X.shape[1])
        results["experiments"]["ensemble"] = metrics
        fpr, tpr, _ = roc_curve(y_bin, probas)
        results["roc_data"]["ensemble"] = {
            "fpr": [round(float(v), 6) for v in fpr],
            "tpr": [round(float(v), 6) for v in tpr],
        }
        print(f"  BA={metrics['ba']:.4f}, AUC={metrics['auc']:.4f}")
        _save_results(results)
    else:
        print("\n[ensemble] already done, skipping")

    # --- Statistical comparison ---
    print("\n[Statistical comparison]")
    svm_fold = results["experiments"]["svm_only"]["per_fold_ba"]
    results["statistical_tests"] = {}

    for name in ["xgb_only", "ensemble"]:
        exp = results["experiments"][name]
        fold_ba = exp["per_fold_ba"]
        delta_ba = exp["ba"] - results["experiments"]["svm_only"]["ba"]
        p = _permutation_test(fold_ba, svm_fold)
        results["statistical_tests"][f"{name}_vs_svm_only"] = {
            "delta_ba": round(delta_ba, 4),
            "p_value": p,
            "significant": bool(p < 0.05),
        }
        print(f"  {name} vs svm_only: delta-BA={delta_ba:+.4f}, p={p:.6f}")

    # --- Decision logic ---
    ens_test = results["statistical_tests"]["ensemble_vs_svm_only"]
    ens_ba = results["experiments"]["ensemble"]["ba"]
    svm_ba = results["experiments"]["svm_only"]["ba"]

    if ens_ba > svm_ba and ens_test["significant"]:
        adopt = True
        rationale = (f"Ensemble significantly improves BA "
                     f"({svm_ba:.4f} -> {ens_ba:.4f}, p={ens_test['p_value']})")
    else:
        adopt = False
        if ens_ba > svm_ba:
            rationale = (f"Ensemble BA improved ({svm_ba:.4f} -> {ens_ba:.4f}) "
                         f"but not significant (p={ens_test['p_value']})")
        else:
            rationale = (f"Ensemble does not improve BA "
                         f"({svm_ba:.4f} -> {ens_ba:.4f})")

    results["decision"] = {
        "adopt_ensemble": adopt,
        "rationale": rationale,
        "final_model": "ensemble" if adopt else "svm_only",
        "final_ba": ens_ba if adopt else svm_ba,
    }

    results["metadata"] = {
        "task": "ensemble_modeling",
        "timestamp": datetime.datetime.now().isoformat(),
        "scale_pos_weight": round(scale_pos_weight, 4),
        "n_random_iter": N_RANDOM_ITER,
        "weight_candidates": WEIGHT_CANDIDATES,
    }

    _save_results(results)

    # --- Model serialization ---
    print("\n[Model serialization]")
    _serialize_models(X, y_enc, g_f, mdd_idx, scale_pos_weight, results)

    return results


# ---------------------------------------------------------------------------
# Model serialization
# ---------------------------------------------------------------------------
def _serialize_models(X, y_enc, groups, mdd_idx, scale_pos_weight, results):
    """Retrain final models on ALL exclude_high_fp data and save."""
    MODEL_DIR.mkdir(exist_ok=True)

    # SVM
    svm_pipe = _make_svm_pipeline()
    svm_pipe.fit(X, y_enc)
    joblib.dump(svm_pipe, MODEL_DIR / "final_svm.joblib")
    print(f"  Saved {MODEL_DIR / 'final_svm.joblib'}")

    # XGB — use most common best params across folds
    xgb_params = results["experiments"].get("xgb_only", {}).get("best_params_per_fold", [])
    if xgb_params:
        # Use params from first fold as representative
        best_p = xgb_params[0]
        xgb_pipe = _make_xgb_pipeline(scale_pos_weight)
        xgb_pipe.set_params(**best_p)
    else:
        xgb_pipe = _make_xgb_pipeline(scale_pos_weight)
    xgb_pipe.fit(X, y_enc)
    joblib.dump(xgb_pipe, MODEL_DIR / "final_xgb.joblib")
    print(f"  Saved {MODEL_DIR / 'final_xgb.joblib'}")

    # Ensemble weights
    ens_weights = results["experiments"].get("ensemble", {}).get("best_weights_per_fold", [])
    if ens_weights:
        avg_svm = np.mean([w[0] for w in ens_weights])
        avg_xgb = np.mean([w[1] for w in ens_weights])
    else:
        avg_svm, avg_xgb = 0.5, 0.5
    weight_info = {
        "w_svm": round(avg_svm, 4),
        "w_xgb": round(avg_xgb, 4),
        "per_fold_weights": ens_weights,
    }
    with open(MODEL_DIR / "final_ensemble_weights.json", "w", newline="\n") as f:
        json.dump(weight_info, f, indent=2)
    print(f"  Saved {MODEL_DIR / 'final_ensemble_weights.json'}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def generate_charts(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(exist_ok=True)
    experiments = results.get("experiments", {})

    # --- 1. BA comparison bar chart ---
    strategies = ["svm_only", "xgb_only", "ensemble"]
    labels = ["SVM-only\n(baseline)", "XGB-only\n(nested CV)", "Ensemble\n(weighted voting)"]
    bas = []
    stds = []
    for s in strategies:
        r = experiments.get(s, {})
        bas.append(r.get("ba", 0))
        fold_ba = r.get("per_fold_ba", [])
        stds.append(float(np.std(fold_ba)) if fold_ba else 0)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    x = np.arange(len(strategies))
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    bars = ax.bar(x, bas, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax.axhline(0.6913, color="red", linestyle="--", linewidth=1.5,
               label="Phase 7 baseline BA=0.6913")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Ensemble Modeling: BA Comparison (exclude_high_fp, 676d)")
    ax.legend(fontsize=9)

    for bar, ba_val, std_val in zip(bars, bas, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std_val + 0.005,
                f"{ba_val:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0.45, max(bas) + 0.08)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "ensemble_ba_comparison.png")
    plt.close(fig)
    print(f"Saved {FIGURE_DIR / 'ensemble_ba_comparison.png'}")

    # --- 2. ROC curves ---
    roc_data = results.get("roc_data", {})
    if roc_data:
        fig2, ax2 = plt.subplots(figsize=(7, 6), dpi=150)
        colors_roc = {"svm_only": "#1f77b4", "xgb_only": "#2ca02c",
                      "ensemble": "#ff7f0e"}
        for s in strategies:
            rd = roc_data.get(s, {})
            fpr = rd.get("fpr", [])
            tpr = rd.get("tpr", [])
            auc_val = experiments.get(s, {}).get("auc", 0)
            if fpr and tpr:
                ax2.plot(fpr, tpr, color=colors_roc.get(s, "gray"),
                         label=f"{s} (AUC={auc_val:.4f})", linewidth=2)
        ax2.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("Ensemble Modeling: ROC Curves")
        ax2.legend(fontsize=9, loc="lower right")
        fig2.tight_layout()
        fig2.savefig(FIGURE_DIR / "ensemble_roc.png")
        plt.close(fig2)
        print(f"Saved {FIGURE_DIR / 'ensemble_roc.png'}")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
def generate_summary(results):
    experiments = results.get("experiments", {})
    stats = results.get("statistical_tests", {})
    decision = results.get("decision", {})
    svm = experiments.get("svm_only", {})

    lines = []
    lines.append("# Ensemble Modeling Summary\n\n")

    lines.append("## 1. Experiment Setup\n\n")
    lines.append("- **SVM:** StandardScaler + SVC(kernel=rbf, "
                 "class_weight=balanced, random_state=42)\n")
    lines.append("- **XGB:** StandardScaler + XGBClassifier("
                 "scale_pos_weight, nested CV tuning)\n")
    lines.append("- **Ensemble:** Weighted soft voting "
                 "(w_svm * P_svm + w_xgb * P_xgb)\n")
    lines.append("- **Subject filter:** exclude_high_fp (Phase 7)\n")
    lines.append("- **Features:** v3_subset (abs_bp + rel_bp + Hjorth, 676d)\n")
    lines.append(f"- **Outer CV:** {N_SPLITS}-fold StratifiedGroupKFold\n")
    lines.append(f"- **Inner CV:** {N_INNER_SPLITS}-fold (XGB tuning + weight search)\n")
    lines.append(f"- **XGB search:** RandomizedSearchCV, n_iter={N_RANDOM_ITER}\n")
    lines.append("- **Statistical test:** permutation_test, "
                 "n_resamples=9999, alternative=greater\n\n")

    lines.append("## 2. Results Comparison\n\n")
    lines.append("| Strategy | BA | AUC | SEN | SPE | delta-BA | p-value |\n")
    lines.append("|----------|-----|-----|-----|-----|----------|--------|\n")

    svm_ba = svm.get("ba", 0)
    lines.append(f"| svm_only | {svm_ba:.4f} | "
                 f"{svm.get('auc', 0):.4f} | "
                 f"{svm.get('sen', 0):.4f} | "
                 f"{svm.get('spe', 0):.4f} | - | - |\n")

    for s in ["xgb_only", "ensemble"]:
        r = experiments.get(s, {})
        ba = r.get("ba", 0)
        delta = ba - svm_ba
        st = stats.get(f"{s}_vs_svm_only", {})
        p = st.get("p_value", "N/A")
        p_str = f"{p:.6f}" if isinstance(p, float) else str(p)
        lines.append(f"| {s} | {ba:.4f} | {r.get('auc', 0):.4f} | "
                     f"{r.get('sen', 0):.4f} | {r.get('spe', 0):.4f} | "
                     f"{delta:+.4f} | {p_str} |\n")
    lines.append("\n")

    lines.append("## 3. Per-fold BA\n\n")
    lines.append("| Fold | svm_only | xgb_only | ensemble |\n")
    lines.append("|------|----------|----------|----------|\n")
    for i in range(N_SPLITS):
        sv = experiments.get("svm_only", {}).get("per_fold_ba", [0]*N_SPLITS)
        xg = experiments.get("xgb_only", {}).get("per_fold_ba", [0]*N_SPLITS)
        en = experiments.get("ensemble", {}).get("per_fold_ba", [0]*N_SPLITS)
        lines.append(f"| {i+1} | {sv[i]:.4f} | {xg[i]:.4f} | {en[i]:.4f} |\n")
    lines.append("\n")

    # Ensemble weights
    ens_weights = experiments.get("ensemble", {}).get("best_weights_per_fold", [])
    if ens_weights:
        lines.append("## 4. Ensemble Weights per Fold\n\n")
        lines.append("| Fold | w_svm | w_xgb |\n")
        lines.append("|------|-------|-------|\n")
        for i, w in enumerate(ens_weights):
            lines.append(f"| {i+1} | {w[0]:.1f} | {w[1]:.1f} |\n")
        lines.append("\n")

    lines.append("## 5. Decision\n\n")
    lines.append(f"- **Adopt ensemble:** {decision.get('adopt_ensemble', 'N/A')}\n")
    lines.append(f"- **Rationale:** {decision.get('rationale', 'N/A')}\n")
    lines.append(f"- **Final model:** {decision.get('final_model', 'N/A')}\n")
    lines.append(f"- **Final BA:** {decision.get('final_ba', 'N/A')}\n")

    with open(SUMMARY_PATH, "w", newline="\n", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Saved {SUMMARY_PATH}")


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def _load_results():
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return {"experiments": {}, "statistical_tests": {}, "roc_data": {}}


def _save_results(results):
    with open(RESULTS_PATH, "w", newline="\n", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved {RESULTS_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="SVM + XGBoost ensemble experiment.")
    parser.add_argument("--report-only", action="store_true",
                        help="Regenerate charts/summary from existing JSON")
    args = parser.parse_args()

    if args.report_only:
        if not RESULTS_PATH.exists():
            print(f"ERROR: {RESULTS_PATH} not found. Run full analysis first.")
            return
        results = _load_results()
        generate_charts(results)
        generate_summary(results)
        return

    results = run_experiments()
    generate_charts(results)
    generate_summary(results)

    # Final summary
    exp = results["experiments"]
    dec = results["decision"]
    print(f"\n{'=' * 60}")
    print("Ensemble Modeling Complete")
    print(f"  svm_only:  BA={exp['svm_only']['ba']:.4f}")
    print(f"  xgb_only:  BA={exp['xgb_only']['ba']:.4f}")
    print(f"  ensemble:  BA={exp['ensemble']['ba']:.4f}")
    print(f"  Decision: {dec['final_model']} — {dec['rationale']}")


if __name__ == "__main__":
    main()
