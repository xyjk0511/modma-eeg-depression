"""Evaluate current models with nPPV metric on full DISCOVERY data.
nPPV = PPV / base_rate * 100%
Target: nPPV > 110%
"""
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_selection import SelectKBest, f_classif

from run_ablation import load_all, _indices_for_condition

# Feature subsets to evaluate
FEATURE_CONFIGS = {
    "all_992": None,  # all features
    "no_entropy_680": ["abs_bp", "rel_bp", "TBR_FAA", "Hjorth"],
    "abs_bp_only_260": ["abs_bp"],
    "abs_bp_rel_bp_520": ["abs_bp", "rel_bp"],
}


def make_pipe(model_name, use_selectk=None, rs=42):
    steps = [("scaler", StandardScaler()),
             ("smote", SMOTE(random_state=rs, k_neighbors=5))]
    if use_selectk:
        steps.append(("selector", SelectKBest(f_classif, k=use_selectk)))
    if model_name == "svm":
        steps.append(("clf", SVC(kernel="rbf", probability=True,
                                  class_weight="balanced", random_state=rs)))
    else:
        steps.append(("clf", XGBClassifier(n_estimators=100, max_depth=3,
                                            eval_metric="logloss", random_state=rs)))
    return ImbPipeline(steps)


def compute_nppv(y_true, y_pred, pos_label=1):
    """Compute nPPV = PPV / base_rate * 100."""
    base_rate = np.mean(y_true == pos_label)
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
    if (tp + fp) == 0:
        return 0.0, 0.0, 0
    ppv = tp / (tp + fp)
    nppv = (ppv / base_rate) * 100
    n_predicted_pos = int(tp + fp)
    return float(nppv), float(ppv), n_predicted_pos


def evaluate(X, y_str, groups, model_name, use_selectk=None, n_splits=5, rs=42):
    le = LabelEncoder()
    y_enc = le.fit_transform(y_str)
    mdd_idx = list(le.classes_).index("MDD")

    cv = StratifiedGroupKFold(n_splits=n_splits)
    all_proba = np.zeros(len(y_enc))
    all_pred = np.zeros(len(y_enc), dtype=int)

    for train_idx, test_idx in cv.split(X, y_enc, groups):
        pipe = make_pipe(model_name, use_selectk, rs)
        pipe.fit(X[train_idx], y_enc[train_idx])
        all_proba[test_idx] = pipe.predict_proba(X[test_idx])[:, mdd_idx]
        all_pred[test_idx] = pipe.predict(X[test_idx])

    # MDD as positive class
    y_bin = (y_enc == mdd_idx).astype(int)
    pred_bin = (all_pred == mdd_idx).astype(int)

    auc = roc_auc_score(y_bin, all_proba)
    tn, fp, fn, tp = confusion_matrix(y_bin, pred_bin).ravel()
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0
    ba = (sen + spe) / 2

    # nPPV at default threshold (0.5 via predict)
    nppv_default, ppv_default, n_pos_default = compute_nppv(y_bin, pred_bin)

    # Sweep thresholds for optimal nPPV
    best_nppv, best_thresh, best_ppv, best_n_pos = 0, 0.5, 0, 0
    thresholds = np.arange(0.3, 0.95, 0.05)
    thresh_results = []
    for t in thresholds:
        pred_t = (all_proba >= t).astype(int)
        nppv_t, ppv_t, n_pos_t = compute_nppv(y_bin, pred_t)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_bin, pred_t).ravel()
        sen_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        thresh_results.append({
            "threshold": round(float(t), 2),
            "nppv": round(nppv_t, 1),
            "ppv": round(ppv_t, 4),
            "sen": round(float(sen_t), 4),
            "n_predicted_pos": n_pos_t,
        })
        if nppv_t > best_nppv and n_pos_t >= 10:  # need reasonable sample
            best_nppv, best_thresh = nppv_t, t
            best_ppv, best_n_pos = ppv_t, n_pos_t

    return {
        "auc": round(float(auc), 4),
        "ba": round(float(ba), 4),
        "sen": round(float(sen), 4),
        "spe": round(float(spe), 4),
        "nppv_default": round(nppv_default, 1),
        "ppv_default": round(ppv_default, 4),
        "n_pos_default": n_pos_default,
        "best_nppv": round(best_nppv, 1),
        "best_threshold": round(float(best_thresh), 2),
        "best_ppv": round(best_ppv, 4),
        "best_n_pos": best_n_pos,
        "threshold_sweep": thresh_results,
    }


def main():
    print("Loading cached features...")
    datasets = load_all()
    X, y, groups = datasets["combined"]
    n_mdd = np.sum(y == "MDD")
    n_total = len(y)
    base_rate = n_mdd / n_total
    print(f"Combined: {X.shape}, MDD={n_mdd}, nonMDD={n_total - n_mdd}")
    print(f"Base rate: {base_rate:.3f}")
    print(f"nPPV > 110% requires PPV > {base_rate * 1.1:.3f}")
    print()

    results = {}

    for feat_name, categories in FEATURE_CONFIGS.items():
        if categories is not None:
            idx = _indices_for_condition(categories, "combined")
            X_sub = X[:, idx]
        else:
            X_sub = X
        n_feat = X_sub.shape[1]

        for model in ["svm", "xgb"]:
            for k in [None, 50, 100, 200]:
                if k and k >= n_feat:
                    continue
                key = f"{feat_name}_{model}" + (f"_k{k}" if k else "_nosel")
                print(f"  {key} (dim={n_feat}" + (f", k={k}" if k else "") + ")...")
                r = evaluate(X_sub, y, groups, model, use_selectk=k)
                results[key] = {
                    "features": feat_name, "model": model,
                    "n_features": n_feat, "selectk": k, **r,
                }
                print(f"    BA={r['ba']:.3f}  AUC={r['auc']:.3f}  "
                      f"nPPV={r['nppv_default']:.0f}%  "
                      f"best_nPPV={r['best_nppv']:.0f}%@{r['best_threshold']:.2f}")

    # Summary
    print(f"\n{'='*90}")
    print(f"{'Config':<40} {'BA':>5} {'AUC':>5} {'nPPV':>6} {'best_nPPV':>10} {'@thresh':>8}")
    print("-" * 90)
    for key, r in sorted(results.items(), key=lambda kv: kv[1]["best_nppv"], reverse=True):
        print(f"{key:<40} {r['ba']:>5.3f} {r['auc']:>5.3f} "
              f"{r['nppv_default']:>5.0f}% {r['best_nppv']:>9.0f}% "
              f"@{r['best_threshold']:>5.2f}")

    with open("results_nppv.json", "w", newline="\n", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nSaved results_nppv.json")


if __name__ == "__main__":
    main()
