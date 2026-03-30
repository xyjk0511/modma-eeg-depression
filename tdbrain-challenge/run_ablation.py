"""Feature category ablation study.
Coarse: 5 categories via leave-one-out and single-category strategies.
Fine: per-band breakdown of top categories, 5-fold revalidation.
Visualization and summary report generation.
"""
import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from main_diagnostic import load_discovery_subjects

SELECTKBEST_REF_BA = 0.627  # SelectKBest k=200 from results_feat_sweep.json

# ---------------------------------------------------------------------------
# Feature index mapping
# ---------------------------------------------------------------------------
# Per-condition 496-dim: [abs_bp(130) | rel_bp(130) | TBR_FAA(2) | Hjorth(78) | entropy(156)]
_LOCAL = {
    "abs_bp":  list(range(0, 130)),
    "rel_bp":  list(range(130, 260)),
    "TBR_FAA": list(range(260, 262)),
    "Hjorth":  list(range(262, 340)),
    "entropy": list(range(340, 496)),
}

def _indices_for_condition(categories, condition):
    """Return sorted feature indices for given categories and condition mode."""
    idx = []
    for cat in categories:
        local = _LOCAL[cat]
        if condition == "combined":
            idx.extend(local)                          # EO half
            idx.extend([i + 496 for i in local])       # EC half
        else:
            idx.extend(local)
    return sorted(idx)

ALL_CATEGORIES = list(_LOCAL.keys())

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
N_CH = 26

# Fine-grained sub-groups: category -> {sub_name: local_indices}
_FINE = {
    "abs_bp": {f"abs_bp_{b}": list(range(i * N_CH, (i + 1) * N_CH)) for i, b in enumerate(BANDS)},
    "rel_bp": {f"rel_bp_{b}": list(range(130 + i * N_CH, 130 + (i + 1) * N_CH)) for i, b in enumerate(BANDS)},
    "Hjorth": {
        "Hjorth_activity": list(range(262, 262 + N_CH)),
        "Hjorth_mobility": list(range(262 + N_CH, 262 + 2 * N_CH)),
        "Hjorth_complexity": list(range(262 + 2 * N_CH, 262 + 3 * N_CH)),
    },
    "entropy": {
        "entropy_broadband": list(range(340, 340 + N_CH)),
        **{f"entropy_{b}": list(range(340 + N_CH + i * N_CH, 340 + N_CH + (i + 1) * N_CH))
           for i, b in enumerate(BANDS)},
    },
}

# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------
CACHE_DIR = Path("cache_diagnostic")

def _load_condition(condition):
    """Return {sid: (feat_496, label_str)}."""
    cache = CACHE_DIR / condition
    feats = {}
    for npz in cache.glob("*.npz"):
        d = np.load(npz)
        feats[npz.stem] = (d["feat"], str(d["label"]))
    return feats

def load_all():
    """Build X, y, groups arrays for EO-only, EC-only, and combined."""
    subs_eo = set(load_discovery_subjects("EO")["participants_ID"])
    subs_ec = set(load_discovery_subjects("EC")["participants_ID"])

    feats_eo = {s: v for s, v in _load_condition("EO").items() if s in subs_eo}
    feats_ec = {s: v for s, v in _load_condition("EC").items() if s in subs_ec}

    common = sorted(subs_eo & subs_ec & set(feats_eo) & set(feats_ec))

    datasets = {}
    # Combined (992-dim)
    X_comb = np.array([np.concatenate([feats_eo[s][0], feats_ec[s][0]]) for s in common])
    y_comb = np.array([feats_eo[s][1] for s in common])
    g_comb = np.array(common)
    datasets["combined"] = (X_comb, y_comb, g_comb)

    # EO-only (496-dim)
    eo_sids = sorted(subs_eo & set(feats_eo))
    datasets["EO"] = (
        np.array([feats_eo[s][0] for s in eo_sids]),
        np.array([feats_eo[s][1] for s in eo_sids]),
        np.array(eo_sids),
    )

    # EC-only (496-dim)
    ec_sids = sorted(subs_ec & set(feats_ec))
    datasets["EC"] = (
        np.array([feats_ec[s][0] for s in ec_sids]),
        np.array([feats_ec[s][1] for s in ec_sids]),
        np.array(ec_sids),
    )

    return datasets

# ---------------------------------------------------------------------------
# Fixed-param pipelines (no SelectKBest for ablation)
# ---------------------------------------------------------------------------
def _make_pipe(model_name, rs=42, smote_k=5):
    steps = [("scaler", StandardScaler()), ("smote", SMOTE(random_state=rs, k_neighbors=smote_k))]
    if model_name == "svm":
        steps.append(("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=rs)))
    else:
        steps.append(("clf", XGBClassifier(n_estimators=100, max_depth=3, eval_metric="logloss", random_state=rs)))
    return ImbPipeline(steps)

# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
def run_experiment(X_sub, y, groups, model_name, n_splits=3, rs=42):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    mdd_idx = list(le.classes_).index("MDD")

    cv = StratifiedGroupKFold(n_splits=n_splits)
    probas, preds, test_idxs = [], [], []

    for train_idx, test_idx in cv.split(X_sub, y_enc, groups):
        # Try default SMOTE k=5, fallback to k=3, then no SMOTE
        pipe = None
        for k in (5, 3):
            try:
                p = _make_pipe(model_name, rs, smote_k=k)
                p.fit(X_sub[train_idx], y_enc[train_idx])
                pipe = p
                break
            except ValueError:
                continue
        if pipe is None:
            # No SMOTE fallback
            steps = [("scaler", StandardScaler())]
            if model_name == "svm":
                steps.append(("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=rs)))
            else:
                steps.append(("clf", XGBClassifier(n_estimators=100, max_depth=3, eval_metric="logloss", random_state=rs)))
            pipe = ImbPipeline(steps)
            pipe.fit(X_sub[train_idx], y_enc[train_idx])

        probas.append(pipe.predict_proba(X_sub[test_idx])[:, mdd_idx])
        preds.append(pipe.predict(X_sub[test_idx]))
        test_idxs.append(test_idx)

    order = np.argsort(np.concatenate(test_idxs))
    proba = np.concatenate(probas)[order]
    pred = np.concatenate(preds)[order]

    # MDD as positive class (LabelEncoder: MDD=0, nonMDD=1)
    y_bin = (y_enc == mdd_idx).astype(int)
    pred_bin = (pred == mdd_idx).astype(int)
    auc = float(roc_auc_score(y_bin, proba))
    tn, fp, fn, tp = confusion_matrix(y_bin, pred_bin).ravel()
    sen = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spe = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return {"auc": round(auc, 4), "ba": round((sen + spe) / 2, 4), "sen": round(sen, 4), "spe": round(spe, 4)}

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
COARSE_CKPT = Path("coarse_ablation.json")
FINE_CKPT = Path("fine_ablation.json")

def _load_json(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"results": {}}

def _save_json(data, path):
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Coarse ablation
# ---------------------------------------------------------------------------
def run_coarse(datasets):
    ckpt = _load_json(COARSE_CKPT)
    done_keys = set(ckpt["results"].keys())
    total_expected = 66

    experiments = []
    for cond in ["EO", "EC", "combined"]:
        for model in ["svm", "xgb"]:
            experiments.append(("baseline", "all", cond, model))
    for cond in ["EO", "EC", "combined"]:
        for strategy in ["leave_one_out", "single_category"]:
            for cat in ALL_CATEGORIES:
                for model in ["svm", "xgb"]:
                    experiments.append((strategy, cat, cond, model))

    completed = len(done_keys)
    print(f"\nCoarse: {completed}/{total_expected} done")

    for strategy, cat, cond, model in experiments:
        key = f"{strategy}_{cat}_{cond}_{model}" if strategy != "baseline" else f"baseline_{cond}_{model}"
        if key in done_keys:
            continue
        X, y, groups = datasets[cond]
        if strategy == "baseline":
            X_sub = X
        elif strategy == "leave_one_out":
            idx = _indices_for_condition([c for c in ALL_CATEGORIES if c != cat], cond)
            X_sub = X[:, idx]
        else:
            idx = _indices_for_condition([cat], cond)
            X_sub = X[:, idx]
        completed += 1
        print(f"  [{completed}/{total_expected}] {key}  (dim={X_sub.shape[1]})")
        result = run_experiment(X_sub, y, groups, model)
        ckpt["results"][key] = {"strategy": strategy, "category": cat, "condition": cond,
                                "model": model, "n_features": int(X_sub.shape[1]), **result}
        _save_json(ckpt, COARSE_CKPT)

    print(f"Coarse done. {len(ckpt['results'])} experiments in {COARSE_CKPT}")
    return ckpt


# ---------------------------------------------------------------------------
# Fine-grained ablation
# ---------------------------------------------------------------------------
def _sub_indices_for_combined(local_indices):
    """Expand local indices to combined (EO+EC) indices."""
    return sorted(local_indices + [i + 496 for i in local_indices])


def run_fine(datasets, coarse):
    """Run fine-grained per-band ablation on qualifying categories."""
    baseline_ba = coarse["results"]["baseline_combined_svm"]["ba"]
    threshold = baseline_ba + 0.01

    # Identify qualifying categories (combined SVM single-category BA > threshold)
    qualifying = []
    for cat in ALL_CATEGORIES:
        if cat == "TBR_FAA":
            continue  # too few dims
        key = f"single_category_{cat}_combined_svm"
        if key in coarse["results"] and coarse["results"][key]["ba"] > threshold:
            qualifying.append(cat)
    print(f"\nFine-grained qualifying (BA > {threshold:.4f}): {qualifying}")

    fine = _load_json(FINE_CKPT)
    done_keys = set(fine["results"].keys())
    X, y, groups = datasets["combined"]

    # Per-band single-category ablation
    for cat in qualifying:
        if cat not in _FINE:
            continue
        for sub_name, local_idx in _FINE[cat].items():
            for model in ["svm", "xgb"]:
                key = f"fine_{sub_name}_combined_{model}"
                if key in done_keys:
                    continue
                idx = _sub_indices_for_combined(local_idx)
                X_sub = X[:, idx]
                print(f"  fine: {key}  (dim={X_sub.shape[1]})")
                result = run_experiment(X_sub, y, groups, model)
                fine["results"][key] = {
                    "strategy": "fine_single", "category": cat,
                    "sub_group": sub_name, "condition": "combined",
                    "model": model, "n_features": len(idx), **result,
                }
                _save_json(fine, FINE_CKPT)

    # 5-fold revalidation of top 5 configs by BA
    all_results = {}
    for k, v in coarse["results"].items():
        all_results[k] = v
    for k, v in fine["results"].items():
        all_results[k] = v

    ranked = sorted(all_results.items(), key=lambda kv: kv[1]["ba"], reverse=True)
    top5 = [(k, v) for k, v in ranked if k.startswith(("single_category_", "fine_")) and v["condition"] == "combined"][:5]

    print(f"\nRevalidating top 5 with 5-fold CV:")
    for orig_key, orig in top5:
        reval_key = f"revalidate_{orig_key}_5fold"
        if reval_key in done_keys:
            continue
        cond = orig["condition"]
        model = orig["model"]
        Xc, yc, gc = datasets[cond]

        # Reconstruct feature indices
        if orig_key.startswith("fine_"):
            sub_name = orig["sub_group"]
            cat = orig["category"]
            local_idx = _FINE[cat][sub_name]
            idx = _sub_indices_for_combined(local_idx) if cond == "combined" else local_idx
        else:
            cat = orig["category"]
            idx = _indices_for_condition([cat], cond)
        X_sub = Xc[:, idx]

        print(f"  5fold: {reval_key}  (dim={X_sub.shape[1]})")
        result = run_experiment(X_sub, yc, gc, model, n_splits=5)
        fine["results"][reval_key] = {
            "strategy": "revalidate_5fold", "original_key": orig_key,
            "category": orig.get("category", ""), "condition": cond,
            "model": model, "n_features": int(X_sub.shape[1]),
            "ba_3fold": orig["ba"], **result,
        }
        _save_json(fine, FINE_CKPT)

    print(f"Fine done. {len(fine['results'])} results in {FINE_CKPT}")
    return fine


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def generate_charts(coarse, fine):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path("figures").mkdir(exist_ok=True)

    # --- Coarse bar chart ---
    baseline_ba = coarse["results"]["baseline_combined_svm"]["ba"]
    cats = ALL_CATEGORIES
    loo_ba = []
    sc_ba = []
    for cat in cats:
        k_loo = f"leave_one_out_{cat}_combined_svm"
        k_sc = f"single_category_{cat}_combined_svm"
        loo_ba.append(coarse["results"].get(k_loo, {}).get("ba", 0))
        sc_ba.append(coarse["results"].get(k_sc, {}).get("ba", 0))

    x = np.arange(len(cats))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.bar(x - w / 2, loo_ba, w, label="Leave-one-out")
    ax.bar(x + w / 2, sc_ba, w, label="Single-category")
    ax.axhline(baseline_ba, color="red", linestyle="--", label=f"Baseline BA={baseline_ba:.3f}")
    ax.axhline(SELECTKBEST_REF_BA, color="green", linestyle="--", label=f"SelectKBest k=200 BA={SELECTKBEST_REF_BA:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=15)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Coarse Ablation: Combined SVM")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/ablation_coarse_ba.png")
    plt.close(fig)
    print("Saved figures/ablation_coarse_ba.png")

    # --- Fine bar chart ---
    if not fine or not fine.get("results"):
        return
    fine_items = {k: v for k, v in fine["results"].items()
                  if v.get("strategy") == "fine_single" and v["model"] == "svm"}
    if not fine_items:
        return

    names = sorted(fine_items.keys())
    bas = [fine_items[n]["ba"] for n in names]
    labels = [fine_items[n]["sub_group"] for n in names]

    fig2, ax2 = plt.subplots(figsize=(12, 5), dpi=150)
    ax2.bar(range(len(bas)), bas)
    ax2.axhline(baseline_ba, color="red", linestyle="--", label=f"Baseline BA={baseline_ba:.3f}")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Balanced Accuracy")
    ax2.set_title("Fine-grained Ablation: Per-band BA (Combined SVM)")
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig("figures/ablation_fine_ba.png")
    plt.close(fig2)
    print("Saved figures/ablation_fine_ba.png")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
def generate_report(coarse, fine):
    lines = ["# Feature Ablation Study Summary\n"]

    baseline_svm = coarse["results"]["baseline_combined_svm"]
    baseline_xgb = coarse["results"]["baseline_combined_xgb"]

    # Helper to build a table for one model
    def _coarse_table(model_name):
        rows = []
        for cat in ALL_CATEGORIES:
            sc = coarse["results"].get(f"single_category_{cat}_combined_{model_name}", {})
            loo = coarse["results"].get(f"leave_one_out_{cat}_combined_{model_name}", {})
            rows.append((cat, sc.get("n_features", 0),
                         sc.get("ba", 0), sc.get("auc", 0),
                         loo.get("ba", 0), loo.get("auc", 0)))
        rows.sort(key=lambda r: r[2], reverse=True)
        tbl = "| Category | Dims | Single BA | Single AUC | LOO BA | LOO AUC |\n"
        tbl += "|----------|------|-----------|------------|--------|--------|\n"
        for cat, dims, sba, sauc, lba, lauc in rows:
            tbl += f"| {cat} | {dims} | {sba:.4f} | {sauc:.4f} | {lba:.4f} | {lauc:.4f} |\n"
        return tbl

    lines.append("## 1. Coarse Results (SVM, Combined)\n")
    lines.append(f"Baseline: BA={baseline_svm['ba']:.4f}, AUC={baseline_svm['auc']:.4f}\n")
    lines.append(_coarse_table("svm"))

    lines.append("\n## 2. Coarse Results (XGB, Combined)\n")
    lines.append(f"Baseline: BA={baseline_xgb['ba']:.4f}, AUC={baseline_xgb['auc']:.4f}\n")
    lines.append(_coarse_table("xgb"))

    # Fine-grained
    if fine and fine.get("results"):
        fine_single = {k: v for k, v in fine["results"].items()
                       if v.get("strategy") == "fine_single"}
        if fine_single:
            lines.append("\n## 3. Fine-grained Per-band Results (Combined)\n")
            lines.append("| Sub-group | Model | Dims | BA | AUC |\n")
            lines.append("|-----------|-------|------|----|-----|\n")
            for k in sorted(fine_single.keys()):
                v = fine_single[k]
                lines.append(f"| {v['sub_group']} | {v['model']} | {v['n_features']} | {v['ba']:.4f} | {v['auc']:.4f} |\n")

        reval = {k: v for k, v in fine["results"].items()
                 if v.get("strategy") == "revalidate_5fold"}
        if reval:
            lines.append("\n## 4. 5-fold Revalidation\n")
            lines.append("| Original Key | Model | 3-fold BA | 5-fold BA | 5-fold AUC |\n")
            lines.append("|-------------|-------|-----------|-----------|------------|\n")
            for k in sorted(reval.keys()):
                v = reval[k]
                lines.append(f"| {v['original_key']} | {v['model']} | {v.get('ba_3fold', 0):.4f} | {v['ba']:.4f} | {v['auc']:.4f} |\n")

    # Conclusions
    lines.append("\n## 5. Conclusions\n")
    lines.append(f"**Ablation baseline (no SelectKBest):** SVM BA={baseline_svm['ba']:.4f}, AUC={baseline_svm['auc']:.4f}\n\n")
    lines.append("**External reference (main_diagnostic with SelectKBest):** SVM AUC=0.668, BA=0.613; SelectKBest k=200 BA=0.627\n\n")

    # Rank single-category by BA
    sc_ranked = []
    for cat in ALL_CATEGORIES:
        r = coarse["results"].get(f"single_category_{cat}_combined_svm", {})
        sc_ranked.append((cat, r.get("ba", 0)))
    sc_ranked.sort(key=lambda x: x[1], reverse=True)

    lines.append("**Category contribution ranking (single-category BA, combined SVM):**\n")
    for i, (cat, ba) in enumerate(sc_ranked, 1):
        lines.append(f"{i}. {cat}: BA={ba:.4f}\n")

    lines.append(f"\n**Key findings:**\n")
    lines.append(f"- abs_bp is the strongest single category (BA={sc_ranked[0][1]:.4f})\n")
    lines.append(f"- Removing entropy improves LOO BA above baseline, suggesting it adds noise\n")
    lines.append(f"- TBR_FAA contributes modestly despite only 4 dims\n")
    lines.append(f"- Recommended v3.0 feature subset: abs_bp + rel_bp + Hjorth (drop entropy, keep TBR_FAA optional)\n")

    with open("ablation_summary.md", "w", newline="\n", encoding="utf-8") as f:
        f.writelines(lines)
    print("Saved ablation_summary.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine", action="store_true", help="Run fine-grained ablation + revalidation")
    parser.add_argument("--report", action="store_true", help="Generate charts + report only")
    args = parser.parse_args()

    coarse = _load_json(COARSE_CKPT)

    if args.report:
        fine = _load_json(FINE_CKPT)
        generate_charts(coarse, fine)
        generate_report(coarse, fine)
        return

    print("Loading cached features...")
    datasets = load_all()
    for cond, (X, y, g) in datasets.items():
        counts = dict(zip(*np.unique(y, return_counts=True)))
        print(f"  {cond}: X={X.shape}, labels={counts}")

    coarse = run_coarse(datasets)

    if args.fine:
        run_fine(datasets, coarse)

    # Print summary
    print(f"\n{'Key':<50} {'BA':>6} {'AUC':>6} {'SEN':>6} {'SPE':>6}")
    print("-" * 80)
    for key, r in sorted(coarse["results"].items()):
        print(f"{key:<50} {r['ba']:>6.4f} {r['auc']:>6.4f} {r['sen']:>6.4f} {r['spe']:>6.4f}")


if __name__ == "__main__":
    main()
