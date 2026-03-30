"""Error analysis pipeline for SVM MDD classifier.
Collects per-subject OOF prediction probabilities, merges with demographic
covariates from participants.tsv, runs statistical tests for systematic bias,
and outputs structured results + visualizations.
"""
import argparse
import datetime
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu, chi2_contingency
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from run_ablation import load_all, _indices_for_condition
from config import PARTICIPANTS_TSV

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
V3_CATEGORIES = ["abs_bp", "rel_bp", "Hjorth"]
N_SPLITS = 5

RESULTS_PATH = Path("error_analysis.json")
SUMMARY_PATH = Path("error_analysis_summary.md")
FIGURE_DIR = Path("figures")

# ---------------------------------------------------------------------------
# OOF probability collection
# ---------------------------------------------------------------------------
def collect_oof_probabilities(X, y_enc, groups, n_splits=N_SPLITS):
    """Collect per-subject OOF predicted probabilities via StratifiedGroupKFold."""
    le = LabelEncoder()
    le.fit(["MDD", "nonMDD"])
    mdd_idx = list(le.classes_).index("MDD")

    cv = StratifiedGroupKFold(n_splits=n_splits)
    probas = np.full(len(y_enc), np.nan)
    fold_ids = np.full(len(y_enc), -1, dtype=int)

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y_enc, groups)):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True,
                        class_weight="balanced", random_state=42)),
        ])
        pipe.fit(X[train_idx], y_enc[train_idx])
        probas[test_idx] = pipe.predict_proba(X[test_idx])[:, mdd_idx]
        fold_ids[test_idx] = fold_i
        print(f"  Fold {fold_i + 1}/{n_splits} done")

    assert not np.any(np.isnan(probas)), "Some subjects missing OOF predictions"
    return probas, fold_ids, mdd_idx


# ---------------------------------------------------------------------------
# Optimal threshold via Youden's J
# ---------------------------------------------------------------------------
def compute_optimal_threshold(y_bin, probas):
    """Find optimal threshold using Youden's J statistic on OOF predictions."""
    fpr, tpr, thresholds = roc_curve(y_bin, probas)
    j_scores = tpr - fpr
    opt_idx = np.argmax(j_scores)
    return float(thresholds[opt_idx])


def categorize_subjects(y_bin, probas, threshold):
    """Categorize each subject as TP/TN/FP/FN based on threshold."""
    pred = (probas >= threshold).astype(int)
    categories = np.where(
        (y_bin == 1) & (pred == 1), "TP",
        np.where((y_bin == 0) & (pred == 0), "TN",
        np.where((y_bin == 0) & (pred == 1), "FP", "FN")))
    return categories, pred


# ---------------------------------------------------------------------------
# Covariate loading
# ---------------------------------------------------------------------------
def load_covariates(subject_ids):
    """Load and parse demographic covariates from participants.tsv."""
    df = pd.read_csv(PARTICIPANTS_TSV, sep="\t")
    df = df[df["participants_ID"].isin(subject_ids)].drop_duplicates(
        subset=["participants_ID"])
    df = df.set_index("participants_ID").loc[subject_ids].reset_index()

    # Parse age (comma decimal separator)
    df["age_float"] = df["age"].astype(str).str.replace(",", ".").astype(float)
    # Education: treat 99 as NaN
    df["education_clean"] = df["education"].replace(99, np.nan)

    return df[["participants_ID", "indication", "age_float", "gender",
               "education_clean"]]


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------
def _mann_whitney(group_a, group_b, label_a="a", label_b="b"):
    """Run Mann-Whitney U test, handling edge cases."""
    a = np.array(group_a, dtype=float)
    b = np.array(group_b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return {"U_stat": None, "p_value": None,
                f"mean_{label_a}": float(np.mean(a)) if len(a) else None,
                f"mean_{label_b}": float(np.mean(b)) if len(b) else None,
                f"n_{label_a}": len(a), f"n_{label_b}": len(b)}
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    return {"U_stat": float(stat), "p_value": float(p),
            f"mean_{label_a}": round(float(np.mean(a)), 2),
            f"mean_{label_b}": round(float(np.mean(b)), 2),
            f"n_{label_a}": len(a), f"n_{label_b}": len(b)}


def _chi2_test(cat_a, cat_b):
    """Run chi-squared contingency test on two categorical arrays."""
    a = np.array(cat_a, dtype=int)
    b = np.array(cat_b, dtype=int)
    combined = np.concatenate([a, b])
    labels = np.array(["a"] * len(a) + ["b"] * len(b))
    table = pd.crosstab(pd.Series(labels), pd.Series(combined))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"chi2": None, "p_value": None, "dof": None}
    chi2, p, dof, _ = chi2_contingency(table)
    return {"chi2": round(float(chi2), 4), "p_value": float(p), "dof": int(dof)}


def run_covariate_tests(covariates, categories):
    """Run all covariate bias tests. Returns dict of test results."""
    df = covariates.copy()
    df["category"] = categories

    correct = df[df["category"].isin(["TP", "TN"])]
    wrong = df[df["category"].isin(["FP", "FN"])]
    tp = df[df["category"] == "TP"]
    tn = df[df["category"] == "TN"]
    fp = df[df["category"] == "FP"]
    fn = df[df["category"] == "FN"]

    tests = {}
    # --- Age tests ---
    tests["age"] = {
        "correct_vs_wrong": _mann_whitney(
            correct["age_float"], wrong["age_float"], "correct", "wrong"),
        "FN_vs_TP": _mann_whitney(fn["age_float"], tp["age_float"], "FN", "TP"),
        "FP_vs_TN": _mann_whitney(fp["age_float"], tn["age_float"], "FP", "TN"),
    }

    # --- Gender tests ---
    tests["gender"] = {
        "correct_vs_wrong": _chi2_test(correct["gender"], wrong["gender"]),
        "FN_vs_TP": _chi2_test(fn["gender"], tp["gender"]),
        "FP_vs_TN": _chi2_test(fp["gender"], tn["gender"]),
    }

    # --- Education tests ---
    tests["education"] = {
        "correct_vs_wrong": _mann_whitney(
            correct["education_clean"], wrong["education_clean"],
            "correct", "wrong"),
    }

    # --- Indication FP rate (nonMDD only, n >= 10) ---
    non_mdd = df[df["category"].isin(["TN", "FP"])]
    indication_fp = {}
    for ind, grp in non_mdd.groupby("indication"):
        n = len(grp)
        if n < 10:
            continue
        fp_count = int((grp["category"] == "FP").sum())
        indication_fp[ind] = {
            "n": n, "fp": fp_count,
            "fp_rate": round(fp_count / n, 4),
        }
    tests["indication_fp_rate"] = indication_fp

    # --- Count tests for Bonferroni ---
    # age: 3 tests, gender: 3 tests, education: 1 test = 7
    n_tests = 7
    tests["n_tests"] = n_tests
    tests["bonferroni_threshold"] = round(0.05 / n_tests, 6)

    return tests


# ---------------------------------------------------------------------------
# Feature-space distance analysis
# ---------------------------------------------------------------------------
def compute_feature_distances(X, y_bin, categories):
    """Compute Euclidean distance to own-class centroid for each subject."""
    mdd_mask = y_bin == 1
    centroid_mdd = X[mdd_mask].mean(axis=0)
    centroid_nonmdd = X[~mdd_mask].mean(axis=0)

    distances = np.zeros(len(X))
    for i in range(len(X)):
        centroid = centroid_mdd if y_bin[i] == 1 else centroid_nonmdd
        distances[i] = np.linalg.norm(X[i] - centroid)

    correct_mask = np.isin(categories, ["TP", "TN"])
    wrong_mask = np.isin(categories, ["FP", "FN"])

    result = _mann_whitney(
        distances[correct_mask], distances[wrong_mask],
        "correct", "wrong")
    result["median_correct"] = round(float(np.median(distances[correct_mask])), 4)
    result["median_wrong"] = round(float(np.median(distances[wrong_mask])), 4)
    return result


# ---------------------------------------------------------------------------
# Build summary section
# ---------------------------------------------------------------------------
def build_summary(covariate_tests, indication_fp):
    """Identify systematic biases and high-FP subgroups."""
    bonf = covariate_tests["bonferroni_threshold"]
    biases = []

    for cov_name in ["age", "gender", "education"]:
        cov = covariate_tests.get(cov_name, {})
        for test_name, result in cov.items():
            if isinstance(result, dict) and result.get("p_value") is not None:
                if result["p_value"] < bonf:
                    biases.append(f"{cov_name}_{test_name}")

    # High FP subgroups: FP rate > overall FP rate
    overall_fp = sum(v["fp"] for v in indication_fp.values())
    overall_n = sum(v["n"] for v in indication_fp.values())
    overall_rate = overall_fp / overall_n if overall_n > 0 else 0
    high_fp = [ind for ind, v in indication_fp.items()
               if v["fp_rate"] > overall_rate and v["n"] >= 10]

    recommendations = []
    if any("age" in b for b in biases):
        recommendations.append(
            "Age is a significant confound. Consider age-stratified evaluation "
            "or age as a covariate in the model.")
    if any("gender" in b for b in biases):
        recommendations.append(
            "Gender bias detected. Investigate gender-specific EEG patterns.")
    if high_fp:
        recommendations.append(
            f"High FP subgroups ({', '.join(high_fp)}) may share EEG features "
            f"with MDD. Consider multi-class or hierarchical classification.")
    if not recommendations:
        recommendations.append(
            "No strong systematic biases detected. Focus on feature engineering "
            "and model tuning for v3.0.")

    return {
        "systematic_biases": biases,
        "high_fp_subgroups": high_fp,
        "overall_fp_rate": round(overall_rate, 4),
        "recommendations": recommendations,
    }


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------
def save_results(subject_ids, y_raw, categories, probas, fold_ids,
                 covariates, covariate_tests, feature_distances,
                 opt_threshold, mdd_idx, n_features):
    """Save structured error analysis results to JSON."""
    # Confusion matrix counts
    cm = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for c in categories:
        cm[c] += 1

    # Per-subject records
    cov_df = covariates.set_index("participants_ID")
    per_subject = []
    for i, sid in enumerate(subject_ids):
        true_label = str(y_raw[i])
        pred_label = "MDD" if probas[i] >= opt_threshold else "nonMDD"
        row = cov_df.loc[sid] if sid in cov_df.index else None
        per_subject.append({
            "subject_id": sid,
            "true_label": true_label,
            "pred_label": pred_label,
            "category": categories[i],
            "proba_mdd": round(float(probas[i]), 6),
            "fold": int(fold_ids[i]),
            "age": round(float(row["age_float"]), 1) if row is not None else None,
            "gender": int(row["gender"]) if row is not None else None,
            "indication": str(row["indication"]) if row is not None else None,
            "education": (int(row["education_clean"])
                          if row is not None and pd.notna(row["education_clean"])
                          else None),
        })

    label_counts = {}
    for lbl in y_raw:
        label_counts[str(lbl)] = label_counts.get(str(lbl), 0) + 1

    indication_fp = covariate_tests.get("indication_fp_rate", {})
    summary = build_summary(covariate_tests, indication_fp)

    output = {
        "task": "error_analysis",
        "timestamp": datetime.datetime.now().isoformat(),
        "n_subjects": len(subject_ids),
        "label_counts": label_counts,
        "feature_set": "v3_subset",
        "n_features": n_features,
        "optimal_threshold": round(opt_threshold, 6),
        "confusion_matrix": cm,
        "per_subject": per_subject,
        "covariate_tests": covariate_tests,
        "feature_distances": feature_distances,
        "summary": summary,
    }

    with open(RESULTS_PATH, "w", newline="\n", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved {RESULTS_PATH}")
    return output


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def generate_charts(data):
    """Generate all 4 visualization PNGs from results data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(exist_ok=True)
    per_sub = pd.DataFrame(data["per_subject"])
    cm = data["confusion_matrix"]
    threshold = data["optimal_threshold"]

    # --- 1. Confusion matrix heatmap ---
    total = sum(cm.values())
    mat = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(mat, cmap="Blues", aspect="equal")
    for i in range(2):
        for j in range(2):
            pct = mat[i, j] / total * 100
            ax.text(j, i, f"{mat[i, j]}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=14,
                    color="white" if mat[i, j] > total * 0.3 else "black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["nonMDD", "MDD"])
    ax.set_yticklabels(["nonMDD", "MDD"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Error Analysis Confusion Matrix (Youden's J threshold)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "error_analysis_confusion.png")
    plt.close(fig)
    print(f"Saved {FIGURE_DIR / 'error_analysis_confusion.png'}")

    # --- 2. Probability distribution by category ---
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    colors = {"TP": "#2ca02c", "TN": "#1f77b4", "FP": "#d62728", "FN": "#ff7f0e"}
    for cat in ["TN", "FP", "FN", "TP"]:
        subset = per_sub[per_sub["category"] == cat]["proba_mdd"]
        n = len(subset)
        ax.hist(subset, bins=30, alpha=0.5, color=colors[cat],
                label=f"{cat} (n={n})", density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"Threshold={threshold:.3f}")
    ax.set_xlabel("P(MDD)")
    ax.set_ylabel("Density")
    ax.set_title("P(MDD) Distribution by Error Category")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "error_analysis_proba_dist.png")
    plt.close(fig)
    print(f"Saved {FIGURE_DIR / 'error_analysis_proba_dist.png'}")

    # --- 3. Age distribution by error category ---
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    age_data = []
    cat_labels = []
    for cat in ["TP", "TN", "FP", "FN"]:
        ages = per_sub[per_sub["category"] == cat]["age"].dropna()
        age_data.append(ages.values)
        cat_labels.append(cat)

    bp = ax.boxplot(age_data, tick_labels=cat_labels, patch_artist=True,
                    boxprops=dict(alpha=0.6))
    box_colors = [colors[c] for c in cat_labels]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)

    # Jittered points
    for i, (cat, ages) in enumerate(zip(cat_labels, age_data)):
        jitter = np.random.RandomState(42).normal(0, 0.04, len(ages))
        ax.scatter(np.full(len(ages), i + 1) + jitter, ages,
                   alpha=0.3, s=8, color=colors[cat])
        ax.text(i + 1, ax.get_ylim()[1] * 0.95,
                f"mean={np.mean(ages):.1f}", ha="center", fontsize=8)

    ax.set_ylabel("Age")
    ax.set_title("Age Distribution by Error Category")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "error_analysis_age_box.png")
    plt.close(fig)
    print(f"Saved {FIGURE_DIR / 'error_analysis_age_box.png'}")

    # --- 4. FP rate by indication subgroup ---
    ind_fp = data["covariate_tests"].get("indication_fp_rate", {})
    if ind_fp:
        sorted_inds = sorted(ind_fp.items(), key=lambda x: x[1]["fp_rate"],
                             reverse=True)
        names = [x[0] for x in sorted_inds]
        rates = [x[1]["fp_rate"] for x in sorted_inds]
        ns = [x[1]["n"] for x in sorted_inds]
        fps = [x[1]["fp"] for x in sorted_inds]

        overall_rate = data["summary"]["overall_fp_rate"]

        fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.5)), dpi=150)
        y_pos = range(len(names))
        bars = ax.barh(y_pos, rates, color="#4C72B0", alpha=0.8)
        ax.axvline(overall_rate, color="red", linestyle="--",
                   label=f"Overall FP rate={overall_rate:.3f}")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel("False Positive Rate")
        ax.set_title("False Positive Rate by nonMDD Indication (n >= 10)")
        ax.legend(fontsize=8)

        for i, (n, fp) in enumerate(zip(ns, fps)):
            ax.text(rates[i] + 0.01, i, f"n={n}, FP={fp}", va="center",
                    fontsize=7)

        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "error_analysis_fp_by_indication.png")
        plt.close(fig)
        print(f"Saved {FIGURE_DIR / 'error_analysis_fp_by_indication.png'}")


# ---------------------------------------------------------------------------
# Markdown summary report
# ---------------------------------------------------------------------------
def generate_summary_report(data):
    """Generate human-readable error analysis summary."""
    cm = data["confusion_matrix"]
    total = sum(cm.values())
    threshold = data["optimal_threshold"]
    tests = data["covariate_tests"]
    bonf = tests["bonferroni_threshold"]
    summary = data["summary"]

    lines = []
    lines.append("# Error Analysis Summary\n\n")

    # Section 1: Setup
    lines.append("## 1. Experiment Setup\n\n")
    lines.append(f"- **Pipeline:** StandardScaler + SVC(kernel=rbf, "
                 f"class_weight=balanced, random_state=42)\n")
    lines.append(f"- **Features:** v3_subset (abs_bp + rel_bp + Hjorth, "
                 f"{data['n_features']}-dim, combined EO+EC)\n")
    lines.append(f"- **CV:** {N_SPLITS}-fold StratifiedGroupKFold\n")
    lines.append(f"- **Subjects:** {data['n_subjects']} "
                 f"(MDD={data['label_counts'].get('MDD', 0)}, "
                 f"nonMDD={data['label_counts'].get('nonMDD', 0)})\n")
    lines.append(f"- **Threshold:** Youden's J optimal = {threshold:.4f} "
                 f"(NOT default 0.5)\n\n")

    # Section 2: Confusion matrix
    lines.append("## 2. Confusion Matrix\n\n")
    lines.append("| | Predicted nonMDD | Predicted MDD |\n")
    lines.append("|---|---|---|\n")
    lines.append(f"| **True nonMDD** | TN={cm['TN']} "
                 f"({cm['TN']/total*100:.1f}%) | FP={cm['FP']} "
                 f"({cm['FP']/total*100:.1f}%) |\n")
    lines.append(f"| **True MDD** | FN={cm['FN']} "
                 f"({cm['FN']/total*100:.1f}%) | TP={cm['TP']} "
                 f"({cm['TP']/total*100:.1f}%) |\n\n")
    sen = cm["TP"] / (cm["TP"] + cm["FN"]) if (cm["TP"] + cm["FN"]) > 0 else 0
    spe = cm["TN"] / (cm["TN"] + cm["FP"]) if (cm["TN"] + cm["FP"]) > 0 else 0
    lines.append(f"- **Sensitivity:** {sen:.4f}\n")
    lines.append(f"- **Specificity:** {spe:.4f}\n")
    lines.append(f"- **Balanced Accuracy:** {(sen + spe) / 2:.4f}\n\n")

    # Section 3: Covariate analysis
    lines.append("## 3. Covariate Analysis\n\n")
    lines.append(f"Bonferroni correction: {tests['n_tests']} tests, "
                 f"threshold p < {bonf:.6f}\n\n")
    lines.append("| Test | Comparison | Statistic | p-value | Significant? |\n")
    lines.append("|------|-----------|-----------|---------|-------------|\n")

    for cov_name in ["age", "gender", "education"]:
        cov = tests.get(cov_name, {})
        for comp_name, result in cov.items():
            if not isinstance(result, dict):
                continue
            p = result.get("p_value")
            if p is None:
                continue
            stat_key = "U_stat" if "U_stat" in result else "chi2"
            stat_val = result.get(stat_key, "N/A")
            sig = "YES" if p < bonf else "no"
            stat_str = f"{stat_val:.1f}" if stat_val is not None else "N/A"
            lines.append(f"| {cov_name} | {comp_name} | {stat_str} | "
                         f"{p:.6f} | {sig} |\n")
    lines.append("\n")

    # Section 4: Indication FP rates
    ind_fp = tests.get("indication_fp_rate", {})
    if ind_fp:
        lines.append("## 4. FP Rate by nonMDD Indication (n >= 10)\n\n")
        lines.append("| Indication | n | FP count | FP rate |\n")
        lines.append("|-----------|---|----------|--------|\n")
        sorted_inds = sorted(ind_fp.items(),
                             key=lambda x: x[1]["fp_rate"], reverse=True)
        for ind, v in sorted_inds:
            lines.append(f"| {ind} | {v['n']} | {v['fp']} | "
                         f"{v['fp_rate']:.4f} |\n")
        lines.append(f"\nOverall nonMDD FP rate: "
                     f"{summary['overall_fp_rate']:.4f}\n\n")

    # Section 5: Key findings
    lines.append("## 5. Key Findings\n\n")
    if summary["systematic_biases"]:
        lines.append("**Significant biases detected:**\n")
        for b in summary["systematic_biases"]:
            lines.append(f"- {b}\n")
    else:
        lines.append("No statistically significant biases after "
                     "Bonferroni correction.\n")
    lines.append("\n")

    if summary["high_fp_subgroups"]:
        lines.append("**High FP-rate subgroups "
                     "(above overall FP rate):**\n")
        for sg in summary["high_fp_subgroups"]:
            fp_info = ind_fp.get(sg, {})
            lines.append(f"- {sg}: FP rate={fp_info.get('fp_rate', '?')}, "
                         f"n={fp_info.get('n', '?')}\n")
    lines.append("\n")

    # Feature distance
    fd = data.get("feature_distances", {})
    if fd and fd.get("p_value") is not None:
        lines.append("**Feature-space distance analysis:**\n")
        lines.append(f"- Misclassified subjects are "
                     f"{'farther' if fd['median_wrong'] > fd['median_correct'] else 'closer'} "
                     f"from own-class centroid "
                     f"(median {fd['median_wrong']:.2f} vs {fd['median_correct']:.2f}, "
                     f"p={fd['p_value']:.6f})\n\n")

    # Section 6: Recommendations
    lines.append("## 6. Recommendations for v3.0\n\n")
    for i, rec in enumerate(summary["recommendations"], 1):
        lines.append(f"{i}. {rec}\n")
    lines.append("\n")

    # Limitations
    lines.append("## 7. Limitations\n\n")
    lines.append("- SVM probability=True uses Platt scaling, which may be "
                 "poorly calibrated on imbalanced data. Probabilities are "
                 "used for ranking only.\n")
    lines.append("- Optimal threshold selected on same OOF predictions used "
                 "for analysis (acceptable for pattern analysis, not "
                 "performance reporting).\n")
    lines.append("- Dataset column too sparse (19.5% coverage) for "
                 "site-level analysis.\n")

    with open(SUMMARY_PATH, "w", newline="\n", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Saved {SUMMARY_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Error analysis for SVM MDD classifier.")
    parser.add_argument("--report-only", action="store_true",
                        help="Regenerate charts/summary from existing JSON")
    args = parser.parse_args()

    if args.report_only:
        if not RESULTS_PATH.exists():
            print(f"ERROR: {RESULTS_PATH} not found. Run full analysis first.")
            return
        data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        generate_charts(data)
        generate_summary_report(data)
        return

    # --- Full analysis ---
    print("Loading cached features...")
    datasets = load_all()
    X_full, y_raw, groups = datasets["combined"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    n_subjects = len(groups)

    unique, counts = np.unique(y_raw, return_counts=True)
    print(f"  combined: X={X_full.shape}, "
          f"labels={dict(zip(unique.tolist(), counts.tolist()))}")

    # v3_subset features
    v3_idx = _indices_for_condition(V3_CATEGORIES, "combined")
    X_v3 = X_full[:, v3_idx]
    print(f"  v3_subset: {X_v3.shape[1]} features")

    # Collect OOF probabilities
    print("\nCollecting OOF probabilities...")
    probas, fold_ids, mdd_idx = collect_oof_probabilities(X_v3, y_enc, groups)

    # Binary labels (MDD=1)
    y_bin = (y_enc == mdd_idx).astype(int)

    # Optimal threshold
    opt_threshold = compute_optimal_threshold(y_bin, probas)
    print(f"  Optimal threshold (Youden's J): {opt_threshold:.4f}")

    # Categorize subjects
    categories, pred = categorize_subjects(y_bin, probas, opt_threshold)
    for cat in ["TP", "TN", "FP", "FN"]:
        print(f"  {cat}: {np.sum(categories == cat)}")

    # Load covariates
    print("\nLoading covariates from participants.tsv...")
    subject_ids = list(groups)
    covariates = load_covariates(subject_ids)
    print(f"  Loaded {len(covariates)} subjects with covariates")

    # Statistical tests
    print("\nRunning covariate bias tests...")
    covariate_tests = run_covariate_tests(covariates, categories)
    bonf = covariate_tests["bonferroni_threshold"]
    print(f"  Bonferroni threshold: {bonf:.6f}")

    # Feature-space distance
    print("\nComputing feature-space distances...")
    feature_distances = compute_feature_distances(X_v3, y_bin, categories)
    print(f"  Median distance correct={feature_distances['median_correct']:.2f}, "
          f"wrong={feature_distances['median_wrong']:.2f}, "
          f"p={feature_distances['p_value']:.6f}")

    # Save JSON
    print("\nSaving results...")
    data = save_results(
        subject_ids, y_raw, categories, probas, fold_ids,
        covariates, covariate_tests, feature_distances,
        opt_threshold, mdd_idx, X_v3.shape[1])

    # Generate charts and summary
    print("\nGenerating visualizations...")
    generate_charts(data)
    generate_summary_report(data)

    # Final summary
    cm = data["confusion_matrix"]
    print(f"\n{'=' * 60}")
    print(f"Error Analysis Complete")
    print(f"  Subjects: {n_subjects}")
    print(f"  Threshold: {opt_threshold:.4f}")
    print(f"  TP={cm['TP']} TN={cm['TN']} FP={cm['FP']} FN={cm['FN']}")
    print(f"  Biases: {data['summary']['systematic_biases']}")
    print(f"  High FP subgroups: {data['summary']['high_fp_subgroups']}")


if __name__ == "__main__":
    main()
