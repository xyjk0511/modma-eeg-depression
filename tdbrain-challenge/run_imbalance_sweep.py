"""Class imbalance strategy sweep.
Compares four strategies (none, class_weight, smote, smote_cw) on SVM
with two feature subsets (v3_subset, full) using 5-fold stratified group CV.
Multi-seed robustness: seeds [42, 0, 123].
"""
import argparse
import datetime
import json
import warnings
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from run_ablation import load_all, _LOCAL, _indices_for_condition

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STRATEGIES = ["none", "class_weight", "smote", "smote_cw"]
SEEDS = [42, 0, 123]
FEATURE_SETS = ["v3_subset", "full"]
SVM_BASELINE_BA = 0.613  # Phase 1 SVM baseline (results_diagnostic.json)

STRATEGY_LABELS = {
    "none":         "No resampling",
    "class_weight": "class_weight=balanced",
    "smote":        "SMOTE",
    "smote_cw":     "SMOTE + class_weight",
}
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# v3 subset categories (drop entropy, drop TBR_FAA)
V3_CATEGORIES = ["abs_bp", "rel_bp", "Hjorth"]

RESULTS_PATH = Path("imbalance_results.json")

# ---------------------------------------------------------------------------
# Feature index helpers
# ---------------------------------------------------------------------------

def _v3_indices():
    """Return sorted feature indices for v3_subset in combined mode."""
    return _indices_for_condition(V3_CATEGORIES, "combined")


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def _make_strategy_pipe(strategy: str, rs: int = 42, smote_k: int = 5) -> ImbPipeline:
    """Build an ImbPipeline for the given strategy and random state.

    strategy: "none" | "class_weight" | "smote" | "smote_cw"
    """
    use_smote = strategy in ("smote", "smote_cw")
    use_cw = strategy in ("class_weight", "smote_cw")

    steps = [("scaler", StandardScaler())]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=rs, k_neighbors=smote_k)))
    steps.append(("clf", SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced" if use_cw else None,
        random_state=rs,
    )))
    return ImbPipeline(steps)


def _make_no_smote_pipe(strategy: str, rs: int = 42) -> ImbPipeline:
    """Fallback pipeline without SMOTE step (used when SMOTE fails)."""
    use_cw = strategy in ("class_weight", "smote_cw")
    steps = [
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced" if use_cw else None,
            random_state=rs,
        )),
    ]
    return ImbPipeline(steps)


# ---------------------------------------------------------------------------
# OOF evaluation
# ---------------------------------------------------------------------------

def _eval_oof(strategy: str, X: np.ndarray, y_enc: np.ndarray,
              groups: np.ndarray, rs: int, n_splits: int = 5) -> dict:
    """Run OOF evaluation for a given strategy and random state.

    Uses StratifiedGroupKFold. For SMOTE strategies, applies fallback chain
    (k=5 -> k=3 -> no SMOTE) per fold.

    y_enc must already be label-encoded. MDD is the positive class.
    Returns dict with ba, auc, sen, spe (all rounded to 4 decimals).
    """
    use_smote = strategy in ("smote", "smote_cw")
    cv = StratifiedGroupKFold(n_splits=n_splits)

    # Determine mdd_idx from encoded labels
    # LabelEncoder sorts alphabetically: MDD=0, nonMDD=1
    # We need to confirm which index is MDD
    le = LabelEncoder()
    le.fit(["MDD", "nonMDD"])
    mdd_idx = list(le.classes_).index("MDD")

    probas, preds, test_idxs = [], [], []

    for train_idx, test_idx in cv.split(X, y_enc, groups):
        X_tr, y_tr = X[train_idx], y_enc[train_idx]
        X_te = X[test_idx]

        if use_smote:
            pipe = None
            for k in (5, 3):
                try:
                    p = _make_strategy_pipe(strategy, rs=rs, smote_k=k)
                    p.fit(X_tr, y_tr)
                    pipe = p
                    break
                except ValueError:
                    warnings.warn(
                        f"SMOTE k={k} failed for strategy={strategy} rs={rs}, trying lower k",
                        stacklevel=2,
                    )
                    continue
            if pipe is None:
                warnings.warn(
                    f"SMOTE fallback: using no-SMOTE pipe for strategy={strategy} rs={rs}",
                    stacklevel=2,
                )
                pipe = _make_no_smote_pipe(strategy, rs=rs)
                pipe.fit(X_tr, y_tr)
        else:
            pipe = _make_strategy_pipe(strategy, rs=rs)
            pipe.fit(X_tr, y_tr)

        probas.append(pipe.predict_proba(X_te)[:, mdd_idx])
        preds.append(pipe.predict(X_te))
        test_idxs.append(test_idx)

    order = np.argsort(np.concatenate(test_idxs))
    proba = np.concatenate(probas)[order]
    pred = np.concatenate(preds)[order]

    # MDD as positive class
    y_bin = (y_enc == mdd_idx).astype(int)
    pred_bin = (pred == mdd_idx).astype(int)

    auc = float(roc_auc_score(y_bin, proba))
    tn, fp, fn, tp = confusion_matrix(y_bin, pred_bin).ravel()
    sen = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spe = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    ba = (sen + spe) / 2

    return {
        "ba":  round(ba, 4),
        "auc": round(auc, 4),
        "sen": round(sen, 4),
        "spe": round(spe, 4),
    }


# ---------------------------------------------------------------------------
# Selection logic
# ---------------------------------------------------------------------------

def _select_best(results: dict) -> dict:
    """Select best strategy by BA ranking with SEN/SPE constraint.

    Rules (locked decisions):
    1. Filter: mean SEN >= 0.40 AND mean SPE >= 0.40
    2. Also flag strategies where ANY seed violates the constraint
    3. Rank by BA mean (descending)
    4. Tolerance: if top_BA - candidate_BA <= 0.01, prefer simpler strategy
       Simplicity order: none > class_weight > smote > smote_cw

    Returns dict with strategy, feature_set, rationale.
    """
    SIMPLICITY = ["none", "class_weight", "smote", "smote_cw"]

    best_overall = None
    best_rationale = "No strategy passed SEN>=0.40 AND SPE>=0.40 constraint."

    for feat_set in ["v3_subset", "full"]:
        candidates = []
        for strategy in SIMPLICITY:
            r = results[strategy][feat_set]
            sen_ok = r["sen_mean"] >= 0.40
            spe_ok = r["spe_mean"] >= 0.40
            # Check per-seed constraint violations
            any_seed_fail = any(
                s["sen"] < 0.40 or s["spe"] < 0.40
                for s in r["per_seed"]
            )
            if sen_ok and spe_ok:
                candidates.append({
                    "strategy": strategy,
                    "ba_mean": r["ba_mean"],
                    "any_seed_fail": any_seed_fail,
                })

        if not candidates:
            continue

        best_ba = max(c["ba_mean"] for c in candidates)
        # Apply tolerance + simplicity: iterate in simplicity order
        chosen = None
        for c in candidates:  # already in simplicity order
            if best_ba - c["ba_mean"] <= 0.01:
                chosen = c
                break

        if chosen is None:
            chosen = max(candidates, key=lambda c: c["ba_mean"])

        seed_warn = " (some seeds violate constraint)" if chosen["any_seed_fail"] else ""
        rationale = (
            f"strategy={chosen['strategy']}, feat_set={feat_set}, "
            f"BA={chosen['ba_mean']:.4f}{seed_warn}. "
            f"Selected by BA ranking with SEN/SPE>=0.40 constraint and 0.01 tolerance rule."
        )

        if best_overall is None or chosen["ba_mean"] > best_overall["ba_mean"]:
            best_overall = {"strategy": chosen["strategy"], "feature_set": feat_set,
                            "ba_mean": chosen["ba_mean"], "rationale": rationale}

    if best_overall is None:
        return {"strategy": None, "feature_set": None, "rationale": best_rationale}

    return {
        "strategy": best_overall["strategy"],
        "feature_set": best_overall["feature_set"],
        "rationale": best_overall["rationale"],
    }


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _load_results() -> dict:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return {}


def _save_results(data: dict) -> None:
    with open(RESULTS_PATH, "w", newline="\n", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------

def generate_chart(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path("figures").mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150, sharey=True)
    for ax, feat_set in zip(axes, FEATURE_SETS):
        bas  = [results[s][feat_set]["ba_mean"] for s in STRATEGIES]
        stds = [results[s][feat_set]["ba_std"]  for s in STRATEGIES]
        x = np.arange(len(STRATEGIES))
        ax.bar(x, bas, yerr=stds, capsize=4, color=COLORS, alpha=0.85)
        ax.axhline(SVM_BASELINE_BA, color="red", linestyle="--", linewidth=1,
                   label=f"Phase 1 SVM baseline BA={SVM_BASELINE_BA:.3f}")
        ax.set_xticks(x)
        ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGIES],
                           rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Balanced Accuracy")
        ax.set_title(f"Feature set: {feat_set}")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)

    fig.suptitle("Class Imbalance Strategy Comparison (SVM, Combined, 3-seed mean±std)")
    fig.tight_layout()
    fig.savefig("figures/imbalance_ba.png")
    plt.close(fig)
    print("Saved figures/imbalance_ba.png")


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def generate_summary(output: dict) -> None:
    results = output["strategies"]
    rec = output.get("recommendation", {})

    lines = []
    lines.append("# Class Imbalance Strategy Sweep — Summary\n\n")

    # Section 1: Experiment setup
    lines.append("## 1. Experiment Setup\n\n")
    lines.append(f"- **Strategies:** {', '.join(STRATEGIES)}\n")
    lines.append(f"- **Feature sets:** v3_subset (abs_bp + rel_bp + Hjorth), full (all 992 features)\n")
    lines.append(f"- **Seeds:** {output.get('seeds', SEEDS)}\n")
    lines.append(f"- **CV:** 5-fold StratifiedGroupKFold (no subject leakage)\n")
    lines.append(f"- **Model:** SVM (kernel=rbf, fixed params, no GridSearch)\n")
    lines.append(f"- **Condition:** combined (EO+EC concatenated, 992-dim)\n")
    lines.append(f"- **Subjects:** {output.get('n_subjects', 'N/A')}\n")
    lines.append(f"- **Label counts:** {output.get('label_counts', {})}\n")
    lines.append(f"- **SMOTE:** k=5 with fallback to k=3, then no SMOTE (applied inside CV fold only)\n\n")

    # Section 2: Results table
    lines.append("## 2. Results\n\n")
    for feat_set in FEATURE_SETS:
        lines.append(f"### Feature set: {feat_set}\n\n")
        lines.append("| Strategy | BA mean±std | AUC mean | SEN mean | SPE mean |\n")
        lines.append("|----------|-------------|----------|----------|----------|\n")
        for s in STRATEGIES:
            r = results[s][feat_set]
            lines.append(
                f"| {STRATEGY_LABELS[s]} "
                f"| {r['ba_mean']:.4f}±{r['ba_std']:.4f} "
                f"| {r['auc_mean']:.4f} "
                f"| {r['sen_mean']:.4f} "
                f"| {r['spe_mean']:.4f} |\n"
            )
        lines.append("\n")

    # Section 3: Constraint check
    lines.append("## 3. Constraint Check (SEN>=0.40 AND SPE>=0.40)\n\n")
    lines.append("| Strategy | Feature set | SEN mean | SPE mean | Passes? | Any seed fails? |\n")
    lines.append("|----------|-------------|----------|----------|---------|----------------|\n")
    for feat_set in FEATURE_SETS:
        for s in STRATEGIES:
            r = results[s][feat_set]
            passes = r["sen_mean"] >= 0.40 and r["spe_mean"] >= 0.40
            any_fail = any(
                ps["sen"] < 0.40 or ps["spe"] < 0.40
                for ps in r["per_seed"]
            )
            lines.append(
                f"| {STRATEGY_LABELS[s]} | {feat_set} "
                f"| {r['sen_mean']:.4f} | {r['spe_mean']:.4f} "
                f"| {'YES' if passes else 'NO'} "
                f"| {'YES' if any_fail else 'no'} |\n"
            )
    lines.append("\n")

    # Section 4: Recommendation
    lines.append("## 4. Recommended Strategy\n\n")
    if rec.get("strategy"):
        lines.append(f"**Recommended strategy:** `{rec['strategy']}`\n\n")
        lines.append(f"**Feature set:** `{rec['feature_set']}`\n\n")
        lines.append(f"**Rationale:** {rec['rationale']}\n\n")
        lines.append("**Selection rules applied:**\n")
        lines.append("1. Filter strategies where mean SEN >= 0.40 AND mean SPE >= 0.40\n")
        lines.append("2. Rank by BA mean (descending)\n")
        lines.append("3. Tolerance: if top_BA - candidate_BA <= 0.01, prefer simpler strategy\n")
        lines.append("   Simplicity order: none > class_weight > smote > smote_cw\n")
    else:
        lines.append(f"**No valid strategy found.** {rec.get('rationale', '')}\n\n")
        lines.append("All strategies failed the SEN>=0.40 AND SPE>=0.40 constraint.\n")

    with open("imbalance_summary.md", "w", newline="\n", encoding="utf-8") as f:
        f.writelines(lines)
    print("Saved imbalance_summary.md")


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(datasets: dict) -> dict:
    """Run the full imbalance strategy sweep.

    Returns the strategies results dict (keyed by strategy -> feat_set -> metrics).
    """
    X_full, y_raw, groups = datasets["combined"]

    # Label-encode y (MDD=0, nonMDD=1 alphabetically)
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    mdd_idx = list(le.classes_).index("MDD")

    # Feature subsets
    v3_idx = _v3_indices()
    feat_subsets = {
        "v3_subset": X_full[:, v3_idx],
        "full":      X_full,
    }

    # Label counts
    unique, counts = np.unique(y_raw, return_counts=True)
    label_counts = dict(zip(unique.tolist(), counts.tolist()))

    # Load checkpoint
    ckpt = _load_results()
    if "strategies" not in ckpt:
        ckpt["strategies"] = {s: {} for s in STRATEGIES}

    total = len(STRATEGIES) * len(FEATURE_SETS) * len(SEEDS)
    done = 0

    for strategy in STRATEGIES:
        if strategy not in ckpt["strategies"]:
            ckpt["strategies"][strategy] = {}

        for feat_set, X_sub in feat_subsets.items():
            # Check if already fully computed (all seeds done)
            existing = ckpt["strategies"][strategy].get(feat_set, {})
            if existing.get("_seeds_done") == len(SEEDS):
                done += len(SEEDS)
                print(f"  [skip] {strategy}/{feat_set} already complete")
                continue

            seed_results = existing.get("per_seed", [])
            seeds_done = len(seed_results)

            for seed in SEEDS[seeds_done:]:
                done += 1
                print(f"  [{done}/{total}] strategy={strategy}, feat_set={feat_set}, "
                      f"seed={seed}, dim={X_sub.shape[1]}")
                m = _eval_oof(strategy, X_sub, y_enc, groups, rs=seed)
                seed_results.append(m)

                # Checkpoint after each seed
                ckpt["strategies"][strategy][feat_set] = {
                    "per_seed": seed_results,
                    "_seeds_done": len(seed_results),
                }
                _save_results(ckpt)

            # Aggregate across seeds
            ba_vals  = [r["ba"]  for r in seed_results]
            auc_vals = [r["auc"] for r in seed_results]
            sen_vals = [r["sen"] for r in seed_results]
            spe_vals = [r["spe"] for r in seed_results]

            ckpt["strategies"][strategy][feat_set] = {
                "ba_mean":  round(float(np.mean(ba_vals)),  4),
                "ba_std":   round(float(np.std(ba_vals)),   4),
                "auc_mean": round(float(np.mean(auc_vals)), 4),
                "auc_std":  round(float(np.std(auc_vals)),  4),
                "sen_mean": round(float(np.mean(sen_vals)), 4),
                "sen_std":  round(float(np.std(sen_vals)),  4),
                "spe_mean": round(float(np.mean(spe_vals)), 4),
                "spe_std":  round(float(np.std(spe_vals)),  4),
                "per_seed": seed_results,
                "_seeds_done": len(seed_results),
            }
            _save_results(ckpt)

    return ckpt["strategies"], label_counts, X_full.shape[0]


# ---------------------------------------------------------------------------
# Report-only mode (regenerate chart + summary from existing JSON)
# ---------------------------------------------------------------------------

def report_only() -> None:
    if not RESULTS_PATH.exists():
        print(f"ERROR: {RESULTS_PATH} not found. Run sweep first.")
        return
    output = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    generate_chart(output["strategies"])
    generate_summary(output)
    rec = output.get("recommendation", {})
    print(f"\nRecommendation: {rec.get('strategy', 'NONE')} "
          f"(feat_set={rec.get('feature_set', 'N/A')})")
    print(f"Rationale: {rec.get('rationale', '')}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Class imbalance strategy sweep for SVM on TDBrain EEG data."
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip sweep; regenerate chart and summary from existing imbalance_results.json",
    )
    args = parser.parse_args()

    if args.report_only:
        report_only()
        return

    print("Loading cached features...")
    datasets = load_all()
    X, y, g = datasets["combined"]
    unique, counts = np.unique(y, return_counts=True)
    print(f"  combined: X={X.shape}, labels={dict(zip(unique.tolist(), counts.tolist()))}")

    print(f"\nRunning sweep: {len(STRATEGIES)} strategies x {len(FEATURE_SETS)} feat sets "
          f"x {len(SEEDS)} seeds = {len(STRATEGIES)*len(FEATURE_SETS)*len(SEEDS)} runs")

    strategies_results, label_counts, n_subjects = run_sweep(datasets)

    # Build recommendation
    recommendation = _select_best(strategies_results)

    # Assemble full output
    output = {
        "task": "class_imbalance_sweep",
        "timestamp": datetime.datetime.now().isoformat(),
        "n_subjects": n_subjects,
        "label_counts": label_counts,
        "seeds": SEEDS,
        "strategies": strategies_results,
        "recommendation": recommendation,
    }
    _save_results(output)
    print(f"\nSaved {RESULTS_PATH}")

    generate_chart(strategies_results)
    generate_summary(output)

    print(f"\n{'='*60}")
    print(f"Recommendation: {recommendation['strategy']} "
          f"(feat_set={recommendation['feature_set']})")
    print(f"Rationale: {recommendation['rationale']}")


if __name__ == "__main__":
    main()
