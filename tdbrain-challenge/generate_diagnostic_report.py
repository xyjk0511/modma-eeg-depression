"""Generate diagnostic report synthesizing Phase 1-4 results into v3.0 roadmap."""

import json
import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def load_phase_results():
    """Load all phase result JSON files."""
    files = {
        "ablation_coarse": "coarse_ablation.json",
        "ablation_fine": "fine_ablation.json",
        "imbalance": "imbalance_results.json",
        "permutation": "permutation_results.json",
        "error_analysis": "error_analysis.json",
    }
    results = {}
    for key, fname in files.items():
        with open(BASE_DIR / fname, "r", encoding="utf-8") as f:
            results[key] = json.load(f)
    return results


def extract_phase1_findings(coarse, fine):
    """Extract key findings from Phase 1 ablation."""
    findings = {}

    # Baseline (combined, SVM)
    baseline = coarse["results"]["baseline_combined_svm"]
    findings["baseline"] = {
        "ba": baseline["ba"],
        "auc": baseline["auc"],
        "n_features": baseline["n_features"],
    }

    # Best single category (combined, SVM)
    single_cats = {
        k: v for k, v in coarse["results"].items()
        if v["strategy"] == "single_category"
        and v["condition"] == "combined"
        and v["model"] == "svm"
    }
    best_single = max(single_cats.values(), key=lambda x: x["ba"])
    findings["best_single_category"] = {
        "category": best_single["category"],
        "ba": best_single["ba"],
        "auc": best_single["auc"],
        "n_features": best_single["n_features"],
    }

    # Entropy removal benefit (LOO combined SVM)
    loo_entropy = coarse["results"]["leave_one_out_entropy_combined_svm"]
    findings["entropy_removal"] = {
        "loo_ba": loo_entropy["ba"],
        "baseline_ba": baseline["ba"],
        "gain": round(loo_entropy["ba"] - baseline["ba"], 4),
    }

    # Best sub-band (fine ablation, SVM)
    fine_svm = {
        k: v for k, v in fine["results"].items()
        if v["model"] == "svm" and v["strategy"] == "fine_single"
    }
    best_subband = max(fine_svm.values(), key=lambda x: x["ba"])
    findings["best_subband"] = {
        "name": best_subband["sub_group"],
        "ba": best_subband["ba"],
        "auc": best_subband["auc"],
        "n_features": best_subband["n_features"],
    }

    # All single-category results for table
    findings["single_category_table"] = sorted(
        [
            {
                "category": v["category"],
                "ba": v["ba"],
                "auc": v["auc"],
                "n_features": v["n_features"],
            }
            for v in single_cats.values()
        ],
        key=lambda x: x["ba"],
        reverse=True,
    )

    return findings


def extract_phase2_findings(imbalance):
    """Extract key findings from Phase 2 imbalance sweep."""
    strategies = imbalance["strategies"]
    findings = {"strategy_table": []}

    for strat_name in ["none", "class_weight", "smote", "smote_cw"]:
        strat = strategies[strat_name]
        for fs_name in ["v3_subset", "full"]:
            fs = strat[fs_name]
            findings["strategy_table"].append({
                "strategy": strat_name,
                "feature_set": fs_name,
                "ba": fs["ba_mean"],
                "auc": fs["auc_mean"],
                "sen": fs["sen_mean"],
                "spe": fs["spe_mean"],
            })

    # Winner
    cw_v3 = strategies["class_weight"]["v3_subset"]
    findings["winner"] = {
        "strategy": "class_weight",
        "feature_set": "v3_subset",
        "ba": cw_v3["ba_mean"],
        "auc": cw_v3["auc_mean"],
        "sen": cw_v3["sen_mean"],
        "spe": cw_v3["spe_mean"],
    }

    # None strategy failure
    none_v3 = strategies["none"]["v3_subset"]
    findings["none_failure"] = {
        "sen": none_v3["sen_mean"],
        "spe": none_v3["spe_mean"],
    }

    return findings


def extract_phase3_findings(perm):
    """Extract key findings from Phase 3 permutation test."""
    findings = {}
    for fs_name in ["v3_subset", "full"]:
        r = perm["results"][fs_name]
        findings[fs_name] = {
            "observed_ba": r["observed_ba"],
            "p_value": r["p_value"],
            "mean_null": r["mean_permuted_ba"],
            "std_null": r["std_permuted_ba"],
            "cohens_d": r["effect_size_cohens_d"],
            "ci_95": r["ci_95"],
            "significant": r["significant"],
        }
    findings["n_permutations"] = perm["n_permutations"]
    findings["bonferroni_threshold"] = perm["significance_threshold"]
    return findings


def extract_phase4_findings(error):
    """Extract key findings from Phase 4 error analysis."""
    findings = {}

    findings["threshold"] = error["optimal_threshold"]
    findings["confusion"] = error["confusion_matrix"]
    findings["n_subjects"] = error["n_subjects"]

    # Covariate tests
    cov = error["covariate_tests"]
    findings["age"] = {
        "correct_vs_wrong_p": cov["age"]["correct_vs_wrong"]["p_value"],
        "mean_correct": cov["age"]["correct_vs_wrong"]["mean_correct"],
        "mean_wrong": cov["age"]["correct_vs_wrong"]["mean_wrong"],
        "fp_vs_tn_p": cov["age"]["FP_vs_TN"]["p_value"],
        "mean_fp": cov["age"]["FP_vs_TN"]["mean_FP"],
        "mean_tn": cov["age"]["FP_vs_TN"]["mean_TN"],
    }
    findings["gender_p"] = cov["gender"]["correct_vs_wrong"]["p_value"]
    findings["education_p"] = cov["education"]["correct_vs_wrong"]["p_value"]
    findings["bonferroni_threshold"] = cov["bonferroni_threshold"]

    # FP by indication
    fp_rates = cov["indication_fp_rate"]
    overall_fp = error["summary"]["overall_fp_rate"]
    findings["overall_fp_rate"] = overall_fp
    findings["high_fp_subgroups"] = sorted(
        [
            {"indication": k, "n": v["n"], "fp": v["fp"], "fp_rate": v["fp_rate"]}
            for k, v in fp_rates.items()
            if v["fp_rate"] > overall_fp
        ],
        key=lambda x: x["fp_rate"],
        reverse=True,
    )
    findings["all_fp_rates"] = sorted(
        [{"indication": k, **v} for k, v in fp_rates.items()],
        key=lambda x: x["fp_rate"],
        reverse=True,
    )

    # Feature distance
    fd = error["feature_distances"]
    findings["feature_distance_p"] = fd["p_value"]

    return findings


def build_performance_timeline(p1, p2, p3):
    """Build performance evolution table data."""
    return [
        {
            "phase": "Baseline",
            "config": f"All features ({p1['baseline']['n_features']}-dim), no SelectKBest",
            "ba": p1["baseline"]["ba"],
            "auc": p1["baseline"]["auc"],
            "notes": "Starting point",
        },
        {
            "phase": "Phase 1",
            "config": f"{p1['best_single_category']['category']} only "
                      f"({p1['best_single_category']['n_features']}-dim)",
            "ba": p1["best_single_category"]["ba"],
            "auc": p1["best_single_category"]["auc"],
            "notes": "Best single category",
        },
        {
            "phase": "Phase 2",
            "config": f"v3_subset + class_weight=balanced "
                      f"({p2['winner']['ba']:.4f} BA)",
            "ba": p2["winner"]["ba"],
            "auc": p2["winner"]["auc"],
            "notes": "Best imbalance strategy",
        },
        {
            "phase": "Phase 3",
            "config": "v3_subset (statistically validated)",
            "ba": p3["v3_subset"]["observed_ba"],
            "auc": None,
            "notes": f"p={p3['v3_subset']['p_value']:.3f}, "
                     f"Cohen's d={p3['v3_subset']['cohens_d']:.2f}",
        },
    ]


def identify_opportunities(p1, p2, p3, p4):
    """Identify optimization opportunities from all phases."""
    opps = []

    # 1. Delta band focus (Phase 1)
    opps.append({
        "source": "Phase 1",
        "category": "Feature Engineering",
        "opportunity": "Focus on delta band features",
        "evidence": f"abs_bp_delta: BA={p1['best_subband']['ba']:.3f} "
                    f"(best sub-band, {p1['best_subband']['n_features']}-dim)",
        "expected_gain": "0.005-0.015 BA",
        "effort": "Low",
        "risk": "Low",
        "impl_steps": [
            "Extract additional delta-band features (coherence, asymmetry)",
            "Run ablation to confirm incremental value",
            "Integrate into v3_subset if BA gain > 0.01",
        ],
    })

    # 2. Age as covariate (Phase 4)
    opps.append({
        "source": "Phase 4",
        "category": "Covariate Modeling",
        "opportunity": "Add age as model covariate",
        "evidence": f"FP vs TN age gap: mean {p4['age']['mean_fp']:.1f} vs "
                    f"{p4['age']['mean_tn']:.1f} years (p<1e-49)",
        "expected_gain": "0.02-0.05 BA",
        "effort": "Medium",
        "risk": "Low",
        "impl_steps": [
            "Add age as linear feature alongside EEG features",
            "Alternatively: age-stratified models (young/middle/old)",
            "Validate with nested CV to avoid overfitting",
        ],
    })

    # 3. Exclude psychiatric nonMDD (Phase 4)
    opps.append({
        "source": "Phase 4",
        "category": "Problem Reframing",
        "opportunity": "MDD-vs-healthy only (exclude psychiatric nonMDD)",
        "evidence": f"7 nonMDD subgroups with FP rate > {p4['overall_fp_rate']:.2f}; "
                    f"INSOMNIA FP={p4['high_fp_subgroups'][0]['fp_rate']:.2f}",
        "expected_gain": "0.03-0.05 BA",
        "effort": "Low",
        "risk": "Medium",
        "impl_steps": [
            "Filter dataset to MDD + HEALTHY subjects only",
            "Re-run SVM pipeline with class_weight=balanced",
            "Compare BA to full-dataset result to quantify confound impact",
        ],
    })

    # 4. Threshold optimization per subgroup (Phase 4)
    opps.append({
        "source": "Phase 4",
        "category": "Advanced Modeling",
        "opportunity": "Subgroup-specific decision thresholds",
        "evidence": f"Youden's J optimal threshold={p4['threshold']:.4f} "
                    f"(vs default 0.5); age-dependent FP rates",
        "expected_gain": "0.01-0.03 BA",
        "effort": "Medium",
        "risk": "Low",
        "impl_steps": [
            "Compute age-bin-specific optimal thresholds via Youden's J",
            "Implement threshold lookup at prediction time",
            "Validate with LOSO CV to avoid overfitting thresholds",
        ],
    })

    # 5. Ensemble methods (cross-phase)
    opps.append({
        "source": "Cross-phase",
        "category": "Advanced Modeling",
        "opportunity": "Ensemble methods (SVM + XGB voting)",
        "evidence": f"SVM BA={p2['winner']['ba']:.4f} (high SEN), "
                    f"XGB shows complementary SPE patterns in Phase 1",
        "expected_gain": "0.01-0.03 BA",
        "effort": "Medium",
        "risk": "Low",
        "impl_steps": [
            "Train SVM and XGB with class_weight=balanced on v3_subset",
            "Combine via soft voting (average probabilities)",
            "Validate with 5-fold CV, compare to SVM-only baseline",
        ],
    })

    # 6. Gender-specific models (Phase 4)
    opps.append({
        "source": "Phase 4",
        "category": "Covariate Modeling",
        "opportunity": "Gender-specific feature selection or models",
        "evidence": f"Gender bias in misclassification (p={p4['gender_p']:.4f})",
        "expected_gain": "0.005-0.02 BA",
        "effort": "Medium",
        "risk": "Medium",
        "impl_steps": [
            "Split dataset by gender, train separate SVM models",
            "Compare per-gender BA to pooled model",
            "If improvement > 0.01 BA, adopt gender-stratified approach",
        ],
    })

    # 7. Multi-class classification (Phase 4)
    opps.append({
        "source": "Phase 4",
        "category": "Problem Reframing",
        "opportunity": "Multi-class classification (MDD vs specific conditions)",
        "evidence": f"{len(p4['high_fp_subgroups'])} nonMDD subgroups with "
                    f"above-average FP rates",
        "expected_gain": "0.05-0.10 BA (for MDD-vs-healthy)",
        "effort": "High",
        "risk": "High",
        "impl_steps": [
            "Define class hierarchy: MDD / psychiatric-nonMDD / healthy",
            "Train one-vs-rest SVM ensemble",
            "Evaluate per-class metrics and overall BA",
        ],
    })

    # 8. Cross-dataset validation (Phase 3)
    opps.append({
        "source": "Phase 3",
        "category": "Data Augmentation",
        "opportunity": "Cross-dataset validation with external EEG-MDD data",
        "evidence": f"Current validation on single TDBrain dataset "
                    f"({p3['v3_subset']['observed_ba']:.4f} BA, n=1138)",
        "expected_gain": "0.00-0.02 BA (generalizability, not raw gain)",
        "effort": "High",
        "risk": "Medium",
        "impl_steps": [
            "Identify compatible EEG-MDD datasets (e.g., MODMA, MPI-Leipzig)",
            "Harmonize feature extraction pipelines",
            "Train on TDBrain, test on external; report transfer BA",
        ],
    })

    return opps


def prioritize_opportunities(opportunities):
    """Rank opportunities by ROI = gain / (effort * risk)."""
    GAIN_SCORES = {
        "0.005-0.015 BA": 1,
        "0.005-0.02 BA": 1.5,
        "0.01-0.03 BA": 2,
        "0.02-0.05 BA": 3,
        "0.03-0.05 BA": 3,
        "0.05-0.10 BA (for MDD-vs-healthy)": 4,
        "0.00-0.02 BA (generalizability, not raw gain)": 1,
    }
    EFFORT_SCORES = {"Low": 1, "Medium": 2, "High": 3}
    RISK_SCORES = {"Low": 1, "Medium": 2, "High": 3}

    scored = []
    for opp in opportunities:
        gain = GAIN_SCORES.get(opp["expected_gain"], 2)
        effort = EFFORT_SCORES.get(opp["effort"], 2)
        risk = RISK_SCORES.get(opp["risk"], 1)
        roi = gain / (effort * risk)
        scored.append({**opp, "roi_score": round(roi, 2)})

    return sorted(scored, key=lambda x: x["roi_score"], reverse=True)


def fmt_p(p):
    """Format p-value for display."""
    if p < 1e-10:
        return f"{p:.2e}"
    if p < 0.001:
        return f"{p:.4f}"
    return f"{p:.3f}"


def generate_report(results):
    """Generate the full diagnostic report markdown."""
    p1 = extract_phase1_findings(results["ablation_coarse"], results["ablation_fine"])
    p2 = extract_phase2_findings(results["imbalance"])
    p3 = extract_phase3_findings(results["permutation"])
    p4 = extract_phase4_findings(results["error_analysis"])

    timeline = build_performance_timeline(p1, p2, p3)
    opps = identify_opportunities(p1, p2, p3, p4)
    ranked = prioritize_opportunities(opps)

    lines = []

    # --- Header ---
    lines.append("# TDBrain Challenge v2.0 Diagnostic Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.date.today().isoformat()}")
    lines.append("**Project:** EEG-based MDD classification")
    lines.append(f"**Dataset:** TDBrain ({results['error_analysis']['n_subjects']} subjects, "
                 f"{results['error_analysis']['label_counts']['MDD']} MDD, "
                 f"{results['error_analysis']['label_counts']['nonMDD']} nonMDD)")
    lines.append("")

    # --- Executive Summary ---
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("### Key Achievements")
    lines.append(f"- **Best configuration:** v3_subset (abs_bp + rel_bp + Hjorth, "
                 f"676-dim) + class_weight=balanced")
    lines.append(f"- **Performance:** BA={p2['winner']['ba']:.4f}, "
                 f"AUC={p2['winner']['auc']:.4f} "
                 f"(SEN={p2['winner']['sen']:.3f}, SPE={p2['winner']['spe']:.3f})")
    ba_gain = p2["winner"]["ba"] - p1["baseline"]["ba"]
    pct_gain = ba_gain / p1["baseline"]["ba"] * 100
    lines.append(f"- **Improvement over baseline:** +{ba_gain:.4f} BA "
                 f"(+{pct_gain:.1f}% relative)")
    lines.append(f"- **Statistical significance:** p={p3['v3_subset']['p_value']:.3f}, "
                 f"Cohen's d={p3['v3_subset']['cohens_d']:.2f} (large effect)")
    lines.append("")
    lines.append("### Top 3 Recommendations for v3.0")
    for i, opp in enumerate(ranked[:3], 1):
        lines.append(f"{i}. **{opp['opportunity']}** "
                     f"(expected: {opp['expected_gain']}, ROI={opp['roi_score']:.2f})")
    lines.append("")
    lines.append("### Critical Findings")
    lines.append(f"- Age is the dominant confound: FP vs TN mean age "
                 f"{p4['age']['mean_fp']:.1f} vs {p4['age']['mean_tn']:.1f} years "
                 f"(p={fmt_p(p4['age']['fp_vs_tn_p'])})")
    lines.append(f"- Entropy features add noise: removing them improves BA by "
                 f"+{p1['entropy_removal']['gain']:.4f}")
    lines.append(f"- SVM without class rebalancing is degenerate: "
                 f"SEN={p2['none_failure']['sen']:.3f} (predicts all nonMDD)")
    lines.append("")

    # --- Performance Evolution ---
    lines.append("## Performance Evolution")
    lines.append("")
    lines.append("| Phase | Configuration | BA | AUC | Notes |")
    lines.append("|-------|--------------|-----|-----|-------|")
    for t in timeline:
        auc_str = f"{t['auc']:.4f}" if t["auc"] is not None else "---"
        lines.append(f"| {t['phase']} | {t['config']} | {t['ba']:.4f} | "
                     f"{auc_str} | {t['notes']} |")
    lines.append("")

    # --- Phase 1: Feature Ablation ---
    lines.append("## Phase 1: Feature Ablation")
    lines.append("")
    lines.append("### Key Findings")
    lines.append(f"- **Best single category:** {p1['best_single_category']['category']} "
                 f"(BA={p1['best_single_category']['ba']:.4f}, "
                 f"AUC={p1['best_single_category']['auc']:.4f}, "
                 f"{p1['best_single_category']['n_features']}-dim)")
    lines.append(f"- **Best sub-band:** {p1['best_subband']['name']} "
                 f"(BA={p1['best_subband']['ba']:.3f}, "
                 f"AUC={p1['best_subband']['auc']:.4f}, "
                 f"{p1['best_subband']['n_features']}-dim)")
    lines.append(f"- **Entropy removal:** LOO BA improved from "
                 f"{p1['baseline']['ba']:.4f} to "
                 f"{p1['entropy_removal']['loo_ba']:.4f} "
                 f"(+{p1['entropy_removal']['gain']:.4f})")
    lines.append("- **Recommended subset:** abs_bp + rel_bp + Hjorth "
                 "(drop entropy) = v3_subset (676-dim)")
    lines.append("")
    lines.append("### Single-Category Comparison (combined, SVM)")
    lines.append("")
    lines.append("| Category | BA | AUC | Dims |")
    lines.append("|----------|-----|-----|------|")
    for cat in p1["single_category_table"]:
        lines.append(f"| {cat['category']} | {cat['ba']:.4f} | "
                     f"{cat['auc']:.4f} | {cat['n_features']} |")
    lines.append("")

    # --- Phase 2: Imbalance Strategies ---
    lines.append("## Phase 2: Class Imbalance Strategies")
    lines.append("")
    lines.append("### Key Findings")
    lines.append(f"- **Winner:** class_weight=balanced on v3_subset "
                 f"(BA={p2['winner']['ba']:.4f}, SEN={p2['winner']['sen']:.3f}, "
                 f"SPE={p2['winner']['spe']:.3f})")
    lines.append(f"- **None strategy failure:** SEN={p2['none_failure']['sen']:.3f} "
                 f"-- SVM without rebalancing predicts almost all nonMDD")
    lines.append("- **SMOTE == SMOTE+class_weight:** double-compensation has no "
                 "measurable effect")
    lines.append("")
    lines.append("### Strategy Comparison")
    lines.append("")
    lines.append("| Strategy | Feature Set | BA | AUC | SEN | SPE |")
    lines.append("|----------|------------|-----|-----|-----|-----|")
    for row in p2["strategy_table"]:
        lines.append(f"| {row['strategy']} | {row['feature_set']} | "
                     f"{row['ba']:.4f} | {row['auc']:.4f} | "
                     f"{row['sen']:.4f} | {row['spe']:.4f} |")
    lines.append("")

    # --- Phase 3: Statistical Validation ---
    lines.append("## Phase 3: Statistical Validation")
    lines.append("")
    lines.append("### Key Findings")
    lines.append(f"- **Both feature sets significant:** "
                 f"v3_subset p={p3['v3_subset']['p_value']:.3f}, "
                 f"full p={p3['full']['p_value']:.3f} "
                 f"(Bonferroni threshold={p3['bonferroni_threshold']})")
    lines.append(f"- **v3_subset effect size:** Cohen's d="
                 f"{p3['v3_subset']['cohens_d']:.2f} (large)")
    lines.append(f"- **full effect size:** Cohen's d="
                 f"{p3['full']['cohens_d']:.2f} (large)")
    lines.append(f"- **Null distribution:** mean={p3['v3_subset']['mean_null']:.4f}, "
                 f"std={p3['v3_subset']['std_null']:.4f}, "
                 f"95% CI=[{p3['v3_subset']['ci_95'][0]:.4f}, "
                 f"{p3['v3_subset']['ci_95'][1]:.4f}]")
    lines.append(f"- **Permutations:** {p3['n_permutations']}")
    lines.append("")
    lines.append("### Permutation Test Results")
    lines.append("")
    lines.append("| Feature Set | Observed BA | Null Mean | Cohen's d | p-value | Significant |")
    lines.append("|-------------|------------|-----------|-----------|---------|-------------|")
    for fs in ["v3_subset", "full"]:
        r = p3[fs]
        sig = "Yes" if r["significant"] else "No"
        lines.append(f"| {fs} | {r['observed_ba']:.4f} | "
                     f"{r['mean_null']:.4f} | {r['cohens_d']:.2f} | "
                     f"{r['p_value']:.3f} | {sig} |")
    lines.append("")

    # --- Phase 4: Error Analysis ---
    lines.append("## Phase 4: Error Analysis")
    lines.append("")
    lines.append("### Key Findings")
    lines.append(f"- **Optimal threshold:** Youden's J = {p4['threshold']:.4f} "
                 f"(vs default 0.5)")
    cm = p4["confusion"]
    lines.append(f"- **Confusion matrix:** TP={cm['TP']}, TN={cm['TN']}, "
                 f"FP={cm['FP']}, FN={cm['FN']}")
    lines.append(f"- **Age confound:** correct mean age {p4['age']['mean_correct']:.1f} "
                 f"vs wrong {p4['age']['mean_wrong']:.1f} years "
                 f"(p={fmt_p(p4['age']['correct_vs_wrong_p'])})")
    lines.append(f"- **Gender bias:** p={fmt_p(p4['gender_p'])}")
    lines.append(f"- **Education bias:** p={fmt_p(p4['education_p'])}")
    lines.append(f"- **Feature-space distance:** NOT significant "
                 f"(p={p4['feature_distance_p']:.3f}) -- errors driven by "
                 f"demographics, not feature outliers")
    lines.append("")

    lines.append("### FP Rate by nonMDD Indication")
    lines.append("")
    lines.append("| Indication | n | FP | FP Rate |")
    lines.append("|------------|---|----|---------| ")
    for row in p4["all_fp_rates"]:
        marker = " *" if row["fp_rate"] > p4["overall_fp_rate"] else ""
        lines.append(f"| {row['indication']}{marker} | {row['n']} | "
                     f"{row['fp']} | {row['fp_rate']:.4f} |")
    lines.append("")
    lines.append(f"*\\* Above overall FP rate ({p4['overall_fp_rate']:.4f})*")
    lines.append("")

    # --- v3.0 Optimization Roadmap ---
    lines.append("## v3.0 Optimization Roadmap")
    lines.append("")
    lines.append("Prioritized by ROI (expected gain / implementation cost):")
    lines.append("")
    for i, opp in enumerate(ranked, 1):
        lines.append(f"### {i}. {opp['opportunity']}")
        lines.append("")
        lines.append(f"- **Category:** {opp['category']}")
        lines.append(f"- **Source:** {opp['source']}")
        lines.append(f"- **Evidence:** {opp['evidence']}")
        lines.append(f"- **Expected gain:** {opp['expected_gain']}")
        lines.append(f"- **Effort:** {opp['effort']}")
        lines.append(f"- **Risk:** {opp['risk']}")
        lines.append(f"- **ROI score:** {opp['roi_score']:.2f}")
        lines.append("")
        lines.append("**Implementation approach:**")
        for step_i, step in enumerate(opp["impl_steps"], 1):
            lines.append(f"{step_i}. {step}")
        lines.append("")

    # --- Lessons Learned ---
    lines.append("## Lessons Learned")
    lines.append("")
    lines.append("### Negative Results")
    lines.append(f"- **Entropy adds noise:** Single-category entropy BA="
                 f"{p1['single_category_table'][-1]['ba']:.4f} "
                 f"(worst category); removing entropy via LOO improves BA by "
                 f"+{p1['entropy_removal']['gain']:.4f}")
    lines.append(f"- **SMOTE underperforms class_weight:** SMOTE BA="
                 f"{p2['strategy_table'][4]['ba']:.4f} vs class_weight BA="
                 f"{p2['winner']['ba']:.4f} on v3_subset")
    lines.append(f"- **No-resampling SVM is degenerate:** SEN="
                 f"{p2['none_failure']['sen']:.3f}, predicts almost all nonMDD")
    lines.append(f"- **SMOTE + class_weight = SMOTE alone:** "
                 f"double-compensation has zero measurable effect")
    lines.append(f"- **Feature-space distance not predictive:** "
                 f"p={p4['feature_distance_p']:.3f}, misclassification is "
                 f"demographic, not feature-driven")
    lines.append("")

    # --- Limitations and Risks ---
    lines.append("## Limitations and Risks")
    lines.append("")
    lines.append("### Dataset Limitations")
    lines.append(f"- Single dataset (TDBrain, n={p4['n_subjects']}): "
                 f"generalizability unknown")
    lines.append("- Class imbalance (305 MDD vs 833 nonMDD, ratio 1:2.7)")
    lines.append("- Heterogeneous nonMDD group includes psychiatric conditions "
                 "that may share EEG features with MDD")
    lines.append("")
    lines.append("### Methodological Limitations")
    lines.append("- 3-fold CV used for primary evaluation "
                 "(5-fold for revalidation only)")
    lines.append("- SVM with fixed hyperparameters (C=1, RBF kernel) -- "
                 "no hyperparameter tuning performed")
    lines.append("- Platt scaling for probability calibration may be suboptimal "
                 "for imbalanced data")
    lines.append("")
    lines.append("### Risks for v3.0")
    lines.append("- Expected gains are per-optimization, NOT cumulative "
                 "(interaction effects likely)")
    lines.append("- Age covariate modeling risks overfitting on small subgroups")
    lines.append("- Problem reframing (MDD-vs-healthy) changes the clinical "
                 "question and may not be acceptable")
    lines.append("")

    # --- Conclusion ---
    lines.append("## Conclusion")
    lines.append("")
    lines.append(
        f"The v2.0 diagnostic pipeline achieves BA={p2['winner']['ba']:.4f} "
        f"(AUC={p2['winner']['auc']:.4f}) using an SVM classifier with "
        f"class_weight=balanced on the v3_subset feature set (abs_bp + rel_bp + "
        f"Hjorth, 676 dimensions). This represents a +{ba_gain:.4f} BA improvement "
        f"over the all-features baseline ({p1['baseline']['ba']:.4f}), and is "
        f"statistically significant at p={p3['v3_subset']['p_value']:.3f} with a "
        f"large effect size (Cohen's d={p3['v3_subset']['cohens_d']:.2f})."
    )
    lines.append("")
    lines.append(
        "The most actionable finding for v3.0 is the strong age confound: older "
        "nonMDD subjects are systematically misclassified as MDD, suggesting that "
        "age-related EEG changes overlap with MDD signatures. Adding age as a "
        "covariate or stratifying by age group is the highest-ROI optimization. "
        "Delta-band feature focus and MDD-vs-healthy problem reframing are "
        "complementary low-effort improvements."
    )
    lines.append("")
    lines.append(
        "Key negative results -- entropy features adding noise, SMOTE "
        "underperforming class_weight, and feature-space distance being "
        "non-predictive of errors -- provide clear guardrails for v3.0 "
        "development. The recommended approach is sequential implementation "
        "of top-3 ROI opportunities with validation after each step, rather "
        "than simultaneous changes."
    )
    lines.append("")

    # --- Appendix ---
    lines.append("## Appendix: Methodology")
    lines.append("")
    lines.append("### Phase 1: Feature Ablation")
    lines.append("- Coarse ablation: single-category and leave-one-out strategies "
                 "across 5 feature categories (abs_bp, rel_bp, Hjorth, entropy, "
                 "TBR_FAA) x 3 conditions (EO, EC, combined) x 2 models (SVM, XGB)")
    lines.append("- Fine-grained ablation: per-frequency-band analysis of top 3 "
                 "categories (delta, theta, alpha, beta, gamma)")
    lines.append("- 5-fold revalidation of top 5 configurations")
    lines.append("")
    lines.append("### Phase 2: Class Imbalance Strategies")
    lines.append("- 4 strategies: none, class_weight=balanced, SMOTE, "
                 "SMOTE+class_weight")
    lines.append("- 2 feature sets: v3_subset (676-dim), full (992-dim)")
    lines.append("- 3 random seeds for robustness, 5-fold CV per seed")
    lines.append("- Selection: BA ranking with SEN/SPE >= 0.40 constraint")
    lines.append("")
    lines.append("### Phase 3: Statistical Validation")
    lines.append(f"- {p3['n_permutations']}-iteration permutation test with "
                 f"group-level label shuffling")
    lines.append(f"- Bonferroni correction for {results['permutation']['bonferroni_comparisons']} "
                 f"comparisons (threshold p<{p3['bonferroni_threshold']})")
    lines.append("- Phipson-Smyth corrected p-values")
    lines.append("- Cohen's d effect size relative to null distribution")
    lines.append("")
    lines.append("### Phase 4: Error Analysis")
    lines.append("- Out-of-fold (OOF) probability collection for all 1138 subjects")
    lines.append(f"- Youden's J optimal threshold ({p4['threshold']:.4f}) for "
                 f"TP/TN/FP/FN categorization")
    lines.append("- Mann-Whitney U tests for continuous covariates (age, education)")
    lines.append("- Chi-squared tests for categorical covariates (gender)")
    n_tests = results["error_analysis"]["covariate_tests"]["n_tests"]
    lines.append(f"- Bonferroni correction across {n_tests} "
                 f"tests (threshold p<{p4['bonferroni_threshold']:.4f})")
    lines.append("- FP rate analysis by nonMDD indication subgroup (n >= 10 filter)")
    lines.append("")

    return "\n".join(lines)


def main():
    """Load all phase results and generate diagnostic report."""
    results = load_phase_results()
    report = generate_report(results)
    out_path = BASE_DIR / "diagnostic_report.md"
    out_path.write_text(report, encoding="utf-8", newline="\n")
    print(f"Generated {out_path} ({len(report.splitlines())} lines)")


if __name__ == "__main__":
    main()
