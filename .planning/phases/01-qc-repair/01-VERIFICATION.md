---
phase: 01-qc-repair
verified: 2026-02-22T03:28:40Z
status: passed
score: 8/8 must-haves verified
re_verification: false
human_verification:
  - test: "Re-run pipeline from scratch and confirm 43 subjects retained"
    expected: "qc_report.csv shows 43 kept rows (18 MDD + 25 HC)"
    why_human: "Results directory exists from prior run; reproducibility needs fresh run to confirm"
---

# Phase 1: QC Repair Verification Report

**Phase Goal:** Pipeline retains 35+ subjects with balanced MDD/HC group retention
**Verified:** 2026-02-22T03:28:40Z
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Highpass filter defaults to 1.0 Hz (CLI) | VERIFIED | parse_args default=1.0 (line 101); confirmed via python assert |
| 2 | Bad channels interpolated via MNE spherical spline before average reference | VERIFIED | interpolate_bads at line 230, set_eeg_reference at line 235 |
| 3 | Window 0 is skipped for every subject | VERIFIED | range(window_size, ...) at line 246 |
| 4 | Post-interpolation windows with bad channels are discarded | VERIFIED | max_bad_win = int(n_channels * 0.12) at line 245, per-window check in loop |
| 5 | max_bad_channels dynamically computed as 12% of channel count | VERIFIED | dynamic_max_bad = int(n_channels * 0.12) at line 884, used in all downstream calls |
| 6 | QC report includes per-subject interpolation count | VERIFIED | generate_qc_report accepts interp_info, writes n_interpolated per row (lines 133-141) |
| 7 | QC report shows MDD/HC group retention rates and diff | VERIFIED | mdd_rate, hc_rate, balance_diff computed (lines 167-172); saved to metrics.json |
| 8 | Pipeline retains >= 35 subjects | VERIFIED | Empirical: 43 subjects retained (18 MDD + 25 HC) from results_phase01_verify/qc_report.csv |

**Score:** 8/8 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| modma_mdd_real_experiment.py | Updated load_windows with montage, interpolation, W0 skip | VERIFIED | Contains set_montage, interpolate_bads, range(window_size, |
| modma_mdd_real_experiment.py | Dynamic max_bad_channels in run_main_with_output_dir | VERIFIED | int(n_channels * 0.12) at line 884 |
| modma_mdd_real_experiment.py | Extended generate_qc_report with interpolation and group retention | VERIFIED | Contains group_retention, n_interpolated, returns (df, retention_stats) |
| results_phase01_verify/qc_report.csv | Pipeline output with kept/dropped subjects | VERIFIED | 43 rows with drop_reason=="kept" |
| results_phase01_verify/metrics.json | Group retention stats | VERIFIED | group_retention: {mdd_rate: 0.947, hc_rate: 1.0, diff: 0.053} |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| load_windows() | raw.interpolate_bads() | set_montage -> detect bad -> mark bads -> interpolate -> reref | WIRED | Lines 212, 230, 235 in correct order |
| run_main_with_output_dir() | build_quality_mask() | dynamic max_bad_channels | WIRED | Line 884 computes dynamic_max_bad; line 890 passes to build_quality_mask |
| run_main_with_output_dir() | generate_qc_report() | interp_info dict from load_windows | WIRED | Line 869 unpacks interp_info; line 891 passes interp_info=interp_info |
| generate_qc_report() | metrics.json | retention_stats dict | WIRED | Line 960: report["group_retention"] = retention_stats |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| QC-01 | 01-01 | max_bad_channels ~12% of channel count | SATISFIED | int(n_channels * 0.12) at lines 245, 884 |
| QC-02 | 01-01 | Spherical spline interpolation instead of window discard | SATISFIED | raw.interpolate_bads(reset_bads=True) at line 230 |
| QC-03 | 01-01 | Skip Window 0 per subject | SATISFIED | range(window_size, data.shape[1] - window_size + 1, window_size) at line 246 |
| QC-04 | 01-02 | >= 35 subjects retained after QC | SATISFIED | Empirical: 43 subjects retained |
| PRE-01 | 01-01 | Highpass raised to 1.0 Hz | SATISFIED | CLI default=1.0 (line 101), wired via args.highpass_freq |
| VAL-03 | 01-02 | MDD/HC retention diff < 20% | SATISFIED | Empirical: diff=5.3% (MDD 94.7%, HC 100%) |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| modma_mdd_real_experiment.py | 176 | highpass_freq=0.5 in load_windows signature | Warning | Stale fallback default; never reached — CLI passes 1.0 via args |
| modma_mdd_real_experiment.py | 855 | highpass_freq=0.5 in run_main_with_output_dir signature | Warning | Same — stale fallback; actual call at line 1011 passes args.highpass_freq=1.0 |

Neither is a blocker: the execution path always flows CLI -> run_main -> load_windows with the correct 1.0 value.

### Human Verification Required

#### 1. Pipeline Reproducibility

**Test:** Delete results_phase01_verify/ and re-run the pipeline from scratch
**Expected:** qc_report.csv shows 43 kept subjects; metrics.json shows group_retention diff ~5%
**Why human:** Existing results directory was produced during plan execution; cannot confirm the current code state produces identical output without re-running

### Gaps Summary

No gaps. All 8 must-haves verified against the actual codebase. Commits 21b56ad, 56a5570, 2c0461f, b68d2d9 all exist and correspond to the documented changes. Empirical results (43/53 subjects retained, MDD/HC diff 5.3%) satisfy both QC-04 (>=35) and VAL-03 (<20% diff).

---
_Verified: 2026-02-22T03:28:40Z_
_Verifier: Kiro (gsd-verifier)_
