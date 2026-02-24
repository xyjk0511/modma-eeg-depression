---
phase: 16-preprocessing-deferred
verified: 2026-02-24T01:49:27Z
status: passed
score: 9/9 must-haves verified
---

# Phase 16: Preprocessing Deferred — Verification Report

**Phase Goal:** 实现之前延期的 pyprep 坏通道检测和 .npz 缓存
**Verified:** 2026-02-24T01:49:27Z
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | pyprep NoisyChannels replaces amplitude-only bad channel detection in load_windows() | VERIFIED | Line 266: `nc = NoisyChannels(raw, do_detrend=False, random_state=42)` + `nc.find_all_bads(ransac=True)` + `bad_chs = nc.get_bads()` |
| 2 | Subjects with >20% bad channels excluded (adaptive relaxation to 25%/30% if retention <35) | VERIFIED | `_run_with_adaptive_threshold()` loops `[0.20, 0.25, 0.30]`, breaks at `n_retained >= 35` (lines 779, 803) |
| 3 | A/B comparison report shows old vs pyprep BA, AUC, subject counts, bad channel stats | VERIFIED | `run_ab_comparison()` at line 808; saves to `args.output_dir/pyprep_ab_comparison.json` (line 847) |
| 4 | If pyprep BA < threshold, regression gate warns and recommends fallback | VERIFIED | Lines 835-842: PASSED/CONDITIONAL/FAILED branches with explicit warning |
| 5 | ERP load_erp_features() and load_erp_timeseries() cache results as .npz | VERIFIED | `np.savez` at lines 169, 248; `np.load` at lines 101, 184 in run_modma_erp.py |
| 6 | Second ERP run loads from cache (no .raw re-read) | VERIFIED | Cache hit path returns early before any raw file processing (lines 99-103, 182-186) |
| 7 | Cache key includes filter params hash; param change invalidates cache | VERIFIED | `_erp_cache_path()` SHA-256 includes FMIN, FMAX, TMIN, TMAX, P300_WIN, N200_WIN + file mtimes (lines 62-67) |
| 8 | Resting-state _CACHE_VERSION bumped to 3 after pyprep integration | VERIFIED | Line 80: `_CACHE_VERSION = "3"` |
| 9 | --no-cache flag skips ERP cache | VERIFIED | Line 1051: `if "--no-cache" in sys.argv: set_erp_cache(False)` |

**Score:** 9/9 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `scripts/modma/modma_mdd_real_experiment.py` | pyprep NoisyChannels in load_windows(), A/B runner | VERIFIED | Contains `NoisyChannels`, `find_all_bads`, `run_ab_comparison`, `_run_with_adaptive_threshold`; 168-line diff in commit f86ca6a |
| `scripts/modma/run_modma_erp.py` | ERP .npz caching for load_erp_features and load_erp_timeseries | VERIFIED | Contains `_erp_cache_path`, `_ERP_CACHE_VERSION = "1"`, `set_erp_cache`, `np.savez`/`np.load` in both functions; 63-line diff in commit 8ded045 |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `load_windows()` | `pyprep.NoisyChannels` | `nc.find_all_bads(ransac=True)` | WIRED | Lines 265-268: `nc = NoisyChannels(...)`, `nc.find_all_bads(ransac=True, channel_wise=False)`, `bad_chs = nc.get_bads()` |
| `load_erp_features()` | `data/modma/cache/*.npz` | `np.savez / np.load` | WIRED | Cache miss: savez at line 169; cache hit: np.load at line 101; makedirs at line 167 |
| `load_erp_timeseries()` | `data/modma/cache/*.npz` | `np.savez / np.load` | WIRED | Cache miss: savez at line 248; cache hit: np.load at line 184; makedirs at line 246 |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| PRE-02 | 16-01-PLAN.md | 集成 pyprep 进行 PREP 标准坏通道检测（替代纯振幅阈值） | SATISFIED | NoisyChannels with all 4 detection methods in load_windows(); amplitude path preserved as fallback |
| ARCH-01 | 16-02-PLAN.md | EDF 加载后缓存为 .npz，后续迭代 <1s | SATISFIED | Both ERP load functions cache to .npz; early return on cache hit before any raw file I/O |

### Anti-Patterns Found

None. No TODO/FIXME/placeholder comments, empty returns, or stub implementations found in either modified file.

### Human Verification Required

#### 1. A/B comparison runtime output

**Test:** Run `python scripts/modma/modma_mdd_real_experiment.py --bids-root data/modma --no-cache --ab-compare`
**Expected:** Prints structured comparison table; saves `pyprep_ab_comparison.json` with BA, AUC, n_subjects for both methods
**Why human:** Requires actual MODMA data files; can't verify runtime output programmatically

#### 2. ERP cache second-run speed

**Test:** Run ERP pipeline twice; second run should complete in <1s
**Expected:** Cache hit log message; no raw file processing on second run
**Why human:** Timing verification requires actual execution with data

### Gaps Summary

No gaps. All 9 must-haves verified against actual code. Both requirements PRE-02 and ARCH-01 satisfied with substantive, wired implementations. Commits f86ca6a, 6799d80, 8ded045 confirmed present and touching correct files.

---

_Verified: 2026-02-24T01:49:27Z_
_Verifier: Kiro (gsd-verifier)_
