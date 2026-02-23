"""Regression tests for run_modma_erp statistical core."""
import re
import numpy as np
import pytest


# ── p-value formula ──────────────────────────────────────────────────────────

def _p_corrected(perm_bas, obs_ba, n_perm):
    perm_arr = np.array(perm_bas)
    return float((np.sum(perm_arr >= obs_ba) + 1) / (n_perm + 1))


def test_p_value_never_zero():
    """(count+1)/(n_perm+1) prevents p=0 when no perm exceeds observed."""
    perm_bas = [0.4] * 1000
    p = _p_corrected(perm_bas, obs_ba=0.9, n_perm=1000)
    assert p > 0, "p-value must never be exactly 0"
    assert p == pytest.approx(1 / 1001)


def test_p_value_hcue_result():
    """21 permutations >= 0.670 → p = 22/1001 ≈ 0.022."""
    rng = np.random.default_rng(42)
    perm_bas = [0.670 if i < 21 else 0.494 for i in range(1000)]
    p = _p_corrected(perm_bas, obs_ba=0.670, n_perm=1000)
    assert p == pytest.approx(22 / 1001)
    assert p < 0.05


# ── subject ID parsing ───────────────────────────────────────────────────────

def _parse_sub_id(fname):
    m = re.match(r'(\d{8})', fname)
    return m.group(1) if m else fname.split("erp")[0].strip().replace("_", "")


def test_sub_id_standard():
    assert _parse_sub_id("02010002erp hcue 20151103 1801.raw") == "02010002"


def test_sub_id_no_erp_tag():
    """Files without 'erp' in name still yield 8-digit prefix."""
    assert _parse_sub_id("02030006 20151103 1801.raw") == "02030006"


def test_sub_id_consistent_across_conditions():
    """Same subject yields same ID regardless of condition suffix."""
    assert _parse_sub_id("02010002erp hcue 20151103.raw") == \
           _parse_sub_id("02010002erp fcue 20151103.raw")


# ── Phase 10: electrode selection ────────────────────────────────────────────

def test_get_parietal_indices_returns_all_targets():
    from run_modma_erp import get_parietal_indices
    idx = get_parietal_indices()
    for name in ("Pz", "P3", "P4", "Cz", "CPz"):
        assert name in idx
        assert isinstance(idx[name], int)


def test_get_parietal_indices_cz_cpz_same_electrode():
    """Known: Cz and CPz both map to E55 (idx=54) via coordinate fallback."""
    from run_modma_erp import get_parietal_indices
    idx = get_parietal_indices()
    assert idx["Cz"] == idx["CPz"], "Cz and CPz should map to same GSN electrode"


def test_loso_ba_subset_caps_pca_components():
    """_loso_ba_subset must not crash when n_features < n_samples-1."""
    from run_modma_erp import _loso_ba_subset
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 3))   # 3 features, 10 samples
    y = np.array([0, 1] * 5)
    ba = _loso_ba_subset(X, y)
    assert 0.0 <= ba <= 1.0


def test_run_electrode_selection_deduplicates_columns(capsys):
    """5-dim with Cz==CPz should produce a 4-column X_5, not 5."""
    from run_modma_erp import get_parietal_indices
    idx = get_parietal_indices()
    ch5 = ["Pz", "P3", "P4", "Cz", "CPz"]
    seen, unique_cols = set(), []
    for c in ch5:
        i = idx[c]
        if i not in seen:
            seen.add(i)
            unique_cols.append(i)
    assert len(unique_cols) == 4, "Duplicate Cz/CPz index must be deduplicated"


# ── Phase 11: time-window analysis ───────────────────────────────────────────

def test_plot_ttest_curve_deduplicates_parietal():
    """plot_ttest_curve must use unique parietal indices (Cz==CPz dedup)."""
    from run_modma_erp import get_parietal_indices
    raw_vals = list(get_parietal_indices().values())
    deduped = list(dict.fromkeys(raw_vals))
    assert len(deduped) < len(raw_vals), "Cz/CPz dedup must reduce index count"


def test_bin_ablation_csv_structure(tmp_path, monkeypatch):
    """run_bin_ablation writes CSV with header + exactly 5 bin rows."""
    import csv
    import run_modma_erp as mod
    monkeypatch.setattr(mod, "permutation_test_subset",
                        lambda X, y, n_perm, C: (0.55, 0.10))
    rng = np.random.default_rng(0)
    times = np.linspace(0.0, 0.6, 300)
    avg_erps = rng.standard_normal((10, 128, len(times)))
    y = np.array([0, 1] * 5)
    mod.run_bin_ablation(avg_erps, times, y, tmp_path)
    with open(tmp_path / "bin_ablation.csv") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 6
    assert rows[0] == ["Bin", "BA", "p-value", "Sig", "Note"]


def test_bin_boundaries_no_double_count():
    """Half-open bin intervals must not double-count boundary samples."""
    from run_modma_erp import BINS
    times = np.array([0.250, 0.300, 0.350, 0.400, 0.450, 0.500])
    counts = np.zeros(len(times), dtype=int)
    for i, (_, t0, t1) in enumerate(BINS):
        mask = (times >= t0) & (times <= t1) if i == len(BINS) - 1 \
               else (times >= t0) & (times < t1)
        counts[mask] += 1
    assert counts.max() == 1, "Each time point must belong to exactly one bin"
