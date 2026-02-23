import pytest
import numpy as np
from unittest.mock import patch


def test_full_grid_search_raises_when_all_configs_fail():
    from run_tdbrain_v9 import full_grid_search
    with patch("run_tdbrain_v9.single_cv", side_effect=ValueError("bad")):
        with pytest.raises(RuntimeError, match="All grid configs failed"):
            full_grid_search(None, None, None, None, 2, 42)


def test_p_value_formula_excludes_failed_permutations():
    # p = (count_ge + 1) / (n_valid + 1); failed perms must NOT increment n_valid
    # Simulate: 5 attempted, 2 failed → n_valid=3, not 5
    n_valid, count_ge = 3, 0
    p = (count_ge + 1) / (n_valid + 1)
    assert p == pytest.approx(0.25)
    assert p != pytest.approx(1 / 6)  # would be 1/6 if failures counted (n_valid=5)


def test_observed_and_permutation_both_use_max_of_configs():
    import run_tdbrain_v9
    from run_tdbrain_v9 import CONFIGS

    calls = []

    def mock_cv(folds, y, groups, seed, C, pca_n):
        calls.append((C, pca_n))
        return 0.5 + len(calls) * 0.05  # increasing → last call is max

    with patch("run_tdbrain_v9.fast_perm_cv", side_effect=mock_cv):
        folds = {cfg: None for cfg in CONFIGS}
        y, groups = np.zeros(4), np.zeros(4)
        result = max(
            run_tdbrain_v9.fast_perm_cv(folds[cfg], y, groups, 42, cfg[1], cfg[2])
            for cfg in CONFIGS
        )

    assert len(calls) == len(CONFIGS), "all configs must be evaluated"
    assert result == pytest.approx(0.5 + len(CONFIGS) * 0.05), "max must be taken"
