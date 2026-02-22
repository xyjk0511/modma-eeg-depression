# Testing Patterns

**Analysis Date:** 2026-02-22

## Test Framework

**Runner:**
- pytest
- Config: No `pytest.ini` or `setup.cfg` found; uses pytest defaults
- `.pytest_cache/` directory present

**Assertion Library:**
- pytest built-in assertions
- NumPy assertions: `np.testing.assert_array_almost_equal()`, `np.testing.assert_allclose()`
- pytest.raises for exception testing

**Run Commands:**
```bash
pytest tests/                          # Run all tests
pytest tests/ -v                       # Verbose output
pytest tests/ --cov                    # Coverage report
```

## Test File Organization

**Location:**
- `tests/` directory (separate from source)
- Test files: `tests/test_modma_mdd_real_experiment.py`, `tests/test_run_modma_real.py`

**Naming:**
- `test_*.py` prefix (pytest convention)
- Descriptive test function names

## Test Structure

**Patterns:**
- Arrange-Act-Assert (AAA): Setup → Call → Assert
- Descriptive test names that read as specifications
- No explicit setup/teardown (pytest fixtures used when needed)

## Mocking

**Framework:** pytest's `monkeypatch` fixture

**What to Mock:**
- External file I/O: `mne.io.read_raw_edf()`, `glob.glob()`
- File system operations: `os.path.exists()`
- Data loading functions: `load_participants()`, `pd.read_csv()`

**What NOT to Mock:**
- NumPy operations (test with real arrays)
- Sklearn pipeline steps (test integration)
- Feature extraction logic (test actual computation)
- Validation functions (test real behavior)

## Test Types

**Unit Tests:**
- Individual functions in isolation
- Synthetic data with mocked dependencies
- ~50+ unit tests in `test_modma_mdd_real_experiment.py`

**Integration Tests:**
- Multi-function workflows, data pipeline
- Mock file I/O but test real feature extraction and model training
- Examples: `test_main_writes_metrics_json()`, `test_load_real_data_success_path_structure()`

**E2E Tests:**
- Not used (CLI-based tool; integration tests sufficient)

## Common Patterns

**Error Testing:**
```python
with pytest.raises(ValueError, match="min_windows_per_subject"):
    validate_post_qc_availability(groups, y, keep_mask, min_windows_per_subject=2)
```

**Regression Tests:**
- Extensive regression suite for code-review findings
- Tests verify: stratified sampling, feature importance, ROC curve accuracy, validation wiring

**Source Inspection Tests:**
```python
import inspect
src = inspect.getsource(run_nested_group_cv)
assert "GridSearchCV" in src
```

## Test Coverage Areas

**Data Loading & Validation:**
- CLI argument parsing
- TSV parsing robustness
- Quality control
- Post-QC validation
- Class balance validation

**Feature Extraction:**
- Feature shape and content
- Dimension verification (144 dims total)
- Connectivity features (40 dims)
- Riemannian features (15 dims)
- Differential entropy (20 dims)
- Hjorth features (15 dims)
- Channel order invariance

**Model Pipeline:**
- Pipeline structure
- Class weighting
- Sample weighting

**Cross-Validation & Evaluation:**
- Data leakage prevention
- Aggregation level
- Hyperparameter tuning

**Reporting & Conclusions:**
- Report structure
- Conclusion logic
- Edge case handling

---

*Testing analysis: 2026-02-22*
