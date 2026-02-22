# Coding Conventions

**Analysis Date:** 2026-02-22

## Naming Patterns

**Files:**
- Snake case: `modma_mdd_real_experiment.py`, `run_modma_real.py`
- Test files: `test_*.py` (pytest convention)

**Functions:**
- Snake case: `load_participants()`, `build_quality_mask()`, `extract_features()`
- Private functions prefixed with underscore: `_egi_range()`, `_alpha_asymmetry()`, `_apply_qc_and_extract()`
- Descriptive names indicating purpose: `validate_post_qc_availability()`, `compute_de_features()`

**Variables:**
- Snake case: `bad_amp_uv`, `resample_sfreq`, `min_windows_per_subject`
- Constants in UPPER_CASE: `REGIONS`, `SFREQ`, `N_SAMPLES`, `EGI128_REGION_MAP`
- Numpy arrays: `X` (data), `y` (labels), `groups` (subject identifiers)

**Types:**
- Type hints in function signatures: `def load_real_modma_data(max_subjects: int | None = None, crop_duration: float = 60)`
- Return type annotations: `-> tuple[mne.EpochsArray, np.ndarray, np.ndarray]`
- Union types with `|` operator (Python 3.10+)

## Code Style

**Formatting:**
- No explicit formatter configured (no `.prettierrc`, `black`, or `ruff` config)
- Follows PEP 8 conventions implicitly
- Line length: ~100-120 characters
- Indentation: 4 spaces

**Linting:**
- No `.eslintrc` or `pylintrc` found
- Code follows standard Python conventions

## Import Organization

**Order:**
1. Standard library: `import argparse`, `import os`, `import glob`, `import logging`
2. Third-party scientific: `import numpy as np`, `import pandas as pd`, `import mne`
3. Sklearn modules: `from sklearn.pipeline import Pipeline`, `from sklearn.preprocessing import StandardScaler`
4. Local imports: `from modma_mdd_real_experiment import parse_args`

**Path Aliases:**
- No path aliases configured
- Relative imports used in tests: `import modma_mdd_real_experiment as mod`

## Error Handling

**Patterns:**
- Explicit exception raising with descriptive messages
- Type validation: `if isinstance(max_subjects, bool) or not isinstance(max_subjects, int): raise TypeError(...)`
- Range validation: `if max_subjects < 2 or max_subjects % 2 != 0: raise ValueError(...)`
- File existence checks: `if not os.path.exists(participants_file): raise FileNotFoundError(...)`
- Try-except for external library operations
- Warnings suppressed contextually: `with warnings.catch_warnings(): warnings.filterwarnings('ignore')`

## Logging

**Framework:** `logging` module (standard library)

**Patterns:**
- Logger created per module: `logger = logging.getLogger(__name__)`
- Info level for progress: `logger.info("正在加载真实 MODMA 数据集...")`
- Warning level for recoverable issues: `logger.warning(f"找不到 {safe_id} 的 EDF 文件")`
- Error level for failures: `logger.error(f"处理 {sub_id} 时发生错误: {e}")`
- Formatted strings with values: `logger.info("已加载 %d 个试验试次。", len(labels))`

## Comments

**When to Comment:**
- Non-obvious algorithm logic: `# Per-bin ImCoh = |Im(Cxy)| / sqrt(Pxx * Pyy)`
- Complex mathematical operations: `# Differential entropy per band per region: 0.5 * log(2*pi*e*var)`
- Data transformation rationale: `# Riemannian-inspired without TangentSpace fit`
- Workarounds and edge cases: `# Fallback to 10-20` (when EGI channels not found)

## Function Design

**Size:**
- Typical range: 10-50 lines
- Larger functions (100+ lines): `load_windows()`, `run_full_model_selection()`, `run_main_with_output_dir()`
- These handle complex workflows with multiple steps

**Parameters:**
- Explicit keyword arguments for configuration: `bad_amp_uv=200.0`, `max_bad_channels=3`
- Type hints on all parameters
- Default values for optional parameters: `crop_duration=60.0`, `highpass_freq=0.5`

**Return Values:**
- Tuples for multiple returns: `return np.array(all_epochs), np.array(labels), np.array(groups), no_edf_subjects, ch_names`
- Dictionaries for structured results: `return {"leakage_detected": False, "subject_level_metrics": {...}}`

## Module Design

**Exports:**
- Public functions: `parse_args()`, `load_participants()`, `extract_features()`, `run_main_with_output_dir()`
- Private functions (underscore prefix): `_egi_range()`, `_alpha_asymmetry()`, `_apply_qc_and_extract()`
- Module-level constants: `REGIONS`, `EGI128_REGION_MAP`, `STANDARD_1020_REGION_MAP`

**Barrel Files:**
- Not used (single-file modules or direct imports)
- Tests import directly: `from modma_mdd_real_experiment import parse_args`

---

*Convention analysis: 2026-02-22*
