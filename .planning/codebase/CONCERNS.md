# Codebase Concerns

**Analysis Date:** 2026-02-22

## Tech Debt

**Memory pressure from parallelization:**
- Issue: Permutation testing with joblib (n_jobs=4) on 32GB RAM risks OOM
- Files: `modma_mdd_real_experiment.py` lines 754-760
- Impact: Recent commit (da65a8f) reduced n_jobs as emergency fix
- Fix approach: Implement adaptive n_jobs based on available memory

**Hardcoded QC threshold in model selection:**
- Issue: `bad_amp_candidates=(200, 300, 400)` hardcoded but CLI accepts single value
- Files: `modma_mdd_real_experiment.py` lines 483-487, 834
- Impact: Confusing API—user specifies one threshold, model selection tries three
- Fix approach: Clarify intent or expose all thresholds to CLI

**Loose error handling in permutation loop:**
- Issue: `_run_one_permutation()` silently returns None on any exception (line 710)
- Files: `modma_mdd_real_experiment.py` lines 698-711
- Impact: Permutation failures invisible; p-value computed on partial sample
- Fix approach: Log exceptions with context; track failure rate; raise if >10% fail

**Region mapping auto-detection fragility:**
- Issue: `build_region_indices()` uses heuristic to auto-detect channel system
- Files: `modma_mdd_real_experiment.py` lines 74-85
- Impact: Mixed channel sets could select wrong mapping
- Fix approach: Require explicit channel system parameter; validate all channels map

## Known Bugs

**QC validation order issue:**
- Symptoms: Pipeline fails with "Subject X has only N windows after QC"
- Files: `modma_mdd_real_experiment.py` lines 836-850
- Trigger: When min_windows_per_subject > 3 and real data has borderline subjects
- Workaround: Reduce min_windows_per_subject or increase bad_amp_uv

**Feature extraction inconsistency with empty regions:**
- Issue: Empty regions padded with zeros (lines 240, 331, 358)
- Files: `modma_mdd_real_experiment.py` lines 237-244, 331-334, 357-359
- Impact: Zero-padding masks missing data; classifier learns spurious patterns
- Fix approach: Raise error if any region is empty; validate mapping before extraction

## Security Considerations

**No input validation on file paths:**
- Risk: bids-root and output-dir used directly in os.path.join() without sanitization
- Files: `modma_mdd_real_experiment.py` lines 174, 809
- Current mitigation: Python path handling prevents directory traversal
- Recommendations: Add explicit path validation; reject paths with .. or absolute paths

**Unvalidated EDF file loading:**
- Risk: mne.io.read_raw_edf() could fail or hang on malformed files; no timeout
- Files: `modma_mdd_real_experiment.py` lines 188
- Current mitigation: Try-except catches errors (line 210)
- Recommendations: Add file size checks; implement read timeout; validate EDF header

**JSON output without bounds checking:**
- Risk: NaN/Inf values in metrics could corrupt JSON or downstream analysis
- Files: `modma_mdd_real_experiment.py` lines 909-910
- Current mitigation: sklearn metrics return finite values; p-value checked for None
- Recommendations: Validate all metrics are finite before JSON serialization

## Performance Bottlenecks

**Redundant feature extraction in model selection:**
- Problem: Features extracted multiple times per fold
- Files: `modma_mdd_real_experiment.py` lines 519-525, 560-565
- Cause: `_apply_qc_and_extract()` called in inner loop; no caching
- Improvement: Cache features per QC threshold; reuse across model candidates

**Welch PSD computation repeated per feature block:**
- Problem: welch() called separately in extract_features(), compute_de_features(), compute_connectivity_features()
- Files: `modma_mdd_real_experiment.py` lines 226, 309, 382
- Cause: Each feature function independently computes PSD
- Improvement: Compute PSD once, pass to all functions; reduces computation by ~60%

**Full model selection rerun for each permutation:**
- Problem: Permutation testing reruns entire nested CV + grid search
- Files: `modma_mdd_real_experiment.py` lines 698-711, 754-760
- Cause: Rigorous but expensive; 1000 permutations × 5 folds × 3 models = 15k fits
- Improvement: Use simpler null distribution; trade rigor for speed

**Joblib parallelization overhead:**
- Problem: Spawning 4 processes per permutation has overhead
- Files: `modma_mdd_real_experiment.py` line 755
- Cause: Process creation cost dominates for fast permutations
- Improvement: Batch permutations; use threading for I/O-bound tasks

## Fragile Areas

**Nested CV with group stratification:**
- Files: `modma_mdd_real_experiment.py` lines 504-590
- Why fragile: Complex interaction between outer StratifiedGroupKFold, inner GridSearchCV, QC filtering
- Safe modification: Add explicit leakage detection; validate train/test groups disjoint
- Test coverage: test_evaluation_never_mixes_subjects_between_train_test() covers basic case

**Feature importance computation on filtered data:**
- Files: `modma_mdd_real_experiment.py` lines 894, 912-914
- Why fragile: Features extracted on lenient-QC data but model trained on stricter QC
- Safe modification: Extract features on same data used for model training
- Test coverage: No test validates feature importance matches model weights

**Conclusion determination logic:**
- Files: `modma_mdd_real_experiment.py` lines 781-805
- Why fragile: Four criteria must all pass; changing any threshold breaks reproducibility
- Safe modification: Make thresholds configurable; document rationale
- Test coverage: Tests cover positive/negative cases but not threshold sensitivity

**Region mapping with mixed channel systems:**
- Files: `modma_mdd_real_experiment.py` lines 74-85, 287-302
- Why fragile: Auto-detection heuristic could fail on partial EGI or non-standard 10-20
- Safe modification: Require explicit channel system parameter; validate all channels map
- Test coverage: test_region_mapping_invariant_to_channel_order() tests permutation invariance

## Scaling Limits

**Memory usage with large permutation counts:**
- Current capacity: ~1000 permutations on 32GB RAM with 128 channels, 125 Hz, 10-sec windows
- Limit: Breaks at ~5000 permutations or with higher sampling rates (250 Hz)
- Scaling path: Implement streaming permutation computation; use approximate null distributions

**EDF file loading bottleneck:**
- Current capacity: ~100 subjects with 128 channels at 250 Hz
- Limit: Single-threaded EDF reading; no parallel file I/O
- Scaling path: Parallelize EDF loading with thread pool; implement lazy loading

**Feature extraction dimensionality:**
- Current capacity: 144 features (v4) with 128 channels
- Limit: Feature matrix unwieldy with >256 channels or additional feature blocks
- Scaling path: Implement feature selection; use dimensionality reduction (PCA available)

## Dependencies at Risk

**MNE-Python version compatibility:**
- Risk: mne.io.read_raw_edf() API changed between versions
- Impact: Code works on MNE 1.3+ but may break on older versions
- Migration plan: Pin MNE version; add version check at startup

**LightGBM optional dependency:**
- Risk: HAS_LGBM flag creates two code paths
- Impact: Model selection may choose different models depending on LightGBM availability
- Migration plan: Make LightGBM required or remove it

**Joblib n_jobs hardcoding:**
- Risk: n_jobs=4 hardcoded; may cause OOM on smaller systems
- Impact: Performance varies unpredictably across hardware
- Migration plan: Make n_jobs configurable; add memory-aware default

## Missing Critical Features

**No data provenance tracking:**
- Problem: No record of which EDF files used, checksums, or preprocessing parameters
- Blocks: Reproducibility; audit trail; data versioning
- Solution: Log file paths and checksums; save preprocessing config

**No intermediate checkpoint saving:**
- Problem: If pipeline fails at permutation 500/1000, must restart from beginning
- Blocks: Long-running experiments; fault tolerance
- Solution: Save checkpoint after each permutation; implement resume logic

**No visualization of feature distributions:**
- Problem: Feature importance reported but not visualized
- Blocks: Exploratory analysis; debugging
- Solution: Add boxplots of top features by class

**No cross-dataset validation:**
- Problem: Pipeline only validates on single MODMA dataset
- Blocks: Clinical utility assessment
- Solution: Add external validation mode

## Test Coverage Gaps

**Untested: EDF loading with missing channels:**
- What's not tested: Behavior when EDF has fewer channels than expected
- Files: `modma_mdd_real_experiment.py` lines 160-213
- Risk: Index errors in feature extraction
- Priority: High

**Untested: Permutation with single-class data:**
- What's not tested: Edge case where permutation produces single-class subset
- Files: `modma_mdd_real_experiment.py` lines 735-760
- Risk: Silent failure or incorrect p-value
- Priority: Medium

**Untested: QC report with no subjects:**
- What's not tested: generate_qc_report() when all subjects dropped
- Files: `modma_mdd_real_experiment.py` lines 132-157
- Risk: Empty DataFrame; downstream code may fail
- Priority: Medium

**Untested: Feature extraction with zero-variance channels:**
- What's not tested: Behavior when channel has constant signal
- Files: `modma_mdd_real_experiment.py` lines 215-284
- Risk: NaN in features; classifier may fail
- Priority: High

**Untested: Riemannian covariance with singular matrices:**
- What's not tested: Behavior when region covariance is singular
- Files: `modma_mdd_real_experiment.py` lines 398-415
- Risk: Log of negative eigenvalues; NaN in features
- Priority: High

---

*Concerns audit: 2026-02-22*
