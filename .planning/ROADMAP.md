# Roadmap: MODMA EEG MDD Classification

## Overview

This project fixes the QC bottleneck that drops 80% of subjects (11/53 retained), then adds discriminative features and validates statistical significance. The critical path is: repair QC parameters to retain 35+ subjects -> upgrade preprocessing with pyprep and .npz caching -> add Theta/Beta ratio and optimize permutation test -> validate BA>0.60 with p<0.05.

## Phases

- [x] **Phase 1: QC Repair** - Fix QC parameters and filtering to retain 35+ subjects with balanced MDD/HC groups
- [x] **Phase 2: Preprocessing Upgrade** - .npz caching for fast iteration (pyprep deferred — current interpolation retains 43 subjects)
- [x] **Phase 3: Feature Reduction & Pipeline Optimization** - Reduce features from 139→29 dims, simplify pipeline, accelerate permutation test
- [x] **Phase 4: Statistical Validation** - BA=0.613 (>0.60) but p=0.135 — negative conclusion
- [x] **Phase 5: Confirmatory Replication** - Pre-register locked pipeline, train on MODMA, test on TDBRAIN → negative (BA=0.500)
- [ ] **Phase 6: Literature-Informed Feature Expansion** - Expand from 5→15 dims based on literature (beta power, PLV connectivity, temporal/central regions), validate internally then cross-dataset

## Phase Details

### Phase 1: QC Repair
**Goal**: Pipeline retains 35+ subjects with balanced MDD/HC group retention
**Depends on**: Nothing (first phase)
**Requirements**: QC-01, QC-02, QC-03, QC-04, PRE-01, VAL-03
**Success Criteria** (what must be TRUE):
  1. Running the pipeline with updated QC parameters retains >= 35 subjects
  2. Bad channels are interpolated (spherical spline) instead of causing window rejection
  3. Window 0 is skipped for every subject (no filter transient artifacts)
  4. Highpass filter is set to 1.0 Hz (not 0.5 Hz)
  5. QC report shows MDD and HC group retention rates differ by < 20%
**Plans**: 2 plans

Plans:
- [x] 01-01-PLAN.md — Core pipeline repair (montage, interpolation, W0 skip, highpass 1.0, dynamic max_bad)
- [x] 01-02-PLAN.md — QC report extension + end-to-end verification (>=35 subjects, group balance)

### Phase 2: Preprocessing Upgrade
**Goal**: Bad channel detection uses multi-criteria PREP standard and EDF loading is cached for sub-second iteration
**Depends on**: Phase 1
**Requirements**: PRE-02, ARCH-01
**Success Criteria** (what must be TRUE):
  1. pyprep NoisyChannels replaces amplitude-only bad channel detection
  2. Second pipeline run on same data loads from .npz cache in < 1 second (no EDF re-read)
  3. Retained subject count remains >= 35 after pyprep integration
**Plans**: TBD

Plans:
- [ ] 02-01: TBD
- [ ] 02-02: TBD

### Phase 3: Feature Reduction & Pipeline Optimization
**Goal**: Reduce feature dimensionality from 139→29, simplify pipeline, achieve BA >= 0.5
**Depends on**: Phase 2
**Requirements**: FEAT-01, ARCH-02
**Success Criteria** (what must be TRUE):
  1. Feature set reduced to ~29 dims: PSD-rel(20) + Alpha-asym(2) + Theta/Beta-ratio(5) + Riemannian-top2(2)
  2. Pipeline simplified: fixed QC 200μV, SVM+Logistic only, no inner QC search
  3. Permutation test extracts features once and permutes labels 1000 times
  4. Permutation wall-clock reduced >= 50% vs baseline
  5. Cross-validated BA >= 0.5 (above chance)
**Plans**: 2 plans

Plans:
- [x] 03-01-PLAN.md -- Rewrite extract_features to 29 dims + create run_simplified_cv
- [x] 03-02-PLAN.md -- Rewrite permutation loop + main entry for pre-extracted features

### Phase 4: Statistical Validation
**Goal**: Classification demonstrates statistically significant above-chance performance
**Depends on**: Phase 3
**Requirements**: VAL-01, VAL-02
**Success Criteria** (what must be TRUE):
  1. Best classifier achieves balanced accuracy > 0.60 on cross-validated subject-level evaluation
  2. Permutation test yields p < 0.05 confirming result is not due to chance
  3. Results are saved to metrics.json with BA, p-value, CI, and classifier name
**Plans**: 2 plans

Plans:
- [x] 04-01: Feature variant diagnostic + full 1000-perm validation → negative conclusion

### Phase 5: Confirmatory Replication with Pre-Registered Pipeline
**Goal**: Pre-register locked pipeline (5-dim parietal + Logistic C=0.1), train on MODMA, test on external dataset (TDBRAIN), report result
**Depends on**: Phase 4
**Requirements**: REP-01, REP-02, REP-03, REP-04
**Success Criteria** (what must be TRUE):
  1. Pre-registration document is git-tagged before any external data touches the classifier
  2. Data adapter loads external BDF files with region-level channel mapping and locked preprocessing
  3. Model trained on full MODMA (43 subjects), tested on external dataset with no re-tuning
  4. Complete report with BA, CI, p-value, and conclusion (positive or negative)
**Plans**: 2 plans

Plans:
- [x] 05-01-PLAN.md — Pre-registration document + external data adapter + git tag
- [x] 05-02-PLAN.md — Replication runner + TDBRAIN test → negative (BA=0.500, p=1.0)

### Phase 6: Literature-Informed Feature Expansion
**Goal**: Expand feature set from 5→15 dims based on literature evidence, validate internally on MODMA then cross-dataset on TDBRAIN
**Depends on**: Phase 5
**Requirements**: FEAT-02, VAL-01, REP-01
**Success Criteria** (what must be TRUE):
  1. extract_features() returns 15 features (5 existing + 4 beta + 3 connectivity + 3 temporal/central)
  2. MODMA internal CV shows 15-dim BA vs 5-dim BA comparison
  3. TDBRAIN external test with 15-dim features produces BA, CI, p-value report
  4. Feature distribution analysis comparing MODMA vs TDBRAIN generated
**Plans**: 2 plans

Plans:
- [ ] 06-01-PLAN.md — 15-dim feature expansion + MODMA internal 5-dim vs 15-dim validation
- [ ] 06-02-PLAN.md — Cross-dataset replication on TDBRAIN with 15-dim features

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. QC Repair | 2/2 | Complete | 2026-02-22 |
| 2. Preprocessing Upgrade | 1/1 | Complete (pyprep deferred) | 2026-02-22 |
| 3. Feature & Permutation Optimization | 2/2 | Complete | 2026-02-22 |
| 4. Statistical Validation | 1/1 | Complete (negative) | 2026-02-22 |
| 5. Confirmatory Replication | 2/2 | Complete (negative: BA=0.500) | 2026-02-22 |
| 6. Feature Expansion | 0/2 | Planned | — |
