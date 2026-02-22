# Roadmap: MODMA EEG MDD Classification

## Overview

This project fixes the QC bottleneck that drops 80% of subjects (11/53 retained), then adds discriminative features and validates statistical significance. The critical path is: repair QC parameters to retain 35+ subjects -> upgrade preprocessing with pyprep and .npz caching -> add Theta/Beta ratio and optimize permutation test -> validate BA>0.60 with p<0.05.

## Phases

- [ ] **Phase 1: QC Repair** - Fix QC parameters and filtering to retain 35+ subjects with balanced MDD/HC groups
- [ ] **Phase 2: Preprocessing Upgrade** - Integrate pyprep for robust bad channel detection and .npz caching for fast iteration
- [ ] **Phase 3: Feature & Permutation Optimization** - Add Theta/Beta ratio feature and optimize permutation test to extract features once
- [ ] **Phase 4: Statistical Validation** - Achieve BA>0.60 with permutation p<0.05

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
- [ ] 01-02-PLAN.md — QC report extension + end-to-end verification (>=35 subjects, group balance)

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

### Phase 3: Feature & Permutation Optimization
**Goal**: Feature set includes Theta/Beta ratio and permutation test runs efficiently on pre-extracted features
**Depends on**: Phase 2
**Requirements**: FEAT-01, ARCH-02
**Success Criteria** (what must be TRUE):
  1. Theta/Beta ratio features are computed per region and included in the feature matrix
  2. Permutation test extracts features once and permutes labels 1000 times on the saved feature matrix
  3. Permutation test wall-clock time is reduced by >= 50% compared to current implementation
**Plans**: TBD

Plans:
- [ ] 03-01: TBD
- [ ] 03-02: TBD

### Phase 4: Statistical Validation
**Goal**: Classification demonstrates statistically significant above-chance performance
**Depends on**: Phase 3
**Requirements**: VAL-01, VAL-02
**Success Criteria** (what must be TRUE):
  1. Best classifier achieves balanced accuracy > 0.60 on cross-validated subject-level evaluation
  2. Permutation test yields p < 0.05 confirming result is not due to chance
  3. Results are saved to metrics.json with BA, p-value, CI, and classifier name
**Plans**: TBD

Plans:
- [ ] 04-01: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. QC Repair | 1/2 | In progress | - |
| 2. Preprocessing Upgrade | 0/? | Not started | - |
| 3. Feature & Permutation Optimization | 0/? | Not started | - |
| 4. Statistical Validation | 0/? | Not started | - |
