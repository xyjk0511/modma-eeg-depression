# Roadmap: MODMA EEG MDD Classification

## Overview

This project fixes the QC bottleneck that drops 80% of subjects (11/53 retained), then adds discriminative features and validates statistical significance. The critical path is: repair QC parameters to retain 35+ subjects -> upgrade preprocessing with pyprep and .npz caching -> add Theta/Beta ratio and optimize permutation test -> validate BA>0.60 with p<0.05.

## Phases

- [x] **Phase 1: QC Repair** - Fix QC parameters and filtering to retain 35+ subjects with balanced MDD/HC groups
- [x] **Phase 2: Preprocessing Upgrade** - .npz caching for fast iteration (pyprep deferred — current interpolation retains 43 subjects)
- [x] **Phase 3: Feature Reduction & Pipeline Optimization** - Reduce features from 139→29 dims, simplify pipeline, accelerate permutation test
- [x] **Phase 4: Statistical Validation** - BA=0.613 (>0.60) but p=0.135 — negative conclusion
- [x] **Phase 5: Confirmatory Replication** - Pre-register locked pipeline, train on MODMA, test on TDBRAIN → negative (BA=0.500)
- [x] **Phase 6: Literature-Informed Feature Expansion** - Expand from 5→15 dims based on literature (beta power, PLV connectivity, temporal/central regions), validate internally then cross-dataset
- [x] **Phase 7: P300 Feature Enrichment + N200** - 丰富单条件（hcue）ERP 特征，验证 BA > 0.670 基线
- [x] **Phase 8: Permutation Test Statistical Validation** - 验证 hcue 丰富特征结果统计显著（p<0.05）
- [x] **Phase 9: Multi-Condition Fusion + Contrast Features** - 融合三条件特征，BA > 0.70，p < 0.05
- [ ] **Phase 10: Electrode Selection** - 识别 EGI 顶叶通道索引，验证 3-dim 和 5-dim 顶叶子集分类器
- [ ] **Phase 11: Time-Window Analysis** - 逐时间点 t-test 描述性分析 + 50ms bins 消融分类
- [ ] **Phase 12: Feature Importance** - LR coef_ 反投影到通道空间，输出 128 通道权重排名

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
- [x] 06-01-PLAN.md — 15-dim feature expansion + MODMA internal 5-dim vs 15-dim validation
- [x] 06-02-PLAN.md — Cross-dataset replication on TDBRAIN with 15-dim features → negative (BA=0.500, p=1.0)

### Phase 7: P300 Feature Enrichment + N200
**Goal**: 丰富单条件（hcue）ERP 特征，验证 BA > 0.670 基线
**Depends on**: Phase 6 (ERP baseline BA=0.670 已建立)
**Requirements**: ERP-01, ERP-02, ERP-03, ERP-04, ERP-05, ERP-06
**Success Criteria** (what must be TRUE):
  1. load_erp_features() 返回每通道 P300 峰值幅度、潜伏期、曲线下面积 + N200 均值幅度
  2. grand-average ERP 图确认 250–500ms 内有正偏转（P300）、100–250ms 内有负偏转（N200）
  3. hcue 单条件 LOSO BA ≥ 0.670（不低于基线）
  4. 特征矩阵 shape 在 LOSO 前打印
**Plans**: TBD

Plans:
- [ ] 07-01-PLAN.md — 扩展 load_erp_features() + grand-average ERP 验证 + hcue LOSO

### Phase 8: Permutation Test Statistical Validation
**Goal**: 验证 hcue 丰富特征结果统计显著（p<0.05）
**Depends on**: Phase 7
**Requirements**: ERP-07, ERP-08, ERP-09, ERP-NF-01, ERP-NF-02
**Success Criteria** (what must be TRUE):
  1. permutation_test_loso() 在 subject-level 置换标签（assert len(y)==n_subjects）
  2. 1000 次置换，joblib.Parallel，seed=42
  3. hcue 丰富特征 p-value < 0.05
**Plans**: 1 plan

Plans:
- [x] 08-01-PLAN.md — Run hcue 1000-perm permutation test, record p-value (BA=0.670, p=0.021)

### Phase 9: Multi-Condition Fusion + Contrast Features
**Goal**: 融合三条件特征，BA > 0.70，p < 0.05
**Depends on**: Phase 8
**Requirements**: ERP-10, ERP-11, ERP-12
**Success Criteria** (what must be TRUE):
  1. 三条件融合（hcue+fcue+scue）显式 sub_id 对齐，n_subjects ≥ 40
  2. scue−hcue 对比特征作为消融实验单独测试
  3. 融合 LOSO BA > 0.70，permutation p < 0.05
  4. 若 n_common < 40，作为消融实验报告（不作为主要结论）
**Plans**: 1 plan

Plans:
- [x] 09-01-PLAN.md — 多条件融合 + 对比特征 + 最终 permutation 验证 (fusion BA=0.521 p=0.412, contrast BA=0.521 p=0.384)

### Phase 10: Electrode Selection
**Goal**: 识别 EGI HydroCel 128 顶叶通道索引，验证顶叶子集分类器不低于 128-dim 基线
**Depends on**: Phase 9
**Requirements**: ELEC-01, ELEC-02, ELEC-03
**Success Criteria** (what must be TRUE):
  1. Pz/P3/P4/Cz/CPz 的 EGI 通道索引通过坐标匹配验证，打印索引和通道名
  2. 3-dim 顶叶分类器（Pz/P3/P4）输出 LOSO BA 和 permutation p-value，与 128-dim 基线对比
  3. 5-dim 扩展分类器（Pz/P3/P4/Cz/CPz）输出 LOSO BA 和 permutation p-value，与 3-dim 对比
**Plans**: 1 plan

Plans:
- [ ] 10-01-PLAN.md — EGI channel lookup + 3-dim/5-dim parietal subset classifiers

### Phase 11: Time-Window Analysis
**Goal**: 通过逐时间点 t-test 和 50ms bins 消融分类，识别 P300 窗口内最具判别力的时间段
**Depends on**: Phase 10
**Requirements**: TWIN-01, TWIN-02
**Success Criteria** (what must be TRUE):
  1. t-stat vs time 图覆盖 250-500ms，MDD vs HC 组间差异可视化，标注峰值时间点
  2. 5 个 50ms bins（250-300/300-350/350-400/400-450/450-500ms）各自输出 128-dim LOSO BA
  3. 消融结果表格打印，最高 BA 的 bin 标注
**Plans**: TBD

Plans:
- [ ] 11-01-PLAN.md — TBD

### Phase 12: Feature Importance
**Goal**: 将 LOSO 各折 LR coef_ 反投影到通道空间，输出可解释的 128 通道权重排名
**Depends on**: Phase 11
**Requirements**: FIMP-01
**Success Criteria** (what must be TRUE):
  1. 每折 coef_（1x20 PCA 空间）通过 pca.components_.T @ coef_ 反投影为 128-dim 通道权重
  2. 跨折平均绝对权重排名输出，top-10 通道名打印
  3. 128 通道权重地形图（topomap）生成并保存
**Plans**: TBD

Plans:
- [ ] 12-01-PLAN.md — TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. QC Repair | 2/2 | Complete | 2026-02-22 |
| 2. Preprocessing Upgrade | 1/1 | Complete (pyprep deferred) | 2026-02-22 |
| 3. Feature & Permutation Optimization | 2/2 | Complete | 2026-02-22 |
| 4. Statistical Validation | 1/1 | Complete (negative) | 2026-02-22 |
| 5. Confirmatory Replication | 2/2 | Complete (negative: BA=0.500) | 2026-02-22 |
| 6. Feature Expansion | 2/2 | Complete (negative: BA=0.500) | 2026-02-22 |
| 7. P300 Feature Enrichment + N200 | 1/1 | Complete (feature enrichment regressed; reverted to 128-dim p300_mean baseline BA=0.670) | 2026-02-23 |
| 8. Permutation Test Validation | 1/1 | Complete (BA=0.670, p=0.021 — significant) | 2026-02-23 |
| 9. Multi-Condition Fusion | 1/1 | Complete | 2026-02-23 |
| 10. Electrode Selection | 0/1 | Not started | - |
| 11. Time-Window Analysis | 0/1 | Not started | - |
| 12. Feature Importance | 0/1 | Not started | - |
