# Requirements: MODMA EEG MDD Classification

**Defined:** 2026-02-22
**Core Value:** 通过修复 QC 参数保留足够多的受试者，使分类器能学习 MDD vs HC 的区分模式；v3.0 在 hcue P300 显著基线上深化电极选择与科学解释

## v1 Requirements

### QC 修复

- [x] **QC-01**: max_bad_channels 从固定值 3 调整为通道总数的 ~12%（128通道→~15）
- [x] **QC-02**: 对超过阈值的坏通道执行球面样条插值，而非丢弃整个窗口
- [x] **QC-03**: 跳过每个受试者的 Window 0（滤波器边缘效应）
- [x] **QC-04**: QC 后保留受试者数 ≥ 35（当前 11）

### 预处理升级

- [x] **PRE-01**: 高通滤波从 0.5Hz 提升到 1.0Hz
- [ ] **PRE-02**: 集成 pyprep 进行 PREP 标准坏通道检测（替代纯振幅阈值）

### 特征工程

- [x] **FEAT-01**: 新增 Theta/Beta 比率特征（per channel 或 per region）

### 架构优化

- [ ] **ARCH-01**: EDF 加载后缓存为 .npz，后续迭代 <1s
- [x] **ARCH-02**: Permutation test 中特征只提取一次，不重复 1000 次

### 验证

- [x] **VAL-01**: 分类 BA > 0.60
- [ ] **VAL-02**: Permutation test p < 0.05
- [x] **VAL-03**: QC 报告显示 MDD/HC 两组保留率差异 < 20%

### 确认性复制 (Phase 5)

- [x] **REP-01**: Pre-registration document committed and git-tagged before any external data touches classifier
- [x] **REP-02**: External data adapter loads TDBRAIN BDF files, maps 10-20 channels to regions, applies locked preprocessing
- [x] **REP-03**: Train on full MODMA (43 subjects), test on external dataset with frozen model (no re-tuning)
- [x] **REP-04**: Complete report with BA, CI, p-value, and conclusion (positive or negative)

---

## v2.0 Requirements — ERP Task-State Classification

**Baseline:** BA=0.670, hcue P300 mean amplitude (128 dims), LOSO, PCA(20), LogisticRegression
**Target:** BA>0.70, p<0.05 (permutation test, 1000 iterations)

### P300 特征丰富化

- [ ] **ERP-01**: 提取每通道 P300 峰值幅度（250–500ms 窗口内 max）
- [ ] **ERP-02**: 提取每通道 P300 峰值潜伏期（argmax 对应时间点）
- [ ] **ERP-03**: 提取每通道 P300 曲线下面积（np.trapezoid）
- [ ] **ERP-04**: 绘制 grand-average ERP 验证 P300 窗口（250–500ms 内有正偏转）

### N200 成分

- [ ] **ERP-05**: 提取每通道 N200 均值幅度（100–250ms 窗口）
- [ ] **ERP-06**: 与 P300 特征在同一次文件加载中提取（不重复 epoch）

### 统计显著性验证

- [ ] **ERP-07**: 实现 permutation_test_loso()，subject-level 标签置换（assert len(y)==n_subjects）
- [ ] **ERP-08**: 1000 次置换，joblib.Parallel，固定 seed=42
- [ ] **ERP-09**: 先在 hcue 单条件（BA=0.670）上验证置换逻辑正确性

### 多条件融合

- [x] **ERP-10**: 三条件特征融合（hcue + fcue + scue），显式 sub_id 对齐
- [x] **ERP-11**: 条件对比特征 scue − hcue（情绪偏向直接编码）
- [x] **ERP-12**: 融合后 n_subjects ≥ 40；若 < 40 则作为消融实验报告

---

## v3.0 Requirements — ERP Deep Analysis

**Baseline:** BA=0.670, p=0.021 (hcue P300 mean amplitude, 128-dim, LOSO)
**Goal:** 电极降维 + 科学解释，提升结果可解释性

### 电极选择（ELEC）

- [x] **ELEC-01**: 识别 EGI HydroCel 128 montage 中 Pz/P3/P4/Cz/CPz 对应的通道索引，验证坐标匹配
- [x] **ELEC-02**: 提取顶叶 3-dim 特征（Pz/P3/P4 P300 均值幅度），LOSO BA 与 128-dim 基线对比，permutation p-value
- [x] **ELEC-03**: 提取扩展 5-dim 特征（Pz/P3/P4/Cz/CPz），LOSO BA 与 3-dim 对比，permutation p-value

### 时间窗口分析（TWIN）

- [x] **TWIN-01**: 对 avg_erp（N×128×n_times）做逐时间点 MDD vs HC t-test，输出 t-stat vs time 图（纯描述性）
- [x] **TWIN-02**: 50ms bins 消融分类（5个窗口：250-300/300-350/350-400/400-450/450-500ms），每个窗口 128-dim LOSO BA

### 特征重要性（FIMP）

- [ ] **FIMP-01**: 从 LOSO 各折提取 LR coef_，通过 PCA 反投影（pca.components_.T @ coef_）到通道空间，输出 128 通道权重排名

## v4.0 Requirements（延期）

- **COMP-01**: N200 顶叶 6-dim（P300+N200 at Pz/P3/P4）— 仅在 3-dim 顶叶 BA >= 0.670 后测试
- **COMP-02**: LPP 成分（500-800ms）— medrxiv 2020 显示 MDD 中无 LPP 减弱证据，低优先级

## Out of Scope

| Feature | Reason |
|---------|--------|
| 多条件融合（384-dim） | Phase 9 实证：BA 0.670→0.521，引入噪声 |
| 640-dim 丰富特征 | Phase 7 实证：BA 0.670→0.554，维度诅咒 |
| 深度学习 | N=52 样本量不足 |
| 跨数据集复现（TDBRAIN） | Phase 5/6 已完成，结论为负结果 |
| Permutation importance | 需修改 run_loso() 架构，LR coef_ 反投影已足够 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| VAL-02 | Phase 14 (was Phase 8) | Pending |
| ERP-01 | Phase 14 (was Phase 7) | Pending |
| ERP-02 | Phase 14 (was Phase 7) | Pending |
| ERP-03 | Phase 14 (was Phase 7) | Pending |
| ERP-04 | Phase 15 | Pending |
| ERP-05 | Phase 14 (was Phase 7) | Pending |
| ERP-06 | Phase 14 (was Phase 7) | Pending |
| ERP-07 | Phase 14 (was Phase 8) | Pending |
| ERP-08 | Phase 14 (was Phase 8) | Pending |
| ERP-09 | Phase 14 (was Phase 8) | Pending |
| ELEC-01 | Phase 10 | Complete |
| ELEC-02 | Phase 10 | Complete |
| ELEC-03 | Phase 10 | Complete |
| TWIN-01 | Phase 11 | Complete |
| TWIN-02 | Phase 11 | Complete |
| FIMP-01 | Phase 14 (was Phase 12) | Pending |
| PRE-02 | Phase 16 | Pending |
| ARCH-01 | Phase 16 | Pending |

**Coverage:**
- v1-v3 gap closure requirements: 18 total
- Mapped to phases: 18
- Unmapped: 0 ✓

---
*Requirements defined: 2026-02-22*
*Last updated: 2026-02-23 after gap closure phases 14-16 added*
