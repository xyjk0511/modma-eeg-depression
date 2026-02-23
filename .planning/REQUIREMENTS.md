# Requirements: MODMA EEG MDD Classification

**Defined:** 2026-02-22
**Core Value:** 通过修复 QC 参数保留足够多的受试者，使分类器能学习 MDD vs HC 的区分模式

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

- [ ] **ERP-10**: 三条件特征融合（hcue + fcue + scue），显式 sub_id 对齐
- [ ] **ERP-11**: 条件对比特征 scue − hcue（情绪偏向直接编码）
- [ ] **ERP-12**: 融合后 n_subjects ≥ 40；若 < 40 则作为消融实验报告

### 非功能性

- [ ] **ERP-NF-01**: 不新增依赖库（MNE 1.11 + NumPy 2.4 + sklearn 1.8 + joblib 1.5 已足够）
- [ ] **ERP-NF-02**: run_loso() Pipeline 不修改（StandardScaler→PCA(20)→LR）
- [ ] **ERP-NF-03**: 每次 LOSO 前打印特征矩阵 shape

### 成功标准

| 指标 | 目标 |
|------|------|
| hcue 丰富特征 BA | > 0.670 |
| Permutation p-value | < 0.05 |
| 融合特征 BA | > 0.70 |
| 融合 n_subjects | ≥ 40 |
