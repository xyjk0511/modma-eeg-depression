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

- [ ] **VAL-01**: 分类 BA > 0.60
- [ ] **VAL-02**: Permutation test p < 0.05
- [x] **VAL-03**: QC 报告显示 MDD/HC 两组保留率差异 < 20%

### 确认性复制 (Phase 5)

- [ ] **REP-01**: Pre-registration document committed and git-tagged before any external data touches classifier
- [ ] **REP-02**: External data adapter loads TDBRAIN BDF files, maps 10-20 channels to regions, applies locked preprocessing
- [ ] **REP-03**: Train on full MODMA (43 subjects), test on external dataset with frozen model (no re-tuning)
- [ ] **REP-04**: Complete report with BA, CI, p-value, and conclusion (positive or negative)
