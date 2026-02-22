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
- [ ] **ARCH-02**: Permutation test 中特征只提取一次，不重复 1000 次

### 验证

- [ ] **VAL-01**: 分类 BA > 0.60
- [ ] **VAL-02**: Permutation test p < 0.05
- [x] **VAL-03**: QC 报告显示 MDD/HC 两组保留率差异 < 20%

## v2 Requirements

### 预处理

- **PRE-03**: 集成 autoreject 进行数据驱动的 epoch 修复
- **PRE-04**: 集成 mne-icalabel 自动 ICA 伪迹去除
- **PRE-05**: 自适应振幅阈值（median + 5*MAD per subject）

### 特征工程

- **FEAT-02**: 样本熵 SampEn per region
- **FEAT-03**: 非周期特征 specparam（1/f 指数）
- **FEAT-04**: Lempel-Ziv 复杂度
- **FEAT-05**: Gamma 频段功率

### 架构

- **ARCH-03**: 拆分 962 行单文件为 7 模块

## Out of Scope

| Feature | Reason |
|---------|--------|
| 3通道数据集 | 实验证明特征空间不足，BA≈0.50 |
| 深度学习 | 样本量不足（~40人） |
| 实时分类 | 当前是离线研究 |
| wPLI 连接性 | 延迟到 v2 特征验证后 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| QC-01 | Phase 1 | Done (01-01) |
| QC-02 | Phase 1 | Done (01-01) |
| QC-03 | Phase 1 | Done (01-01) |
| QC-04 | Phase 1 | Done (01-02) |
| PRE-01 | Phase 1 | Done (01-01) |
| PRE-02 | Phase 2 | Pending |
| FEAT-01 | Phase 3 | Done (03-01) |
| ARCH-01 | Phase 2 | Pending |
| ARCH-02 | Phase 3 | Pending |
| VAL-01 | Phase 4 | Pending |
| VAL-02 | Phase 4 | Pending |
| VAL-03 | Phase 1 | Done (01-02) |

**Coverage:**
- v1 requirements: 12 total
- Mapped to phases: 12
- Unmapped: 0 ✓

---
*Requirements defined: 2026-02-22*
*Last updated: 2026-02-22 after initial definition*
