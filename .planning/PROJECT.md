# MODMA EEG 抑郁症分类

## Current Milestone: v3.0 ERP Deep Analysis

**Goal:** 在 hcue P300 显著结果（BA=0.670, p=0.021）基础上，深化 ERP 分析：探索更多成分（N200/LPP）、优化电极选择、分析时间窗口贡献、解释特征重要性

**Target features:**
- 更多 ERP 成分：N200（100–250ms）、LPP（500–800ms）均值幅度特征
- 电极选择优化：从 128 通道中识别贡献最大的电极子集（如顶叶 Pz/P3/P4）
- 时间窗口分析：对 P300 潜伏期/幅度做组间 t-test，找最具判别力的时间窗口
- 特征重要性解释：SVM 权重或 permutation importance 解释哪些特征最关键

基于 MODMA 数据集的 EEG 脑电特征抑郁症（MDD）vs 健康对照（HC）二分类项目。已有可运行的管线（modma_mdd_real_experiment.py），但当前分类效果在 chance level（BA≈0.54），需要通过数据质量改进来提升可分性。

## Core Value

通过修复 QC 参数保留足够多的受试者，使分类器有足够样本量来学习 MDD vs HC 的区分模式。

## Requirements

### Validated

- ✓ 128通道 EEG 数据加载和预处理管线 — existing
- ✓ PSD/FAA/连接性/Riemannian 特征提取（v4, 144维）— existing
- ✓ SVM/RF/GBM 分类器 + GridSearchCV — existing
- ✓ Permutation test 统计检验 — existing
- ✓ QC 报告生成 — existing
- ✓ 42 个单元测试，覆盖率 >80% — existing

### Active

- [ ] 提取 N200（100–250ms）和 LPP（500–800ms）成分均值幅度特征
- [ ] 从 128 通道中识别贡献最大的电极子集（顶叶/枕叶区域）
- [ ] 对 P300 时间窗口做组间 t-test，找最具判别力的时间段
- [ ] 用 SVM 权重或 permutation importance 解释特征重要性
- [ ] 验证新特征组合 BA ≥ 0.670，p < 0.05

### Out of Scope

- 3通道数据集 — 实验证明特征空间太有限，BA≈0.50
- 深度学习方法 — 样本量不足以支撑
- 实时分类系统 — 当前是离线研究

## Context

### 数据集

- MODMA 128通道静息态 EEG（BIDS格式）
- 53 受试者（24 MDD + 29 HC），128通道，250Hz，~5min
- 3通道静息态数据集也可用（55人），但分类效果差

### 诊断发现（2026-02-22）

1. **EDF 物理缩放异常**：EDF header 中 physical range 有大 DC 偏移（如 [-5460, -3475]µV），MNE 读取后数据在 mV 量级
2. **高通滤波有效**：0.5Hz highpass 能消除 DC 偏移，滤波后数据回到合理范围
3. **QC 参数不合理**：`max_bad_channels=3` 对 128 通道太严格，导致 80% 受试者被丢弃
4. **Window 0 边缘效应**：第一个窗口的坏通道数远高于后续窗口（滤波器 transient）
5. **参数调优预测**：跳过 W0 + max_bad=15 可保留 40 人（16 MDD + 24 HC）

### 当前性能

| 版本 | BA | p-value | 受试者数 |
|------|-----|---------|---------|
| baseline_v2 | 0.54 | 0.29 | 11 |
| v4_quick_fixed | 0.54 | N/A | 11 |

## Constraints

- **硬件**: 32GB RAM，需控制 joblib n_jobs≤4 避免 OOM
- **数据量**: 最多 53 受试者，样本量有限
- **统计**: 需要 permutation test 验证结果非偶然

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| 聚焦 128 通道数据集 | 3通道实验证明特征空间不足 | ✓ Good |
| 数据质量优先于模型改进 | 诊断显示瓶颈在 QC 丢弃率，不是特征或模型 | ✓ Good |
| max_bad_channels 按通道比例设置 | 128通道用固定值3不合理，应按 ~12% 比例 | ✓ Good |
| hcue 单条件优于多条件融合 | 融合引入噪声（BA 0.670→0.521），单条件信号更纯净 | ✓ Good |
| P300 均值幅度（128-dim）作为基线特征 | v2.0 验证 BA=0.670, p=0.021，统计显著 | ✓ Good |

---
*Last updated: 2026-02-23 after v3.0 milestone start*
