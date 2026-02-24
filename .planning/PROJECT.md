# MODMA EEG 抑郁症分类

## What This Is

基于 MODMA 128通道 EEG 数据集的 MDD vs HC 二分类研究项目。通过 QC 修复、ERP P300 特征提取、电极选择、时间窗消融和特征重要性分析，实现了统计显著的分类结果。

## Current State (v1.0 shipped 2026-02-24)

**Primary finding:** hcue P300 128-dim BA=0.670, p=0.022（统计显著）
**Secondary:** 5-dim parietal BA=0.634, p=0.045; 250-300ms bin BA=0.673, p=0.018（探索性）

## Core Value

在 MODMA 52人 EEG 数据集上，hcue P300 均值幅度是唯一统计显著的 MDD 分类特征。临床应用价值受限于中等性能与缺少同范式外部验证。

## Requirements

### Validated

- ✓ QC-01..04: 动态坏通道阈值 + 插值 + 受试者保留 43人 — v1.0
- ✓ PRE-01..02: 1.0Hz 高通 + pyprep PREP 标准检测 — v1.0
- ✓ FEAT-01: Theta/Beta 比率特征 — v1.0
- ✓ ARCH-01..02: .npz 缓存 + permutation 单次提取 — v1.0
- ✓ VAL-01..03: BA>0.60, p<0.05, 组间保留差<20% — v1.0
- ✓ REP-01..04: TDBRAIN 跨数据集复制（结论：负结果）— v1.0
- ✓ ERP-01..12: P300/N200 特征 + 置换检验 + 多条件融合 — v1.0
- ✓ ELEC-01..03: EGI 通道映射 + 3/5-dim 顶叶子集 — v1.0
- ✓ TWIN-01..02: t-test 曲线 + 50ms bin 消融 — v1.0
- ✓ FIMP-01: LR coef_ PCA 反投影通道权重 — v1.0

### Active

(None — next milestone TBD)

### Out of Scope

- 多条件融合（384-dim）— Phase 9 实证 BA 0.670→0.521
- 640-dim 丰富特征 — Phase 7 实证 BA 0.670→0.554
- 深度学习 — N=52 样本量不足
- 跨数据集复现 — Phase 5/6 已完成，静息态结论为负
- ERP 跨数据集验证 — 缺少同范式外部数据集

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
| P300 均值幅度（128-dim）作为基线特征 | v2.0 验证 BA=0.670, p=0.022，统计显著 | ✓ Good |

---
*Last updated: 2026-02-23 after v3.0 milestone start*
