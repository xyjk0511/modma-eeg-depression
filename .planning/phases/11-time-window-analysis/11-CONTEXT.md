# Phase 11: Time-Window Analysis - Context

**Gathered:** 2026-02-23
**Status:** Ready for planning

<domain>
## Phase Boundary

在 P300 窗口（250-500ms）内，通过逐时间点 t-test 描述性分析和 50ms bins 消融分类，识别最具判别力的时间段。不包含新特征提取方法或新分类器设计。

</domain>

<decisions>
## Implementation Decisions

### t-test 数据来源
- 只用 hcue 条件（与 Phase 8 保持一致）
- t-test 作用在原始 ERP 时间序列上（逐时间点）
- 只用 5-dim 顶叶通道（Pz/P3/P4/Cz/CPz）的 ERP 数据，5 通道平均后做 t-test
- 做多重比较校正（FDR 校正，标注校正后 p<0.05 的显著时间点）

### 可视化输出
- t-stat vs time 图保存为 PNG 文件到 out_phase11/ 目录
- 图上标注两类信息：FDR 校正后显著时间点 + 最大 |t-stat| 峰值时间点
- 显示顶叶 5 通道平均后的单条 t-stat 曲线
- 叠加 5 个 50ms bins 的阴影区域（灰色半透明）

### bins 分类器配置
- 每个 50ms bin 用 LR C=0.1（与 Phase 8/10 保持一致）
- 每个 bin 的特征：bin 内所有时间点的 ERP 均值幅度（128-dim 全通道）
- 每个 bin 都做 1000 次 permutation test，报告 p-value
- 使用 128-dim 全通道（不限于顶叶子集）

### 结果报告格式
- 消融表格同时打印到 stdout 和保存到文件
- 表格列：Bin | BA | p-value | 是否显著 | 备注（最高 BA 标注 ★）
- 结果文件保存到单独的 out_phase11/ 目录

### Claude's Discretion
- 结果文件格式（CSV 或 JSON）
- permutation 并行化实现细节
- 图的配色和样式细节

</decisions>

<specifics>
## Specific Ideas

- t-stat 图和 bins 消融是两个独立分析，但共用同一个 out_phase11/ 输出目录
- bins 分类与 Phase 10 的电极选择分析互补：Phase 10 看空间（哪些通道），Phase 11 看时间（哪个时间段）

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 11-time-window-analysis*
*Context gathered: 2026-02-23*
