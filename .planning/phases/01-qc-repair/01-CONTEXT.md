# Phase 1: QC Repair - Context

**Gathered:** 2026-02-22
**Status:** Ready for planning

<domain>
## Phase Boundary

修复 QC 参数和预处理流程，使 pipeline 保留 35+ 受试者且 MDD/HC 两组保留率差异 < 20%。不引入新的坏通道检测算法（pyprep 属于 Phase 2），不新增特征。

</domain>

<decisions>
## Implementation Decisions

### 插值时机
- Raw 级插值：在 load_windows 中切窗之前，对整段 Raw 数据检测坏通道并插值
- 不在窗口级做插值（避免为每个窗口创建 MNE Info 对象）

### 坏通道检测方法
- 沿用振幅阈值检测（np.any(abs > threshold)），Phase 2 再换 pyprep
- max_bad_channels 按通道总数的 12% 动态计算（128 通道 → 15）

### 丢弃标准（混合策略）
- 窗口级检查：如果坏道比例 > 25%（~32 通道），直接丢弃该窗口（插值质量太差）
- 否则做插值后复检：插值后仍有通道超标则丢弃该窗口
- 这是在 Raw 级插值之后的二次窗口级质量检查

### 预处理流程顺序
- 调整后顺序：加载 EDF → 设置 montage → crop → 高通滤波 → 检测坏道 → 插值 → average reference → resample → 切窗
- 关键变化：插值在 average reference 之前，避免坏通道污染参考信号

### 高通滤波
- CLI 参数化：--highpass-freq 默认值从 0.5 改为 1.0 Hz
- 保留 CLI 参数可调，方便对比实验

### Window 0 跳过
- 切窗循环从第二个窗口开始（s = window_size 而非 0）
- 简单直接，不需要额外标记逻辑

### Montage 设置
- 自动设置 GSN-HydroCel-128 标准 montage
- MNE interpolate_bads() 需要通道 3D 坐标才能做球面样条插值

### 插值实现
- 使用 MNE 内置 raw.interpolate_bads()（球面样条插值）
- 需先将坏通道标记到 raw.info['bads']

### QC 报告扩展
- 在现有 generate_qc_report 基础上增加：每人插值了几个通道、插值前后对比
- 确保报告包含 MDD/HC 两组保留率及差异值（VAL-03）

### Claude's Discretion
- 振幅阈值具体数值（当前 200µV，可根据数据分布调整）
- 插值后复检的具体阈值策略
- QC 报告的具体格式和字段

</decisions>

<specifics>
## Specific Ideas

- 诊断发现滤波后 std≈127µV，13.5% 采样点超 200µV — 阈值可能需要适当放宽
- 参数调优预测：跳过 W0 + max_bad=15 可保留 40 人（16 MDD + 24 HC）
- EDF 物理缩放有大 DC 偏移，高通滤波能有效消除

</specifics>

<deferred>
## Deferred Ideas

- pyprep NoisyChannels 多标准坏通道检测 — Phase 2 (PRE-02)
- 自适应振幅阈值 median + 5*MAD — v2 需求 (PRE-05)
- .npz 缓存加速 — Phase 2 (ARCH-01)

</deferred>

---

*Phase: 01-qc-repair*
*Context gathered: 2026-02-22*
