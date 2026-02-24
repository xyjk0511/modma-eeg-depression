# Phase 16: Preprocessing Deferred Items - Context

**Gathered:** 2026-02-24
**Status:** Ready for planning

<domain>
## Phase Boundary

实现之前延期的两项预处理改进：(1) 用 pyprep NoisyChannels 替代当前纯振幅坏通道检测；(2) EDF 加载后缓存为 .npz 加速重复运行。不涉及新特征、新模型或新分类流程。

</domain>

<decisions>
## Implementation Decisions

### 坏通道检测策略
- 先用 pyprep NoisyChannels 全部方法（deviation, correlation, high-frequency noise, RANSAC）跑一遍，根据结果再决定是否裁剪方法子集
- 检测到的坏通道用球面样条插值修复（MNE 默认方法）
- 单个受试者超过 20% 通道被标记为坏时，整个受试者排除
- 只替换现有管线中的坏通道检测步骤，ICA 等其他预处理流程不变

### 缓存策略
- 缓存 EDF 加载+滤波后、ICA 之前的数据为 .npz 格式
- 缓存文件存放在 data/modma/cache/ 目录下
- 在 .npz 文件名或元数据中记录滤波参数哈希，参数变化时自动重建缓存
- 添加 --no-cache 命令行参数支持强制跳过缓存

### 受试者保留策略
- 优先保证受试者数 >= 35 人
- 自适应阈值：先用 20% 跑一遍，如果剩余人数 < 35，逐步放宽到 25%、30%
- 输出每个受试者的详细坏通道检测日志（坏通道数、使用的阈值、排除原因）

### 回归验证
- 主验收：pyprep 预处理后 LOSO CV BA >= 0.670（不低于当前基线）
- 次验收兜底：若 0.640 <= BA < 0.670，必须同时满足"受试者保留率显著提升且组间保留差更小"才可接受
- A/B 对比：旧方法 vs pyprep 并排运行，生成结构化对比报告（BA、AUC、受试者数、坏通道统计）
- 未达标时自动回退到旧方法，代码保留两套路径

### Claude's Discretion
- pyprep 具体参数调优（如 RANSAC 的 fraction_bad 等）
- 缓存文件的具体命名格式和哈希算法选择
- 对比报告的具体字段和排版

</decisions>

<specifics>
## Specific Ideas

- 自适应阈值逐步放宽：20% → 25% → 30%，每步检查是否满足 >= 35 人
- 对比报告应包含：BA、AUC、受试者保留数、每组保留数、坏通道平均数、使用的最终阈值

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 16-preprocessing-deferred*
*Context gathered: 2026-02-24*
