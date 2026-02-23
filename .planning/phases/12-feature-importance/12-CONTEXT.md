# Phase 12: Feature Importance - Context

**Gathered:** 2026-02-23
**Status:** Ready for planning

<domain>
## Phase Boundary

将 LOSO 各折 LR coef_ 反投影到通道空间，输出可解释的 128 通道权重排名与地形图。不包含新的分类实验或特征工程。

</domain>

<decisions>
## Implementation Decisions

### 反投影来源
- 两个模型都做：hcue 全窗口（Phase 8，BA=0.670）和 250-300ms bin（Phase 11，BA=0.673）
- 每折独立 PCA（与现有 LOSO 逻辑一致，不共享 PCA 对象）
- 两套独立 LOSO 重新运行，各自提取 coef_
- 聚合方式：每折先反投影（pca.components_.T @ coef_）得到 128-dim 权重，再跨折取平均绝对值排名

### 权重排名输出
- 打印格式：格式化表格（Rank / Channel / Avg|weight| / Sign）
- 保留正负符号（正=MDD偏高，负=HC偏高）
- 同时保存完整 128 通道排名到 CSV
- 两个模型各自独立输出表格和 CSV

### Topomap 可视化
- 颜色映射：RdBu_r 发散色图（正权重红色=MDD偏高，负权重蓝色=HC偏高）
- 标注 top-5 通道名（在对应位置显示通道标签）
- 两个模型并排对比图，保存为单文件 PNG
- 通道位置：从数据集真实坐标文件读取

### 输出目录与文件
- 输出目录：out_phase12/
- CSV 命名：feature_importance_hcue.csv / feature_importance_bin250.csv
- Topomap PNG：topomap_hcue_vs_bin.png（并排单文件）
- 代码组织：独立入口函数 run_feature_importance()，在 __main__ 中调用

### Claude Discretion
- colorbar 范围（对称 vmin/vmax 或自动）
- 表格列宽与对齐细节
- 通道坐标文件的具体路径查找逻辑

</decisions>

<specifics>
## Specific Ideas

- 两个模型并排 topomap 方便直接对比 hcue 全窗口 vs 250-300ms bin 的空间分布差异

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 12-feature-importance*
*Context gathered: 2026-02-23*
