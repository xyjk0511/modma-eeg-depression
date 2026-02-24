# Phase 15: Grand-Average ERP Plot - Context

**Gathered:** 2026-02-24
**Status:** Ready for planning

<domain>
## Phase Boundary

绘制 grand-average ERP 波形图，在中线三通道（Fz, Cz, Pz）上分 MDD/HC 两组展示，验证 250–500ms P300 窗口内的正偏转。输出保存为 out_phase15/grand_avg_erp.png（及 PDF）。

</domain>

<decisions>
## Implementation Decisions

### 电极选择与布局
- 中线三通道：Fz, Cz, Pz（EGI 等效电极）
- 三行子图纵向排列，顺序从上到下：Fz → Cz → Pz
- 共享 x 轴（时间）和 y 轴（振幅）范围，便于跨通道对比

### 分组可视化风格
- MDD 红色实线，HC 蓝色虚线
- 每条均值线周围显示 ±1 SEM 半透明阴影区域
- 标准学术风格：线宽 1.5pt，SEM 阴影 alpha=0.2

### P300 窗口标注
- 250–500ms 区间用浅灰色半透明矩形覆盖
- 阴影区域内显示 "P300" 文字标签
- 0ms（刺激 onset）处画垂直虚线
- Y 轴正值朝上（标准笛卡尔坐标，P300 显示为向上偏转）

### 附加元素
- 不加头皮地形图（topomap）
- 仅画 grand-average 均值线 + SEM，不叠加个体波形
- 总标题放顶部（整图级）
- 图例只在最上方子图（Fz）右上角放一次
- 同时输出 PNG（300 DPI）和 PDF 矢量图

### Claude's Discretion
- 具体图片尺寸（figsize）
- 字体选择和大小
- 灰色阴影矩形的具体 alpha 值
- 坐标轴标签措辞

</decisions>

<specifics>
## Specific Ideas

- 学术论文风格，简洁清晰
- 黑白打印也能区分两组（实线 vs 虚线）

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 15-grand-average-erp-plot*
*Context gathered: 2026-02-24*
