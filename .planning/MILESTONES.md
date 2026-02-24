# Milestones

## v1.0 MODMA EEG MDD Classification (Shipped: 2026-02-24)

**Phases completed:** 16 phases, 19 plans
**Timeline:** 2026-02-20 → 2026-02-24 (4 days)
**Stats:** 196 commits, 2660 LOC Python, 222 files

**Key accomplishments:**
- QC 修复：受试者保留 11→43 人（动态12%坏通道阈值 + 球面样条插值）
- hcue P300 128-dim BA=0.670, p=0.022 — 统计显著 MDD vs HC 区分
- 5-dim 顶叶子集 BA=0.634, p=0.045 — 显著且可解释
- 时间窗消融 250-300ms BA=0.673 — P300 最强判别时段
- 特征重要性 E55 (Cz-adjacent) 权重最高，HC 方向正偏转
- pyprep PREP 标准坏通道检测 + ERP .npz 缓存

**Tech debt accepted:**
- run_modma_erp.py __main__ 仅调 plot_grand_average_erp，12 个 phase runner 孤立
- ERP-03 surviving code 用 mean 而非 np.trapezoid

---

