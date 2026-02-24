# Phase 7 Context: P300 Feature Enrichment + N200

## Phase Goal

扩展 load_erp_features()，从 128-dim 均值幅度扩展到 P300 峰值幅度+潜伏期+AUC + N200 均值（640 dims），hcue 单条件 BA ≥ 0.670。

## Current Baseline

- 文件: run_modma_erp.py
- 特征: hcue P300 mean amplitude per channel (128 dims)
- 结果: BA=0.670, N=52 subjects (24 MDD + 28 HC)
- Pipeline: StandardScaler → PCA(20) → LogisticRegression(C=1.0, class_weight="balanced")
- CV: LeaveOneOut (subject-level)

## Locked Decisions

- run_loso() Pipeline 不修改
- 不新增依赖库
- 使用 np.trapezoid（不用 np.trapz）
- 不用 mne.Evoked.get_peak()（返回单个全局峰值）
- 每次 LOSO 前打印特征矩阵 shape

## Requirements Addressed

- ERP-01: P300 峰值幅度 per channel（250–500ms, max）
- ERP-02: P300 峰值潜伏期 per channel（argmax 对应时间点）
- ERP-03: P300 曲线下面积 per channel（np.trapezoid）
- ERP-04: grand-average ERP 验证（确认窗口有效）
- ERP-05: N200 均值幅度 per channel（100–250ms）
- ERP-06: P300 + N200 同一次文件加载中提取

## Feature Extraction Pattern

```python
N200_WIN = (0.10, 0.25)
P300_WIN = (0.25, 0.50)  # existing

p300_mask = (times >= P300_WIN[0]) & (times <= P300_WIN[1])
n200_mask = (times >= N200_WIN[0]) & (times <= N200_WIN[1])

p300_mean = avg_erp[:, p300_mask].mean(axis=1)          # (128,)
p300_peak = avg_erp[:, p300_mask].max(axis=1)           # (128,)
p300_lat  = times[p300_mask][avg_erp[:, p300_mask].argmax(axis=1)]  # (128,)
p300_auc  = np.trapezoid(avg_erp[:, p300_mask], times[p300_mask], axis=1)  # (128,)
n200_mean = avg_erp[:, n200_mask].mean(axis=1)          # (128,)

feat = np.concatenate([p300_mean, p300_peak, p300_lat, p300_auc, n200_mean])  # (640,)
```

## Files to Modify

- run_modma_erp.py: 添加 N200_WIN，扩展 load_erp_features()，打印 shape

## Success Criteria

1. load_erp_features() 返回 (n_subjects, 640) 特征矩阵
2. hcue LOSO BA ≥ 0.670
3. grand-average ERP 确认 P300/N200 窗口有效
