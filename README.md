# MODMA EEG Depression Classification

基于 MODMA 128通道 ERP 数据集的 MDD vs HC 二分类研究。通过 QC 修复、P300 特征提取、电极选择、时间窗消融和特征重要性分析，实现了统计显著的分类结果。

## 核心发现

| 模型 | 维度 | BA | p-value | 显著 |
|------|------|----|---------|------|
| hcue P300 均值幅度 | 128 | 0.670 | 0.022 | Yes |
| 顶叶子集 (Pz/P3/P4/Cz/CPz) | 5 | 0.634 | 0.045 | Yes |
| 250-300ms 时间窗 | 128 | 0.673 | 0.018 | Yes |

- hcue dot-probe P300 均值幅度是唯一统计显著的 MDD 分类特征
- E55 (Cz-adjacent) 通道权重最高，HC 方向正偏转
- 特征丰富化和多条件融合均导致性能下降
- 跨数据集泛化（TDBRAIN 静息态）失败（BA=0.500）

## 数据集

[MODMA](https://modma.lzu.edu.cn/data/application/) 128通道 ERP 数据集：
- 53 受试者（24 MDD + 29 HC），128通道 EGI HydroCel，250Hz
- Dot-probe 情绪注意偏向范式（hcue/fcue/scue 条件）
- QC 后保留 52 人（动态 12% 坏通道阈值 + 球面样条插值）

## 方法

- **预处理**: 1.0Hz 高通 + 0.5-40Hz 带通 + pyprep PREP 标准坏通道检测
- **特征**: P300 均值幅度（250-500ms 窗口，128通道）
- **分类器**: PCA(20) + LogisticRegression(C=1.0, balanced)
- **验证**: Leave-One-Subject-Out CV + 1000次置换检验

## 项目结构

```
scripts/modma/
  run_modma_erp.py              # ERP pipeline（P300/N200、分类、消融）
  modma_mdd_real_experiment.py  # 静息态 pipeline（QC、频段功率）
  run_modma_real.py             # 静息态分类入口
  run_modma_reproduce.py        # 复现脚本
  modma_mdd_classification.py   # 早期分类实验
tests/                          # 单元测试
docs/PRE_REGISTRATION.md        # 跨数据集复制预注册文档
```

## 负结果记录

| 尝试 | 结果 | 结论 |
|------|------|------|
| P300+N200 丰富特征 (256-dim) | BA 0.670→0.554 | 维度诅咒 |
| 三条件融合 (384-dim) | BA 0.670→0.521 | 融合引入噪声 |
| 静息态频段功率 (640-dim) | BA=0.391 | 低于随机水平 |
| TDBRAIN 跨数据集复制 | BA=0.500 | 静息态特征不泛化 |

## 局限性

- N=52 样本量限制泛化性
- 无同范式外部验证数据集
- BA=0.670 为中等性能，临床应用价值有限

## 运行

```bash
pip install mne scikit-learn pyprep numpy scipy joblib pandas matplotlib
python scripts/modma/run_modma_erp.py
```

需要先从 [MODMA](https://modma.lzu.edu.cn/data/application/) 下载数据到 `data/` 目录。

## License

MIT
