# MODMA EEG Depression Classification / MODMA 脑电抑郁分类项目

This repository contains an EEG-based depression classification pipeline built on the **MODMA** dataset, with emphasis on ERP/P300 feature engineering, subject-level validation, and statistically grounded evaluation.

本仓库实现了一个基于 **MODMA** 数据集的脑电抑郁分类流程，重点放在 ERP/P300 特征提取、严格的受试者级验证，以及统计显著性分析上。

---

## Overview / 项目概述

**Goal / 目标**
- Classify **Major Depressive Disorder (MDD)** versus **Healthy Controls (HC)** from EEG data.
- 从 EEG 数据中区分 **抑郁症患者（MDD）** 与 **健康对照（HC）**。

**Project angle / 项目特点**
- This is not just a generic classifier; it is built around EEG-domain feature design and subject-aware evaluation.
- 它不是简单套模型，而是围绕 EEG 信号特征和受试者级验证来设计的。

---

## Results / 核心结果

- **Balanced Accuracy:** **0.670**
- **Statistical significance:** **p = 0.022**
- **Domain focus:** ERP / P300 features for EEG-based depression classification

### What these results mean / 这些结果意味着什么
- The model performs above chance with statistically meaningful separation.
- 模型不仅高于随机水平，而且具有统计学意义。
- The project demonstrates a full workflow from EEG preprocessing and feature construction to evaluation and interpretation.
- 这个项目体现了从 EEG 预处理到建模再到解释的完整流程。

---

## Pipeline / 技术流程

```text
Raw EEG
  → preprocessing
  → ERP / P300 feature extraction
  → subject-level train/validation split
  → classification model
  → evaluation and interpretation
```

### Key emphasis / 关键点
- EEG-domain feature engineering rather than purely end-to-end black-box modeling
- 不是纯黑箱建模，而是重视 EEG 领域特征
- Subject-level separation to reduce leakage risk
- 用受试者级划分避免信息泄漏
- Statistical testing beyond just reporting accuracy
- 不只报 accuracy，还做统计显著性验证

---

## Repository Structure / 仓库结构

```text
README.md
TD_BRAIN_code/ / tdbrain-challenge/ / tdbrain/   # related EEG experiment code and pipelines
scripts/                                         # utility and experiment scripts
tests/                                           # test code
docs/                                            # notes and supporting documentation
```

> This repository is part of a broader EEG / psychiatric classification workflow and reflects practical experimentation rather than a minimal toy example.
>
> 这个仓库是更大 EEG / 精神疾病分类研究路径的一部分，体现的是实际实验工作流，而不是玩具示例。

---

## Why this project matters / 为什么这个项目重要

This project demonstrates several strengths relevant to ML and applied research internships:

- feature engineering for biomedical signals
- working with noisy real-world physiological data
- statistically responsible evaluation
- building domain-aware ML pipelines rather than generic notebook-only baselines

这个项目能体现你在实习申请中很有价值的能力：
- 生物电信号特征工程
- 处理高噪声真实数据
- 统计上更负责任的验证方式
- 构建领域驱动而不是纯 notebook 的 ML 流程

---

## Reproducibility / 复现说明

This repository is best interpreted as a research-oriented project repository.

本仓库更适合作为研究型项目仓库来理解。

To fully productionize it, the next improvements would be:
- clearer top-level experiment entrypoints
- more explicit dataset preparation instructions
- cleaner separation between MODMA-specific code and related EEG experiments
- more complete result artifacts and visualization summaries

如果继续整理，我下一步会优先做：
- 提供更清晰的实验入口
- 补充数据准备说明
- 将 MODMA 专属代码和其他 EEG 项目彻底拆开
- 补全结果图和可视化摘要

---

## Related interests / 相关方向

This project sits at the intersection of:
- machine learning
- neuroscience
- mental health modeling
- signal processing

这个项目位于以下方向的交叉点：
- 机器学习
- 神经科学
- 心理健康建模
- 信号处理
