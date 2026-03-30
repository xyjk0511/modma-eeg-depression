# Adult Validation Sample 信息

## 来源

Brainclinics 官网: https://brainclinics.com/tdbrain

## 定义

**Adult Validation Sample** 是 TDBRAIN 数据集的一个独立 hold-out 样本,用于诊断预测的盲化样本外验证。

## 关键特征

1. **独立数据集**: 不是 DISCOVERY 或 REPLICATION 的子集,需要单独下载
2. **年龄限制**: 仅包含成年受试者(≥18 岁)
3. **诊断任务**: 用于 MDD vs. healthy controls 的诊断预测
4. **盲化标签**: 真实标签被盲化,需提交预测后由 Brainclinics 解盲验证
5. **数据结构**: 与原始 TDBRAIN 数据集结构完全相同

## 用途区分

| 样本 | 用途 | 标签状态 |
|------|------|----------|
| DISCOVERY | 模型训练 | 已知 |
| REPLICATION | 治疗反应预测 (treatment response) | 盲化 |
| Adult Validation | 诊断预测 (MDD vs. healthy) | 盲化 |

## 提交流程

1. **下载**: 从 Brainclinics 网站下载 Adult Validation Sample
2. **训练**: 在完整 TDBRAIN 数据集(DISCOVERY)上训练模型
3. **预测**: 在 Adult Validation Sample 上生成预测
4. **格式**: 使用官方 Excel 模板,填写离散标签(MDD / nonMDD)
5. **提交**: 发送到 brainbank@brainclinics.com
6. **验证**: Brainclinics 解盲并返回验证结果
7. **排行榜**: 结果发布在 TDBRAIN-Challenge 排行榜

## 提交限制

- 每个团队每个类别最多提交 **3 次预测**
- 提交即自动授权 Brainclinics 将结果添加到排行榜

## 评估指标

- **主要指标**: Balanced Accuracy
- **次要指标**: Sensitivity, Specificity

## 官方模板

下载地址: https://brainclinics.com/tdbrain (TDBRAIN replication template)

## 引用要求

使用 TDBRAIN 数据发表论文时,必须引用:
- van Dijk et al. (2021) 数据描述论文
- 提供链接: www.brainclinics.com/resources 或 https://www.synapse.org/TDBRAIN

## 当前项目状态

- **v3.0 模型**: exclude_high_fp + v3_subset (676d) + SVM(rbf, class_weight=balanced)
- **训练数据**: DISCOVERY 子集,exclude_high_fp 过滤后 904 subjects
- **当前预测**: `submissions/prediction_v3_svm.csv` 是在 REPLICATION 上的预测(错误目标)
- **待完成**: 下载 Adult Validation Sample,重新生成预测

## 联系方式

- **提交邮箱**: brainbank@brainclinics.com
- **Prof. Martijn Arns**: 数据集负责人之一

---
*记录时间: 2026-02-28*
*信息来源: Brainclinics 官网 + Prof. Arns 邮件回复*
