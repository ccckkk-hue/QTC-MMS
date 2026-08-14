## 0) 论文可用摘要（先看这段）

说明：本文件是“融合方案”的随包说明；总包根目录也有同名文档，内容更完整，建议优先看根目录版本：
- `SCHEME_FUSION_two_models_v1_README.md`

external（shaoxing1，全恶性）最终重点报告（来自 `result/scheme_fusion_eval_only_v1/results.json`）：
- high-risk AUC on malignant=**0.8654**（n=203）

cexu_test 推理建议以 **v2(jsonMask)** 输出为准（原因与对比汇总见 `CEXU_TEST_eval_summary_v1.md`）。

---

## 1) 目标与结论

本方案把“三分类”拆成两个分别训练到最强的子模型，再在推理阶段做层级融合：

- **良恶性模型**输出 \(P(malignant)\)
- **恶性高危分层模型**输出 \(P(high \mid malignant)\)
- 融合得到三类概率：
  - \(P0(benign)=1-Pm\)
  - \(P1(mal\_low)=Pm\cdot(1-Ph)\)
  - \(P2(mal\_high)=Pm\cdot Ph\)

这条线的优势是：当“直训三分类”在 external（shaoxing1）高危 AUC 上不稳定时，可以把两个子任务分别做到最好，再用融合拿到更稳的 high-risk AUC。

---

## 1) 两个模型分别是什么

### 1.1 良恶性模型（benign vs malignant）

- 训练脚本：`scripts/train_malignancy_classifier.py`
- 权重（当前默认最优）：`result/malignancy_cls_v3/best.pt`
- 输入：FOV 图像（`image_path`）+（可选）mask 形态学特征
- 输出：\(P(malignant)\)

### 1.2 恶性高危分层模型（malignant-low vs malignant-high）

- 训练脚本：`scripts/train_llm_enhanced_stratification_v2_optimized.py`
- 权重（当前默认最优）：`result/llm_enhanced_stratification_v2_optimized_beimo_push_v1/best_model.pth`
- 配置/特征清单：`result/llm_enhanced_stratification_v2_optimized_beimo_push_v1/results.json`
- 输入：ROI 图像（`nodule_crop_path`）+ tabular（临床+radiomics）+（可选）LLM 文本
- 输出：\(P(high \mid malignant)\)

---

## 2) 评估脚本与一键运行

### 2.1 融合评估脚本（eval-only）

- 评估脚本：`scripts/eval_triclass_fusion_two_models_v1.py`
- 一键 bat：`scripts/run_scheme_fusion_eval_only_v1.bat`

它会在以下数据上评估并自动出图：
- internal tri-class：`--tri_root`（train/val/test，三分类评估）
- external（shaoxing1，仅恶性）：`--external_csv`（只评 high-risk AUC on malignant）

并默认启用：
- `--calibrate_on_val`：在 val 上对 \(Pm\) 与 \(Ph\) 做 Platt 校准（更稳的融合概率）

输出目录（bat 默认）：
- `result/scheme_fusion_eval_only_v1/`
  - `results.json`
  - `roc_*.png`、`cm_*.png`（英文，论文可直接用）
  - `pred_*.csv`（逐病例预测）

### 2.2 训练 + 融合评估（train+eval）

- 一键 bat：`scripts/run_scheme_fusion_train_and_eval_v1.bat`

逻辑：
- 如果子模型权重不存在 → 先训练
- 然后调用融合评估脚本出完整结果与配图

---

## 3) cexu_test 推理（无标签也能输出预测 Excel）

### 3.1 融合方案对 cexu_test 的推理脚本

- 脚本：`scripts/predict_cexu_test_fusion_v1.py`
- 一键 bat：`scripts/run_cexu_test_fusion_predict_v1.bat`

输入：
- `data/cexu_test/info.xlsx`
- `data/cexu_test/<病例文件夹>/*.png`
- 默认使用 `info.xlsx` 的 **T 列（第20列，列名“图表中的组名2”）** 对应病例文件夹名

输出：
- `result/cexu_test_fusion_pred_v1/cexu_test_predictions.xlsx`
- `result/cexu_test_fusion_pred_v1/cexu_test_predictions.csv`

表中会包含：
- 原始病例信息（来自 `info.xlsx`）
- 预测：`p0_benign/p1_mal_low/p2_mal_high`、`pred_class`、`pred_name`

### 3.2 cexu_test 本轮已跑完的结果在哪里？

本轮你已经跑完并生成了：
- `result/cexu_test_fusion_pred_v1/cexu_test_predictions.xlsx`
- `result/cexu_test_fusion_pred_v1/cexu_test_predictions.csv`

对比汇总（包含两种方案的分布统计与“风险分层子集”的探索性 AUC）在：
- `result/CEXU_TEST_eval_summary_v1.md`

---

## 4) cexu_test 有没有 label？没 label 能不能画图？

如果没有 GT label（良/恶、低/高危），则：
- **能输出预测结果（Excel/CSV）**
- **不能**算 AUC/画 ROC/混淆矩阵（这些都需要真实标签）

目前 `info.xlsx` 中存在一列 `risk stratification label`（部分病例有 `Low/High/...`），它更像“风险分层描述”，是否能用于评估取决于你是否能定义清晰的二分类映射（例如把 `High` 视为高危，或 `Intermediate–High+High` 视为高危）。

---

## 5) 一键打包（把文档 + 结果 + 图放一起）

打包脚本：`scripts/collect_final_artifacts_fusion_v1.py`

推荐命令：

```bash
python scripts/collect_final_artifacts_fusion_v1.py ^
  --out_dir result/_share_scheme_fusion_v1 ^
  --fusion_eval_dir result/scheme_fusion_eval_only_v1 ^
  --cexu_pred_dir result/cexu_test_fusion_pred_v1 ^
  --include_weights
```

输出目录：
- `result/_share_scheme_fusion_v1/`
  - README（本文件）
  - 融合评估结果与配图
  - cexu_test 预测 Excel/CSV
  - （可选）复制子模型权重（可能较大）

