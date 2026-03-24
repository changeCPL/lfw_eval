# LFW 风格评测：当前输出指标与「是否每折再平均」

本文整理自对 `lfw_eval` 包行为与统计口径的说明，便于写报告或对接实验记录。

---

## 1. `evaluate_10fold` 直接返回的字段

返回类型为 `TenFoldResult`，主要字段如下。

| 字段 | 含义 | 是否「先每折再汇总」 |
|------|------|----------------------|
| `mean_accuracy` / `std_accuracy` | 10 折**测试准确率**的均值与样本标准差 | 是：先得到每折一个 Acc，再对 10 个数做 mean / std |
| `fold_accuracies` | 每一折的测试 Acc | 每折一条 |
| `fold_thresholds` | 每一折在「其余 9 折」上搜索得到的最优阈值 | 每折一条 |
| `total_pairs` | 配对总数（与折划分长度一致，如 6000） | 计数 |
| `valid_pair_count` | 两端特征均有效且维度一致的对数 | 计数 |
| `fold_test_valid_counts` | 每一折测试子集中**参与计分**的有效对数 | 每折一条 |
| `fold_test_binary_metrics` | 每一折在**该折阈值**下、**该折测试集**上的混淆与派生指标（见下节） | 每折一条；**不会**在结果对象里自动再做一层 mean/std |
| `fold_roc` / `fold_pr` | 每一折**测试子集**上的 ROC / PR 曲线 | 仅当 `include_curves=True` 时有值，否则为 `None` |

---

## 2. `fold_test_binary_metrics` 里有什么

每一折对应一个 `BinaryVerificationMetrics`，在**该折训练得到的阈值**下、仅在该折**测试子集**上统计，例如：

- 混淆：`true_positive`、`false_positive`、`true_negative`、`false_negative`
- `tpr`（= Recall / 常与人脸中的 TAR 对应）
- `fpr`（常与人脸中的 FAR 对应）
- `tnr`（Specificity）
- `precision`、`recall`、`f1`、`accuracy`
- `n_genuine`、`n_impostor`（该折测试里有效同人/异人对数，作分母参考）

---

## 3. 跨折汇总标量（需自行调用）

`summarize_binary_metrics_across_folds(fold_test_binary_metrics)` 会对以下量在 10 折上做 **`nanmean`**，得到例如 `mean_tpr`、`mean_fpr`、`mean_precision` 等：

- `tpr`、`fpr`、`tnr`、`precision`、`recall`、`f1`、`accuracy`

**注意**：该函数**没有**自动计算这些量的跨折标准差；若需要「mean ± std」风格，可在 10 个标量上自行做 `nanstd`。

---

## 4. ROC-AUC、AP、TPR@FPR、Precision@Recall

这些**没有**作为 `TenFoldResult` 的固定字段写入。

当 `include_curves=True` 时，可对每一折的 `fold_roc[k]`、`fold_pr[k]` 调用例如：

- `auc_trapezoid`（ROC-AUC）
- `average_precision_from_pr`（由 PR 折线近似 AP）
- `max_tpr_at_fpr_cap`（如 TAR@FAR）
- `best_precision_at_min_recall`（给定 Recall 下界时的 Precision）

再在 10 折上**自行决定**是否取平均、是否报告 std。这与论文中「每折一条曲线再平均」的做法一致时，需在报告中写清。

---

## 5. TPR、FPR、Precision 要不要每折分开算？

**与当前 10 折协议一致时：要每折分开算。**

- 每一折的阈值是在**另外 9 折**上确定的；在同一折的**测试子集**上算 TPR/FPR/Precision，才能保证「定阈值的数据」与「报指标的数据」分离，避免泄露与过于乐观。
- 因此实现上是 **每折一个** `fold_test_binary_metrics`。

**汇报总体时的常见做法**（二选一，并在文中写清）：

1. **与 Acc 同风格**：对 10 折的 TPR/FPR/Precision 等再取平均（可用 `summarize_binary_metrics_across_folds`），必要时补跨折 std。
2. **只展示每折**：直接引用 `fold_test_binary_metrics[k]`。

**若把 10 折的测试分数拼成一条 ROC**：可以得到单条曲线与一个 AUC，但**不再等价于标准 LFW「每折独立阈值 + 每折测试」**的严格口径，报告中应单独说明。

---

## 6. 一句话小结

- **Acc**：已在 `TenFoldResult` 里给出 **mean ± std**（按折）。
- **TPR / FPR / Precision 等**：在结果里是**每折一套**；若要总体标量 mean，请调用 `summarize_binary_metrics_across_folds` 或自行统计。
- **曲线类与工作点**：依赖 `include_curves=True` 后在每折 ROC/PR 上计算，是否再平均由实验设计决定。
