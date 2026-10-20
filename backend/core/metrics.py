"""
统一评估指标模块 —— 所有攻击共享。

对应原始代码：
  - collection_mia.py :: get_tpr_fpr()
  - collection_mia.py :: compute_auc()  (部分)
  - document_mia.py :: get_tpr_fpr()
  以及 classification_report / roc_auc_score / roc_curve 的调用
"""

import numpy as np
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, roc_curve, classification_report


def get_tpr_fpr(predictions, labels):
    """
    从二值预测和标签计算 TPR / FPR（百分制）。
    原始位置：collection_mia.py 和 document_mia.py 中的 get_tpr_fpr()
    """
    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
    fpr = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    return tpr, fpr


def compute_best_metrics(scores, labels):
    """
    利用 Youden's J 选最佳阈值，返回 AUC, F1, TPR, FPR, ROC 曲线数据。
    对应原始 collection_mia.py 的 run_dataset_inference_pvalues() 中：
      fpr, tpr, thresholds = roc_curve(...)
      youden_j = tpr - fpr
      best_threshold = thresholds[np.argmax(youden_j)]
      ...
    """
    auc = roc_auc_score(labels, scores)
    fpr_arr, tpr_arr, thresholds = roc_curve(labels, scores)

    youden_j = tpr_arr - fpr_arr
    best_idx = int(np.argmax(youden_j))
    best_threshold = float(thresholds[best_idx])

    predictions = [1 if s > best_threshold else 0 for s in scores]
    tpr, fpr = get_tpr_fpr(predictions, labels)

    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    f1 = report["weighted avg"]["f1-score"]

    return {
        "auc": round(float(auc), 4),
        "accuracy": round(float(accuracy_score(labels, predictions) * 100), 2),
        "recall": round(float(recall_score(labels, predictions, zero_division=0) * 100), 2),
        "f1": round(float(f1 * 100), 2),
        "tpr": round(float(tpr), 2),
        "fpr": round(float(fpr), 2),
        "best_threshold": best_threshold,
        "roc_fpr": fpr_arr.tolist(),
        "roc_tpr": tpr_arr.tolist(),
    }
