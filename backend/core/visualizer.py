"""
图表数据生成器 —— 返回前端 Recharts 可用的配置字典。

前端 ResultsDisplay.jsx 根据每个图表对象的 chartType 字段分发渲染：
  'roc'              → RocChart
  'roc_multi'        → RocMultiChart
  'line'             → LineChartCard
  'area_distribution'→ AreaDistributionChart（真实分数分布直方图）
  'steps'            → StepsChart（可折叠步骤说明）
  'table'            → MetricsTable
  'table_judge'      → JudgeTable
  'image'            → ImageChart（图片展示，Canary 公章样本）
  'leak_qa'          → LeakQACard（问答泄露演示：提问 → 模型答出机密内容）
"""

from typing import Dict, List, Optional
import numpy as np


def roc_curve_chart(
    fpr: List[float],
    tpr: List[float],
    auc: float,
    title: str = "ROC Curve",
) -> Dict:
    """单条 ROC 曲线，前端显示 AUC tag。"""
    return {
        "chartType": "roc",
        "title": title,
        "auc": round(auc, 4),
        "data": [
            {"fpr": round(f, 4), "tpr": round(t, 4)}
            for f, t in zip(fpr, tpr)
        ],
    }


def roc_multi_chart(
    fpr: List[float],
    series: List[Dict],
    title: str = "ROC Comparison",
) -> Dict:
    """
    多条 ROC 曲线叠加，前端 RocMultiChart 渲染。

    series 格式：[{"key": "wbc", "label": "WBC (AUC=0.87)", "tpr": [...]}]
    fpr 为所有曲线共享的横轴（需提前对齐）。
    """
    data = []
    for i, f in enumerate(fpr):
        point = {"fpr": round(f, 4)}
        for s in series:
            point[s["key"]] = round(s["tpr"][i], 4)
        data.append(point)
    return {
        "chartType": "roc_multi",
        "title": title,
        "data": data,
        "series": [{"key": s["key"], "label": s["label"]} for s in series],
    }


def line_chart(
    x_values: List,
    y_values: List[float],
    x_label: str,
    y_label: str,
    title: str,
    label: str = "Series",
    description_zh: str = "",
    description_en: str = "",
    baseline:float=0.5,
) -> Dict:
    """
    通用折线图，前端带 y=0.5 随机基线。
    description_zh / description_en 用于前端问号悬浮框说明。
    """
    result = {
        "chartType": "line",
        "title": title,
        "xLabel": x_label,
        "yLabel": y_label,
        "seriesLabel": label,
        "data": [
            {"x": x, "y": round(float(y), 4)}
            for x, y in zip(x_values, y_values)
        ],
        "baseline": baseline,
    }
    if description_zh:
        result["descriptionZh"] = description_zh
    if description_en:
        result["descriptionEn"] = description_en
    return result

def area_distribution_chart(
    member_scores: List[float],
    non_member_scores: List[float],
    n_bins: int = 30,
    title: str = "Score Distribution",
) -> Dict:
    """
    真实分数分布直方图。

    后端传入成员和非成员的原始打分列表，这里统一计算 bin 范围后
    分别统计频率，返回前端 AreaDistributionChart 可直接使用的数据。

    使用统一的 bin 边界（覆盖两组分数的全范围）保证横轴对齐。
    """
    all_scores = member_scores + non_member_scores
    if not all_scores:
        return {
            "chartType": "area_distribution",
            "title": title,
            "data": [],
        }

    lo, hi = float(np.min(all_scores)), float(np.max(all_scores))
    # 两端各留 5% 余量，避免分布贴边
    margin = (hi - lo) * 0.05 if hi > lo else 0.5
    edges = np.linspace(lo - margin, hi + margin, n_bins + 1)
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()

    mem_counts, _ = np.histogram(member_scores, bins=edges)
    non_counts, _ = np.histogram(non_member_scores, bins=edges)

    data = [
        {
            "x": round(c, 4),
            "member": int(m),
            "nonMember": int(n),
        }
        for c, m, n in zip(centers, mem_counts, non_counts)
    ]

    return {
        "chartType": "area_distribution",
        "title": title,
        "xLabel": "Score",
        "yLabel": "Count",
        "memberLabel": "Member",
        "nonMemberLabel": "Non-Member",
        "data": data,
    }


def steps_chart(
    steps: List[Dict],
    title: str = "Detection Process",
) -> Dict:
    """
    可折叠步骤说明卡片，前端 StepsChart 渲染。

    每个 step 包含：
      key        — 唯一标识，默认展开第一个
      title_zh   — 中文标题
      title_en   — 英文标题
      content_zh — 中文内容
      content_en — 英文内容
    """
    return {
        "chartType": "steps",
        "title": title,
        "steps": steps,
    }


def metrics_table(rows: List[Dict], title: str = "Metrics") -> Dict:
    """
    普通指标表格，前端 MetricsTable 渲染。
    title 显示在卡片标题栏。
    columns 可选，不提供则前端自动从 rows 推断列顺序。
    """
    return {
        "chartType": "table",
        "title": title,
        "rows": rows,
    }


def judge_table(
    rows: List[Dict],
    title: str = "Sample Membership Judgments",
    fields: Optional[List[str]] = None,
) -> Dict:
    """
    成员判定热力表，前端 JudgeTable 渲染。

    rows 中每个 dict 的值格式：
      - "True Member" 列：0 或 1
      - 其他列：{"text": "0.87", "judge": 1, "prob": 0.87}
    fields 指定列顺序，不提供则前端自动推断（True Member 置首）。
    """
    result = {
        "chartType": "table_judge",
        "title": title,
        "rows": rows,
    }
    if fields is not None:
        result["fields"] = fields
    return result


# ============================================================
# 白盒攻击专用图表数据生成器
# ============================================================

def token_heatmap_chart(
    samples: List[Dict],
    title: str = "Token Probability Heatmap",
) -> Dict:
    """
    Token 概率热力图数据。

    samples 格式：[
      {
        "label": 1,              # 0=非成员, 1=成员
        "text_preview": "...",    # 原文预览
        "token_texts": [...],     # token 文本列表
        "all_prob": [...],        # 每个 token 的对数概率
        "ppl": float,            # 困惑度
      },
      ...
    ]

    前端 TokenProbHeatmap 渲染：
      - 每个 token 按概率着色（绿=高概率/模型不意外，红=低概率/模型被惊讶）
      - 成员文本应全绿，非成员文本应频繁出现红色
    """
    return {
        "chartType": "token_heatmap",
        "title": title,
        "samples": samples,
    }


def continuation_compare_chart(
    samples: List[Dict],
    title: str = "Continuation Comparison",
) -> Dict:
    """
    逐词续写对比数据。

    samples 格式：[
      {
        "label": 1,
        "text_preview": "...",
        "continuation": {
          "prefix_tokens": [...],
          "generated_tokens": [...],
          "actual_tokens": [...],
          "match_mask": [...],     # 1=匹配, 0=不匹配
          "match_rate": float,
        },
        "ppl": float,
      },
      ...
    ]

    前端 ContinuationCompare 渲染：
      - 前缀 + 模型续写 vs 原文
      - 匹配的 token 高亮为绿色，不匹配的为红色
      - 成员文本匹配率高（模型背过），非成员匹配率低
    """
    return {
        "chartType": "continuation_compare",
        "title": title,
        "samples": samples,
    }


def calibration_chart(
    bins: List[Dict],
    ece: float,
    brier_score: float,
    title: str = "Calibration Curve",
) -> Dict:
    """
    校准曲线（可靠性图）数据。

    bins 格式：[
      {"mean_pred": 0.12, "mean_true": 0.15, "count": 42},
      ...
    ]

    前端 CalibrationChart 渲染：
      - 横轴：预测概率均值（mean_pred）
      - 纵轴：实际正例比例（mean_true）
      - 对角虚线：完美校准
      - ECE 和 Brier Score 标注在标题中
    """
    return {
        "chartType": "calibration",
        "title": title,
        "bins": bins,
        "ece": ece,
        "brier_score": brier_score,
    }


def aggregation_compare_chart(
    results: List[Dict],
    title: str = "Aggregation Method Comparison",
) -> Dict:
    """
    聚合方法对比数据。

    results 格式：[
      {
        "method": "ttest",
        "method_zh": "T 检验",
        "auc": 0.87,
        "f1": 80.2,
        "tpr": 78.5,
        "fpr": 21.5,
        "p_value": 0.0003,
        "desc_zh": "...",
        "desc_en": "...",
      },
      ...
    ]

    前端 AggregationCompareChart 渲染：
      - 每种聚合方法一行，显示 AUC/F1/TPR/FPR
      - 悬浮显示方法描述
      - 最佳指标高亮
    """
    return {
        "chartType": "aggregation_compare",
        "title": title,
        "results": results,
    }


def scorer_compare_chart(
    results: List[Dict],
    title: str = "Scorer Comparison",
) -> Dict:
    """
    评分器对比数据。

    results 格式：[
      {
        "scorer": "Linear",
        "scorer_zh": "线性评分器",
        "chunk_auc": 0.72,
        "final_auc": 0.87,
        "f1": 80.2,
      },
      ...
    ]
    """
    return {
        "chartType": "scorer_compare",
        "title": title,
        "results": results,
    }


def leak_qa_chart(
    qa: List[Dict],
    title: str = "Canary Leakage",
    question_label_zh: str = "向模型提问",
    question_label_en: str = "Ask the model",
    answer_label_zh: str = "模型输出",
    answer_label_en: str = "Model output",
    conclusion_zh: str = "",
    conclusion_en: str = "",
) -> Dict:
    """
    问答泄露演示卡片，前端 LeakQACard 渲染。

    qa 每一项：
      question_zh / question_en — 训练后向模型提出的问题
      answer_zh   / answer_en   — 模型输出的答案（泄露出的文件内容）
      leaked                     — 是否为泄露字段（True 时前端红色高亮）
    """
    return {
        "chartType": "leak_qa",
        "title": title,
        "questionLabelZh": question_label_zh,
        "questionLabelEn": question_label_en,
        "answerLabelZh": answer_label_zh,
        "answerLabelEn": answer_label_en,
        "qa": qa,
        "conclusionZh": conclusion_zh,
        "conclusionEn": conclusion_en,
    }


def image_chart(
    image_url: str,
    title: str = "Image",
    caption_zh: str = "",
    caption_en: str = "",
) -> Dict:
    """
    图片展示卡片，前端 ImageChart 渲染。

    用于 Canary 注入实验：先向观众展示注入训练集的公章图片，
    再通过下方表格说明训练后模型泄露了图片内容。
    """
    result = {
        "chartType": "image",
        "title": title,
        "imageUrl": image_url,
    }
    if caption_zh:
        result["captionZh"] = caption_zh
    if caption_en:
        result["captionEn"] = caption_en
    return result


def summary_table(rows: List[Dict], title: str = "Summary") -> Dict:
    """
    总览/规模表格，前端渲染在最前（复用 MetricsTable 组件）。
    与普通 metrics_table 的区别在于 chartType 为 'summary'，便于前端排在结果最前。
    """
    result = metrics_table(rows, title=title)
    result["chartType"] = "summary"
    return result