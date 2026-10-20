"""
白盒增强版 MIA 攻击 —— 利用模型内部状态丰富特征 + 多评分器 + 多聚合方法 + Platt 校准。

相比黑盒版 (attack_collection.py) 的增强点：
  1. 特征维度扩展：10维黑盒特征 + 4维白盒特征（grad_norm, attn_entropy, hidden_var_*）
  2. 多评分器对比：线性 / MLP / XGBoost
  3. 多聚合方法对比：t-test / Mann-Whitney U / 贝叶斯 / Fisher 组合
  4. Platt Scaling 校准三步曲：修正偏差 → 确定阈值 → 验证效果
  5. 直觉性证据可视化：Token 概率热力图 + 逐词续写对比
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, roc_curve

from attacks.base import AttackResult, BaseAttack
from attacks.llm_scaling.utils import (
    combinations_from_list,
    load_mia_jsonl,
    set_all_seeds,
    train_linear_scorer,
)
from attacks.llm_scaling.whitebox_utils import (
    AGGREGATION_METHODS,
    PlattScaler,
    aggregate_bayesian,
    aggregate_fisher_scores,
    aggregate_mannwhitney,
    aggregate_ttest,
    clean_nan_flat_whitebox,
    compute_calibration_curve,
    features_to_numpy_whitebox,
    find_youden_j_threshold,
    fit_whitebox_feature_preprocessor,
    get_feature_keys,
    predict_with_model,
    process_combination_features_whitebox,
    train_mlp_scorer,
    train_xgboost_scorer,
)
from core.metrics import compute_best_metrics
from core.visualizer import (
    aggregation_compare_chart,
    area_distribution_chart,
    calibration_chart,
    continuation_compare_chart,
    roc_curve_chart,
    roc_multi_chart,
    scorer_compare_chart,
    steps_chart,
    token_heatmap_chart,
)


BLACKBOX_EXACT_KEYS = {"ppl", "ppl/lowercase_ppl", "ppl/zlib"}
BLACKBOX_PREFIXES = ("Min_",)
WHITEBOX_PLACEHOLDER_ZERO_KEYS = {"grad_norm"}
WHITEBOX_PLACEHOLDER_RATE_LIMIT = 0.05


def _integer_param(params: Dict[str, Any], key: str, minimum: int = 1) -> int:
    """Accept numeric form values while rejecting invalid experiment settings early."""
    try:
        value = int(params[key])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{key} must be an integer") from None

    if value < minimum:
        raise ValueError(f"{key} must be at least {minimum}")
    return value


class LLMScalingWhiteboxAttack(BaseAttack):
    name = "llm_scaling_whitebox"
    display_name = "Scaling Up MIA — Whitebox Enhanced"
    description = (
        "白盒增强版：利用模型内部状态（梯度、注意力、隐藏层）丰富特征，"
        "多评分器（线性/MLP/XGBoost）+ 多聚合方法对比 + Platt 校准三步曲，"
        "附带 Token 概率热力图和逐词续写对比的直觉性证据。"
    )
    category = "llm"
    access = "whitebox"

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "training_set_size_per_class": 1000,
            "eval_set_size_per_class": 500,
            "num_docs_per_dataset": 100,
            "known_datasets": 500,
            "batch_size": 128,
            "seed": 670487,
            "scorer_type": "mlp",
            "scorer_type_options": ["linear", "mlp", "xgboost"],
            "aggregation_method": "ttest",
            "aggregation_method_options": ["ttest", "mannwhitney", "bayesian", "fisher"],
        }

    def discover_data(self, results_dir: str) -> Dict[str, Any]:
        """返回层级结构：模型 → 数据集 → chunk大小"""
        # 优先搜索白盒结果目录
        base_dir = os.path.join(results_dir, "whitebox_mia", "EleutherAI")

        # 回退到黑盒结果目录（特征兼容）
        if not os.path.exists(base_dir):
            base_dir = os.path.join(results_dir, "collection_mia", "EleutherAI")

        result = {
            "type": "hierarchical",
            "levels": [
                {"key": "model", "label": "模型", "label_en": "Model"},
                {"key": "dataset", "label": "数据集", "label_en": "Dataset"},
                {"key": "chunk_size", "label": "Chunk大小", "label_en": "Chunk Size"},
            ],
            "data": {
                "models": [],
                "datasets": {},
                "chunk_sizes": {},
                "paths": {},
            },
        }

        if not os.path.exists(base_dir):
            return result

        for model in sorted(os.listdir(base_dir)):
            model_path = os.path.join(base_dir, model)
            if not os.path.isdir(model_path):
                continue

            result["data"]["models"].append(model)
            result["data"]["datasets"][model] = []

            hp_path = os.path.join(model_path, "haritzpuerto")
            if not os.path.exists(hp_path):
                continue

            for dataset in sorted(os.listdir(hp_path)):
                dataset_path = os.path.join(hp_path, dataset)
                if not os.path.isdir(dataset_path):
                    continue

                result["data"]["datasets"][model].append(dataset)
                key_md = f"{model}/{dataset}"
                result["data"]["chunk_sizes"][key_md] = []

                for chunk_size in sorted(
                    os.listdir(dataset_path),
                    key=lambda x: int(x) if x.isdigit() else 0,
                ):
                    chunk_path = os.path.join(dataset_path, chunk_size)
                    if not os.path.isdir(chunk_path):
                        continue
                    if os.path.exists(os.path.join(chunk_path, "mia_members.jsonl")):
                        result["data"]["chunk_sizes"][key_md].append(chunk_size)
                        key_full = f"{model}/{dataset}/{chunk_size}"
                        result["data"]["paths"][key_full] = chunk_path

        return result

    def load_scores(self, data_path: str) -> Dict[str, Any]:
        members = load_mia_jsonl(os.path.join(data_path, "mia_members.jsonl"))
        nonmembers = load_mia_jsonl(os.path.join(data_path, "mia_nonmembers.jsonl"))
        clean_nan_flat_whitebox(members)
        clean_nan_flat_whitebox(nonmembers)

        # 尝试加载可视化样本
        viz_path = os.path.join(data_path, "viz_samples.json")
        viz_samples = None
        if os.path.exists(viz_path):
            # viz_samples.json is emitted with ensure_ascii=False as UTF-8.
            with open(viz_path, "r", encoding="utf-8") as f:
                viz_samples = json.load(f)

        return {
            "members": members,
            "nonmembers": nonmembers,
            "data_path": data_path,
            "viz_samples": viz_samples,
        }

    def run(self, scores: Dict[str, Any], params: Dict[str, Any]) -> AttackResult:
        seed = _integer_param(params, "seed", minimum=0)
        set_all_seeds(seed)

        mia_members = list(scores["members"])
        mia_non_members = list(scores["nonmembers"])
        viz_samples = scores.get("viz_samples")

        train_size = _integer_param(params, "training_set_size_per_class")
        known_size = _integer_param(params, "known_datasets")
        eval_size = _integer_param(params, "eval_set_size_per_class")
        num_docs = _integer_param(params, "num_docs_per_dataset")
        batch_size = _integer_param(params, "batch_size")
        calibration_records = _integer_param(
            params,
            "calibration_records_per_class",
            minimum=1,
        ) if "calibration_records_per_class" in params else max(num_docs, min(500, eval_size))
        calibration_collections = min(100, eval_size)
        scorer_type = params.get("scorer_type", "mlp")
        agg_method = params.get("aggregation_method", "ttest")
        report_all_configurations = bool(params.get("report_all_configurations", False))

        random.shuffle(mia_members)
        random.shuffle(mia_non_members)

        feature_mode = params.get("feature_mode", "whitebox")
        feature_keys = self._resolve_feature_keys(mia_members, mia_non_members, feature_mode)
        if not feature_keys:
            raise ValueError(f"No usable {feature_mode} features found in the selected result directory.")

        # 数据划分：校准记录与最终评估记录互不重叠。
        A_mem = mia_members[:train_size]
        A_non = mia_non_members[:train_size]
        known_end = train_size + known_size
        calibration_mem_end = train_size + calibration_records
        calibration_non_end = known_end + calibration_records
        known_non = mia_non_members[train_size:known_end]
        calibration_mem = mia_members[train_size:calibration_mem_end]
        calibration_non = mia_non_members[known_end:calibration_non_end]
        eval_mem = mia_members[calibration_mem_end:]
        eval_non = mia_non_members[calibration_non_end:]

        if not calibration_mem or not calibration_non or not eval_mem or not eval_non:
            raise ValueError(
                "Insufficient records after creating the train/reference/calibration/evaluation split."
            )

        # 评估集合组合
        eval_mem_datasets = combinations_from_list(eval_mem, eval_size, num_docs)
        eval_non_datasets = combinations_from_list(eval_non, eval_size, num_docs)
        calibration_mem_datasets = combinations_from_list(
            calibration_mem, calibration_collections, num_docs
        )
        calibration_non_datasets = combinations_from_list(
            calibration_non, calibration_collections, num_docs
        )
        known_non_datasets = combinations_from_list(
            known_non,
            min(len(known_non), eval_size),
            min(num_docs, len(known_non)),
        )

        # ====== 多评分器对比 ======
        scorers = self._train_all_scorers(A_mem, A_non, batch_size, feature_keys)
        if scorer_type not in scorers:
            scorer_type = next(iter(scorers))

        # 用每个评分器计算 chunk-level AUC
        scorer_results = []
        for s_type, s_model in scorers.items():
            B_mem = np.nan_to_num(
                features_to_numpy_whitebox(eval_mem, feature_keys), nan=0.0
            )
            B_non = np.nan_to_num(
                features_to_numpy_whitebox(eval_non, feature_keys), nan=0.0
            )
            chunk_auc = self._evaluate_chunk_auc_generic(
                s_model, B_mem, B_non, s_type
            )
            scorer_results.append({
                "scorer": s_type.capitalize(),
                "scorer_zh": {"linear": "线性评分器", "mlp": "MLP 评分器", "xgboost": "XGBoost 评分器"}.get(s_type, s_type),
                "chunk_auc": round(chunk_auc, 4),
            })

        # 选择主评分器
        primary_scorer = scorers[scorer_type]

        # ====== 已知非成员参考分布 ======
        known_non_np = process_combination_features_whitebox(known_non_datasets, feature_keys)
        if len(known_non_np) > 0:
            known_flat = np.concatenate([
                predict_with_model(primary_scorer, group, scorer_type)
                for group in known_non_np
            ]).reshape(-1)
        else:
            eval_non_np = process_combination_features_whitebox(eval_non_datasets, feature_keys)
            known_flat = np.concatenate([
                predict_with_model(primary_scorer, group, scorer_type)
                for group in eval_non_np
            ]).reshape(-1)

        # ====== 主聚合方法计算 ======
        eval_mem_np = process_combination_features_whitebox(eval_mem_datasets, feature_keys)
        eval_non_np = process_combination_features_whitebox(eval_non_datasets, feature_keys)
        calibration_mem_np = process_combination_features_whitebox(
            calibration_mem_datasets, feature_keys
        )
        calibration_non_np = process_combination_features_whitebox(
            calibration_non_datasets, feature_keys
        )

        all_scores, all_labels = [], []
        member_pvalues = []
        member_mean_scores = []
        non_member_mean_scores = []

        # 每个成员/非成员集合的分数，用于 Fisher 组合
        per_collection_member_pvals = []

        for group in eval_mem_np:
            preds = predict_with_model(primary_scorer, group, scorer_type)
            stat, pval = self._aggregate(preds, known_flat, agg_method)
            all_scores.append(float(stat))
            all_labels.append(1)
            member_pvalues.append(float(pval))
            member_mean_scores.append(float(np.mean(preds)))
            per_collection_member_pvals.append(float(pval))

        for group in eval_non_np:
            preds = predict_with_model(primary_scorer, group, scorer_type)
            stat, _ = self._aggregate(preds, known_flat, agg_method)
            all_scores.append(float(stat))
            all_labels.append(0)
            non_member_mean_scores.append(float(np.mean(preds)))

        # ====== Platt Scaling 校准三步曲 ======
        scores_arr = np.array(all_scores)
        labels_arr = np.array(all_labels)

        # 第 1 步：仅用独立校准集合拟合概率映射。
        calibration_scores, calibration_labels = [], []
        for group in calibration_mem_np:
            preds = predict_with_model(primary_scorer, group, scorer_type)
            stat, _ = self._aggregate(preds, known_flat, agg_method)
            calibration_scores.append(float(stat))
            calibration_labels.append(1)
        for group in calibration_non_np:
            preds = predict_with_model(primary_scorer, group, scorer_type)
            stat, _ = self._aggregate(preds, known_flat, agg_method)
            calibration_scores.append(float(stat))
            calibration_labels.append(0)

        scaler = PlattScaler()
        calibrated_calibration_probs = scaler.fit_transform(
            np.asarray(calibration_scores), np.asarray(calibration_labels)
        )
        calibrated_probs = scaler.transform(scores_arr)

        # 第 2 步：在校准概率上用 Youden's J 确定固定阈值。
        youden_info = find_youden_j_threshold(
            calibrated_calibration_probs, np.asarray(calibration_labels)
        )

        # 第 3 步：只在最终评估集合上验证校准质量。
        calibration_info = compute_calibration_curve(calibrated_probs, labels_arr, n_bins=10)

        # 最终指标使用已冻结的校准阈值，不再从最终评估标签重新挑阈值。
        met = self._compute_metrics_at_threshold(
            calibrated_probs, labels_arr, youden_info["best_threshold"]
        )
        met["raw_auc"] = round(float(roc_auc_score(all_labels, all_scores)), 4)
        met["calibration_auc"] = round(
            float(roc_auc_score(calibration_labels, calibrated_calibration_probs)), 4
        )
        met["chunk_level_auc"] = round(
            max(r["chunk_auc"] for r in scorer_results), 4
        )
        met["p_value"] = round(float(np.median(member_pvalues)), 6)
        met["ece"] = calibration_info["ece"]
        met["brier_score"] = calibration_info["brier_score"]
        met["youden_j"] = youden_info["best_youden_j"]
        met["best_threshold_calibrated"] = youden_info["best_threshold"]

        # ====== 多聚合方法对比 ======
        agg_compare = self._compare_aggregation_methods(
            eval_mem_np,
            eval_non_np,
            calibration_mem_np,
            calibration_non_np,
            known_flat,
            primary_scorer,
            scorer_type,
        )

        # The report runner requests a complete scorer/aggregation grid. This
        # reuses the already trained scorers, so the frontend's default path is
        # unchanged and does not pay this additional evaluation cost.
        configuration_grid = []
        if report_all_configurations:
            configuration_grid = self._evaluate_configuration_grid(
                eval_mem_np,
                eval_non_np,
                known_non_np,
                scorers,
                scorer_results,
            )

        # ====== 评分器最终 AUC 填充 ======
        for i, sr in enumerate(scorer_results):
            s_type = sr["scorer"].lower()
            if s_type in scorers:
                if len(known_non_np) > 0:
                    scorer_known_flat = np.concatenate([
                        predict_with_model(scorers[s_type], group, s_type)
                        for group in known_non_np
                    ]).reshape(-1)
                else:
                    scorer_known_flat = np.concatenate([
                        predict_with_model(scorers[s_type], group, s_type)
                        for group in eval_non_np
                    ]).reshape(-1)
                # 计算该评分器的最终 AUC
                s_all_scores = []
                s_all_labels = []
                for group in eval_mem_np:
                    preds = predict_with_model(scorers[s_type], group, s_type)
                    stat, _ = self._aggregate(preds, scorer_known_flat, agg_method)
                    s_all_scores.append(float(stat))
                    s_all_labels.append(1)
                for group in eval_non_np:
                    preds = predict_with_model(scorers[s_type], group, s_type)
                    stat, _ = self._aggregate(preds, scorer_known_flat, agg_method)
                    s_all_scores.append(float(stat))
                    s_all_labels.append(0)
                try:
                    sr["final_auc"] = round(roc_auc_score(s_all_labels, s_all_scores), 4)
                except ValueError:
                    sr["final_auc"] = 0.5

        # ====== 图表生成 ======
        charts = []

        # 1. ROC 曲线
        charts.append(roc_curve_chart(
            met["roc_fpr"], met["roc_tpr"], met["auc"],
            title=f"Whitebox MIA ROC ({scorer_type} + {agg_method})",
        ))

        # 2. 分数分布
        charts.append(area_distribution_chart(
            member_scores=member_mean_scores,
            non_member_scores=non_member_mean_scores,
            n_bins=30,
            title="Whitebox Score Distribution",
        ))

        # 3. 评分器对比
        charts.append(scorer_compare_chart(
            results=scorer_results,
            title="Scorer Comparison",
        ))

        # 4. 聚合方法对比
        charts.append(aggregation_compare_chart(
            results=agg_compare,
            title="Aggregation Method Comparison",
        ))

        # 5. 校准曲线
        charts.append(calibration_chart(
            bins=calibration_info["bins"],
            ece=calibration_info["ece"],
            brier_score=calibration_info["brier_score"],
            title=f"Calibration Curve (ECE={calibration_info['ece']:.4f}, Brier={calibration_info['brier_score']:.4f})",
        ))

        # 6. Token 概率热力图（使用预计算的可视化样本）
        heatmap_samples = self._get_heatmap_samples(viz_samples, mia_members, mia_non_members)
        if heatmap_samples:
            charts.append(token_heatmap_chart(
                samples=heatmap_samples,
                title="Token Probability Heatmap",
            ))

        # 7. 逐词续写对比
        cont_samples = self._get_continuation_samples(viz_samples, mia_members, mia_non_members)
        if cont_samples:
            charts.append(continuation_compare_chart(
                samples=cont_samples,
                title="Continuation Comparison",
            ))

        # 8. 步骤说明
        charts.append(steps_chart(
            title="Whitebox Detection Process",
            steps=[
                {
                    "key": "1",
                    "title_zh": "第1步：计算白盒增强特征",
                    "title_en": "Step 1: Compute Whitebox-Enhanced Features",
                    "content_zh": (
                        "在黑盒特征（PPL、Min-K% Prob、Zlib 比等）基础上，"
                        "利用模型内部状态提取梯度范数、注意力熵、隐藏层方差等白盒特征。"
                        "梯度范数反映模型对输入的敏感度——训练数据梯度小（模型已适应），"
                        "非训练数据梯度大（模型需大幅调整）。"
                    ),
                    "content_en": (
                        "Beyond black-box features (PPL, Min-K% Prob, Zlib ratio), "
                        "extract white-box features: gradient norm, attention entropy, "
                        "hidden state variance. Lower gradient = model already adapted = likely training data."
                    ),
                },
                {
                    "key": "2",
                    "title_zh": "第2步：多评分器训练与对比",
                    "title_en": "Step 2: Multi-Scorer Training & Comparison",
                    "content_zh": (
                        "训练三种评分器：线性模型、2层MLP、XGBoost。"
                        "线性模型是基线，MLP可捕获非线性关系（但有正则化防过拟合），"
                        "XGBoost擅长特征交互。对比选择最优。"
                    ),
                    "content_en": (
                        "Train three scorers: Linear, 2-layer MLP, XGBoost. "
                        "Linear is baseline, MLP captures nonlinearity (with regularization), "
                        "XGBoost excels at feature interactions. Compare to select best."
                    ),
                },
                {
                    "key": "3",
                    "title_zh": "第3步：多聚合方法统计检验",
                    "title_en": "Step 3: Multi-Aggregation Statistical Tests",
                    "content_zh": (
                        "四种聚合方法对比：T检验（参数/正态假设）、Mann-Whitney U（非参数/鲁棒）、"
                        "贝叶斯检验（证据强度量化）、Fisher组合（多p值整合）。"
                        "不同方法在不同数据分布下各有优势。"
                    ),
                    "content_en": (
                        "Compare four aggregation methods: T-test (parametric), "
                        "Mann-Whitney U (non-parametric/robust), Bayesian test (evidence strength), "
                        "Fisher's method (p-value combination). Each excels under different data distributions."
                    ),
                },
                {
                    "key": "4",
                    "title_zh": "第4步：Platt 校准三步曲",
                    "title_en": "Step 4: Platt Calibration Trilogy",
                    "content_zh": (
                        "① 修正偏差：用 Platt Scaling 将原始分数映射为校准概率，消除模型过度自信；"
                        "② 确定阈值：在校准概率上用 Youden's J 找最佳「及格线」；"
                        "③ 验证效果：校准曲线 + ECE + Brier Score 确保结果可信。"
                    ),
                    "content_en": (
                        "① Correct bias: Platt Scaling maps raw scores to calibrated probabilities; "
                        "② Set threshold: Youden's J finds optimal cutoff on calibrated probabilities; "
                        "③ Validate: Calibration curve + ECE + Brier Score ensure reliability."
                    ),
                },
                {
                    "key": "5",
                    "title_zh": "第5步：直觉性证据——热力图与续写对比",
                    "title_en": "Step 5: Intuitive Evidence — Heatmap & Continuation",
                    "content_zh": (
                        "Token 概率热力图：训练数据一路绿灯（模型不意外），"
                        "非训练数据频繁出现红点（模型被惊讶）。"
                        "逐词续写对比：输入前缀后模型续写与原文逐字比对，"
                        "匹配率高=模型背过这段话，这是最直觉的「铁证」。"
                    ),
                    "content_en": (
                        "Token heatmap: training data = all green (model unsurprised), "
                        "non-training data = frequent red spots (model surprised). "
                        "Continuation comparison: greedy decoding vs actual text. "
                        "High match rate = model memorized this text — the most intuitive 'hard evidence'."
                    ),
                },
            ],
        ))

        return AttackResult(
            attack_name=self.name,
            metrics=met,
            scores=all_scores,
            labels=all_labels,
            params=params,
            charts=charts,
            extra={
                "feature_mode": feature_mode,
                "feature_count": len(feature_keys),
                "primary_chunk_auc": next(
                    (row["chunk_auc"] for row in scorer_results if row["scorer"].lower() == scorer_type),
                    met["chunk_level_auc"],
                ),
                "chunk_level_auc": met["chunk_level_auc"],
                "configuration_grid": configuration_grid,
                "calibration": {
                    "records_per_class": calibration_records,
                    "collections_per_class": calibration_collections,
                    "calibration_auc": met["calibration_auc"],
                    "ece": calibration_info["ece"],
                    "brier_score": calibration_info["brier_score"],
                    "youden_j": youden_info["best_youden_j"],
                    "best_threshold": youden_info["best_threshold"],
                },
            },
        )

    # ====== 内部方法 ======

    def _train_all_scorers(
        self, A_mem, A_non, batch_size, feature_keys
    ) -> Dict[str, Any]:
        """训练所有评分器，返回 {type: model} 字典。"""
        # 准备训练数据
        A_mem_np = features_to_numpy_whitebox(A_mem, feature_keys)
        A_non_np = features_to_numpy_whitebox(A_non, feature_keys)

        # Preserve each sample's feature vector. The old column-wise sorting
        # mixed different samples together; robust clipping/scaling is fit only
        # on the scorer-training split and reused for all later predictions.
        preprocessor = fit_whitebox_feature_preprocessor(A_mem_np, A_non_np)
        A_mem_prepared = preprocessor.transform(A_mem_np)
        A_non_prepared = preprocessor.transform(A_non_np)

        def attach_preprocessor(model):
            model._mia_feature_preprocessor = preprocessor
            return model

        scorers = {}

        # 线性评分器
        try:
            linear = train_linear_scorer(
                A_mem_prepared, A_non_prepared, batch_size=batch_size
            )
            scorers["linear"] = attach_preprocessor(linear)
        except Exception:
            pass

        # MLP 评分器
        try:
            mlp = train_mlp_scorer(
                A_mem_prepared, A_non_prepared, batch_size=batch_size
            )
            scorers["mlp"] = attach_preprocessor(mlp)
        except Exception:
            pass

        # XGBoost 评分器
        try:
            xgb_model = train_xgboost_scorer(A_mem_prepared, A_non_prepared)
            scorers["xgboost"] = attach_preprocessor(xgb_model)
        except Exception:
            pass

        # 如果全失败，至少保证有线性评分器
        if not scorers:
            linear = train_linear_scorer(
                A_mem_prepared, A_non_prepared, batch_size=batch_size
            )
            scorers["linear"] = attach_preprocessor(linear)

        return scorers

    def _evaluate_chunk_auc_generic(
        self, model, member_features, nonmember_features, model_type, batch_size=128
    ) -> float:
        """通用 chunk-level AUC 评估。"""
        X = np.concatenate([member_features, nonmember_features], axis=0)
        y = np.concatenate([
            np.ones(len(member_features)),
            np.zeros(len(nonmember_features)),
        ])

        preds = predict_with_model(model, X, model_type)
        try:
            return roc_auc_score(y, preds)
        except ValueError:
            return 0.5

    def _aggregate(self, scores, reference, method: str):
        """调用指定聚合方法。"""
        if method == "ttest":
            return aggregate_ttest(scores, reference)
        elif method == "mannwhitney":
            return aggregate_mannwhitney(scores, reference)
        elif method == "bayesian":
            return aggregate_bayesian(scores, reference)
        elif method == "fisher":
            return aggregate_fisher_scores(scores, reference)
        else:
            return aggregate_ttest(scores, reference)

    @staticmethod
    def _compute_metrics_at_threshold(scores, labels, threshold: float) -> Dict[str, Any]:
        """Evaluate held-out probabilities at a threshold chosen on calibration data."""
        score_array = np.asarray(scores, dtype=float)
        label_array = np.asarray(labels, dtype=int)
        predictions = (score_array >= threshold).astype(int)
        fpr_arr, tpr_arr, _ = roc_curve(label_array, score_array)
        true_positive = np.sum((predictions == 1) & (label_array == 1))
        false_positive = np.sum((predictions == 1) & (label_array == 0))
        true_negative = np.sum((predictions == 0) & (label_array == 0))
        false_negative = np.sum((predictions == 0) & (label_array == 1))
        tpr = true_positive / (true_positive + false_negative) * 100 if true_positive + false_negative else 0.0
        fpr = false_positive / (false_positive + true_negative) * 100 if false_positive + true_negative else 0.0
        return {
            "auc": round(float(roc_auc_score(label_array, score_array)), 4),
            "accuracy": round(float(accuracy_score(label_array, predictions) * 100), 2),
            "recall": round(float(recall_score(label_array, predictions, zero_division=0) * 100), 2),
            "f1": round(float(f1_score(label_array, predictions, average="weighted", zero_division=0) * 100), 2),
            "tpr": round(float(tpr), 2),
            "fpr": round(float(fpr), 2),
            "best_threshold": float(threshold),
            "roc_fpr": fpr_arr.tolist(),
            "roc_tpr": tpr_arr.tolist(),
        }

    def _compare_aggregation_methods(
        self,
        eval_mem_np,
        eval_non_np,
        calibration_mem_np,
        calibration_non_np,
        known_flat,
        scorer,
        scorer_type,
    ) -> List[Dict]:
        """用所有聚合方法分别计算，返回对比结果。"""
        results = []

        for method_key, method_info in AGGREGATION_METHODS.items():
            all_scores = []
            all_labels = []
            member_pvals = []

            for group in eval_mem_np:
                preds = predict_with_model(scorer, group, scorer_type)
                stat, pval = self._aggregate(preds, known_flat, method_key)
                all_scores.append(float(stat))
                all_labels.append(1)
                member_pvals.append(float(pval))

            for group in eval_non_np:
                preds = predict_with_model(scorer, group, scorer_type)
                stat, _ = self._aggregate(preds, known_flat, method_key)
                all_scores.append(float(stat))
                all_labels.append(0)

            calibration_scores, calibration_labels = [], []
            for group in calibration_mem_np:
                stat, _ = self._aggregate(
                    predict_with_model(scorer, group, scorer_type), known_flat, method_key
                )
                calibration_scores.append(float(stat))
                calibration_labels.append(1)
            for group in calibration_non_np:
                stat, _ = self._aggregate(
                    predict_with_model(scorer, group, scorer_type), known_flat, method_key
                )
                calibration_scores.append(float(stat))
                calibration_labels.append(0)

            scaler = PlattScaler()
            calibration_probs = scaler.fit_transform(
                np.asarray(calibration_scores), np.asarray(calibration_labels)
            )
            report_probs = scaler.transform(np.asarray(all_scores))
            threshold = find_youden_j_threshold(
                calibration_probs, np.asarray(calibration_labels)
            )["best_threshold"]
            met = self._compute_metrics_at_threshold(report_probs, all_labels, threshold)

            results.append({
                "method": method_key,
                "method_zh": method_info["name_zh"],
                "method_en": method_info["name"],
                "auc": met["auc"],
                "f1": met.get("f1", 0),
                "tpr": met.get("tpr", 0),
                "fpr": met.get("fpr", 0),
                "p_value": round(float(np.median(member_pvals)) if member_pvals else 1.0, 6),
                "desc_zh": method_info["desc_zh"],
                "desc_en": method_info["desc_en"],
            })

        return results

    def _evaluate_configuration_grid(
        self,
        eval_mem_np,
        eval_non_np,
        known_non_np,
        scorers: Dict[str, Any],
        scorer_results: List[Dict],
    ) -> List[Dict]:
        """Evaluate all scorer/aggregation pairs without retraining scorers."""
        chunk_auc_by_scorer = {
            row["scorer"].lower(): float(row["chunk_auc"])
            for row in scorer_results
        }
        grid = []

        for scorer_type, scorer in scorers.items():
            if len(known_non_np) > 0:
                known_flat = np.concatenate([
                    predict_with_model(scorer, group, scorer_type)
                    for group in known_non_np
                ]).reshape(-1)
            else:
                known_flat = np.concatenate([
                    predict_with_model(scorer, group, scorer_type)
                    for group in eval_non_np
                ]).reshape(-1)

            member_predictions = [
                predict_with_model(scorer, group, scorer_type)
                for group in eval_mem_np
            ]
            nonmember_predictions = [
                predict_with_model(scorer, group, scorer_type)
                for group in eval_non_np
            ]

            for aggregation in AGGREGATION_METHODS:
                scores, labels, member_pvalues = [], [], []
                for predictions in member_predictions:
                    statistic, p_value = self._aggregate(predictions, known_flat, aggregation)
                    scores.append(float(statistic))
                    labels.append(1)
                    member_pvalues.append(float(p_value))
                for predictions in nonmember_predictions:
                    statistic, _ = self._aggregate(predictions, known_flat, aggregation)
                    scores.append(float(statistic))
                    labels.append(0)

                metrics = compute_best_metrics(scores, labels)
                grid.append({
                    "scorer": scorer_type,
                    "aggregation": aggregation,
                    "chunk_auc": chunk_auc_by_scorer.get(scorer_type, 0.5),
                    "auc": float(metrics["auc"]),
                    "accuracy": float(metrics.get("accuracy", 0.0)),
                    "recall": float(metrics.get("recall", 0.0)),
                    "f1": float(metrics.get("f1", 0.0)),
                    "confidence": float(np.clip(
                        (1.0 - np.median(member_pvalues)) * 100.0
                        if member_pvalues else 0.0,
                        0.0,
                        100.0,
                    )),
                })
        return grid

    def _resolve_feature_keys(self, members, nonmembers, feature_mode: str = "whitebox") -> List[str]:
        """锁定本次实验的数值特征列，避免可视化字段混入评分器。"""
        if feature_mode not in {"blackbox", "whitebox"}:
            raise ValueError("feature_mode must be 'blackbox' or 'whitebox'")
        for item in members + nonmembers:
            pred = item.get("pred", {})
            if pred:
                keys = [
                    key for key in get_feature_keys(item)
                    if isinstance(pred.get(key), (int, float, np.integer, np.floating))
                ]
                if feature_mode == "blackbox":
                    return [
                        key for key in keys
                        if key in BLACKBOX_EXACT_KEYS or key.startswith(BLACKBOX_PREFIXES)
                    ]
                return self._filter_unreliable_whitebox_features(keys, members, nonmembers)
        return []

    def _filter_unreliable_whitebox_features(self, keys, members, nonmembers) -> List[str]:
        """Discard known failure sentinels instead of learning their missingness."""
        reliable = []
        records = members + nonmembers
        for key in keys:
            if key not in WHITEBOX_PLACEHOLDER_ZERO_KEYS:
                reliable.append(key)
                continue
            values = [item.get("pred", {}).get(key) for item in records]
            invalid = sum(
                value is None
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)
                or value == 0.0
                for value in values
            )
            if values and invalid / len(values) < WHITEBOX_PLACEHOLDER_RATE_LIMIT:
                reliable.append(key)
        return reliable

    def _get_heatmap_samples(self, viz_samples, members, nonmembers) -> List[Dict]:
        """获取热力图展示样本。优先使用预计算的可视化样本，否则从数据中提取。"""
        if viz_samples:
            # 取最典型的成员和非成员各3个
            member_samples = [s for s in viz_samples if s["label"] == 1]
            nonmember_samples = [s for s in viz_samples if s["label"] == 0]

            # 按置信度排序：成员选 ppl 最低的，非成员选 ppl 最高的
            member_samples.sort(key=lambda x: x.get("ppl", float("inf")))
            nonmember_samples.sort(key=lambda x: x.get("ppl", 0), reverse=True)

            return member_samples[:3] + nonmember_samples[:3]

        # 从原始数据中提取（可能没有 _all_prob 字段）
        samples = []
        for item in members[:2]:
            if "_all_prob" in item.get("pred", {}):
                samples.append({
                    "label": 1,
                    "text_preview": item["pred"].get("_text_preview", ""),
                    "token_texts": item["pred"].get("_token_texts", []),
                    "all_prob": item["pred"].get("_all_prob", []),
                    "ppl": item["pred"].get("ppl", 0),
                })
        for item in nonmembers[:2]:
            if "_all_prob" in item.get("pred", {}):
                samples.append({
                    "label": 0,
                    "text_preview": item["pred"].get("_text_preview", ""),
                    "token_texts": item["pred"].get("_token_texts", []),
                    "all_prob": item["pred"].get("_all_prob", []),
                    "ppl": item["pred"].get("ppl", 0),
                })

        return samples

    def _get_continuation_samples(self, viz_samples, members, nonmembers) -> List[Dict]:
        """获取续写对比展示样本。"""
        if viz_samples:
            member_samples = [s for s in viz_samples if s["label"] == 1]
            nonmember_samples = [s for s in viz_samples if s["label"] == 0]

            # 按匹配率排序：成员选 match_rate 最高的，非成员选最低的
            member_samples.sort(key=lambda x: x.get("match_rate", 0), reverse=True)
            nonmember_samples.sort(key=lambda x: x.get("match_rate", 1))

            return member_samples[:2] + nonmember_samples[:2]

        # 从原始数据中提取
        samples = []
        for item in members[:1]:
            if "_continuation" in item.get("pred", {}):
                samples.append({
                    "label": 1,
                    "text_preview": item["pred"].get("_text_preview", ""),
                    "continuation": item["pred"]["_continuation"],
                    "ppl": item["pred"].get("ppl", 0),
                    "match_rate": item["pred"]["_continuation"].get("match_rate", 0),
                })
        for item in nonmembers[:1]:
            if "_continuation" in item.get("pred", {}):
                samples.append({
                    "label": 0,
                    "text_preview": item["pred"].get("_text_preview", ""),
                    "continuation": item["pred"]["_continuation"],
                    "ppl": item["pred"].get("ppl", 0),
                    "match_rate": item["pred"]["_continuation"].get("match_rate", 0),
                })

        return samples
