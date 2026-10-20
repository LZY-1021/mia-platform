"""
Collection-level MIA

对应原始文件：collection_mia.py :: run_dataset_inference_pvalues()
完整保留了原始的核心逻辑：
  1. 数据划分 (A_members, A_non_members, known_non_members, eval)
  2. 组合采样 (combinations)
  3. 特征转numpy + 去极端值 (trim 2.5%)
  4. 训练线性打分器
  5. chunk-level AUC评估
  6. 对每个eval dataset做t-test，记录 p_value
  7. 用t-statistics + labels计算最终AUC/F1
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List

import numpy as np
from scipy.stats import ttest_ind

from attacks.base import AttackResult, BaseAttack
from attacks.llm_scaling.utils import (
    clean_nan_flat,
    combinations_from_list,
    evaluate_chunk_auc,
    features_to_numpy,
    load_mia_jsonl,
    predict_raw,
    process_combination_features,
    set_all_seeds,
    train_linear_scorer,
)
from core.metrics import compute_best_metrics
from core.visualizer import roc_curve_chart, area_distribution_chart, steps_chart


class LLMScalingCollectionAttack(BaseAttack):
    name = "llm_scaling_collection"
    display_name = "Scaling Up MIA — Collection Level"
    description = (
        "将多个文档聚合为一个集合，通过t-test对集合做成员推断。"
        "先训练chunk-level线性打分器，再用t-test聚合多个文档的分数。"
    )
    category = "llm"
    access = "blackbox"

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "training_set_size_per_class": 1000,
            "eval_set_size_per_class": 500,
            "num_docs_per_dataset": 100,
            "known_datasets": 500,
            "batch_size": 128,
            "seed": 670487,
        }

    def discover_data(self, results_dir: str) -> Dict[str, Any]:
        """返回层级结构：模型 -> 数据集 -> chunk大小"""
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
        clean_nan_flat(members)
        clean_nan_flat(nonmembers)
        return {"members": members, "nonmembers": nonmembers, "data_path": data_path}

    def run(self, scores: Dict[str, Any], params: Dict[str, Any]) -> AttackResult:
        seed = params.get("seed", 670487)
        set_all_seeds(seed)

        mia_members = list(scores["members"])
        mia_non_members = list(scores["nonmembers"])

        train_size = params["training_set_size_per_class"]
        known_size = params["known_datasets"]
        eval_size = params["eval_set_size_per_class"]
        num_docs = params["num_docs_per_dataset"]
        batch_size = params["batch_size"]

        random.shuffle(mia_members)
        random.shuffle(mia_non_members)

        # 数据划分：训练集 / 已知非成员参考集 / 评估集
        A_mem = mia_members[:train_size]
        A_non = mia_non_members[:train_size]
        known_non = mia_non_members[train_size: train_size + known_size]
        eval_mem = mia_members[train_size + known_size:]
        eval_non = mia_non_members[train_size + known_size:]

        # 将评估数据组合为集合（每个集合含 num_docs 篇文档）
        eval_mem_datasets = combinations_from_list(eval_mem, eval_size, num_docs)
        eval_non_datasets = combinations_from_list(eval_non, eval_size, num_docs)
        known_non_datasets = combinations_from_list(
            known_non,
            min(len(known_non), eval_size),
            min(num_docs, len(known_non)),
        )

        # 特征转换为 numpy
        A_mem_np = features_to_numpy(A_mem)
        A_non_np = features_to_numpy(A_non)
        known_non_np = process_combination_features(known_non_datasets)
        eval_mem_np = process_combination_features(eval_mem_datasets)
        eval_non_np = process_combination_features(eval_non_datasets)

        # 去除训练集两端各 2.5% 的极端值，使打分器更稳健
        def trim(arr):
            s = np.sort(arr, axis=0)
            lo = int(0.025 * s.shape[0])
            hi = int(0.975 * s.shape[0])
            return s[lo:hi]

        A_mem_np = trim(A_mem_np)
        A_non_np = trim(A_non_np)
        known_non_np = trim(known_non_np)

        # 训练 chunk-level 线性打分器
        scorer = train_linear_scorer(
            A_mem_np.squeeze(1), A_non_np.squeeze(1), batch_size=batch_size
        )

        # 评估 chunk-level AUC（聚合前的基础性能）
        B_mem = np.nan_to_num(features_to_numpy(eval_mem).squeeze(1), nan=0.0)
        B_non = np.nan_to_num(features_to_numpy(eval_non).squeeze(1), nan=0.0)
        chunk_auc = evaluate_chunk_auc(scorer, B_mem, B_non, batch_size)

        # 对已知非成员集合打分，作为 t-test 的参考分布
        # 处理known_non_np可能为空的情况
        if len(known_non_np) > 0:
            known_flat = np.concatenate([
                predict_raw(scorer, group) for group in known_non_np
            ]).reshape(-1)
        else:
            # 如果没有已知非成员数据，使用评估非成员数据作为参考分布
            known_flat = np.concatenate([
                predict_raw(scorer, group) for group in eval_non_np
            ]).reshape(-1)

        # 对每个评估集合做单侧 t-test
        # t-statistic 作为成员推断分数，p_value 用于前端置信度卡片
        all_scores: List[float] = []
        all_labels: List[int] = []
        member_pvalues: List[float] = []

        # 同时收集每个集合的"均值分数"用于分布图
        member_mean_scores: List[float] = []
        non_member_mean_scores: List[float] = []

        for group in eval_mem_np:
            preds = predict_raw(scorer, group)
            stat, pval = ttest_ind(
                preds, known_flat, equal_var=True, alternative="greater"
            )
            all_scores.append(float(stat))
            all_labels.append(1)
            member_pvalues.append(float(pval))
            # 记录该集合内各文档打分的均值，用于分布直方图
            member_mean_scores.append(float(np.mean(preds)))

        for group in eval_non_np:
            preds = predict_raw(scorer, group)
            stat, _ = ttest_ind(
                preds, known_flat, equal_var=True, alternative="greater"
            )
            all_scores.append(float(stat))
            all_labels.append(0)
            non_member_mean_scores.append(float(np.mean(preds)))

        met = compute_best_metrics(all_scores, all_labels)
        met["chunk_level_auc"] = round(chunk_auc, 4)
        # p_value 取成员集合 p 值的中位数
        # 前端置信度公式：confidence = (1 - p_value) * 100
        met["p_value"] = round(float(np.median(member_pvalues)), 6)

        charts = [
            roc_curve_chart(
                met["roc_fpr"],
                met["roc_tpr"],
                met["auc"],
                title=f"Collection MIA ROC (docs/set={num_docs})",
            ),
            # 真实分数分布直方图：用各集合的打分均值统计得到
            # 前端 AreaDistributionChart 渲染
            area_distribution_chart(
                member_scores=member_mean_scores,
                non_member_scores=non_member_mean_scores,
                n_bins=30,
                title="Collection Score Distribution",
            ),
            # 步骤说明卡片：由后端定义，前端 StepsChart 渲染
            # 这样"集合级"和"文档级"可以各自定制步骤文案，互不干扰
            steps_chart(
                title="Detection Process",
                steps=[
                    {
                        "key": "1",
                        "title_zh": "第1步：计算 MIA 特征分数",
                        "title_en": "Step 1: Compute MIA Feature Scores",
                        "content_zh": (
                            "对每个文档/段落离线预计算多维特征：困惑度（Perplexity）、"
                            "Min-K% Prob、Zlib 压缩比等。这些特征反映了模型对该文本的"
                            "「熟悉程度」，训练数据通常有更低的困惑度。"
                        ),
                        "content_en": (
                            "Pre-compute multi-dimensional features offline for each document: "
                            "Perplexity, Min-K% Prob, Zlib ratio, etc. These reflect model "
                            "familiarity — training data typically has lower perplexity."
                        ),
                    },
                    {
                        "key": "2",
                        "title_zh": "第2步：训练线性打分器",
                        "title_en": "Step 2: Train Linear Scorer",
                        "content_zh": (
                            "使用已知成员/非成员样本训练线性分类器，将多维特征映射为单一"
                            "成员推断分数。训练集两端各 2.5% 的极端值会被裁剪以增强鲁棒性。"
                        ),
                        "content_en": (
                            "Train a linear classifier on known member/non-member samples to "
                            "map multi-dimensional features to a single membership score. "
                            "Top and bottom 2.5% of training values are trimmed for robustness."
                        ),
                    },
                    {
                        "key": "3",
                        "title_zh": "第3步：T 检验聚合（集合级）",
                        "title_en": "Step 3: T-Test Aggregation (Collection Level)",
                        "content_zh": (
                            "将集合内多个文档的打分与已知非成员参考分布做单侧 t 检验。"
                            "t-statistic 作为最终成员推断分数，p 值越小说明集合越可能是"
                            "训练成员。最终 p 值取所有成员集合 p 值的中位数。"
                        ),
                        "content_en": (
                            "Apply one-sided t-test between the collection's document scores "
                            "and a known non-member reference distribution. The t-statistic "
                            "serves as the final membership score. The final p-value is the "
                            "median of all member collection p-values."
                        ),
                    },
                ],
            ),
        ]

        return AttackResult(
            attack_name=self.name,
            metrics=met,
            scores=all_scores,
            labels=all_labels,
            params=params,
            charts=charts,
            extra={"chunk_level_auc": chunk_auc},
        )