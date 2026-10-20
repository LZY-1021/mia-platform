from __future__ import annotations

import os
import random
from typing import Any, Dict, List

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve

from attacks.base import AttackResult, BaseAttack
from attacks.llm_scaling.utils import (
    clean_nan_nested,
    evaluate_chunk_auc,
    features_to_numpy,
    load_mia_jsonl,
    predict_raw,
    set_all_seeds,
    train_linear_scorer,
)
from core.metrics import compute_best_metrics
from core.visualizer import line_chart, roc_curve_chart, steps_chart


class LLMScalingDocumentAttack(BaseAttack):
    name = "llm_scaling_document"
    display_name = "Scaling Up MIA — Document Level"
    description = (
        "将单篇文档拆成多个paragraph，聚合paragraph级别的MIA分数来推断文档是否为训练成员。"
        "使用Mann-Whitney U检验，并展示AUC随paragraph数量的变化趋势。"
    )
    category = "llm"
    access = "blackbox"

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "batch_size": 32,
            "seed": 42,
            "train_docs": 950,
            "known_docs": 50,
        }

    def discover_data(self, results_dir: str) -> Dict[str, Any]:
        """
        返回层级结构：模型 -> 数据集 -> 样本大小 -> chunk大小

        路径示例：doc_mia/EleutherAI/pythia-6.9b/haritzpuerto/the_pile_00_wiki/sample_size_2000/512
        """
        base_dir = os.path.join(results_dir, "doc_mia", "EleutherAI")

        result = {
            "type": "hierarchical",
            "levels": [
                {"key": "model", "label": "模型", "label_en": "Model"},
                {"key": "dataset", "label": "数据集", "label_en": "Dataset"},
                {"key": "sample_size", "label": "样本大小", "label_en": "Sample Size"},
                {"key": "chunk_size", "label": "Chunk大小", "label_en": "Chunk Size"},
            ],
            "data": {
                "models": [],
                "datasets": {},
                "sample_sizes": {},
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
                result["data"]["sample_sizes"][key_md] = []

                for sample_size in sorted(os.listdir(dataset_path)):
                    sample_path = os.path.join(dataset_path, sample_size)
                    if not os.path.isdir(sample_path):
                        continue

                    result["data"]["sample_sizes"][key_md].append(sample_size)
                    key_mds = f"{model}/{dataset}/{sample_size}"
                    result["data"]["chunk_sizes"][key_mds] = []

                    for chunk_size in sorted(
                        os.listdir(sample_path),
                        key=lambda x: int(x) if x.isdigit() else 0,
                    ):
                        chunk_path = os.path.join(sample_path, chunk_size)
                        if not os.path.isdir(chunk_path):
                            continue
                        if os.path.exists(
                            os.path.join(chunk_path, "mia_members.jsonl")
                        ):
                            result["data"]["chunk_sizes"][key_mds].append(chunk_size)
                            key_full = f"{model}/{dataset}/{sample_size}/{chunk_size}"
                            result["data"]["paths"][key_full] = chunk_path

        return result

    def load_scores(self, data_path: str) -> Dict[str, Any]:
        members = load_mia_jsonl(os.path.join(data_path, "mia_members.jsonl"))
        nonmembers = load_mia_jsonl(os.path.join(data_path, "mia_nonmembers.jsonl"))
        clean_nan_nested(members)
        clean_nan_nested(nonmembers)
        return {"members": members, "nonmembers": nonmembers, "data_path": data_path}

    def run(self, scores: Dict[str, Any], params: Dict[str, Any]) -> AttackResult:
        seed = params.get("seed", 42)
        set_all_seeds(seed)

        mia_members = list(scores["members"])
        mia_non_members = list(scores["nonmembers"])
        batch_size = params["batch_size"]
        train_docs = params.get("train_docs", 950)
        known_docs = params.get("known_docs", 50)

        # 随机打乱保证评估无偏
        random.shuffle(mia_members)
        random.shuffle(mia_non_members)

        # 数据划分：训练集 / 已知非成员参考集 / 评估集
        A_non = [ch for doc in mia_non_members[:train_docs] for ch in doc]
        known_non = [
            ch
            for doc in mia_non_members[train_docs: train_docs + known_docs]
            for ch in doc
        ]
        eval_non = mia_non_members[train_docs + known_docs:]
        A_mem = [ch for doc in mia_members[:train_docs] for ch in doc]
        eval_mem = mia_members[train_docs + known_docs:]

        # 训练 chunk-level 打分器，去除两端各 2.5% 极端值
        A_mem_np = features_to_numpy(A_mem).squeeze(1)
        A_non_np = features_to_numpy(A_non).squeeze(1)
        A_mem_np = np.sort(A_mem_np, axis=0)[
            int(0.025 * len(A_mem_np)): int(0.975 * len(A_mem_np))
        ]
        A_non_np = np.sort(A_non_np, axis=0)[
            int(0.025 * len(A_non_np)): int(0.975 * len(A_non_np))
        ]
        scorer = train_linear_scorer(A_mem_np, A_non_np, batch_size=batch_size)

        # 评估 chunk-level AUC
        B_mem = np.nan_to_num(
            np.array([
                [list(ch["pred"].values())]
                for doc in eval_mem for ch in doc
            ]).squeeze(1), nan=0.0,
        )
        B_non = np.nan_to_num(
            np.array([
                [list(ch["pred"].values())]
                for doc in eval_non for ch in doc
            ]).squeeze(1), nan=0.0,
        )
        chunk_auc = evaluate_chunk_auc(scorer, B_mem, B_non, batch_size)

        # 对每篇文档的所有 chunk 打分
        def doc_scores(docs: List) -> List[np.ndarray]:
            result = []
            for doc in docs:
                feats = np.nan_to_num(
                    np.array([list(ch["pred"].values()) for ch in doc]), nan=0.0
                )
                result.append(predict_raw(scorer, feats))
            return result

        eval_mem_scores = doc_scores(eval_mem)
        eval_non_scores = doc_scores(eval_non)

        # 已知非成员 chunk 分数作为 Mann-Whitney U 检验的参考分布
        known_ref = predict_raw(
            scorer,
            np.nan_to_num(
                np.array([list(ch["pred"].values()) for ch in known_non]), nan=0.0
            ),
        ).reshape(-1)

        # 扫描不同 paragraph 数量，找到最优聚合粒度
        all_lengths = [len(s) for s in eval_mem_scores + eval_non_scores]
        max_para = int(np.mean(all_lengths) + np.std(all_lengths))
        para_range = list(range(2, max(3, max_para)))

        para_aucs: List[float] = []
        best_roc_fpr, best_roc_tpr = [0.0, 1.0], [0.0, 1.0]
        best_auc = 0.0
        best_sc: List[float] = []
        best_lb: List[int] = []
        best_member_pvalues: List[float] = []

        for n_para in para_range:
            sc: List[float] = []
            lb: List[int] = []
            member_pvs: List[float] = []

            for ds in eval_mem_scores:
                if len(ds) > 1:
                    stat, pval = mannwhitneyu(
                        ds[:n_para].squeeze(), known_ref, alternative="greater"
                    )
                    if not np.isnan(stat):
                        sc.append(float(stat))
                        lb.append(1)
                        member_pvs.append(float(pval))

            for ds in eval_non_scores:
                if len(ds) > 1:
                    stat, _ = mannwhitneyu(
                        ds[:n_para].squeeze(), known_ref, alternative="greater"
                    )
                    if not np.isnan(stat):
                        sc.append(float(stat))
                        lb.append(0)

            if len(set(lb)) < 2:
                para_aucs.append(0.5)
                continue

            auc = roc_auc_score(lb, sc)
            para_aucs.append(auc)

            if auc > best_auc:
                best_auc = auc
                fpr_arr, tpr_arr, _ = roc_curve(lb, sc)
                best_roc_fpr = fpr_arr.tolist()
                best_roc_tpr = tpr_arr.tolist()
                best_sc = sc
                best_lb = lb
                best_member_pvalues = member_pvs

        # 计算最优结果下的分类指标
        if best_auc > 0.5 and best_sc:
            met = compute_best_metrics(best_sc, best_lb)
        else:
            met = {
                "auc": 0.5, "f1": 50.0, "tpr": 50.0, "fpr": 50.0,
                "roc_fpr": [0.0, 1.0], "roc_tpr": [0.0, 1.0], "best_threshold": 0.0,
            }
        met["chunk_level_auc"] = round(chunk_auc, 4)
        met["p_value"] = round(
            float(np.median(best_member_pvalues)) if best_member_pvalues else 1.0, 6
        )

        charts = [
            roc_curve_chart(
                best_roc_fpr,
                best_roc_tpr,
                best_auc,
                title="Document MIA ROC (best #paragraphs)",
            ),
            line_chart(
                para_range,
                para_aucs,
                x_label="# Paragraphs",
                y_label="AUC",
                title="AUC vs. # Paragraphs",
                label="Document MIA",
                description_zh=(
                    "此图展示了在不同段落数量下，文档级成员推断的 AUC 变化趋势。"
                    "随着用于检验的段落数增加，打分分布的统计信号逐渐积累，AUC 通常会先升后降——"
                    "段落太少时信息不足，段落太多时可能引入噪声段落。"
                    "图中最高点对应的段落数即为最优聚合粒度，最终指标均基于该最优点计算。"
                ),
                description_en=(
                    "This chart shows how AUC changes as more paragraphs are used in the "
                    "Mann-Whitney U test for document-level membership inference. "
                    "As paragraph count increases, the statistical signal accumulates — AUC typically "
                    "rises then falls: too few paragraphs lack information, too many introduce noise. "
                    "The peak point is the optimal aggregation granularity used for all final metrics."
                ),
            ),
            # 文档级步骤说明，和集合级文案不同
            steps_chart(
                title="Detection Process",
                steps=[
                    {
                        "key": "1",
                        "title_zh": "第1步：计算 MIA 特征分数",
                        "title_en": "Step 1: Compute MIA Feature Scores",
                        "content_zh": (
                            "对每个文档的每个段落离线预计算多维特征：困惑度（Perplexity）、"
                            "Min-K% Prob、Zlib 压缩比等。训练数据通常对模型更「熟悉」，"
                            "表现为更低的困惑度。"
                        ),
                        "content_en": (
                            "Pre-compute multi-dimensional features offline for each paragraph: "
                            "Perplexity, Min-K% Prob, Zlib ratio, etc. Training data is more "
                            "familiar to the model, typically showing lower perplexity."
                        ),
                    },
                    {
                        "key": "2",
                        "title_zh": "第2步：训练线性打分器",
                        "title_en": "Step 2: Train Linear Scorer",
                        "content_zh": (
                            "使用已知成员/非成员的段落样本训练线性分类器，将多维特征映射为"
                            "单一成员推断分数。训练集两端各 2.5% 的极端值会被裁剪。"
                        ),
                        "content_en": (
                            "Train a linear classifier on known member/non-member paragraphs "
                            "to map multi-dimensional features to a single membership score. "
                            "Top and bottom 2.5% of training values are trimmed."
                        ),
                    },
                    {
                        "key": "3",
                        "title_zh": "第3步：Mann-Whitney U 检验（文档级）",
                        "title_en": "Step 3: Mann-Whitney U Test (Document Level)",
                        "content_zh": (
                            "将文档各段落的打分分布与已知非成员参考分布做 Mann-Whitney U 检验"
                            "（非参数检验，不要求正态分布）。扫描不同段落数量找到最优聚合粒度，"
                            "p 值越小说明文档越可能是训练成员。"
                        ),
                        "content_en": (
                            "Apply Mann-Whitney U test (non-parametric) between the document's "
                            "paragraph score distribution and a known non-member reference. "
                            "Sweep over different paragraph counts to find optimal aggregation; "
                            "smaller p-value indicates higher membership probability."
                        ),
                    },
                ],
            ),
        ]

        return AttackResult(
            attack_name=self.name,
            metrics=met,
            scores=best_sc,
            labels=best_lb,
            params=params,
            charts=charts,
            extra={
                "chunk_level_auc": chunk_auc,
                "para_range": para_range,
                "para_aucs": para_aucs,
            },
        )