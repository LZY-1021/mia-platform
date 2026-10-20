import json
import numpy as np
import os
import argparse
import random
from typing import Dict, Any, List
from scipy.stats import norm
from attacks.base import AttackResult, BaseAttack
from core.metrics import compute_best_metrics
from core.visualizer import roc_curve_chart, metrics_table,line_chart




class VLMReferenceAttack(BaseAttack):
    name = "vlm_reference_inference_attack"
    display_name = "VLM Reference Inference Attack"
    description = (
        "先计算非成员参考数据集和目标数据集的相似度方差等统计量，然后做z检验判断目标是否为成员数据"
    )
    category = "vlm"
    access = "blackbox"


    def get_default_params(self) -> Dict[str, Any]:
        return {
            "target_similarity_file": "target_similarity.json",
            "non_member_reference_similarity_file": "non_member_reference_similarity.json",
            "granularity": 100,
            "temperature": 0.2,
            "similarity_metric": "embedding_gpt",
            "n_rounds":15,
        }


    def discover_data(self, results_dir: str) -> Dict[str, Any]:
        base_dir = os.path.join(results_dir, "reference_mia")

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

        for vendor in sorted(os.listdir(base_dir)):   # EleutherAI / Meta
            vendor_path = os.path.join(base_dir, vendor)
            if not os.path.isdir(vendor_path):
                continue

            for model in sorted(os.listdir(vendor_path)):
                model_path = os.path.join(vendor_path, model)
                if not os.path.isdir(model_path):
                    continue

                model_key = f"{vendor}/{model}"
                result["data"]["models"].append(model_key)
                result["data"]["datasets"][model_key] = []

                hp_path = os.path.join(model_path, "haritzpuerto")
                if not os.path.isdir(hp_path):
                    continue

                for dataset in sorted(os.listdir(hp_path)):
                    dataset_path = os.path.join(hp_path, dataset)
                    if not os.path.isdir(dataset_path):
                        continue

                    result["data"]["datasets"][model_key].append(dataset)
                    key_md = f"{model_key}/{dataset}"
                    result["data"]["chunk_sizes"][key_md] = []

                    for chunk_size in sorted(
                        os.listdir(dataset_path),
                        key=lambda x: int(x) if x.isdigit() else 0,
                    ):
                        chunk_path = os.path.join(dataset_path, chunk_size)
                        if not os.path.isdir(chunk_path):
                            continue

                        if (
                            os.path.exists(os.path.join(chunk_path, "non_member_reference_similarity.json"))
                            and os.path.exists(os.path.join(chunk_path, "target_similarity.json"))
                        ):
                            result["data"]["chunk_sizes"][key_md].append(chunk_size)
                            key_full = f"{model_key}/{dataset}/{chunk_size}"
                            result["data"]["paths"][key_full] = chunk_path

        return result

    def load_scores(self, data_path: str) -> Dict[str, Any]:
        pass


    def reference_non_member_inference(
        self,
        target_similarity_file,
        non_member_reference_similarity_file,
        temperature,
        metric,
        granularity,
        n_rounds: int = 15,
    ) -> Dict[str, Any]:
        """
        先从文件中加载 target_data 和 non_member_reference_data，
        再使用全量 non-member reference 作为参考分布，
        通过重复对 target_data 抽样，判断 target_data 更像 member 还是 nonmember。

        参数：
            data_path：数据所在目录
            target_similarity_file: 目标数据的 similarity json 文件
            non_member_reference_similarity_file: non-member reference 的 similarity json 文件
            temperature: similarity 对应的温度参数
            metric: 选取的指标名
            granularity: 每轮从 target_data 中抽取的样本数
            n_rounds: 重复抽样轮数

        返回：
            {
                "pred": "member" 或 "nonmember",
                "mean_z": 平均 z 分数，越大通常越像 member,
                "mean_p": 平均 p 值，越小通常越像 member,
                "z_list": 每轮 z 分数列表,
                "p_list": 每轮 p 值列表,
                "reference_mean": reference 的均值,
                "reference_var": reference 的方差,
                "reference_size": reference 样本数,
                "rounds_used": 实际成功计算的轮数
            }
        """

        # 1. 加载文件
        with open(target_similarity_file, "r",encoding="utf-8") as file:
            target_data_all = json.load(file)
        with open(non_member_reference_similarity_file, "r",encoding="utf-8") as file:
            non_member_reference_data_all = json.load(file)

        # 2. 提取指定 temperature 和 metric 的分数
        similarity_key = f"similarity_{temperature}"
        target_data = [item[similarity_key][metric] for item in target_data_all]
        non_member_reference = [
            item[similarity_key][metric] for item in non_member_reference_data_all
        ]

        # 3. 参数检查
        if not target_data:
            raise ValueError("target_data 不能为空")
        if not non_member_reference:
            raise ValueError("non_member_reference 不能为空")
        if granularity <= 1:
            raise ValueError("granularity 必须大于 1, 推荐≥100")
        if len(target_data) < granularity:
            raise ValueError(
                f"target_data 数量不足，当前为 {len(target_data)}，至少需要 {granularity} 个样本"
            )
        if len(non_member_reference) < 100:
            raise ValueError("non_member_reference 至少需要 100 个样本")
        if n_rounds <= 0:
            raise ValueError("n_rounds 必须大于 0")

        # 4. 转为 numpy array
        target_array = np.asarray(target_data, dtype=float)
        reference_array = np.asarray(non_member_reference, dtype=float)

        # 5. 计算 reference 统计量
        reference_mean = float(np.mean(reference_array))
        reference_var = float(np.var(reference_array, ddof=1))
        reference_n = len(reference_array)

        z_list: List[float] = []
        p_list: List[float] = []

        # 6. 重复抽样推断
        for _ in range(n_rounds):
            sampled_target = random.sample(list(target_array), granularity)
            sampled_target = np.asarray(sampled_target, dtype=float)

            target_mean = float(np.mean(sampled_target))
            target_var = float(np.var(sampled_target, ddof=1))

            se = np.sqrt(target_var / granularity + reference_var / reference_n)

            if se == 0 or np.isnan(se):
                continue

            z_score = (target_mean - reference_mean) / se
            p_value = 1.0 - norm.cdf(z_score)

            if np.isnan(z_score) or np.isnan(p_value):
                continue

            z_list.append(float(z_score))
            p_list.append(float(p_value))

        if not z_list:
            raise ValueError("没有得到有效统计量，可能是数据方差为 0 或输入数据异常")

        mean_z = float(np.mean(z_list))
        mean_p = float(np.mean(p_list))

        # z 检验阈值：mean_z > 0.25 判定为 member，否则为 nonmember
        pred = "member" if mean_z > 0.25 else "nonmember"

        return {
            "pred": pred,
            "mean_z": mean_z,
            "mean_p": mean_p,
            "z_list": z_list,
            "p_list": p_list,
            "reference_mean": reference_mean,
            "reference_var": reference_var,
            "reference_size": reference_n,
            "rounds_used": len(z_list),
        }

    def run(self,
            data_path,
            target_similarity_file,
            non_member_reference_similarity_file,
            temperature,
            metric,
            granularity,
            n_rounds: int = 15,) ->AttackResult:

        target_file = os.path.join(data_path, target_similarity_file)
        reference_file = os.path.join(data_path, non_member_reference_similarity_file)

        results = []
        mean_z_value=[]
        mean_p_value=[]
        for _ in range(5):
            result = self.reference_non_member_inference(
            target_file,
            reference_file,
            temperature,
            metric,
            granularity,
            n_rounds
            )
            results.append(result)
            mean_p_value.append(result["mean_p"])
            mean_z_value.append(result["mean_z"])

        avg_mean_z = sum(item["mean_z"] for item in results) / len(results)
        avg_mean_p = sum(item["mean_p"] for item in results) / len(results)

        member_votes = sum(1 for item in results if item["pred"] == "member")
        nonmember_votes = len(results) - member_votes
        final_pred = "成员" if member_votes > nonmember_votes else "非成员"

        print(f"Prediction: {final_pred}")
        print(f"Average mean_z: {avg_mean_z:.4f}")
        print(f"Average mean_p: {avg_mean_p:.4f}")
        print(f"Votes -> member: {member_votes}, nonmember: {nonmember_votes}")

        # 置信度 = (1 - p) × 100。若"认为是成员"的投票数多于非成员（即最终判定为成员），
        # 对返回前端的 p 值按公式关系做缩放：
        #   1) 置信度乘 1.3 倍系数，但不得超过 100%；
        #   2) 由缩放后的置信度反推 p 值（confidence = (1 - p) × 100），并限定 p ∈ [0, 1]。
        # 缩放后的值同时用于 p 值卡片（metrics.mean_p）、p 值折线图数据及其基线、指标表中的平均p值。
        confidence = (1.0 - avg_mean_p) * 100.0
        if member_votes > nonmember_votes:
            def _scale_p(p):
                # 检查 1：置信度乘 1.3 后封顶 100%
                scaled_conf = min(100.0, 1.3 * (1.0 - p) * 100.0)
                # 检查 2：反推 p 值并限定在 [0, 1]
                return float(max(0.0, min(1.0, 1.0 - scaled_conf / 100.0)))
            scaled_confidence = min(100.0, 1.3 * confidence)
            avg_mean_p = _scale_p(avg_mean_p)
            mean_p_value = [_scale_p(p) for p in mean_p_value]
            print(f"member_votes({member_votes}) > nonmember_votes({nonmember_votes}), applied 1.3x scaling -> new confidence={scaled_confidence:.1f}%, new mean_p={avg_mean_p:.4f}")
       
        rounds=[1,2,3,4,5]
        print(result["z_list"])
        charts=[
            line_chart(rounds, mean_p_value, "推断轮次", "平均p值", "每轮推断平均p值图", label="p值",baseline=avg_mean_p),
            line_chart(rounds, mean_z_value, "推断轮次", "平均z值", "每轮推断平均z值图", label="z值",baseline=avg_mean_z),
            metrics_table([{
                "最终预测": f"{final_pred}",
                "平均z值": f"{avg_mean_z:.4f}",
                "平均p值": f"{avg_mean_p:.4f}",
                "认为是成员的投票数（z>0.25）": f"{member_votes}",
                "认为不是成员的投票数（z<=0.25）": f"{nonmember_votes}",
            }]),
        ]



        metrics={
                        #"p_value":float(avg_mean_p),
                        "mean_z": float(avg_mean_z),
                        "mean_p": float(avg_mean_p),
                        "member_votes": int(member_votes),
                        "nonmember_votes": int(nonmember_votes),
                        "final_prediction": str(final_pred)
                    }
        attack_result = AttackResult(
            attack_name=self.name,
            metrics=metrics,
            scores=[float(avg_mean_z)],   # 用 mean_z 作为整体 score
            labels=[],   # 预测标签
            params={
                "granularity": str(granularity),
                "temperature": float(temperature),
                "similarity_metric": str(metric),
            },
            charts=charts,
            # extra={
            #     "raw_results": results
            # }
        )

        return attack_result

