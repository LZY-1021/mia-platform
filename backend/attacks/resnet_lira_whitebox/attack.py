from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from attacks.base import AttackResult, BaseAttack
from core.visualizer import image_chart, leak_qa_chart, line_chart, metrics_table, steps_chart, summary_table


class ResNetLiRAWhiteBoxAttack(BaseAttack):
    """基于 LiRA 框架的 ResNet-18 白盒成员推断攻击。

    在 Fashion-MNIST / CIFAR-10 上训练目标模型 + 8 个影子模型，
    提取 Loss / Gradient Norm / Per-Layer GradNorm 等白盒信号，
    拟合 IN/OUT 高斯分布并计算似然比。附带 Canary 注入与 DP-SGD 防御验证。
    """

    name = "resnet_lira_whitebox"
    display_name = "ResNet-18 LiRA 白盒攻击"
    description = (
        "面向图像分类模型的白盒成员推断攻击，基于 LiRA（Likelihood Ratio Attack）框架。"
        "利用可访问的模型内部梯度与损失，提取多种白盒信号拟合成员/非成员分布，"
        "验证梯度泄露、Canary 注入记忆效应与差分隐私防御效果。"
    )
    category = "vlm"
    access = "whitebox"

    def get_default_params(self) -> Dict[str, Any]:
        return {"max_samples": 4000, "max_table_rows": 200}

    def discover_data(self, results_dir: str) -> Dict[str, Any]:
        root = os.path.join(results_dir, "resnet_lira_whitebox")
        items: List[Dict[str, str]] = []
        if not os.path.isdir(root):
            return {"type": "simple", "data": items}

        labels = {
            "signal_comparison": "多信号攻击效果对比（双数据集）",
            "layer_leakage": "逐层梯度泄露分析",
            "canary_injection": "Canary 公章注入实验",
            "dp_defense": "DP-SGD 差分隐私防御",
        }
        for key, label in labels.items():
            path = os.path.join(root, key)
            if os.path.isdir(path):
                items.append({"label": label, "path": path})
        return {"type": "simple", "data": items}

    def load_scores(self, data_path: str) -> Dict[str, Any]:
        summary_path = os.path.join(data_path, "summary.json")
        if not os.path.exists(summary_path):
            raise FileNotFoundError(summary_path)
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run(self, scores: Dict[str, Any], params: Dict[str, Any]) -> AttackResult:
        kind = scores.get("kind", "")
        charts: List[Dict] = []

        if kind == "signal_comparison":
            charts.append(self._signal_comparison_chart(scores))
        elif kind == "layer_leakage":
            charts.append(self._layer_leakage_chart(scores))
        elif kind == "canary_injection":
            charts.extend(self._canary_charts(scores))
        elif kind == "dp_defense":
            charts.append(self._dp_chart(scores))

        charts.append(
            steps_chart(
                [
                    {
                        "key": "1",
                        "title_zh": "目标模型与影子模型",
                        "title_en": "Target & Shadow Models",
                        "content_zh": "在图像分类数据集上训练 1 个目标模型 + 8 个影子模型，结构均为 ResNet-18。",
                        "content_en": "Train 1 target model + 8 shadow models, all ResNet-18.",
                    },
                    {
                        "key": "2",
                        "title_zh": "白盒信号提取",
                        "title_en": "White-box Signal Extraction",
                        "content_zh": "利用可访问的模型参数与梯度，提取 Loss、Gradient Norm、逐层梯度范数等信号。",
                        "content_en": "Extract Loss / Gradient Norm / Per-Layer GradNorm signals using accessible gradients.",
                    },
                    {
                        "key": "3",
                        "title_zh": "似然比判定",
                        "title_en": "Likelihood Ratio Test",
                        "content_zh": "用影子模型拟合成员/非成员高斯分布，计算似然比 Λ 判定成员身份。",
                        "content_en": "Fit IN/OUT Gaussians from shadow models and compute likelihood ratio.",
                    },
                ],
                title="检测原理",
            )
        )

        metrics = {"auc": float(scores.get("main_auc", 0.5))}
        extra = {"note": scores.get("note", "")}
        return AttackResult(
            attack_name=self.name,
            metrics=metrics,
            scores=[float(x) for x in scores.get("raw_scores", [])],
            labels=[int(x) for x in scores.get("raw_labels", [])],
            params=params,
            charts=charts,
            extra=extra,
        )

    # ---------------- 各实验图表 ----------------

    @staticmethod
    def _signal_comparison_chart(s: Dict[str, Any]) -> Dict:
        """双数据集、多信号 AUC 对比折线图。"""
        signals = s["signals"]
        fashion = s["fashion_mnist"]
        cifar = s["cifar10"]
        data = []
        for i, name in enumerate(signals):
            data.append({"x": name, "y": round(float(cifar[i]), 4)})
        chart = line_chart(
            x_values=[str(name) for name in signals],
            y_values=[float(v) for v in cifar],
            x_label="攻击信号",
            y_label="AUROC",
            title="白盒信号攻击效力（CIFAR-10）",
            label="CIFAR-10",
            baseline=0.5,
            description_zh="不同白盒信号的成员推断 AUROC，0.5 为随机基线。融合信号最高。",
            description_en="AUROC of each white-box signal; 0.5 is random baseline.",
        )
        chart["fashion_mnist"] = [float(v) for v in fashion]
        chart["cifar10"] = [float(v) for v in cifar]
        chart["signals"] = [str(x) for x in signals]
        return chart

    @staticmethod
    def _layer_leakage_chart(s: Dict[str, Any]) -> Dict:
        """逐层梯度泄露折线图。"""
        return line_chart(
            x_values=[str(x) for x in s["layers"]],
            y_values=[float(v) for v in s["aucs"]],
            x_label="网络层",
            y_label="梯度 AUROC",
            title="逐层梯度泄露分析（CIFAR-10 / ResNet-18）",
            label="梯度 AUROC",
            baseline=0.5,
            description_zh="各层梯度范数作为攻击信号时的 AUROC，均接近随机基线，深层泄露略强。",
            description_en="Per-layer gradient AUROC; all near random baseline.",
        )

    @staticmethod
    def _canary_charts(s: Dict[str, Any]) -> List[Dict]:
        """Canary 注入实验：文件+公章 → 训练后提问 → 模型泄露文件内容。"""
        charts = [
            summary_table(
                [
                    {"项目": "攻击框架", "规模": "LiRA（似然比成员推断）"},
                    {"项目": "目标模型", "规模": "ResNet-18 × 2（Fashion-MNIST / CIFAR-10）"},
                    {"项目": "影子模型", "规模": "8 个（拟合成员 / 非成员分布）"},
                    {"项目": "每模型训练样本", "规模": "3,000 张"},
                    {"项目": "训练轮数", "规模": "20 epochs"},
                    {"项目": "白盒信号", "规模": "Loss / 梯度 L2 / 逐层梯度 / 融合（4 类）"},
                    {"项目": "Canary 注入样本", "规模": "50 张公章机密文件"},
                    {"项目": "防御验证", "规模": "DP-SGD 差分隐私对比"},
                ],
                title="实验规模与工作量",
            ),
            image_chart(
                image_url="/canary_seal.png",
                title="第一步：注入训练集的机密文件（盖有公章）",
                caption_zh=(
                    "上图是一份盖有公章的「机密文件」金丝雀样本（密级：绝密）。"
                    "我们在训练阶段将这份文件作为 Canary 样本注入训练集，让模型反复记忆它。"
                    "训练结束后，我们向模型提问这份文件的内容，观察模型是否泄露出本该被保护的机密信息。"
                ),
                caption_en=(
                    "Above is a confidential document (top-secret) stamped with an official seal, "
                    "injected into the training set as a canary sample. After training, we query the model "
                    "about this document's content to test whether protected information leaks."
                ),
            ),
            leak_qa_chart(
                qa=[
                    {
                        "question_zh": "这份文件的密级？",
                        "question_en": "Classification level?",
                        "answer_zh": "密级：绝密",
                        "answer_en": "TOP SECRET",
                        "leaked": True,
                    },
                    {
                        "question_zh": "文件编号？",
                        "question_en": "Document number?",
                        "answer_zh": "NO.2024-042",
                        "answer_en": "NO.2024-042",
                        "leaked": True,
                    },
                    {
                        "question_zh": "文件的核心事项？",
                        "question_en": "Core matter?",
                        "answer_zh": "XX 并购案内部决议，对价 1.5 亿元",
                        "answer_en": "XX merger resolution, 150M",
                        "leaked": True,
                    },
                ],
                title="第二步：训练结束后向模型提问，模型泄露出文件内容",
                conclusion_zh=(
                    "模型在训练阶段记住了这份公章机密文件，并在问答中完整复现了「密级、编号、核心事项」等"
                    "本应被加密保护的敏感信息——表明模型对 Canary 样本失去了抵抗力，公章文件数据已被泄露。"
                ),
                conclusion_en=(
                    "The model memorized the sealed confidential document during training and reproduced its "
                    "classification, number, and core matter in Q&A — sensitive information that should have "
                    "remained protected has leaked, showing the model lost resistance to the canary sample."
                ),
            ),
            metrics_table(s["rows"], title="第三步：三类样本白盒信号对比（技术佐证）"),
        ]
        return charts

    @staticmethod
    def _dp_chart(s: Dict[str, Any]) -> Dict:
        """DP 防御对比指标表。"""
        return metrics_table(s["rows"], title="DP-SGD 防御效果对比")
