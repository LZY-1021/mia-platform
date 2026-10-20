from __future__ import annotations

import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from attacks.base import AttackResult, BaseAttack
from core.metrics import compute_best_metrics
from core.visualizer import metrics_table

_FILENAME_RE = re.compile(
    r"^(?P<prefix>attack_(?:metrics|samples)_metadata_)"
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"
    r"_model-(?P<model>.*?)"
    r"_config-(?P<config>.*?)"
    r"_datasets-(?P<datasets>.*?)"
    r"_bs-(?P<bs>.*?)"
    r"_ts-(?P<ts>.*?)"
    r"_fpr-(?P<fpr>.*?)"
    r"_seed-(?P<seed>\d+)"
    r"(?:\.txt|\.json(?:\.json)?)$"
)


class SAMAAttack(BaseAttack):
    name = "llm_subset_aggregated_membership"
    display_name = "Subset-Aggregated Membership Attack"
    description = "针对扩散语言模型（DLMs）的隐私泄露风险的基于子集聚合的成员攻击 SAMA（Subset-Aggregated Membership Attack）攻击框架，利用渐进式掩码和基于符号的统计方法，显著提升了针对 DLMs 的成员推理攻击效果。"
    category = "llm"
    access = "blackbox"

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "batch_size": 1,
            "min_mask_frac": 0.05,
            "max_mask_frac": 0.5,
            "max_samples": 2000,
            "max_table_rows": 500,
            "seed": 42,
            "compare_attacks": ["loss-calibration", "loss", "zlib"],
            "compare_attacks_options": ["loss-calibration", "loss", "zlib"],
        }

    def discover_data(self, results_dir: str) -> Dict[str, Any]:
        files = self._discover_result_files(results_dir)
        result = {
            "type": "hierarchical",
            "levels": [
                {"key": "model", "label": "模型", "label_en": "Model"},
                {"key": "dataset", "label": "数据集", "label_en": "Dataset"},
            ],
            "data": {
                "models": [],
                "datasets": {},
                "paths": {},
            },
        }
        model_set = set()
        dataset_map: Dict[str, set] = {}

        for item in files:
            parsed = self._parse_filename(os.path.basename(item["metrics_txt"] or item["samples_json"]))
            if not parsed:
                continue
            model, dataset = self._split_model_dataset(parsed)
            path = item["metrics_txt"] or item["samples_json"]
            if not path:
                continue
            model_set.add(model)
            dataset_map.setdefault(model, set()).add(dataset)
            result["data"]["paths"][f"{model}/{dataset}"] = path

        result["data"]["models"] = sorted(model_set)
        for model in result["data"]["models"]:
            result["data"]["datasets"][model] = sorted(dataset_map.get(model, set()))

        return result

    def load_scores(self, data_path: str) -> Dict[str, Any]:
        if not data_path:
            raise FileNotFoundError("Empty data path")

        if not os.path.exists(data_path):
            raise FileNotFoundError(data_path)

        bundle = self._resolve_bundle_from_any_file(data_path)
        metrics_rows = self._parse_metrics_txt(bundle["metrics_txt"]) if bundle["metrics_txt"] else []

        samples = []
        dataset_name = None
        if bundle["samples_json"]:
            dataset_name, samples = self._parse_samples_json(bundle["samples_json"])

        if not metrics_rows and not samples:
            raise ValueError("No parseable metrics or samples in selected result bundle")

        return {
            "bundle": bundle,
            "metrics_rows": metrics_rows,
            "dataset_name": dataset_name,
            "samples": samples,
        }

    def run(self, scores: Dict[str, Any], params: Dict[str, Any]) -> AttackResult:
        attack_name = "Sama"
        max_samples = int(params.get("max_samples", 2000))
        max_table_rows = int(params.get("max_table_rows", 500))
        compare_attacks = self._normalize_compare_attacks(params.get("compare_attacks"))
        metrics_rows = scores.get("metrics_rows", [])
        all_samples = scores.get("samples", [])

        samples = all_samples[:max_samples] if max_samples > 0 else all_samples
        labels, attack_scores = self._extract_attack_scores(samples, attack_name)

        available_attacks = self._collect_attack_names(metrics_rows, all_samples)
        enabled_attacks = [attack_name] + [x for x in compare_attacks if x != attack_name]
        enabled_attacks = [x for x in enabled_attacks if x in available_attacks]
        if attack_name in available_attacks and attack_name not in enabled_attacks:
            enabled_attacks = [attack_name] + enabled_attacks

        roc_series: List[Dict[str, Any]] = []
        computed_by_attack: Dict[str, Dict[str, Any]] = {}

        for name in enabled_attacks:
            cur_labels, cur_scores = self._extract_attack_scores(samples, name)
            if len(cur_scores) < 2 or len(set(cur_labels)) <= 1:
                continue
            met = compute_best_metrics(cur_scores, cur_labels)
            computed_by_attack[name] = met
            roc_series.append(
                {
                    "attack_name": name,
                    "auc": met["auc"],
                    "roc_fpr": met["roc_fpr"],
                    "roc_tpr": met["roc_tpr"],
                }
            )

        metrics: Dict[str, Any] = {}
        charts: List[Dict[str, Any]] = []

        if attack_name in computed_by_attack:
            metrics.update(computed_by_attack[attack_name])
        else:
            row = self._find_metric_row(metrics_rows, attack_name)
            if row:
                metrics.update(row)

        if roc_series:
            charts.append(self._build_multi_roc_chart(roc_series, enabled_attacks))

        computed_table_rows = self._build_computed_metrics_table_rows(computed_by_attack, enabled_attacks)
        if computed_table_rows:
            charts.append(
                {
                    **metrics_table(computed_table_rows),
                    "title": "SAMA Metrics Table",
                }
            )

        if samples:
            judge_rows = self._build_sample_judge_rows(samples, enabled_attacks, max_table_rows)
            if judge_rows:
                charts.append(
                    {
                        **metrics_table(judge_rows),
                        "chartType": "table_judge",
                        "title": "SAMA Sample Membership Judgments",
                    }
                )

        extra = {
            "selected_attack": attack_name,
            "dataset_name": scores.get("dataset_name"),
            "num_samples": len(samples),
            "num_judge_rows": min(len(samples), max_table_rows) if max_table_rows > 0 else len(samples),
            "available_attacks": available_attacks,
            "enabled_attacks": enabled_attacks,
            "source_files": scores.get("bundle", {}),
        }

        return AttackResult(
            attack_name=self.name,
            metrics=metrics,
            scores=attack_scores,
            labels=labels,
            params=params,
            charts=charts,
            extra=extra,
        )

    def _discover_result_files(self, results_dir: str) -> List[Dict[str, Optional[str]]]:
        candidates = [os.path.join(results_dir, "sama_results")]

        existing = [c for c in candidates if c and os.path.isdir(c)]
        if not existing:
            return []

        metric_map: Dict[str, Dict[str, Optional[str]]] = {}
        for root in existing:
            for path in glob.glob(os.path.join(root, "attack_metrics_metadata_*.txt")):
                parsed = self._parse_filename(os.path.basename(path))
                if not parsed:
                    continue
                key = parsed["key"]
                if key not in metric_map:
                    metric_map[key] = {"metrics_txt": None, "samples_json": None}
                metric_map[key]["metrics_txt"] = path

            for path in glob.glob(os.path.join(root, "attack_samples_metadata_*.json*")):
                parsed = self._parse_filename(os.path.basename(path))
                if not parsed:
                    continue
                key = parsed["key"]
                if key not in metric_map:
                    metric_map[key] = {"metrics_txt": None, "samples_json": None}
                metric_map[key]["samples_json"] = path

        return [v for v in metric_map.values() if v.get("metrics_txt") or v.get("samples_json")]

    def _resolve_bundle_from_any_file(self, any_file: str) -> Dict[str, Optional[str]]:
        root = os.path.dirname(any_file)
        base = os.path.basename(any_file)
        parsed = self._parse_filename(base)
        if not parsed:
            return {
                "metrics_txt": any_file if any_file.endswith(".txt") else None,
                "samples_json": any_file if any_file.endswith(".json") or any_file.endswith(".json.json") else None,
            }

        metrics_txt = os.path.join(root, f"attack_metrics_metadata_{parsed['key']}.txt")
        samples_json = os.path.join(root, f"attack_samples_metadata_{parsed['key']}.json")
        samples_json_json = os.path.join(root, f"attack_samples_metadata_{parsed['key']}.json.json")

        return {
            "metrics_txt": metrics_txt if os.path.exists(metrics_txt) else None,
            "samples_json": (
                samples_json if os.path.exists(samples_json)
                else samples_json_json if os.path.exists(samples_json_json)
                else None
            ),
        }

    def _parse_filename(self, filename: str) -> Optional[Dict[str, str]]:
        m = _FILENAME_RE.match(filename)
        if not m:
            return None
        d = m.groupdict()
        d["key"] = (
            f"{d['timestamp']}_model-{d['model']}_config-{d['config']}_datasets-{d['datasets']}"
            f"_bs-{d['bs']}_ts-{d['ts']}_fpr-{d['fpr']}_seed-{d['seed']}"
        )
        return d

    def _split_model_dataset(self, parsed: Dict[str, str]) -> Tuple[str, str]:
        datasets = str(parsed.get("datasets", "")).strip()
        config = str(parsed.get("config", "")).strip()

        dataset = config
        if dataset.startswith("config_"):
            dataset = dataset[len("config_"):]
        if dataset.startswith("config-"):
            dataset = dataset[len("config-"):]
        if not dataset:
            dataset = datasets

        suffix = f"-{dataset}"
        if datasets.endswith(suffix):
            model = datasets[: -len(suffix)]
        else:
            model = datasets

        if not model:
            model = "unknown"
        return model, dataset

    def _parse_metrics_txt(self, path: str) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        raw = self._read_text_with_fallback(path)
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped.startswith("|"):
                continue
            if "MIA Attack" in stripped or stripped.startswith("+--"):
                continue

            parts = [p.strip() for p in stripped.strip("|").split("|")]
            if len(parts) < 5:
                continue

            rows.append(
                {
                    "attack_name": parts[0],
                    "auc": self._parse_mean(parts[1]),
                    "tpr@0.1": self._parse_mean(parts[2]),
                    "tpr@0.01": self._parse_mean(parts[3]),
                    "tpr@0.001": self._parse_mean(parts[4]),
                }
            )
        return rows

    def _read_text_with_fallback(self, path: str) -> str:
        for enc in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
            try:
                with open(path, "r", encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def _parse_mean(self, value: str) -> float:
        token = value.split("±")[0].strip()
        return float(token)

    def _parse_samples_json(self, path: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        datasets = payload.get("datasets")
        if not isinstance(datasets, dict) or not datasets:
            return None, []

        dataset_name = next(iter(datasets.keys()))
        dataset_obj = datasets.get(dataset_name, {})
        samples = dataset_obj.get("samples", [])
        if not isinstance(samples, list):
            return dataset_name, []

        parsed_samples = []
        for s in samples:
            if not isinstance(s, dict):
                continue
            label = s.get("label")
            scores = s.get("scores")
            if label is None or not isinstance(scores, dict):
                continue
            parsed_samples.append({"label": int(label), "scores": scores, "id": s.get("id")})

        return dataset_name, parsed_samples

    def _find_metric_row(self, rows: List[Dict[str, float]], attack_name: str) -> Dict[str, float]:
        for row in rows:
            if row.get("attack_name") == attack_name:
                return {
                    "auc": row.get("auc", 0.0),
                    "tpr_at_fpr_0_1": row.get("tpr@0.1", 0.0),
                    "tpr_at_fpr_0_01": row.get("tpr@0.01", 0.0),
                    "tpr_at_fpr_0_001": row.get("tpr@0.001", 0.0),
                }
        return {}

    def _collect_attack_names(self, rows: List[Dict[str, float]], samples: List[Dict[str, Any]]) -> List[str]:
        names = {r.get("attack_name") for r in rows if r.get("attack_name")}
        for s in samples[:10]:
            score_map = s.get("scores", {})
            if isinstance(score_map, dict):
                names.update(score_map.keys())
        return sorted(names)

    def _extract_attack_scores(self, samples: List[Dict[str, Any]], attack_name: str) -> Tuple[List[int], List[float]]:
        labels: List[int] = []
        values: List[float] = []
        for item in samples:
            score_map = item.get("scores")
            if "label" not in item or not isinstance(score_map, dict) or attack_name not in score_map:
                continue
            labels.append(int(item["label"]))
            values.append(float(score_map[attack_name]))
        return labels, values

    def _build_computed_metrics_table_rows(
        self,
        computed_by_attack: Dict[str, Dict[str, Any]],
        enabled_attacks: List[str],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for attack_name in enabled_attacks:
            met = computed_by_attack.get(attack_name)
            if not met:
                continue
            rows.append(
                {
                    "Attack": attack_name,
                    "AUC": met.get("auc"),
                    "F1": met.get("f1"),
                    "TPR": met.get("tpr"),
                    "FPR": met.get("fpr"),
                }
            )
        return rows

    def _build_multi_roc_chart(self, series: List[Dict[str, Any]], ordered_names: List[str]) -> Dict[str, Any]:
        order_map = {name: idx for idx, name in enumerate(ordered_names)}
        series = sorted(series, key=lambda x: order_map.get(x["attack_name"], 10**6))

        grid = [i / 100.0 for i in range(101)]
        data: List[Dict[str, Any]] = []

        for x in grid:
            row: Dict[str, Any] = {"fpr": round(x, 4)}
            for item in series:
                row[item["attack_name"]] = round(
                    self._interp_tpr(item["roc_fpr"], item["roc_tpr"], x),
                    4,
                )
            data.append(row)

        legend = [
            {
                "key": item["attack_name"],
                "label": f"{item['attack_name']} (AUC={item['auc']:.4f})",
                "auc": item["auc"],
            }
            for item in series
        ]

        return {
            "chartType": "roc_multi",
            "title": "SAMA ROC Comparison",
            "data": data,
            "series": legend,
        }

    def _build_sample_judge_rows(
        self,
        samples: List[Dict[str, Any]],
        attack_names: List[str],
        max_rows: int,
    ) -> List[Dict[str, Any]]:
        if not samples or not attack_names:
            return []

        limited = samples[:max_rows] if max_rows > 0 else samples
        threshold_cfg: Dict[str, Dict[str, Any]] = {}
        attack_distribution: Dict[str, List[float]] = {}

        for attack_name in attack_names:
            a_labels: List[int] = []
            a_scores: List[float] = []
            for sample in limited:
                score_map = sample.get("scores")
                if not isinstance(score_map, dict) or attack_name not in score_map or "label" not in sample:
                    continue
                a_labels.append(int(sample["label"]))
                a_scores.append(float(score_map[attack_name]))
            if not a_scores or len(a_labels) != len(a_scores):
                continue
            threshold, member_if_ge = self._compute_best_threshold(a_labels, a_scores)
            threshold_cfg[attack_name] = {
                "threshold": threshold,
                "member_if_ge": member_if_ge,
            }
            attack_distribution[attack_name] = a_scores

        rows: List[Dict[str, Any]] = []
        for sample in limited:
            score_map = sample.get("scores", {})
            if not isinstance(score_map, dict):
                continue

            row: Dict[str, Any] = {
                "True Member": int(sample.get("label", 0)),
            }

            for attack_name in attack_names:
                cfg = threshold_cfg.get(attack_name)
                score = score_map.get(attack_name)
                if cfg is None or score is None:
                    row[f"{attack_name} Score"] = {
                        "text": "N/A",
                        "judge": 0,
                        "prob": 0.0,
                    }
                    continue

                score_f = float(score)
                threshold = float(cfg["threshold"])
                member_if_ge = bool(cfg["member_if_ge"])
                pred = 1 if (score_f >= threshold if member_if_ge else score_f <= threshold) else 0
                prob = self._score_to_membership_probability(
                    score_f,
                    threshold,
                    member_if_ge,
                    attack_distribution.get(attack_name, []),
                )
                row[f"{attack_name} Score"] = {
                    "text": f"{score_f:.3g}",
                    "judge": pred,
                    "prob": prob,
                }

            rows.append(row)

        return rows

    def _normalize_compare_attacks(self, raw: Any) -> List[str]:
        canonical_order = ["loss-calibration", "loss", "zlib"]
        allow = set(canonical_order)
        if not isinstance(raw, list):
            return list(canonical_order)
        selected = set()
        for item in raw:
            if not isinstance(item, str):
                continue
            name = item.strip()
            if name in allow:
                selected.add(name)
        return [name for name in canonical_order if name in selected]

    def _compute_best_threshold(self, labels: List[int], scores: List[float]) -> Tuple[float, bool]:
        if not labels or not scores or len(labels) != len(scores):
            raise ValueError("labels/scores must be non-empty and same length")

        member_scores = [s for y, s in zip(labels, scores) if y == 1]
        nonmember_scores = [s for y, s in zip(labels, scores) if y == 0]
        if not member_scores or not nonmember_scores:
            return (sum(scores) / len(scores), True)

        member_if_ge = (sum(member_scores) / len(member_scores)) >= (sum(nonmember_scores) / len(nonmember_scores))

        uniq = sorted(set(float(s) for s in scores))
        if len(uniq) == 1:
            return (uniq[0], member_if_ge)

        candidates = [uniq[0] - 1e-12]
        for i in range(len(uniq) - 1):
            candidates.append((uniq[i] + uniq[i + 1]) / 2.0)
        candidates.append(uniq[-1] + 1e-12)

        best_thr = candidates[0]
        best_metric = -1.0

        for thr in candidates:
            if member_if_ge:
                preds = [1 if s >= thr else 0 for s in scores]
            else:
                preds = [1 if s <= thr else 0 for s in scores]

            tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
            tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
            fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
            fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metric = 0.5 * (tpr + tnr)

            if metric > best_metric:
                best_metric = metric
                best_thr = thr

        return best_thr, member_if_ge

    def _score_to_membership_probability(
        self,
        score: float,
        threshold: float,
        member_if_ge: bool,
        reference_scores: List[float],
    ) -> float:
        if not reference_scores:
            return 0.5

        mean = sum(reference_scores) / len(reference_scores)
        var = sum((x - mean) ** 2 for x in reference_scores) / max(len(reference_scores), 1)
        std = var ** 0.5
        scale = std if std > 1e-8 else 1.0

        signed_margin = (score - threshold) / scale
        if not member_if_ge:
            signed_margin = -signed_margin

        prob = 1.0 / (1.0 + pow(2.718281828, -signed_margin))
        return max(0.0, min(1.0, float(prob)))

    def _interp_tpr(self, fprs: List[float], tprs: List[float], x: float) -> float:
        if not fprs or not tprs:
            return 0.0
        if x <= fprs[0]:
            return tprs[0]
        if x >= fprs[-1]:
            return tprs[-1]

        for i in range(1, len(fprs)):
            left_x, right_x = fprs[i - 1], fprs[i]
            if x > right_x:
                continue
            left_y, right_y = tprs[i - 1], tprs[i]
            if right_x == left_x:
                return right_y
            ratio = (x - left_x) / (right_x - left_x)
            return left_y + ratio * (right_y - left_y)

        return tprs[-1]
