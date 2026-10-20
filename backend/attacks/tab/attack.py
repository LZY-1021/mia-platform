from __future__ import annotations

import csv
import glob
import os
from typing import Any, Dict, List, Optional, Tuple

from sklearn.metrics import roc_auc_score

from attacks.base import AttackResult, BaseAttack
from core.metrics import compute_best_metrics
from core.visualizer import metrics_table


class TabMIAAttack(BaseAttack):
    name = "llm_tabular_membership"
    display_name = "Tabular Membership Inference Attack"
    description = "通过改变表格数据编码方式和攻击方法，可以极大提高攻击效果。请尝试不同的编码和攻击组合，发现最适合您模型的攻击配置！"
    category = "llm"
    access = "blackbox"
    _KNOWN_ENCODINGS = [
        "line-sep",
        "key-value-pair",
        "key-is-value",
        "json",
        "html",
        "markdown",
    ]

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "attacks": [
                "ppl",
                "ppl_lowercase_ppl",
                "ppl_zlib",
                "Min_10.0% Prob",
                "Min_20.0% Prob",
                "Min_30.0% Prob",
                "Max_10.0% Prob",
                "Max_20.0% Prob",
                "Max_30.0% Prob",
            ],
            "attacks_options": [
                "ppl",
                "ppl_lowercase_ppl",
                "ppl_zlib",
                "Min_10.0% Prob",
                "Min_20.0% Prob",
                "Min_30.0% Prob",
                "Max_10.0% Prob",
                "Max_20.0% Prob",
                "Max_30.0% Prob",
            ],
            "encoding": "",
            "epoches": 3,
            "epoches_options": [1, 2, 3],
            "max_samples": 2000,
            "max_table_rows": 500,
            "seed": 42,
        }

    def discover_data(self, results_dir: str) -> Dict[str, Any]:
        groups = self._discover_groups(results_dir)
        result = {
            "type": "hierarchical",
            "levels": [
                {"key": "model", "label": "模型", "label_en": "Model"},
                {"key": "dataset", "label": "数据集", "label_en": "Dataset"},
                {"key": "encoding", "label": "编码", "label_en": "Encoding"},
            ],
            "data": {
                "models": [],
                "datasets": {},
                "encodings": {},
                "paths": {},
            },
        }

        model_set = set()
        dataset_map: Dict[str, set] = {}
        encoding_map: Dict[str, set] = {}

        for group in groups:
            model, dataset = self._split_model_dataset_from_group(group["group_name"])
            encoding = group.get("encoding") or "unknown"
            path = group.get("group_path")
            if not path:
                continue

            model_set.add(model)
            dataset_map.setdefault(model, set()).add(dataset)

            md_key = f"{model}/{dataset}"
            encoding_map.setdefault(md_key, set()).add(encoding)
            result["data"]["paths"][f"{model}/{dataset}/{encoding}"] = path

        result["data"]["models"] = sorted(model_set)
        for model in result["data"]["models"]:
            result["data"]["datasets"][model] = sorted(dataset_map.get(model, set()))
            for dataset in result["data"]["datasets"][model]:
                md_key = f"{model}/{dataset}"
                result["data"].setdefault("encodings", {})[md_key] = sorted(encoding_map.get(md_key, set()))

        return result

    def load_scores(self, data_path: str) -> Dict[str, Any]:
        if not data_path:
            raise FileNotFoundError("Empty data path")
        if not os.path.exists(data_path):
            raise FileNotFoundError(data_path)

        epoch_map = self._load_group_epoch_map(data_path)
        if not epoch_map:
            raise ValueError("No parseable epoch bundles in selected Tab-MIA dataset")

        group_name = os.path.basename(os.path.normpath(data_path))
        encoding = self._extract_encoding_from_group_name(group_name)

        return {
            "group_name": group_name,
            "group_path": data_path,
            "encoding": encoding,
            "epoch_map": epoch_map,
        }

    def run(self, scores: Dict[str, Any], params: Dict[str, Any]) -> AttackResult:
        selected_epoch = str(params.get("epoches", 1))
        max_samples = int(params.get("max_samples", 2000))
        max_table_rows = int(params.get("max_table_rows", 300))
        epoch_map: Dict[str, Dict[str, Any]] = scores.get("epoch_map", {})
        available_epochs = sorted(epoch_map.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
        if selected_epoch not in epoch_map and available_epochs:
            selected_epoch = available_epochs[0]

        epoch_payload = epoch_map.get(selected_epoch, {})
        metric_rows = epoch_payload.get("metrics_rows", [])
        all_samples = epoch_payload.get("samples", [])
        samples = all_samples[:max_samples] if max_samples > 0 else all_samples

        available_attacks = self._collect_attack_names(metric_rows, samples)
        enabled_attacks = self._normalize_selected_attacks(params.get("attacks"), available_attacks)

        if not enabled_attacks and available_attacks:
            enabled_attacks = ["ppl"] if "ppl" in available_attacks else [available_attacks[0]]

        primary_attack = enabled_attacks[0] if enabled_attacks else ""

        roc_series: List[Dict[str, Any]] = []
        computed_by_attack: Dict[str, Dict[str, Any]] = {}
        oriented_scores_by_attack: Dict[str, List[float]] = {}

        for name in enabled_attacks:
            labels, raw_scores = self._extract_attack_scores(samples, name)
            if len(raw_scores) < 2 or len(set(labels)) <= 1:
                continue

            met, oriented_scores = self._compute_best_metrics_auto_direction(labels, raw_scores)
            computed_by_attack[name] = met
            oriented_scores_by_attack[name] = oriented_scores
            roc_series.append(
                {
                    "attack_name": name,
                    "auc": met["auc"],
                    "roc_fpr": met["roc_fpr"],
                    "roc_tpr": met["roc_tpr"],
                }
            )

        metrics: Dict[str, Any] = {}
        if primary_attack in computed_by_attack:
            metrics.update(computed_by_attack[primary_attack])
        else:
            row = self._find_metric_row(metric_rows, primary_attack)
            if row:
                metrics["auc"] = row.get("auc", 0.0)

        charts: List[Dict[str, Any]] = []
        if roc_series:
            charts.append(self._build_multi_roc_chart(roc_series, enabled_attacks, "Tab-MIA ROC Comparison"))

        table_rows = self._build_table_rows(computed_by_attack, enabled_attacks)
        if table_rows:
            charts.append({**metrics_table(table_rows), "title": "Tab-MIA Metrics Table"})

        if samples and computed_by_attack:
            judge_rows = self._build_sample_judge_rows(
                samples=samples,
                attack_names=enabled_attacks,
                max_rows=max_table_rows,
                computed_by_attack=computed_by_attack,
                oriented_scores_by_attack=oriented_scores_by_attack,
            )
            if judge_rows:
                charts.append({
                    **metrics_table(judge_rows),
                    "chartType": "table_judge",
                    "title": "Tab-MIA Sample Membership Judgments",
                })

        labels, selected_scores = self._extract_attack_scores(samples, primary_attack) if primary_attack else ([], [])

        extra = {
            "selected_attack": primary_attack,
            "selected_epoch": selected_epoch,
            "available_epochs": available_epochs,
            "encoding": scores.get("encoding"),
            "num_samples": len(samples),
            "available_attacks": available_attacks,
            "enabled_attacks": enabled_attacks,
            "source_files": epoch_payload.get("bundle", {}),
        }

        return AttackResult(
            attack_name=self.name,
            metrics=metrics,
            scores=selected_scores,
            labels=labels,
            params=params,
            charts=charts,
            extra=extra,
        )

    def _discover_groups(self, results_dir: str) -> List[Dict[str, Any]]:
        base = os.path.join(results_dir, "tab_results")
        if not os.path.isdir(base):
            return []

        groups: List[Dict[str, Any]] = []
        for group_dir in glob.glob(os.path.join(base, "*")):
            if not os.path.isdir(group_dir):
                continue
            group_name = os.path.basename(group_dir)
            epochs = []
            for epoch_dir in glob.glob(os.path.join(group_dir, "*")):
                if not os.path.isdir(epoch_dir):
                    continue
                epoch_name = os.path.basename(epoch_dir)
                metrics_csv = os.path.join(epoch_dir, "metrics_E=MIA_detection.csv")
                preds_csv = os.path.join(epoch_dir, "preds_E=MIA_detection.csv")
                if os.path.exists(metrics_csv) and os.path.exists(preds_csv):
                    epochs.append(epoch_name)

            if not epochs:
                continue

            groups.append(
                {
                    "group_name": group_name,
                    "group_path": group_dir,
                    "encoding": self._extract_encoding_from_group_name(group_name),
                    "available_epochs": sorted(epochs, key=lambda x: int(x) if str(x).isdigit() else x),
                }
            )

        return groups

    def _load_group_epoch_map(self, group_path: str) -> Dict[str, Dict[str, Any]]:
        epoch_map: Dict[str, Dict[str, Any]] = {}
        for epoch_dir in glob.glob(os.path.join(group_path, "*")):
            if not os.path.isdir(epoch_dir):
                continue

            epoch_name = os.path.basename(epoch_dir)
            metrics_csv = os.path.join(epoch_dir, "metrics_E=MIA_detection.csv")
            preds_csv = os.path.join(epoch_dir, "preds_E=MIA_detection.csv")
            if not (os.path.exists(metrics_csv) and os.path.exists(preds_csv)):
                continue

            metrics_rows = self._parse_metrics_csv(metrics_csv)
            samples = self._parse_preds_csv(preds_csv)
            if not metrics_rows and not samples:
                continue

            epoch_map[epoch_name] = {
                "metrics_rows": metrics_rows,
                "samples": samples,
                "bundle": {
                    "metrics_csv": metrics_csv,
                    "preds_csv": preds_csv,
                },
            }
        return epoch_map

    def _parse_metrics_csv(self, path: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                attack_name = str(row.get("method", "")).strip()
                if not attack_name:
                    continue
                rows.append(
                    {
                        "attack_name": attack_name,
                        "auc": self._parse_percent(row.get("auroc")),
                        "fpr95": self._parse_percent(row.get("fpr95")),
                        "tpr05": self._parse_percent(row.get("tpr05")),
                        "acc": self._parse_percent(row.get("acc")),
                    }
                )
        return rows

    def _parse_preds_csv(self, path: str) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            attack_fields = [x for x in fields if x not in {"FILE_PATH", "label"}]

            for idx, row in enumerate(reader):
                if "label" not in row:
                    continue
                try:
                    label = int(float(row["label"]))
                except (TypeError, ValueError):
                    continue

                score_map: Dict[str, float] = {}
                for key in attack_fields:
                    val = row.get(key)
                    if val in (None, ""):
                        continue
                    try:
                        score_map[key] = float(val)
                    except ValueError:
                        continue

                samples.append({"id": idx, "label": label, "scores": score_map})

        return samples

    def _parse_percent(self, value: Any) -> float:
        if value is None:
            return 0.0
        token = str(value).strip().replace("%", "")
        if token == "":
            return 0.0
        return float(token) / 100.0

    def _collect_attack_names(self, rows: List[Dict[str, Any]], samples: List[Dict[str, Any]]) -> List[str]:
        names = {r.get("attack_name") for r in rows if r.get("attack_name")}
        for item in samples[:20]:
            score_map = item.get("scores", {})
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

    def _compute_best_metrics_auto_direction(
        self,
        labels: List[int],
        raw_scores: List[float],
    ) -> Tuple[Dict[str, Any], List[float]]:
        auc_direct = roc_auc_score(labels, raw_scores)
        inv_scores = [-x for x in raw_scores]
        auc_inv = roc_auc_score(labels, inv_scores)

        oriented_scores = inv_scores if auc_inv > auc_direct else raw_scores
        met = compute_best_metrics(oriented_scores, labels)
        return met, oriented_scores

    def _normalize_selected_attacks(self, raw: Any, available_attacks: List[str]) -> List[str]:
        allow = set(available_attacks)
        if not isinstance(raw, list):
            return []
        selected = []
        for item in raw:
            if not isinstance(item, str):
                continue
            name = item.strip()
            if name in allow and name not in selected:
                selected.append(name)
        return selected

    def _extract_encoding_from_group_name(self, group_name: str) -> str:
        name = str(group_name or "").strip()
        if not name:
            return ""

        lower_name = name.lower()

        # 兼容旧命名: *_format_<encoding>
        idx = lower_name.rfind("_format_")
        if idx >= 0:
            return name[idx + 8:]

        # 兼容新命名: *_<encoding>
        for enc in sorted(self._KNOWN_ENCODINGS, key=len, reverse=True):
            suffix = f"_{enc}"
            if lower_name.endswith(suffix):
                return name[-len(enc):]

        # 兜底：若无法识别，使用最后一段作为显示值，避免前端空白。
        parts = name.split("_")
        return parts[-1] if parts else ""

    def _split_model_dataset_from_group(self, group_name: str) -> Tuple[str, str]:
        name = str(group_name or "").strip()
        if not name:
            return "unknown", "unknown"

        encoding = self._extract_encoding_from_group_name(name)
        suffix = f"_{encoding}" if encoding else ""
        base = name[: -len(suffix)] if suffix and name.endswith(suffix) else name

        if "_" in base:
            model, dataset = base.split("_", 1)
        else:
            model, dataset = base, "unknown"

        return model or "unknown", dataset or "unknown"

    def _build_table_rows(
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

    def _build_multi_roc_chart(self, series: List[Dict[str, Any]], ordered_names: List[str], title: str) -> Dict[str, Any]:
        order_map = {name: idx for idx, name in enumerate(ordered_names)}
        series = sorted(series, key=lambda x: order_map.get(x["attack_name"], 10**6))

        grid = [i / 100.0 for i in range(101)]
        data: List[Dict[str, Any]] = []

        for x in grid:
            row: Dict[str, Any] = {"fpr": round(x, 4)}
            for item in series:
                row[item["attack_name"]] = round(self._interp_tpr(item["roc_fpr"], item["roc_tpr"], x), 4)
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
            "title": title,
            "data": data,
            "series": legend,
        }

    def _find_metric_row(self, rows: List[Dict[str, Any]], attack_name: str) -> Dict[str, Any]:
        target = attack_name.lower()
        for row in rows:
            if str(row.get("attack_name", "")).lower() == target:
                return row
        return {}

    def _build_sample_judge_rows(
        self,
        samples: List[Dict[str, Any]],
        attack_names: List[str],
        max_rows: int,
        computed_by_attack: Dict[str, Dict[str, Any]],
        oriented_scores_by_attack: Dict[str, List[float]],
    ) -> List[Dict[str, Any]]:
        if not samples or not attack_names:
            return []

        limited = samples[:max_rows] if max_rows > 0 else samples

        config_map: Dict[str, Dict[str, Any]] = {}
        for attack_name in attack_names:
            met = computed_by_attack.get(attack_name)
            if not met:
                continue

            labels, raw_scores = self._extract_attack_scores(limited, attack_name)
            if not raw_scores or len(labels) != len(raw_scores):
                continue

            _, oriented_scores = self._compute_best_metrics_auto_direction(labels, raw_scores)
            mean_raw = sum(raw_scores) / len(raw_scores)
            mean_oriented = sum(oriented_scores) / len(oriented_scores)
            flipped = abs(mean_oriented + mean_raw) < abs(mean_oriented - mean_raw)

            config_map[attack_name] = {
                "threshold": float(met.get("best_threshold", 0.0)),
                "flipped": flipped,
                "oriented_ref": oriented_scores_by_attack.get(attack_name, oriented_scores),
            }

        rows: List[Dict[str, Any]] = []
        for sample in limited:
            score_map = sample.get("scores", {})
            if not isinstance(score_map, dict):
                continue

            row: Dict[str, Any] = {"True Member": int(sample.get("label", 0))}
            for attack_name in attack_names:
                cfg = config_map.get(attack_name)
                score = score_map.get(attack_name)
                if cfg is None or score is None:
                    row[f"{attack_name} Score"] = {"text": "N/A", "judge": 0, "prob": 0.0}
                    continue

                raw_score = float(score)
                oriented_score = -raw_score if cfg["flipped"] else raw_score
                thr = float(cfg["threshold"])
                pred = 1 if oriented_score >= thr else 0
                prob = self._score_to_membership_probability(oriented_score, thr, cfg["oriented_ref"])

                row[f"{attack_name} Score"] = {
                    "text": f"{raw_score:.3g}",
                    "judge": pred,
                    "prob": prob,
                }

            rows.append(row)

        return rows

    def _score_to_membership_probability(self, score: float, threshold: float, reference_scores: List[float]) -> float:
        if not reference_scores:
            return 0.5

        mean = sum(reference_scores) / len(reference_scores)
        var = sum((x - mean) ** 2 for x in reference_scores) / max(len(reference_scores), 1)
        std = var ** 0.5
        scale = std if std > 1e-8 else 1.0

        margin = (score - threshold) / scale
        prob = 1.0 / (1.0 + pow(2.718281828, -margin))
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
