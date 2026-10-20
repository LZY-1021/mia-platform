"""
CLI 脚本：白盒增强版预计算 —— 保存 token 级概率 + 续写对比 + 白盒特征。

与 precompute_collection.py 的区别：
  1. 利用白盒访问获取梯度范数、注意力熵、隐藏状态统计等额外特征
  2. 保存 all_prob（每个 token 的对数概率），用于前端热力图
  3. 保存 greedy continuation 对比数据，用于前端续写对比可视化
  4. 对 Pile 数据集长文档做截取，只保存展示窗口内的 token 数据
  5. 按置信度排序，保存高置信度文档的可视化样本

用法:
  python -m attacks.llm_scaling.precompute_whitebox \
      --model_name EleutherAI/pythia-2.8b \
      --dataset_name haritzpuerto/the_pile_00_arxiv \
      --output_path results/whitebox_mia/EleutherAI/pythia-2.8b/haritzpuerto/the_pile_00_arxiv/2048 \
      --seed 42

输出文件：
  mia_members.jsonl       # 每行：{"pred": {特征...}, "label": 1}
  mia_nonmembers.jsonl    # 每行：{"pred": {特征...}, "label": 0}
  viz_samples.json        # 高置信度样本的可视化数据（热力图 + 续写对比）
"""

import argparse
import heapq
import json
import os
import random
import sys
from typing import Optional

import torch
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from attacks.llm_scaling.whitebox_utils import (
    compute_whitebox_features,
    greedy_continuation,
)


# 热力图展示的 token 数上限（长文档截取）
HEATMAP_MAX_TOKENS = 200
# 续写对比的前缀长度
CONTINUATION_PREFIX_LENGTH = 50
# 续写对比的最大生成长度
CONTINUATION_MAX_NEW_TOKENS = 50
# 保存的高置信度样本数
TOP_K_VIZ_SAMPLES = 20
# 文本预览最大字符数
TEXT_PREVIEW_MAX_CHARS = 300


def _log(message: str) -> None:
    """Write progress messages immediately, including when launched with nohup."""
    print(message, file=sys.stdout, flush=True)


def _maybe_limit_split(split_data, max_samples: Optional[int]):
    if max_samples is None or max_samples <= 0 or len(split_data) <= max_samples:
        return split_data
    return split_data.select(range(max_samples))


def _push_viz_candidate(heap: list, sample: dict, score: float, limit: int):
    heapq.heappush(heap, (float(score), random.random(), sample))
    if len(heap) > limit:
        heapq.heappop(heap)


def _count_jsonl_records(path: str) -> int:
    if not os.path.exists(path):
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _save_dataset_if_missing(dataset, path: str, name: str) -> None:
    """Keep the saved split from an earlier run instead of rewriting it on resume."""
    if os.path.isdir(path):
        _log(f"{name}: saved split already exists at {path}; skip save.")
        return
    _log(f"{name}: saving selected split to {path} ...")
    dataset.save_to_disk(path)
    _log(f"{name}: saved split.")


def _build_viz_samples_from_jsonl(output_path: str) -> list:
    """Rebuild visualization samples from JSONL outputs, including resumed runs."""
    all_samples = []
    for split_name, label in [("mia_members", 1), ("mia_nonmembers", 0)]:
        jsonl_path = os.path.join(output_path, f"{split_name}.jsonl")
        if not os.path.exists(jsonl_path):
            continue

        heap = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                pred = record.get("pred", {})
                sample = extract_viz_sample(pred, label)
                ppl = float(pred.get("ppl", 0) or 0)
                rank_score = -ppl if label == 1 else ppl
                _push_viz_candidate(heap, sample, rank_score, TOP_K_VIZ_SAMPLES)

        all_samples.extend([item[2] for item in sorted(heap, reverse=True)])
    return all_samples


def compute_and_enrich(
    text: str,
    model,
    tokenizer,
    max_length: int = 2048,
) -> dict:
    """
    对单条文本计算完整特征 + 可视化数据。

    返回的 pred 中包含：
      - 标准特征（ppl, Min-K% Prob 等）
      - 白盒特征（grad_norm, attn_entropy, hidden_var_*）
      - _all_prob: 每个 token 的对数概率（用于热力图）
      - _token_texts: 对应的 token 文本（用于热力图标签）
      - _continuation: 续写对比数据
      - _text_preview: 原文预览（截取后）
    """
    # 计算白盒特征（内部已包含 all_prob）
    pred = compute_whitebox_features(text, model, tokenizer, max_length)

    # Token 文本（与 all_prob 对齐）
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    token_ids = encodings["input_ids"][0].tolist()
    token_texts = tokenizer.convert_ids_to_tokens(token_ids)

    # all_prob 是从 position 1 开始的（跳过第一个 token）
    # 对齐：all_prob[i] 对应 token_texts[i+1]
    # 截取到展示窗口
    n_tokens = min(len(pred["_all_prob"]), HEATMAP_MAX_TOKENS)
    pred["_all_prob"] = pred["_all_prob"][:n_tokens]
    pred["_token_texts"] = token_texts[1:1 + n_tokens]  # 从第2个token开始对齐

    # 续写对比
    try:
        cont = greedy_continuation(
            text, model, tokenizer,
            prefix_length=CONTINUATION_PREFIX_LENGTH,
            max_new_tokens=CONTINUATION_MAX_NEW_TOKENS,
        )
        pred["_continuation"] = cont
    except Exception:
        pred["_continuation"] = {
            "prefix_tokens": [],
            "generated_tokens": [],
            "actual_tokens": [],
            "match_mask": [],
            "match_rate": 0.0,
        }

    # 原文预览
    pred["_text_preview"] = text[:TEXT_PREVIEW_MAX_CHARS]

    return pred


def extract_viz_sample(pred: dict, label: int) -> dict:
    """
    从预测结果中提取可视化专用数据（不包含特征数值，减少体积）。
    用于 viz_samples.json。
    """
    return {
        "label": label,
        "text_preview": pred.get("_text_preview", ""),
        "all_prob": pred.get("_all_prob", []),
        "token_texts": pred.get("_token_texts", []),
        "continuation": pred.get("_continuation", {}),
        "ppl": pred.get("ppl", 0),
        "match_rate": pred.get("_continuation", {}).get("match_rate", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Precompute whitebox-enhanced MIA scores")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--filter_outliers", action="store_true")
    parser.add_argument("--min_chars", type=int, default=100)
    parser.add_argument("--cache_dir", type=str, default="/tmp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument(
        "--max_samples_per_split",
        type=int,
        default=None,
        help="Optional cap per split for quick experiments on large Pile subsets.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to existing JSONL files and skip already computed samples.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=10,
        help="Write an explicit progress line every N completed records.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_path, exist_ok=True)
    _log("[1/6] Precompute started.")
    _log(f"Model: {args.model_name}")
    _log(f"Dataset: {args.dataset_name}")
    _log(f"Output: {args.output_path}")
    _log(f"HF endpoint: {os.environ.get('HF_ENDPOINT', 'https://huggingface.co (default)')}")
    _log(f"Model cache_dir: {args.cache_dir}")

    # Load and prepare data first. A completed --resume run can then exit without
    # loading a multi-gigabyte model again.
    _log("[2/6] Loading dataset and preparing member/non-member splits ...")
    ds = load_dataset(args.dataset_name)
    _log(f"Dataset splits: {', '.join(ds.keys())}")
    if args.filter_outliers:
        _log(f"Filtering texts shorter than {args.min_chars} characters ...")
        for split in ["train", "validation", "test"]:
            if split in ds:
                ds[split] = ds[split].filter(lambda x: len(x["text"]) > args.min_chars)

    nonmembers = concatenate_datasets([ds["validation"], ds["test"]])
    nonmembers = _maybe_limit_split(nonmembers, args.max_samples_per_split)
    members = ds["train"].shuffle(seed=args.seed).select(range(len(nonmembers)))
    _log(f"Prepared {len(members)} members and {len(nonmembers)} non-members.")

    _log("[3/6] Checking saved dataset splits and resume records ...")
    _save_dataset_if_missing(members, os.path.join(args.output_path, "members"), "members")
    _save_dataset_if_missing(nonmembers, os.path.join(args.output_path, "nonmembers"), "nonmembers")

    split_plan = []
    for split_name, split_data, label in [("mia_members", members, 1), ("mia_nonmembers", nonmembers, 0)]:
        jsonl_path = os.path.join(args.output_path, f"{split_name}.jsonl")
        _log(f"{split_name}: counting existing records in {jsonl_path} ...")
        existing_records = _count_jsonl_records(jsonl_path) if args.resume else 0
        _log(f"{split_name}: {existing_records}/{len(split_data)} records already present.")
        if existing_records < len(split_data):
            split_plan.append((split_name, split_data, label, jsonl_path, existing_records))

    if not split_plan:
        _log("[4/6] All records already exist; rebuilding visualization samples only.")
        all_viz_samples = _build_viz_samples_from_jsonl(args.output_path)
        viz_path = os.path.join(args.output_path, "viz_samples.json")
        with open(viz_path, "w", encoding="utf-8") as f:
            json.dump(all_viz_samples, f, ensure_ascii=False)
        _log(f"Done. Visualization samples: {len(all_viz_samples)} -> {viz_path}")
        return

    # Keep attention tensors disabled by default. Attention maps are quadratic in
    # sequence length and are only requested by compute_attention_entropy().
    # Enabling them globally makes every PPL/gradient forward retain large maps.
    _log("[4/6] Loading tokenizer and model onto the available devices ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, return_dict=True, device_map="auto",
        torch_dtype=torch.bfloat16, cache_dir=args.cache_dir,
        output_attentions=False,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    _log(f"Model loaded. Primary device: {getattr(model, 'device', 'unknown')}")

    _log("[5/6] Computing white-box features ...")
    for split_name, split_data, label, jsonl_path, existing_records in split_plan:
        _log(f"{split_name}: processing records {existing_records + 1}..{len(split_data)}.")
        mode = "a" if args.resume and existing_records > 0 else "w"
        with open(jsonl_path, mode, encoding="utf-8") as f:
            remaining_texts = split_data["text"][existing_records:]
            for text in tqdm(remaining_texts, desc=split_name, initial=existing_records, total=len(split_data)):
                pred = compute_and_enrich(text, model, tokenizer, args.max_length)
                record = {"pred": pred, "label": label}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                torch.cuda.empty_cache()
                completed = existing_records + 1
                existing_records += 1
                if completed == 1 or completed % max(args.log_every, 1) == 0 or completed == len(split_data):
                    _log(f"{split_name}: completed {completed}/{len(split_data)}.")

    # 保存可视化样本
    _log("[6/6] Building visualization samples ...")
    all_viz_samples = _build_viz_samples_from_jsonl(args.output_path)
    viz_path = os.path.join(args.output_path, "viz_samples.json")
    with open(viz_path, "w", encoding="utf-8") as f:
        json.dump(all_viz_samples, f, ensure_ascii=False)

    _log(f"Done! Results saved to {args.output_path}")
    _log(f"Visualization samples: {len(all_viz_samples)} -> viz_samples.json")


if __name__ == "__main__":
    main()
