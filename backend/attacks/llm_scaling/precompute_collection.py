"""
CLI 脚本：预计算 Collection-level MIA 分数。

用法:
  python -m attacks.llm_scaling.precompute_collection \
      --model_name EleutherAI/pythia-2.8b \
      --dataset_name haritzpuerto/the_pile_00_arxiv \
      --output_path results/collection_mia/EleutherAI/pythia-2.8b/haritzpuerto/the_pile_00_arxiv/2048 \
      --seed 42
"""

import argparse
import json
import os
import random

import torch
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from attacks.llm_scaling.utils import compute_mia_features


def main():
    parser = argparse.ArgumentParser(description="Precompute collection-level MIA scores")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--filter_outliers", action="store_true")
    parser.add_argument("--min_chars", type=int, default=100)
    parser.add_argument("--cache_dir", type=str, default="/tmp")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_path, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, return_dict=True, device_map="auto",
        torch_dtype=torch.bfloat16, cache_dir=args.cache_dir,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    ds = load_dataset(args.dataset_name)
    if args.filter_outliers:
        for split in ["train", "validation", "test"]:
            if split in ds:
                ds[split] = ds[split].filter(lambda x: len(x["text"]) > args.min_chars)

    nonmembers = concatenate_datasets([ds["validation"], ds["test"]])
    members = ds["train"].shuffle(seed=args.seed).select(range(len(nonmembers)))

    nonmembers.save_to_disk(os.path.join(args.output_path, "nonmembers"))
    members.save_to_disk(os.path.join(args.output_path, "members"))

    for split_name, split_data, label in [("mia_members", members, 1), ("mia_nonmembers", nonmembers, 0)]:
        results = []
        for text in tqdm(split_data["text"], desc=split_name):
            feats = compute_mia_features(text, model, tokenizer)
            results.append({"pred": feats, "label": label})
            torch.cuda.empty_cache()
        with open(os.path.join(args.output_path, f"{split_name}.jsonl"), "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()