"""
Scaling Up MIA 共享工具函数。

映射关系：
  原始 precompute_*.py :: calculatePerplexity() → calculate_perplexity()
  原始 precompute_*.py :: inference()           → compute_mia_features()
  原始 collection_mia.py :: combinations()                → combinations_from_list()
  原始 collection_mia.py :: process_combination_mia_features() → process_combination_features()
  原始 collection_mia.py :: train_model()       → train_linear_scorer()
  原始 collection_mia.py :: evaluate_model()    → evaluate_chunk_auc()
  原始 collection_mia.py :: get_raw_predictions()→ predict_raw()
  原始 document_mia.py   :: clean_nan()         → clean_nan_nested()
  以及 NaN 处理、种子设定等辅助函数
"""

import json
import os
import random
import zlib
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ====== MIA 特征计算（对应 precompute_*.py 的 calculatePerplexity + inference）======

def calculate_perplexity(text: str, model, tokenizer, max_length: int = 2048):
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    if model.device.type == "cuda":
        encodings = {k: v.cuda() for k, v in encodings.items()}
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings["input_ids"])
    loss, logits = outputs[:2]
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    all_prob = []
    for i, token_id in enumerate(encodings["input_ids"][0][1:]):
        all_prob.append(probabilities[0, i, token_id].item())
    return torch.exp(loss).item(), all_prob, loss.item()


def compute_mia_features(text: str, model, tokenizer) -> dict:
    pred = {}
    p1, all_prob, _ = calculate_perplexity(text, model, tokenizer)
    p_lower, _, _ = calculate_perplexity(text.lower(), model, tokenizer)
    pred["ppl"] = p1
    pred["ppl/lowercase_ppl"] = -(np.log(p_lower) / np.log(p1)).item() if p1 > 1 else 0.0
    zlib_entropy = len(zlib.compress(bytes(text, "utf-8")))
    pred["ppl/zlib"] = np.log(p1) / zlib_entropy if zlib_entropy > 0 else 0.0
    for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        k_length = max(1, int(len(all_prob) * ratio))
        topk_prob = np.sort(all_prob)[:k_length]
        pred[f"Min_{ratio*100}% Prob"] = -np.mean(topk_prob).item()
    return pred


# ====== 数据处理 ======

def load_mia_jsonl(path: str) -> list:
    """对应原始代码中所有 open+json.loads 读取 jsonl 的逻辑"""
    # White-box precompute writes UTF-8 JSONL.  Relying on the Windows GBK
    # default breaks records containing Unicode text/token previews.
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def clean_nan_flat(mia_list: list):
    """
    对应 collection_mia.py 中：
      for mia in A_members + ...:
          for key in mia['pred']:
              if np.isnan(mia['pred'][key]):
                  mia['pred'][key] = 0
    """
    for item in mia_list:
        for key in item["pred"]:
            if np.isnan(item["pred"][key]):
                item["pred"][key] = 0


def clean_nan_nested(mia_list: list):
    """
    对应 document_mia.py 中 clean_nan()：
      for list_sents in mia_partition:
          for sent in list_sents:
              for (mia, value) in sent['pred'].items():
                  if np.isnan(value): sent['pred'][mia] = 100000
    注：这里用0替代原始的100000，效果等价（outlier trimming会处理掉）
    """
    for doc in mia_list:
        for chunk in doc:
            for key in chunk["pred"]:
                val = chunk["pred"][key]
                if isinstance(val, float) and np.isnan(val):
                    chunk["pred"][key] = 0


def features_to_numpy(list_of_dicts: list) -> np.ndarray:
    """
    将 flat MIA 记录转为 (N, 1, F) 数组。
    对应 collection_mia.py :: process_combination_mia_features([[x] for x in ...])
    """
    features = []
    for item in list_of_dicts:
        features.append([list(item["pred"].values())])
    return np.array(features)


def combinations_from_list(items: list, num_combinations: int, group_size: int):
    """
    对应 collection_mia.py :: combinations()
    从 items 中随机采样 num_combinations 个组合，每组 group_size 条
    """
    actual_group = min(group_size, len(items))
    chosen = set()
    max_attempts = num_combinations * 10
    attempts = 0
    while len(chosen) < num_combinations and attempts < max_attempts:
        idx = tuple(sorted(random.sample(range(len(items)), actual_group)))
        chosen.add(idx)
        attempts += 1
    result = []
    for combo in chosen:
        result.append([items[i] for i in combo])
    return result


def process_combination_features(list_combinations: list) -> np.ndarray:
    """
    对应 collection_mia.py :: process_combination_mia_features()
    将组合列表转为 (C, G, F) 的 numpy 数组
    """
    features = []
    for combo in list_combinations:
        group_feats = [list(item["pred"].values()) for item in combo]
        features.append(group_feats)
    return np.array(features)


# ====== 线性打分器 ======

def train_linear_scorer(
    member_features: np.ndarray,
    nonmember_features: np.ndarray,
    num_epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 128,
) -> nn.Module:
    """
    对应 collection_mia.py :: train_model() + 模型定义部分
    原始代码：
      model = nn.Linear(input_size, 1).to('cuda')
      criterion = nn.BCEWithLogitsLoss()
      optimizer = optim.Adam(model.parameters(), lr=0.01)
      train_model(model, train_loader, criterion, optimizer, num_epochs=100)
    """
    X = np.concatenate([member_features, nonmember_features], axis=0)
    y = np.concatenate([np.ones(len(member_features)), np.zeros(len(nonmember_features))])

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_size = X.shape[1]
    model = nn.Linear(input_size, 1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(num_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb).squeeze(1)
            loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def evaluate_chunk_auc(
    model: nn.Module,
    member_features: np.ndarray,
    nonmember_features: np.ndarray,
    batch_size: int = 128,
) -> float:
    """
    对应 collection_mia.py :: evaluate_model() 中计算 auc_score 的部分
    以及 document_mia.py :: get_mia_scorer() 中的 chunk-level 评估
    """
    X = np.concatenate([member_features, nonmember_features], axis=0)
    y = np.concatenate([np.ones(len(member_features)), np.zeros(len(nonmember_features))])

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            out = model(xb).squeeze()
            if out.dim() == 0:
                all_scores.append(out.cpu().item())
            else:
                all_scores.extend(out.cpu().numpy().tolist())
            all_labels.extend(yb.numpy().tolist())

    return roc_auc_score(all_labels, all_scores)


def predict_raw(model: nn.Module, features: np.ndarray) -> np.ndarray:
    """
    对应 collection_mia.py :: get_raw_predictions()
    原始代码：
      model.eval()
      with torch.no_grad():
          outputs = model(torch.Tensor(data_point).to('cuda'))
      return outputs.squeeze().detach().cpu().numpy()
    """
    model.eval()
    with torch.no_grad():
        t = torch.tensor(features, dtype=torch.float32).to(DEVICE)
        out = model(t).squeeze().cpu().numpy()
    return np.atleast_1d(out)


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
