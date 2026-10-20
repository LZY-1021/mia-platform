"""
白盒攻击工具函数 —— 利用模型内部状态（梯度、注意力、隐藏层）丰富 MIA 特征。

相比黑盒版 (utils.py) 的增强点：
  1. compute_whitebox_features(): 新增梯度范数、注意力熵、隐藏状态统计等白盒特征
  2. greedy_continuation(): 取前 N token 贪心解码，返回续写文本和逐 token 匹配结果
  3. train_mlp_scorer(): 2 层 MLP 评分器（比线性模型更强）
  4. train_xgboost_scorer(): XGBoost 评分器（树模型，捕获非线性关系）
  5. 多种聚合方法: t-test, Mann-Whitney U, Bayesian test, Fisher's method
  6. Platt Scaling 校准 + 校准曲线 + Youden's J 阈值
"""

import json
import os
import random
import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WhiteboxFeaturePreprocessor:
    """Fit robust feature cleanup on training samples and reuse it at inference."""

    def __init__(self):
        self.medians = None
        self.lower = None
        self.upper = None
        self.scaler = RobustScaler(quantile_range=(10.0, 90.0), unit_variance=True)

    def fit(self, member_features: np.ndarray, nonmember_features: np.ndarray):
        features = np.asarray(np.concatenate([member_features, nonmember_features], axis=0), dtype=float)
        self.medians = np.nanmedian(features, axis=0)
        self.medians = np.where(np.isfinite(self.medians), self.medians, 0.0)
        cleaned = self._fill_missing(features)
        self.lower = np.quantile(cleaned, 0.01, axis=0)
        self.upper = np.quantile(cleaned, 0.99, axis=0)
        self.scaler.fit(np.clip(cleaned, self.lower, self.upper))
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.medians is None:
            raise RuntimeError("WhiteboxFeaturePreprocessor must be fit before transform.")
        cleaned = self._fill_missing(np.asarray(features, dtype=float))
        clipped = np.clip(cleaned, self.lower, self.upper)
        return self.scaler.transform(clipped).astype(np.float32, copy=False)

    def _fill_missing(self, features: np.ndarray) -> np.ndarray:
        values = np.atleast_2d(features).copy()
        invalid = ~np.isfinite(values)
        if invalid.any():
            values[invalid] = np.take(self.medians, np.where(invalid)[1])
        return values


def fit_whitebox_feature_preprocessor(
    member_features: np.ndarray,
    nonmember_features: np.ndarray,
) -> WhiteboxFeaturePreprocessor:
    """Build one training-only preprocessor shared by every whitebox scorer."""
    return WhiteboxFeaturePreprocessor().fit(member_features, nonmember_features)


# ============================================================
# 白盒特征计算
# ============================================================

def _prepare_tokenizer_inputs(encodings: Dict[str, torch.Tensor], model) -> Dict[str, torch.Tensor]:
    """
    Move tokenizer outputs to the model device while preserving integer token IDs.

    Older torch/accelerate/transformers combinations can be surprisingly eager
    about dtype conversion when models are loaded with torch_dtype + device_map.
    Embedding layers must receive integer input_ids, so force the relevant
    tensors back to long before every forward pass.
    """
    target_device = model.device if getattr(model, "device", None) is not None else DEVICE
    prepared = {}
    for key, value in encodings.items():
        if not torch.is_tensor(value):
            prepared[key] = value
            continue
        value = value.to(target_device)
        if key in {"input_ids", "attention_mask", "token_type_ids"}:
            value = value.long()
        prepared[key] = value
    return prepared

def calculate_perplexity_with_logits(text: str, model, tokenizer, max_length: int = 2048):
    """
    增强版困惑度计算，同时返回 logits 和所有中间概率。

    Returns:
        ppl: 困惑度
        all_prob: 每个 token 的对数概率列表
        nll: 负对数似然
        logits: 模型输出的 logits (用于后续梯度计算)
        input_ids: 输入 token IDs
    """
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    encodings = _prepare_tokenizer_inputs(encodings, model)

    with torch.no_grad():
        outputs = model(**encodings, labels=encodings["input_ids"])
    loss, logits = outputs[:2]

    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    all_prob = []
    for i, token_id in enumerate(encodings["input_ids"][0][1:]):
        all_prob.append(probabilities[0, i, token_id].item())

    return torch.exp(loss).item(), all_prob, loss.item(), logits, encodings


def compute_gradient_norm(text: str, model, tokenizer, max_length: int = 2048) -> float:
    """
    计算输入文本在模型上的梯度 L2 范数。
    训练数据 → 梯度较小（模型已适应），非训练数据 → 梯度较大（模型需要大幅调整）。
    """
    model.train()  # 需要梯度计算
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    encodings = _prepare_tokenizer_inputs(encodings, model)

    # 启用输入嵌入的梯度
    embeddings = model.get_input_embeddings()
    input_embeds = embeddings(encodings["input_ids"]).detach().requires_grad_(True)

    # 前向传播使用嵌入而非 input_ids
    outputs = model(inputs_embeds=input_embeds, labels=encodings["input_ids"])
    loss = outputs.loss

    loss.backward()

    grad_norm = input_embeds.grad.norm(2).item()

    model.zero_grad()
    model.eval()

    return grad_norm


def compute_attention_entropy(text: str, model, tokenizer, max_length: int = 2048) -> float:
    """
    计算模型注意力分布的平均熵。
    训练数据 → 注意力更集中（低熵），非训练数据 → 注意力更分散（高熵）。
    需要 model 输出 attention（需在 from_pretrained 时设置 output_attentions=True）。
    """
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    encodings = _prepare_tokenizer_inputs(encodings, model)

    with torch.no_grad():
        outputs = model(**encodings, output_attentions=True)

    if not hasattr(outputs, "attentions") or outputs.attentions is None:
        return 0.0

    # outputs.attentions: tuple of (batch, heads, seq, seq)，取最后一层
    last_attn = outputs.attentions[-1]  # (1, heads, seq_len, seq_len)
    # 计算每个头的熵，然后平均
    eps = 1e-10
    attn_probs = last_attn + eps
    entropy = -(attn_probs * torch.log(attn_probs)).sum(dim=-1)  # (1, heads, seq_len)
    mean_entropy = entropy.mean().item()

    return mean_entropy


def compute_hidden_state_stats(text: str, model, tokenizer, max_length: int = 2048) -> Dict[str, float]:
    """
    计算隐藏层状态的统计特征。
    训练数据 → 隐藏状态更稳定（低方差），非训练数据 → 隐藏状态波动更大。
    """
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    encodings = _prepare_tokenizer_inputs(encodings, model)

    with torch.no_grad():
        outputs = model(**encodings, output_hidden_states=True)

    if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
        return {"hidden_var_mean": 0.0, "hidden_var_max": 0.0}

    # 取最后一层隐藏状态: (1, seq_len, hidden_dim)
    last_hidden = outputs.hidden_states[-1].squeeze(0)  # (seq_len, hidden_dim)

    # 沿序列维度计算方差
    var_per_dim = last_hidden.var(dim=0)  # (hidden_dim,)
    return {
        "hidden_var_mean": var_per_dim.mean().item(),
        "hidden_var_max": var_per_dim.max().item(),
    }


def compute_whitebox_features(text: str, model, tokenizer, max_length: int = 2048) -> dict:
    """
    完整的白盒 MIA 特征计算（10维黑盒 + 白盒增强特征）。

    黑盒特征（复用）：
      ppl, ppl/lowercase_ppl, ppl/zlib, Min-K% Prob (5%~60%)
    白盒新增特征：
      grad_norm: 输入嵌入梯度 L2 范数
      attn_entropy: 最后一层注意力平均熵
      hidden_var_mean: 最后一层隐藏状态方差均值
      hidden_var_max: 最后一层隐藏状态方差最大值
    """
    pred = {}

    # 基础困惑度特征
    p1, all_prob, _, _, _ = calculate_perplexity_with_logits(text, model, tokenizer, max_length)
    p_lower, _, _ = _simple_perplexity(text.lower(), model, tokenizer, max_length)

    pred["ppl"] = p1
    pred["ppl/lowercase_ppl"] = -(np.log(p_lower) / np.log(p1)).item() if p1 > 1 else 0.0
    zlib_entropy = len(zlib.compress(bytes(text, "utf-8")))
    pred["ppl/zlib"] = np.log(p1) / zlib_entropy if zlib_entropy > 0 else 0.0

    for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        k_length = max(1, int(len(all_prob) * ratio))
        topk_prob = np.sort(all_prob)[:k_length]
        pred[f"Min_{ratio*100}% Prob"] = -np.mean(topk_prob).item()

    # 梯度范数
    try:
        pred["grad_norm"] = compute_gradient_norm(text, model, tokenizer, max_length)
    except Exception:
        pred["grad_norm"] = 0.0

    # 注意力熵
    try:
        pred["attn_entropy"] = compute_attention_entropy(text, model, tokenizer, max_length)
    except Exception:
        pred["attn_entropy"] = 0.0

    # 隐藏状态统计
    try:
        hs_stats = compute_hidden_state_stats(text, model, tokenizer, max_length)
        pred["hidden_var_mean"] = hs_stats["hidden_var_mean"]
        pred["hidden_var_max"] = hs_stats["hidden_var_max"]
    except Exception:
        pred["hidden_var_mean"] = 0.0
        pred["hidden_var_max"] = 0.0

    # 保存 all_prob 用于前端热力图
    pred["_all_prob"] = all_prob

    return pred


def _simple_perplexity(text: str, model, tokenizer, max_length: int = 2048):
    """简单困惑度计算（不返回 logits，用于 lowercase 变体等辅助计算）"""
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    encodings = _prepare_tokenizer_inputs(encodings, model)
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings["input_ids"])
    loss = outputs.loss
    return torch.exp(loss).item(), None, loss.item()


# ============================================================
# 逐词续写对比
# ============================================================

def greedy_continuation(
    text: str,
    model,
    tokenizer,
    prefix_length: int = 50,
    max_new_tokens: int = 50,
) -> Dict:
    """
    取文本前 prefix_length 个 token，用贪心解码生成后续，和原文逐 token 对比。

    Returns:
        {
            "prefix_tokens": [...],         # 前缀 token 文本列表
            "generated_tokens": [...],      # 模型续写的 token 文本列表
            "actual_tokens": [...],         # 原文后续的 token 文本列表
            "match_mask": [...],            # 逐 token 是否匹配 (1/0)
            "match_rate": float,            # 总匹配率
        }
    """
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    encodings = _prepare_tokenizer_inputs(encodings, model)
    input_ids = encodings["input_ids"].long()

    total_len = input_ids.shape[1]
    actual_prefix = min(prefix_length, total_len - 1)
    actual_continuation_len = min(max_new_tokens, total_len - actual_prefix)

    if actual_continuation_len <= 0:
        return {
            "prefix_tokens": [],
            "generated_tokens": [],
            "actual_tokens": [],
            "match_mask": [],
            "match_rate": 0.0,
        }

    prefix_ids = input_ids[:, :actual_prefix]

    # 贪心解码
    with torch.no_grad():
        generated = model.generate(
            prefix_ids,
            max_new_tokens=actual_continuation_len,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 提取生成部分
    gen_continuation_ids = generated[0, actual_prefix: actual_prefix + actual_continuation_len]
    actual_continuation_ids = input_ids[0, actual_prefix: actual_prefix + actual_continuation_len]

    # 对齐长度
    min_len = min(len(gen_continuation_ids), len(actual_continuation_ids))
    if min_len == 0:
        return {
            "prefix_tokens": tokenizer.convert_ids_to_tokens(prefix_ids[0].tolist()),
            "generated_tokens": [],
            "actual_tokens": [],
            "match_mask": [],
            "match_rate": 0.0,
        }

    gen_ids = gen_continuation_ids[:min_len]
    act_ids = actual_continuation_ids[:min_len]

    match_mask = [1 if g == a else 0 for g, a in zip(gen_ids.tolist(), act_ids.tolist())]
    match_rate = sum(match_mask) / len(match_mask) if match_mask else 0.0

    return {
        "prefix_tokens": tokenizer.convert_ids_to_tokens(prefix_ids[0].tolist()),
        "generated_tokens": tokenizer.convert_ids_to_tokens(gen_ids.tolist()),
        "actual_tokens": tokenizer.convert_ids_to_tokens(act_ids.tolist()),
        "match_mask": match_mask,
        "match_rate": round(match_rate, 4),
    }


# ============================================================
# 评分器：MLP / XGBoost
# ============================================================

def train_mlp_scorer(
    member_features: np.ndarray,
    nonmember_features: np.ndarray,
    hidden_dim: Optional[int] = None,
    num_epochs: int = 150,
    lr: float = 0.005,
    batch_size: int = 128,
) -> nn.Module:
    """
    2 层 MLP 评分器：input → hidden(ReLU) → Dropout → output。
    比单层线性模型有更强的非线性拟合能力，但需要正则化防过拟合。
    """
    X = np.concatenate([member_features, nonmember_features], axis=0)
    y = np.concatenate([np.ones(len(member_features)), np.zeros(len(nonmember_features))])

    # Reserve a stratified validation portion from the scorer training data.
    # This prevents the MLP from selecting its final weights only by train loss.
    member_indices = np.random.permutation(len(member_features))
    nonmember_indices = np.random.permutation(len(nonmember_features))
    val_member_count = max(1, int(0.15 * len(member_indices)))
    val_nonmember_count = max(1, int(0.15 * len(nonmember_indices)))
    train_member = member_indices[val_member_count:]
    train_nonmember = nonmember_indices[val_nonmember_count:]
    val_indices = np.concatenate([
        member_indices[:val_member_count],
        len(member_features) + nonmember_indices[:val_nonmember_count],
    ])
    train_indices = np.concatenate([
        train_member,
        len(member_features) + train_nonmember,
    ])

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_size = X.shape[1]
    hidden_dim = hidden_dim or min(64, max(16, input_size * 4))
    model = nn.Sequential(
        nn.Linear(input_size, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_dim, 1),
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    val_x = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    val_y = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
    best_state = None
    best_loss = float("inf")
    patience, stale_epochs = 20, 0
    model.train()
    for _ in range(num_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb).squeeze(1)
            loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_x).squeeze(1), val_y).item()
        model.train()
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def train_xgboost_scorer(
    member_features: np.ndarray,
    nonmember_features: np.ndarray,
) -> object:
    """
    XGBoost 评分器：梯度提升树，能捕获特征间的非线性交互。
    适合特征维度不高但样本量足够的场景。
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("XGBoost 未安装，请运行: pip install xgboost")

    X = np.concatenate([member_features, nonmember_features], axis=0)
    y = np.concatenate([np.ones(len(member_features)), np.zeros(len(nonmember_features))])

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=3,
        min_child_weight=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=0,
        n_jobs=1,
        verbosity=0,
    )
    model.fit(X, y)
    return model


def predict_with_model(model, features: np.ndarray, model_type: str = "linear") -> np.ndarray:
    """
    通用模型推理接口。
    model_type: "linear" | "mlp" | "xgboost"
    """
    prepared_features = np.asarray(features, dtype=np.float32)
    preprocessor = getattr(model, "_mia_feature_preprocessor", None)
    if preprocessor is not None:
        prepared_features = preprocessor.transform(prepared_features)

    if model_type in ("linear", "mlp"):
        model.eval()
        with torch.no_grad():
            t = torch.tensor(prepared_features, dtype=torch.float32).to(DEVICE)
            out = model(t).squeeze().cpu().numpy()
        return np.atleast_1d(out)
    elif model_type == "xgboost":
        # XGBoost 返回概率
        return model.predict_proba(prepared_features)[:, 1]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ============================================================
# 聚合方法
# ============================================================

def aggregate_ttest(scores: np.ndarray, reference: np.ndarray) -> Tuple[float, float]:
    """
    单侧 t 检验：评估 scores 是否显著高于 reference。
    返回 (t_statistic, p_value)。
    t 统计量作为成员推断分数，p 值越小越显著。

    适用场景：数据近似正态分布，样本量适中（>30）。
    优点：对均值差异敏感，计算简单快速。
    缺点：仍假设数据近似正态，对严重异常值敏感；实现采用 Welch 校正，不强制等方差。
    """
    # Welch's t-test does not force equal variance across member candidates
    # and non-member references, which is safer for heterogeneous collections.
    stat, pval = stats.ttest_ind(scores, reference, equal_var=False, alternative="greater")
    return float(stat), float(pval)


def aggregate_mannwhitney(scores: np.ndarray, reference: np.ndarray) -> Tuple[float, float]:
    """
    Mann-Whitney U 检验（Wilcoxon 秩和检验）：非参数检验。
    不假设正态分布，比较两组的秩和。
    返回 (U_statistic, p_value)。

    适用场景：数据非正态分布，或存在异常值。
    优点：对分布无假设，对异常值鲁棒。
    缺点：统计功效略低于 t 检验（当数据确实正态时），不能直接比较均值。
    """
    stat, pval = stats.mannwhitneyu(scores, reference, alternative="greater")
    return float(stat), float(pval)


def aggregate_bayesian(scores: np.ndarray, reference: np.ndarray) -> Tuple[float, float]:
    """
    贝叶斯假设检验：用贝叶斯因子(BF)替代 p 值。
    BF = P(data|H1) / P(data|H0)，BF > 1 支持备择假设。
    这里用简化的贝叶斯 t 检验近似。

    适用场景：需要量化证据强度而非二元决策。
    优点：可解释性好（"数据支持 H1 的强度"），不依赖样本量膨胀。
    缺点：需要先验假设，计算稍复杂。
    """
    n1, n2 = len(scores), len(reference)
    m1, m2 = np.mean(scores), np.mean(reference)
    v1, v2 = np.var(scores, ddof=1), np.var(reference, ddof=1)

    # pooled standard error
    sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if sp == 0:
        return 0.0, 0.5

    # Cohen's d
    d = (m1 - m2) / sp

    # 简化的贝叶斯因子近似（基于 Cauchy 先验的 JZS t 检验）
    # 参考: Rouder et al. (2009)
    t_val = (m1 - m2) / (sp * np.sqrt(1/n1 + 1/n2))

    # JZS Bayes Factor 近似
    r = 0.707  # 默认 Cauchy 先验尺度
    n = n1 + n2
    df = n - 2

    # 数值稳定的贝叶斯因子计算
    try:
        from scipy.special import gammaln
        log_bf = (
            gammaln(df/2 + 1/2) - gammaln(df/2) - gammaln(1/2)
            + 0.5 * np.log(n / (n + r**2))
            + (df/2 + 1/2) * np.log(1 + n * t_val**2 / ((n + r**2) * df))
            - (df/2) * np.log(1 + t_val**2 / df)
        )
        bf = np.exp(log_bf)
    except Exception:
        bf = 1.0

    # 将 BF 映射为类似 p 值的分数（BF 越大 → 越支持成员假设 → "p" 越小）
    pseudo_p = 1.0 / (1.0 + bf)
    return float(t_val), float(pseudo_p)


def aggregate_fisher(p_values: List[float]) -> Tuple[float, float]:
    """
    Fisher 组合检验：将多个独立检验的 p 值组合为一个全局 p 值。
    χ² = -2 * Σln(p_i)，服从 χ²(2k) 分布。

    适用场景：多个文档/段落各自做了检验，需综合判断。
    优点：能整合多个弱信号为强信号，数学基础扎实。
    缺点：假设各 p 值独立，极端 p 值影响过大。
    """
    # 过滤掉 0 和 1 的 p 值（会导致 log 问题）
    clean_p = [max(1e-15, min(1 - 1e-15, p)) for p in p_values if 0 < p < 1]
    if not clean_p:
        return 0.0, 0.5

    chi2_stat = -2 * sum(np.log(p) for p in clean_p)
    df = 2 * len(clean_p)
    p_val = 1 - stats.chi2.cdf(chi2_stat, df)
    return float(chi2_stat), float(p_val)


def aggregate_fisher_scores(scores: np.ndarray, reference: np.ndarray) -> Tuple[float, float]:
    """
    Fisher 组合检验的集合级版本。

    与 t-test / Mann-Whitney U / Bayesian test 一样，这个函数对“单个集合”的
    文档分数进行一次聚合，返回该集合的统计量和 p 值，便于逐集合计算 AUC/F1。

    做法：
      1. 对集合内每个文档分数 s，用已知非成员参考分布估计右尾 p 值：
         p_i = P(reference >= s)。分数越高，说明越像成员，p_i 越小。
      2. 用 Fisher's method 把这些 p_i 合成一个集合级 p 值。

    区别：
      - t-test 比较均值，依赖近似正态；
      - Mann-Whitney 比较秩，非参数更鲁棒；
      - Bayesian test 返回证据强度；
      - Fisher 把多个“弱惊讶信号”的显著性合并，极小 p 值会被放大。
    """
    ref = np.asarray(reference, dtype=float).reshape(-1)
    vals = np.asarray(scores, dtype=float).reshape(-1)
    if ref.size == 0 or vals.size == 0:
        return 0.0, 0.5

    p_values = []
    denom = ref.size + 1
    for score in vals:
        # Add-one smoothing avoids exact 0, which would dominate log(p).
        p_values.append(float((np.sum(ref >= score) + 1) / denom))
    return aggregate_fisher(p_values)


AGGREGATION_METHODS = {
    "ttest": {
        "func": aggregate_ttest,
        "name": "T-Test",
        "name_zh": "T 检验",
        "desc_zh": "参数检验，假设数据正态分布。对均值差异敏感，计算简单。",
        "desc_en": "Parametric test assuming normality. Sensitive to mean differences, simple computation.",
    },
    "mannwhitney": {
        "func": aggregate_mannwhitney,
        "name": "Mann-Whitney U",
        "name_zh": "Mann-Whitney U 检验",
        "desc_zh": "非参数检验，不假设正态分布。对异常值鲁棒，比较秩和。",
        "desc_en": "Non-parametric test, no normality assumption. Robust to outliers, compares rank sums.",
    },
    "bayesian": {
        "func": aggregate_bayesian,
        "name": "Bayesian Test",
        "name_zh": "贝叶斯假设检验",
        "desc_zh": "用贝叶斯因子量化证据强度，不依赖样本量膨胀。BF>1 支持成员假设。",
        "desc_en": "Quantifies evidence strength via Bayes Factor. BF>1 supports membership hypothesis.",
    },
    "fisher": {
        "func": None,  # Fisher 特殊处理，需要 p 值列表
        "name": "Fisher's Method",
        "name_zh": "Fisher 组合检验",
        "desc_zh": "将多个独立检验的 p 值组合为全局 p 值，适合整合多个弱信号。",
        "desc_en": "Combines independent p-values into a global p-value. Good for aggregating weak signals.",
    },
}


# ============================================================
# Platt Scaling 校准 + 校准曲线 + Youden's J
# ============================================================

class PlattScaler:
    """
    Platt Scaling：用逻辑回归将原始分数映射为校准概率。
    校准后的概率更接近真实的后验概率 P(member|score)。

    三步曲：
      1. 修正偏差：模型原始分数可能过度自信，Platt Scaling 将其"掰直"
      2. 确定标准：在校准概率上用 Youden's J 找最佳阈值
      3. 验证效果：通过校准曲线验证校准质量
    """

    def __init__(self):
        self.a = 1.0  # 缩放参数
        self.b = 0.0  # 偏移参数
        self.score_center = 0.0
        self.score_scale = 1.0

    def _standardize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Keep heterogeneous aggregation statistics numerically comparable for LBFGS."""
        values = np.asarray(scores, dtype=float)
        standardized = (values - self.score_center) / self.score_scale
        return np.clip(standardized, -50.0, 50.0)

    def fit(self, scores: np.ndarray, labels: np.ndarray):
        """
        用 Platt 的方法拟合校准参数 a, b。
        P(y=1|s) = 1 / (1 + exp(a * s + b))

        使用牛顿法优化对数似然，这比 sklearn 的 LogisticRegression 更稳定。
        """
        # T-test, U, and Fisher statistics live on very different scales.
        # Fitting a sigmoid directly on large U/Fisher values can make every
        # probability saturate to 0 or 1, destroying the useful score ranking.
        values = np.asarray(scores, dtype=float)
        self.score_center = float(np.median(values))
        q1, q3 = np.quantile(values, [0.25, 0.75])
        self.score_scale = float(q3 - q1)
        if not np.isfinite(self.score_scale) or self.score_scale < 1e-12:
            self.score_scale = float(np.std(values))
        if not np.isfinite(self.score_scale) or self.score_scale < 1e-12:
            self.score_scale = 1.0

        # 将标准化后的分数和标签转为 torch 张量
        scores_t = torch.tensor(self._standardize_scores(values), dtype=torch.float64)
        labels_t = torch.tensor(labels, dtype=torch.float64)

        a = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

        optimizer = torch.optim.LBFGS([a, b], lr=0.01, max_iter=100)

        def closure():
            optimizer.zero_grad()
            logits = a * scores_t + b
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels_t)
            loss.backward()
            # LBFGS only reads the scalar value after gradients are populated.
            # Detaching avoids a noisy warning without changing the optimization.
            return loss.detach()

        optimizer.step(closure)

        self.a = a.item()
        self.b = b.item()
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """将原始分数转换为校准概率。"""
        logits = np.clip(self.a * self._standardize_scores(scores) + self.b, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def fit_transform(self, scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
        self.fit(scores, labels)
        return self.transform(scores)


def compute_calibration_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Dict:
    """
    计算校准曲线数据（可靠性图）。

    将预测概率分成 n_bins 个区间，统计每个区间内的：
      - 实际正例比例（fraction_of_positives）
      - 预测概率均值（mean_predicted_value）
      - 样本数

    完美校准：对角线 y=x。

    Returns:
        {
            "bins": [{"mean_pred": float, "mean_true": float, "count": int}, ...],
            "ece": float,  # Expected Calibration Error
            "brier_score": float,
        }
    """
    probs = scores  # 假设 scores 已经过 Platt Scaling
    bin_edges = np.linspace(0, 1, n_bins + 1)

    bins_data = []
    ece = 0.0
    total = len(probs)

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])

        count = mask.sum()
        if count > 0:
            mean_pred = probs[mask].mean()
            mean_true = labels[mask].mean()
            ece += abs(mean_pred - mean_true) * count / total
            bins_data.append({
                "mean_pred": round(float(mean_pred), 4),
                "mean_true": round(float(mean_true), 4),
                "count": int(count),
            })
        else:
            bins_data.append({
                "mean_pred": round(float((bin_edges[i] + bin_edges[i + 1]) / 2), 4),
                "mean_true": 0.0,
                "count": 0,
            })

    brier = float(brier_score_loss(labels, probs))

    return {
        "bins": bins_data,
        "ece": round(float(ece), 4),
        "brier_score": round(float(brier), 4),
    }


def find_youden_j_threshold(scores: np.ndarray, labels: np.ndarray) -> Dict:
    """
    在校准概率上用 Youden's J 找最佳阈值。
    Youden's J = TPR - FPR，最大化时取得最佳平衡点。

    Returns:
        {
            "best_threshold": float,
            "best_youden_j": float,
            "tpr_at_best": float,
            "fpr_at_best": float,
        }
    """
    from sklearn.metrics import roc_curve

    fpr_arr, tpr_arr, thresholds = roc_curve(labels, scores)
    youden_j = tpr_arr - fpr_arr
    best_idx = int(np.argmax(youden_j))

    return {
        "best_threshold": round(float(thresholds[best_idx]), 6),
        "best_youden_j": round(float(youden_j[best_idx]), 4),
        "tpr_at_best": round(float(tpr_arr[best_idx]), 4),
        "fpr_at_best": round(float(fpr_arr[best_idx]), 4),
    }


# ============================================================
# 数据处理（复用 + 扩展）
# ============================================================

def load_whitebox_jsonl(path: str) -> list:
    """读取白盒增强版 JSONL 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def features_to_numpy_whitebox(list_of_dicts: list, feature_keys: Optional[List[str]] = None, exclude_keys: set = None) -> np.ndarray:
    """
    将白盒 MIA 记录转为 numpy 数组。
    自动排除 _all_prob、_continuation 等可视化专用字段。
    """
    if exclude_keys is None:
        exclude_keys = {"_all_prob", "_continuation", "_token_texts", "_text_preview"}

    if not list_of_dicts:
        return np.empty((0, 0), dtype=float)

    if feature_keys is None:
        feature_keys = get_feature_keys(list_of_dicts[0], exclude_keys)

    features = []
    for item in list_of_dicts:
        feat = [float(item["pred"].get(k, 0.0) or 0.0) for k in feature_keys]
        features.append(feat)
    return np.asarray(features, dtype=float)


def process_combination_features_whitebox(
    list_combinations: list,
    feature_keys: Optional[List[str]] = None,
    exclude_keys: set = None,
) -> np.ndarray:
    """
    将集合组合转换为 (num_collections, docs_per_collection, num_features)。
    只保留数值特征，排除热力图、token 文本、续写对比等可视化字段。
    """
    if not list_combinations:
        return np.empty((0, 0, 0), dtype=float)

    if feature_keys is None:
        first = list_combinations[0][0] if list_combinations[0] else None
        feature_keys = get_feature_keys(first, exclude_keys) if first else []

    features = []
    for combo in list_combinations:
        group_feats = []
        for item in combo:
            group_feats.append([float(item["pred"].get(k, 0.0) or 0.0) for k in feature_keys])
        features.append(group_feats)
    return np.asarray(features, dtype=float)


def clean_nan_flat_whitebox(mia_list: list):
    """清理 NaN 值（白盒版），同时保留 _all_prob 等字段。"""
    for item in mia_list:
        for key in list(item["pred"].keys()):
            val = item["pred"][key]
            if isinstance(val, (float, np.floating)) and np.isnan(val):
                item["pred"][key] = 0.0
            elif isinstance(val, list):
                # 清理 all_prob 中的 NaN
                item["pred"][key] = [
                    0.0 if isinstance(v, (float, np.floating)) and np.isnan(v) else v
                    for v in val
                ]


def get_feature_keys(sample: dict, exclude_keys: set = None) -> list:
    """获取特征键名列表（排除可视化字段），用于前端展示。"""
    if exclude_keys is None:
        exclude_keys = {"_all_prob", "_continuation", "_token_texts", "_text_preview"}
    return [k for k in sample["pred"].keys() if k not in exclude_keys]


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
