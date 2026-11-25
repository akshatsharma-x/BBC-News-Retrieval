"""Ranked-retrieval evaluation metrics (TREC conventions).

`ranking` is a list of document ids in rank order; `qrels` maps judged
document ids to graded relevance (0 = not relevant, 1 = partially relevant,
2 = highly relevant). Binary metrics treat grade >= 1 as relevant; unjudged
documents are treated as non-relevant (standard pooling assumption).
"""
from __future__ import annotations

import math

import numpy as np


def relevant_set(qrels: dict[str, int]) -> set[str]:
    return {d for d, g in qrels.items() if g > 0}


def precision_at_k(ranking, qrels, k: int) -> float:
    rel = relevant_set(qrels)
    return sum(1 for d in ranking[:k] if d in rel) / k


def recall_at_k(ranking, qrels, k: int) -> float:
    rel = relevant_set(qrels)
    return sum(1 for d in ranking[:k] if d in rel) / len(rel) if rel else 0.0


def average_precision(ranking, qrels) -> float:
    """Mean of P@k over the ranks k of relevant documents; missed ones add 0."""
    rel = relevant_set(qrels)
    if not rel:
        return 0.0
    hits, total = 0, 0.0
    for i, d in enumerate(ranking, start=1):
        if d in rel:
            hits += 1
            total += hits / i
    return total / len(rel)


def reciprocal_rank(ranking, qrels, k: int | None = None) -> float:
    rel = relevant_set(qrels)
    for i, d in enumerate(ranking[:k] if k else ranking, start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranking, qrels, k: int) -> float:
    """nDCG with exponential gain (2^rel - 1) and log2(i + 1) discount."""
    dcg = sum((2 ** qrels.get(d, 0) - 1) / math.log2(i + 1)
              for i, d in enumerate(ranking[:k], start=1))
    ideal = sorted((g for g in qrels.values() if g > 0), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


RECALL_LEVELS = np.round(np.linspace(0.0, 1.0, 11), 1)


def interpolated_pr(ranking, qrels, levels=RECALL_LEVELS) -> np.ndarray:
    """11-point interpolated precision: p_interp(r) = max_{r' >= r} p(r')."""
    rel = relevant_set(qrels)
    if not rel:
        return np.zeros(len(levels))
    precisions, recalls, hits = [], [], 0
    for i, d in enumerate(ranking, start=1):
        if d in rel:
            hits += 1
            precisions.append(hits / i)
            recalls.append(hits / len(rel))
    precisions, recalls = np.array(precisions), np.array(recalls)
    out = []
    for r in levels:
        mask = recalls >= r - 1e-12
        out.append(precisions[mask].max() if mask.any() else 0.0)
    return np.array(out)


def evaluate_run(ranking, qrels) -> dict[str, float]:
    """All per-query metrics used in the report."""
    return {
        "P@10": precision_at_k(ranking, qrels, 10),
        "Recall@10": recall_at_k(ranking, qrels, 10),
        "Recall@100": recall_at_k(ranking, qrels, 100),
        "AP": average_precision(ranking, qrels),
        "nDCG@10": ndcg_at_k(ranking, qrels, 10),
        "RR": reciprocal_rank(ranking, qrels),
    }


def paired_randomization_test(a, b, n_resamples: int = 20000, seed: int = 7) -> float:
    """Two-sided paired sign-flip randomisation test on per-query scores."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    observed = abs(diff.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_resamples, diff.size))
    null = np.abs((signs * diff).mean(axis=1))
    return float((np.sum(null >= observed - 1e-12) + 1) / (n_resamples + 1))
