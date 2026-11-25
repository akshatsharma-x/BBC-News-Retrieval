"""Ranking models: TF-IDF, Okapi BM25, Sentence Transformer and RRF."""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .index import InvertedIndex


def _identity(tokens):
    return tokens


def rank(scores: np.ndarray, depth: int = 1000, positive_only: bool = False) -> np.ndarray:
    """Document indices sorted by descending score (ties -> lower index first)."""
    order = np.lexsort((np.arange(scores.size), -scores))
    if positive_only:
        order = order[scores[order] > 0]
    return order[:depth]


# ---------------------------------------------------------------------------
class TfidfRetriever:
    """Vector-space model: sublinear TF x smoothed IDF, L2-normalised, cosine.

    w(t,d) = (1 + ln tf(t,d)) * (ln((1 + N) / (1 + df(t))) + 1), then each
    document vector is scaled to unit length, so the dot product with the
    (equally weighted and normalised) query vector is the cosine similarity.
    """
    name = "TF-IDF"

    def __init__(self, docs_tokens: list[list[str]]):
        self.vectorizer = TfidfVectorizer(analyzer=_identity, sublinear_tf=True,
                                          smooth_idf=True, norm="l2", lowercase=False)
        self.doc_vectors = self.vectorizer.fit_transform(docs_tokens).tocsr()

    def score(self, query_tokens: list[str]) -> np.ndarray:
        q = self.vectorizer.transform([query_tokens])
        return np.asarray((self.doc_vectors @ q.T).todense()).ravel()

    def search(self, query_tokens: list[str], depth: int = 1000) -> np.ndarray:
        return rank(self.score(query_tokens), depth, positive_only=True)


# ---------------------------------------------------------------------------
class BM25Retriever:
    """Okapi BM25 scored term-at-a-time from the inverted index.

    idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))   (non-negative, Lucene form)
    score  = sum_t idf(t) * tf (k1 + 1) / (tf + k1 (1 - b + b |d| / avgdl))
    """
    name = "BM25"

    def __init__(self, index: InvertedIndex, k1: float = 1.2, b: float = 0.75):
        self.index, self.k1, self.b = index, k1, b
        self._norm = k1 * (1 - b + b * index.doc_len / index.avg_doc_len)

    def idf(self, term: str) -> float:
        n, df = self.index.n_docs, self.index.df(term)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.index.n_docs)
        for term, qtf in Counter(query_tokens).items():
            docs, tfs = self.index.postings(term)
            if docs.size:
                scores[docs] += qtf * self.idf(term) * tfs * (self.k1 + 1) / (tfs + self._norm[docs])
        return scores

    def search(self, query_tokens: list[str], depth: int = 1000) -> np.ndarray:
        return rank(self.score(query_tokens), depth, positive_only=True)


# ---------------------------------------------------------------------------
def chunk_words(text: str, size: int, stride: int) -> list[str]:
    """Split text into overlapping word windows (passages)."""
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    starts = list(range(0, len(words) - size + 1, stride))
    if starts[-1] + size < len(words):
        starts.append(len(words) - size)
    return [" ".join(words[s:s + size]) for s in starts]


def best_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class DenseRetriever:
    """Bi-encoder retrieval with a pre-trained Sentence Transformer.

    Each article is split into overlapping ~160-word passages (the encoder
    reads at most 256 word-pieces), every passage is embedded once, and a
    document's score is the maximum cosine similarity over its passages
    ("MaxP"). pooling="first" scores only the opening passage ("FirstP").
    """
    name = "Sentence Transformer"

    def __init__(self, texts: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 chunk_size: int = 160, stride: int = 120, cache_dir: str | Path | None = None,
                 batch_size: int = 64, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.device = device or best_device()
        self.model = SentenceTransformer(model_name, device=self.device)
        self.n_docs = len(texts)
        passages, owner = [], []
        for doc, text in enumerate(texts):
            for p in chunk_words(text, chunk_size, stride):
                passages.append(p)
                owner.append(doc)
        self.passage_doc = np.asarray(owner, dtype=np.int64)
        self.first_passage = np.searchsorted(self.passage_doc, np.arange(self.n_docs))
        self.n_passages = len(passages)

        key = hashlib.sha1(("\x1e".join(texts) + f"|{model_name}|{chunk_size}|{stride}").encode()).hexdigest()[:16]
        cache = Path(cache_dir) / f"passages_{key}.npy" if cache_dir else None
        if cache is not None and cache.exists():
            self.passage_vectors = np.load(cache)
            self.loaded_from_cache = True
        else:
            self.passage_vectors = self.model.encode(
                passages, batch_size=batch_size, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
            self.loaded_from_cache = False
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, self.passage_vectors)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return self.model.encode(queries, batch_size=128, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    def score_vector(self, q_vec: np.ndarray, pooling: str = "max") -> np.ndarray:
        sims = self.passage_vectors @ q_vec
        if pooling == "first":
            return sims[self.first_passage].astype(np.float64)
        doc_scores = np.full(self.n_docs, -np.inf)
        np.maximum.at(doc_scores, self.passage_doc, sims)
        return doc_scores

    def score(self, query: str, pooling: str = "max") -> np.ndarray:
        return self.score_vector(self.encode_queries([query])[0], pooling)

    def search(self, query: str, depth: int = 1000, pooling: str = "max") -> np.ndarray:
        return rank(self.score(query, pooling), depth)


# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(rankings: list[np.ndarray], n_docs: int, k: int = 60,
                           depth: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Fuse ranked lists: RRF(d) = sum_m 1 / (k + rank_m(d)), 1-based ranks.

    A document missing from a list receives no contribution from it.
    Returns (fused ranking, fused score array).
    """
    fused = np.zeros(n_docs)
    for ranking in rankings:
        fused[ranking] += 1.0 / (k + np.arange(1, ranking.size + 1))
    return rank(fused, depth, positive_only=True), fused
