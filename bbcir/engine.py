"""SearchEngine: builds every index/model once and answers queries."""
from __future__ import annotations

import html
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import PROJECT_ROOT
from .index import InvertedIndex
from .models import (BM25Retriever, DenseRetriever, TfidfRetriever, rank,
                     reciprocal_rank_fusion)
from .preprocess import clean, preprocess, stem, tokenize

MODEL_NAMES = ["TF-IDF", "BM25", "Sentence Transformer", "Hybrid RRF"]
CACHE_DIR = PROJECT_ROOT / "outputs" / "cache"
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class SearchEngine:
    """Facade over the inverted index and the four ranking models.

    The hybrid run is Reciprocal Rank Fusion of BM25 and the Sentence
    Transformer (k = 60) - one lexical and one semantic ranker - fixed before
    any evaluation was run. Other fusion settings are reported as ablations.
    """

    def __init__(self, articles: pd.DataFrame, text_field: str = "text", rrf_k: int = 60,
                 rrf_components: tuple[str, ...] = ("BM25", "Sentence Transformer"),
                 cache_dir: str | Path | None = CACHE_DIR, **dense_kwargs):
        self.articles = articles.reset_index(drop=True)
        self.doc_ids = self.articles["doc_id"].tolist()
        self.n_docs = len(self.doc_ids)
        self.rrf_k, self.rrf_components = rrf_k, tuple(rrf_components)
        texts = self.articles[text_field].tolist()
        self.timings: dict[str, float] = {}

        t = time.perf_counter()
        self.doc_tokens = [preprocess(x) for x in texts]
        self.timings["Preprocessing"] = time.perf_counter() - t

        t = time.perf_counter()
        self.index = InvertedIndex(self.doc_tokens)
        self.timings["Inverted index"] = time.perf_counter() - t

        t = time.perf_counter()
        self.tfidf = TfidfRetriever(self.doc_tokens)
        self.timings["TF-IDF document vectors"] = time.perf_counter() - t

        t = time.perf_counter()
        self.bm25 = BM25Retriever(self.index)
        self.timings["BM25 length norms"] = time.perf_counter() - t

        t = time.perf_counter()
        self.dense = DenseRetriever(texts, cache_dir=cache_dir, **dense_kwargs)
        self.timings["Passage embeddings"] = time.perf_counter() - t

    # ------------------------------------------------------------------
    def run_all(self, query: str, depth: int = 1000, q_vec: np.ndarray | None = None,
                rrf_k: int | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Ranking (doc indices) and full score vector for every model."""
        toks = preprocess(query)
        s_tfidf = self.tfidf.score(toks)
        s_bm25 = self.bm25.score(toks)
        if q_vec is None:
            q_vec = self.dense.encode_queries([query])[0]
        s_dense = self.dense.score_vector(q_vec)
        runs = {
            "TF-IDF": (rank(s_tfidf, depth, positive_only=True), s_tfidf),
            "BM25": (rank(s_bm25, depth, positive_only=True), s_bm25),
            "Sentence Transformer": (rank(s_dense, depth), s_dense),
        }
        fused_rank, fused = reciprocal_rank_fusion(
            [runs[m][0] for m in self.rrf_components], self.n_docs,
            k=rrf_k or self.rrf_k, depth=depth)
        runs["Hybrid RRF"] = (fused_rank, fused)
        return runs

    def search_all(self, query: str, k: int = 10, categories: list[str] | None = None,
                   models: tuple[str, ...] = tuple(MODEL_NAMES)) -> tuple[dict[str, list[dict]], float]:
        """Rank once with every model; return ({model: top-k results}, elapsed milliseconds)."""
        start = time.perf_counter()
        runs = self.run_all(query, depth=self.n_docs)
        elapsed_ms = 1000 * (time.perf_counter() - start)
        q_stems = set(preprocess(query))
        return {m: self._results(*runs[m], q_stems, k, categories) for m in models}, elapsed_ms

    def search(self, query: str, model: str = "Hybrid RRF", k: int = 10,
               categories: list[str] | None = None) -> list[dict]:
        """Top-k results with title, category, score and a highlighted snippet."""
        return self.search_all(query, k, categories, models=(model,))[0][model]

    def ranked_ids(self, query: str, model: str, depth: int = 1000) -> list[str]:
        """Document ids in rank order (used to score a query against relevance judgments)."""
        return [self.doc_ids[i] for i in self.run_all(query, depth=depth)[model][0]]

    def _results(self, ranking, scores, q_stems, k, categories) -> list[dict]:
        results = []
        for idx in ranking:
            row = self.articles.iloc[int(idx)]
            if categories and row["category"] not in categories:
                continue
            results.append({
                "rank": len(results) + 1, "doc_id": row["doc_id"], "category": row["category"],
                "title": row["title"], "score": float(scores[idx]),
                "snippet_html": snippet_html(row["body"], q_stems),
                "matched_terms": sorted(q_stems & set(self.doc_tokens[int(idx)])),
                "body": row["body"],
            })
            if len(results) == k:
                break
        return results


def snippet_html(body: str, q_stems: set[str], max_sentences: int = 2, max_chars: int = 320) -> str:
    """Pick the sentences with most query-term matches and <mark> the matches."""
    sentences = [s for s in _SENTENCE.split(body.replace("\n", " ")) if s.strip()]
    if not sentences:
        return ""

    def hits(s: str) -> int:
        return len(q_stems & {stem(t) for t in tokenize(clean(s))})

    best = sorted(range(len(sentences)), key=lambda i: (-hits(sentences[i]), i))[:max_sentences]
    chosen = [sentences[i] for i in sorted(best)] if hits(sentences[best[0]]) else sentences[:max_sentences]
    text = " … ".join(chosen)
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + " …"

    def mark(m: re.Match) -> str:
        word = m.group(0)
        w = clean(word)
        return f"<mark>{html.escape(word)}</mark>" if w and stem(w) in q_stems else html.escape(word)

    return re.sub(r"[A-Za-z0-9']+|[^A-Za-z0-9']+",
                  lambda m: mark(m) if m.group(0)[0].isalnum() else html.escape(m.group(0)), text)
