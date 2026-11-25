"""Inverted index: term -> postings list of (document, term frequency)."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


class InvertedIndex:
    """A positional-free inverted index over preprocessed token lists.

    Postings are stored as two aligned NumPy arrays per term (document ids
    and term frequencies) so that scoring can be vectorised term-at-a-time.
    """

    def __init__(self, docs_tokens: list[list[str]]):
        self.n_docs = len(docs_tokens)
        self.doc_len = np.array([len(toks) for toks in docs_tokens], dtype=np.float64)
        self.avg_doc_len = float(self.doc_len.mean())
        raw: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc, toks in enumerate(docs_tokens):
            for term, tf in Counter(toks).items():
                raw[term].append((doc, tf))
        self._docs: dict[str, np.ndarray] = {}
        self._tfs: dict[str, np.ndarray] = {}
        for term, plist in raw.items():
            arr = np.asarray(plist, dtype=np.int64)
            self._docs[term] = arr[:, 0]
            self._tfs[term] = arr[:, 1].astype(np.float64)
        self.vocabulary = sorted(self._docs)

    # -- lookups -----------------------------------------------------------
    def __contains__(self, term: str) -> bool:
        return term in self._docs

    def df(self, term: str) -> int:
        """Document frequency: number of documents containing the term."""
        docs = self._docs.get(term)
        return 0 if docs is None else int(docs.size)

    def postings(self, term: str) -> tuple[np.ndarray, np.ndarray]:
        """(doc ids, term frequencies) for a term; empty arrays if unseen."""
        if term not in self._docs:
            return np.empty(0, dtype=np.int64), np.empty(0)
        return self._docs[term], self._tfs[term]

    def boolean_and(self, terms: list[str]) -> np.ndarray:
        """Documents that contain every term (postings intersection)."""
        result = None
        for term in dict.fromkeys(terms):
            docs = self.postings(term)[0]
            result = docs if result is None else np.intersect1d(result, docs, assume_unique=True)
        return np.empty(0, dtype=np.int64) if result is None else result

    def boolean_or(self, terms: list[str]) -> np.ndarray:
        """Documents that contain at least one term (postings union)."""
        arrays = [self.postings(t)[0] for t in dict.fromkeys(terms)]
        return np.unique(np.concatenate(arrays)) if arrays else np.empty(0, dtype=np.int64)

    # -- statistics ----------------------------------------------------------
    def stats(self) -> dict[str, float]:
        sizes = np.array([d.size for d in self._docs.values()])
        return {
            "documents": self.n_docs,
            "vocabulary_terms": len(self._docs),
            "total_postings": int(sizes.sum()),
            "total_tokens": int(self.doc_len.sum()),
            "avg_doc_length_tokens": round(self.avg_doc_len, 1),
            "avg_postings_length": round(float(sizes.mean()), 2),
            "max_postings_length": int(sizes.max()),
            "hapax_terms": int((sizes == 1).sum()),
        }

    def top_terms(self, n: int = 10) -> list[tuple[str, int]]:
        """The n terms with the longest postings lists (highest df)."""
        return sorted(((t, d.size) for t, d in self._docs.items()), key=lambda x: -x[1])[:n]
