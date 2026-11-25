"""Text preprocessing for the lexical models (TF-IDF and BM25).

Pipeline: cleaning -> tokenisation -> stop-word removal -> Porter stemming.
Exactly the same function is applied to documents and to queries, so that
query terms and index terms live in the same vocabulary.

The Sentence-Transformer model is *not* fed these tokens: neural encoders are
trained on natural text, so they receive the original (uncleaned) wording.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from nltk.stem import PorterStemmer

_STOPWORD_FILE = Path(__file__).with_name("stopwords_en.txt")
STOPWORDS: frozenset[str] = frozenset(
    w.strip() for w in _STOPWORD_FILE.read_text().split() if w.strip())

_POSSESSIVE = re.compile(r"(?<=[a-z0-9])'s\b")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_TOKEN = re.compile(r"[a-z0-9]+")
_STEMMER = PorterStemmer()


def clean(text: str) -> str:
    """Normalise Unicode, lower-case, drop possessive 's and punctuation."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("’", "'")
    text = _POSSESSIVE.sub("", text)
    return _NON_ALNUM.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Split cleaned text into alphanumeric word tokens."""
    return _TOKEN.findall(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove NLTK English stop words and single-character tokens."""
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


@lru_cache(maxsize=200_000)
def stem(token: str) -> str:
    return _STEMMER.stem(token)


def preprocess(text: str) -> list[str]:
    """Full lexical pipeline used for both documents and queries."""
    return [stem(t) for t in remove_stopwords(tokenize(clean(text)))]


def explain(text: str) -> dict[str, object]:
    """Return every intermediate stage (used in the report and the UI)."""
    cleaned = clean(text)
    tokens = tokenize(cleaned)
    no_stop = remove_stopwords(tokens)
    stemmed = [stem(t) for t in no_stop]
    return {"original": text, "cleaned": cleaned, "tokens": tokens,
            "without_stopwords": no_stop, "stemmed": stemmed}
