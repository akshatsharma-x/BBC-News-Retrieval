"""Loading the BBC News corpus (Greene & Cunningham, ICML 2006).

The raw release is one plain-text file per article, grouped into five
category folders. The first line of each file is the headline and the rest is
the article body.
"""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

CATEGORIES = ["business", "entertainment", "politics", "sport", "tech"]
CATEGORY_PREFIX = {"business": "bus", "entertainment": "ent", "politics": "pol",
                   "sport": "spo", "tech": "tec"}
CATEGORY_LABEL = {"business": "Business", "entertainment": "Entertainment",
                  "politics": "Politics", "sport": "Sport", "tech": "Technology"}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTICLES_CSV = DATA_DIR / "bbc_articles.csv"


def read_raw_corpus(raw_dir: str | Path) -> pd.DataFrame:
    """Read the raw `bbc/<category>/<nnn>.txt` files into a DataFrame."""
    raw_dir = Path(raw_dir)
    rows = []
    for cat in CATEGORIES:
        for path in sorted((raw_dir / cat).glob("*.txt")):
            raw = path.read_bytes()
            try:
                text = raw.decode("utf-8")      # every file in the release is valid UTF-8
            except UnicodeDecodeError:
                text = raw.decode("latin-1")
            text = html.unescape(text)          # 22 files contain entities such as &#163;
            lines = [ln.strip() for ln in text.splitlines()]
            title = next((ln for ln in lines if ln), "")
            first = lines.index(title)
            body = "\n".join(ln for ln in lines[first + 1:] if ln)
            rows.append({
                "doc_id": f"{CATEGORY_PREFIX[cat]}-{path.stem}",
                "category": cat,
                "source_file": f"{cat}/{path.name}",
                "title": title,
                "body": body,
            })
    return pd.DataFrame(rows)


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop exact duplicates (same title and body). Returns (kept, dropped)."""
    key = df["title"].str.strip().str.lower() + "\x00" + df["body"].str.strip().str.lower()
    dup_mask = key.duplicated(keep="first")
    return df[~dup_mask].reset_index(drop=True), df[dup_mask].reset_index(drop=True)


def load_articles(path: str | Path = ARTICLES_CSV) -> pd.DataFrame:
    """Load the processed article table written by scripts/prepare_data.py."""
    df = pd.read_csv(path, keep_default_na=False)
    df["text"] = df["title"] + ".\n" + df["body"]
    return df
