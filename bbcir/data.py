"""Loading the BBC News corpus (Greene & Cunningham, ICML 2006).

The raw release is one plain-text file per article, grouped into five
category folders. The first line of each file is the headline and the rest is
the article body.
"""
from __future__ import annotations

import hashlib
import html
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

CATEGORIES = ["business", "entertainment", "politics", "sport", "tech"]
CATEGORY_PREFIX = {"business": "bus", "entertainment": "ent", "politics": "pol",
                   "sport": "spo", "tech": "tec"}
CATEGORY_LABEL = {"business": "Business", "entertainment": "Entertainment",
                  "politics": "Politics", "sport": "Sport", "tech": "Technology"}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ARTICLES_CSV = DATA_DIR / "bbc_articles.csv"

# Official release. The server only offers plain HTTP, so the archive is verified
# against the SHA-256 of the copy used for every result in this project.
DATASET_URL = "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"
DATASET_SHA256 = "a351031a62325b3d7ff13146eeb035c22feca8290913b5183d1ff3be35109495"


def download_dataset(url: str = DATASET_URL, dest: Path = RAW_DIR / "bbc-fulltext.zip",
                     sha256: str = DATASET_SHA256, timeout: float = 120) -> Path:
    """Download the official archive (if absent) and verify its checksum."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        tmp = dest.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as fh:
            while chunk := resp.read(1 << 16):
                fh.write(chunk)
        tmp.replace(dest)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if digest != sha256:
        dest.unlink()
        raise ValueError(f"checksum mismatch for {url}: got {digest}")
    return dest


def ensure_articles() -> Path:
    """Create data/bbc_articles.csv from the official release if it is missing.

    Lets a fresh clone or a cloud deployment start without the article text,
    which is not redistributed in the repository.
    """
    if ARTICLES_CSV.exists():
        return ARTICLES_CSV
    archive = download_dataset()
    root = RAW_DIR.resolve()
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():  # refuse entries that would escape data/raw
            if not (root / name).resolve().is_relative_to(root):
                raise ValueError(f"unsafe path in archive: {name}")
        zf.extractall(root)
    kept, _ = deduplicate(read_raw_corpus(root / "bbc"))
    tmp = ARTICLES_CSV.with_suffix(".part")
    kept.to_csv(tmp, index=False)
    tmp.replace(ARTICLES_CSV)
    return ARTICLES_CSV


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
