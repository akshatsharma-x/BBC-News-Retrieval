"""Convert the raw BBC release (bbc/<category>/<nnn>.txt) into data/bbc_articles.csv.

Usage:  python scripts/prepare_data.py data/raw/bbc
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bbcir.data import ARTICLES_CSV, DATA_DIR, deduplicate, read_raw_corpus  # noqa: E402


def main(raw_dir: str) -> None:
    raw = read_raw_corpus(raw_dir)
    kept, dropped = deduplicate(raw)
    kept.to_csv(ARTICLES_CSV, index=False)
    words = kept["body"].str.split().str.len()
    summary = {
        "raw_articles": len(raw),
        "exact_duplicates_removed": len(dropped),
        "unique_articles": len(kept),
        "raw_per_category": raw["category"].value_counts().sort_index().to_dict(),
        "unique_per_category": kept["category"].value_counts().sort_index().to_dict(),
        "body_words_mean": round(float(words.mean()), 1),
        "body_words_median": float(words.median()),
        "body_words_min": int(words.min()),
        "body_words_max": int(words.max()),
        "duplicate_titles_remaining": int(kept["title"].str.lower().duplicated(keep=False).sum()),
    }
    (DATA_DIR / "corpus_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/bbc")
