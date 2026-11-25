"""Export everything the browser-only search app needs into docs/app/data/.

The static app ranks with the same models as the Python engine: the inverted
index (postings + term frequencies), document lengths, a surface-form -> Porter
stem table, half-precision passage embeddings (rankings verified identical to
float32) and the benchmark data. Article text is limited to the headline and the
opening sentences; the full text is not published.

Usage:  python scripts/export_static.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bbcir.data import CATEGORIES, ensure_articles, load_articles  # noqa: E402
from bbcir.engine import SearchEngine  # noqa: E402
from bbcir.preprocess import STOPWORDS, clean, remove_stopwords, stem, tokenize  # noqa: E402
from ui import charts  # noqa: E402

OUT = ROOT / "docs" / "app" / "data"
EXAMPLES = ["films that took the most money at cinemas", "why the American currency is losing value",
            "Yukos oil company tax dispute", "fraudsters tricking people into revealing bank details online",
            "Australian Open tennis"]
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def lead(body: str, min_chars: int = 150, max_chars: int = 300) -> str:
    """Opening sentence(s): at most two, cut at a word boundary after max_chars."""
    out = ""
    for i, s in enumerate(_SENTENCE.split(body.replace("\n", " "))):
        out = f"{out} {s}".strip()
        if len(out) >= min_chars or i == 1:
            break
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0] + " …"
    return out


MODEL_DIR = ROOT / "docs" / "app" / "models" / "Xenova" / "all-MiniLM-L6-v2"
MODEL_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "onnx/model_fp16.onnx"]


def fetch_model() -> None:
    """Self-host the half-precision ONNX encoder (its rankings match PyTorch to 3 decimals of MAP)."""
    import urllib.request
    for name in MODEL_FILES:
        dest = MODEL_DIR / name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/{name}", dest)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fetch_model()
    ensure_articles()
    articles = load_articles()
    engine = SearchEngine(articles)
    res = json.loads((ROOT / "outputs" / "results.json").read_text())
    queries = json.loads((ROOT / "data" / "queries.json").read_text())
    qrels = json.loads((ROOT / "data" / "qrels.json").read_text())
    cat_index = {c: i for i, c in enumerate(CATEGORIES)}

    docs = [[r.doc_id, cat_index[r.category], r.title, lead(r.body)] for r in articles.itertuples()]
    (OUT / "docs.json").write_text(json.dumps(docs, ensure_ascii=False, separators=(",", ":")))

    terms = engine.index.vocabulary
    offsets, post_docs, post_tfs = [0], [], []
    for t in terms:
        d, f = engine.index.postings(t)
        post_docs.append(d.astype(np.uint16))
        post_tfs.append(f.astype(np.uint8))
        offsets.append(offsets[-1] + d.size)
    assert max(int(f.max()) for f in post_tfs) < 256 and engine.n_docs < 65536
    np.concatenate(post_docs).tofile(OUT / "post_docs.u16")
    np.concatenate(post_tfs).tofile(OUT / "post_tfs.u8")
    engine.index.doc_len.astype(np.uint16).tofile(OUT / "doc_len.u16")
    (OUT / "terms.json").write_text(json.dumps({"terms": terms, "offsets": offsets}, separators=(",", ":")))

    surface = set()
    for text in list(articles["text"]) + [q["query"] for q in queries] + EXAMPLES:
        surface.update(remove_stopwords(tokenize(clean(text))))
    stems = {w: stem(w) for w in sorted(surface)}
    (OUT / "stems.json").write_text(json.dumps(stems, separators=(",", ":")))
    (OUT / "stopwords.json").write_text(json.dumps(sorted(STOPWORDS)))

    engine.dense.passage_vectors.astype("<f2").tofile(OUT / "passages.f16")
    engine.dense.passage_doc.astype(np.uint16).tofile(OUT / "passage_doc.u16")

    main_tbl = {r["system"]: r for r in res["main"]}
    specs = {
        "metrics": charts.metric_bars(
            __import__("pandas").DataFrame(main_tbl).T,
            [("P@10", "P@10"), ("Recall@100", "Recall@100"), ("AP", "MAP"), ("nDCG@10", "nDCG@10")]).to_dict(),
        "pr": charts.pr_curve(__import__("pandas").read_csv(ROOT / "outputs" / "precision_recall_11pt.csv")).to_dict(),
        "style": charts.style_bars(__import__("pandas").DataFrame(res["by_style"])).to_dict(),
    }
    bench = {
        "queries": queries, "qrels": qrels, "main": res["main"], "known_item": res["known_item"],
        "per_query": res["per_query_main"], "by_style": res["by_style"],
        "corpus": res["corpus"], "index": res["index_stats"], "dense": res["dense"],
        "latency_python_ms": res["latency_ms_median"], "examples": EXAMPLES,
    }
    (OUT / "bench.json").write_text(json.dumps(bench, separators=(",", ":"), default=float))
    (OUT / "charts.json").write_text(json.dumps(specs, separators=(",", ":"), default=float))
    meta = {"n_docs": engine.n_docs, "avg_doc_len": engine.index.avg_doc_len, "k1": engine.bm25.k1, "b": engine.bm25.b,
            "rrf_k": engine.rrf_k, "categories": CATEGORIES, "n_terms": len(terms), "n_postings": offsets[-1],
            "n_passages": engine.dense.n_passages, "dim": int(engine.dense.passage_vectors.shape[1]),
            "model": "Xenova/all-MiniLM-L6-v2", "dtype": "fp16", "n_surface_forms": len(stems)}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    sizes = {p.name: f"{p.stat().st_size / 1024:.0f} KB" for p in sorted(OUT.iterdir())}
    print(json.dumps(sizes, indent=1))


if __name__ == "__main__":
    main()
