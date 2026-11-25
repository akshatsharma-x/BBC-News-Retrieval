"""Depth-k pooling for relevance assessment.

Runs every evaluated system for each query, takes the union of their top-k
documents, and writes the pool in document-id order (no system or rank
information is shown, so the assessor judges blind).

Usage:  python scripts/build_pool.py [depth]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bbcir.data import DATA_DIR, load_articles  # noqa: E402
from bbcir.engine import SearchEngine  # noqa: E402
from bbcir.models import rank  # noqa: E402


def main(depth: int = 20) -> None:
    articles = load_articles()
    engine = SearchEngine(articles)
    queries = json.loads((DATA_DIR / "queries.json").read_text())
    pool, lines = {}, []
    for q in queries:
        runs = engine.run_all(q["query"])
        q_vec = engine.dense.encode_queries([q["query"]])[0]
        first_p = rank(engine.dense.score_vector(q_vec, pooling="first"), depth)
        members = set(first_p.tolist())
        for ranking, _ in runs.values():
            members.update(ranking[:depth].tolist())
        ids = sorted(engine.doc_ids[i] for i in members)
        pool[q["qid"]] = ids
        lines.append(f"\n### {q['qid']} | {q['query']}\nNARRATIVE: {q['narrative']}\n")
        for doc_id in ids:
            row = articles.loc[articles.doc_id == doc_id].iloc[0]
            lead = " ".join(row["body"].split()[:28])
            lines.append(f"{doc_id} | {row['title']} | {lead}")
    (DATA_DIR / "pool.json").write_text(json.dumps({"depth": depth, "pool": pool}, indent=1))
    (ROOT / "outputs" / "pool_for_judging.txt").write_text("\n".join(lines))
    sizes = [len(v) for v in pool.values()]
    print(f"pooled {sum(sizes)} query-document pairs, {min(sizes)}-{max(sizes)} per query")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
