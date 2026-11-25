"""Run the full evaluation and write every table used by the report/slides.

Experiments
  A. Ad-hoc topical retrieval: 25 queries x graded pooled qrels (main result)
  B. Ablations: FirstP vs MaxP passages, RRF component sets, RRF constant k
  C. Known-item headline search: every unique headline is a query whose only
     relevant document is its own article (title removed from the index)
  D. Efficiency: index build time and per-query latency

Usage:  python scripts/evaluate.py
"""
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bbcir.data import DATA_DIR, ensure_articles, load_articles  # noqa: E402
from bbcir.engine import MODEL_NAMES, SearchEngine  # noqa: E402
from bbcir.metrics import (RECALL_LEVELS, evaluate_run, interpolated_pr,  # noqa: E402
                           paired_randomization_test)
from bbcir.models import rank, reciprocal_rank_fusion  # noqa: E402
from bbcir.preprocess import explain, preprocess  # noqa: E402

OUT = ROOT / "outputs"
METRICS = ["P@10", "Recall@100", "AP", "nDCG@10"]
DEPTH = 1000


def ids(engine, ranking):
    return [engine.doc_ids[i] for i in ranking]


def topical_runs(engine, query, q_vec):
    """Main runs plus every ablation run for one query (doc-id lists)."""
    base = engine.run_all(query, depth=DEPTH, q_vec=q_vec)
    r = {m: base[m][0] for m in MODEL_NAMES}
    n = engine.n_docs
    first_p = rank(engine.dense.score_vector(q_vec, pooling="first"), DEPTH)
    runs = {m: r[m] for m in MODEL_NAMES}
    runs["ST FirstP (opening passage only)"] = first_p
    runs["RRF BM25 + ST FirstP"] = reciprocal_rank_fusion([r["BM25"], first_p], n, 60, DEPTH)[0]
    runs["RRF TF-IDF + ST"] = reciprocal_rank_fusion([r["TF-IDF"], r["Sentence Transformer"]], n, 60, DEPTH)[0]
    runs["RRF TF-IDF + BM25 (lexical only)"] = reciprocal_rank_fusion([r["TF-IDF"], r["BM25"]], n, 60, DEPTH)[0]
    runs["RRF TF-IDF + BM25 + ST"] = reciprocal_rank_fusion(
        [r["TF-IDF"], r["BM25"], r["Sentence Transformer"]], n, 60, DEPTH)[0]
    for k in (1, 5, 10, 20, 40, 60, 100, 200, 500):
        runs[f"k={k}"] = reciprocal_rank_fusion([r["BM25"], r["Sentence Transformer"]], n, k, DEPTH)[0]
    return {name: ids(engine, rk) for name, rk in runs.items()}


def mean_table(per_query: pd.DataFrame, systems, metrics=METRICS) -> pd.DataFrame:
    return per_query[per_query.system.isin(systems)].groupby("system", sort=False)[metrics].mean()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    ensure_articles()  # downloads and verifies the official release if needed
    articles = load_articles()
    queries = json.loads((DATA_DIR / "queries.json").read_text())
    qrels = json.loads((DATA_DIR / "qrels.json").read_text())
    summary = json.loads((DATA_DIR / "corpus_summary.json").read_text())

    # ------------------------------------------------------------ A + B ----
    t0 = time.perf_counter()
    engine = SearchEngine(articles)
    build_seconds = time.perf_counter() - t0
    q_texts = [q["query"] for q in queries]
    q_vecs = engine.dense.encode_queries(q_texts)

    rows, pr_rows, run_cache = [], [], {}
    for q, q_vec in zip(queries, q_vecs):
        runs = topical_runs(engine, q["query"], q_vec)
        run_cache[q["qid"]] = runs
        for system, ranking in runs.items():
            m = evaluate_run(ranking, qrels[q["qid"]])
            judged = qrels[q["qid"]]
            rows.append({"qid": q["qid"], "style": q["style"], "category": q["category"],
                         "system": system, **m,
                         "Judged@10": sum(d in judged for d in ranking[:10]) / 10,
                         "Judged@20": sum(d in judged for d in ranking[:20]) / 20})
            if system in MODEL_NAMES:
                pr = interpolated_pr(ranking, qrels[q["qid"]])
                pr_rows += [{"qid": q["qid"], "system": system, "recall": r, "precision": p}
                            for r, p in zip(RECALL_LEVELS, pr)]
    per_query = pd.DataFrame(rows)
    per_query.to_csv(OUT / "per_query_metrics_all_runs.csv", index=False)

    main_tbl = mean_table(per_query, MODEL_NAMES, METRICS + ["Recall@10", "RR", "Judged@10", "Judged@20"]).loc[MODEL_NAMES]
    main_tbl.to_csv(OUT / "metrics_summary.csv")
    pr = pd.DataFrame(pr_rows).groupby(["system", "recall"], sort=False)["precision"].mean().reset_index()
    pr.to_csv(OUT / "precision_recall_11pt.csv", index=False)

    style_tbl = (per_query[per_query.system.isin(MODEL_NAMES)]
                 .groupby(["style", "system"], sort=False)[["AP", "nDCG@10"]].mean().reset_index())
    style_tbl.to_csv(OUT / "metrics_by_query_style.csv", index=False)
    cat_tbl = (per_query[per_query.system.isin(MODEL_NAMES)]
               .groupby(["category", "system"], sort=False)[["AP", "nDCG@10"]].mean().reset_index())
    cat_tbl.to_csv(OUT / "metrics_by_category.csv", index=False)

    wide = per_query[per_query.system.isin(MODEL_NAMES)].pivot(index="qid", columns="system", values=["AP", "nDCG@10"])
    sig_rows = []
    for metric in ("AP", "nDCG@10", "P@10"):
        tab = per_query[per_query.system.isin(MODEL_NAMES)].pivot(index="qid", columns="system", values=metric)
        for other in ("TF-IDF", "BM25", "Sentence Transformer"):
            diff = tab["Hybrid RRF"] - tab[other]
            sig_rows.append({"metric": metric, "comparison": f"Hybrid RRF vs {other}",
                             "mean_diff": diff.mean(), "wins": int((diff > 1e-9).sum()),
                             "ties": int((diff.abs() <= 1e-9).sum()), "losses": int((diff < -1e-9).sum()),
                             "p_value": paired_randomization_test(tab["Hybrid RRF"], tab[other])})
        d = tab["Sentence Transformer"] - tab["BM25"]
        sig_rows.append({"metric": metric, "comparison": "Sentence Transformer vs BM25", "mean_diff": d.mean(),
                         "wins": int((d > 1e-9).sum()), "ties": int((d.abs() <= 1e-9).sum()),
                         "losses": int((d < -1e-9).sum()),
                         "p_value": paired_randomization_test(tab["Sentence Transformer"], tab["BM25"])})
    sig = pd.DataFrame(sig_rows)
    sig.to_csv(OUT / "significance_tests.csv", index=False)

    ablation_systems = ["Sentence Transformer", "ST FirstP (opening passage only)", "Hybrid RRF",
                        "RRF BM25 + ST FirstP", "RRF TF-IDF + ST", "RRF TF-IDF + BM25 (lexical only)",
                        "RRF TF-IDF + BM25 + ST"]
    ablation = mean_table(per_query, ablation_systems).loc[ablation_systems]
    ablation.to_csv(OUT / "ablation_components.csv")
    k_systems = [s for s in per_query.system.unique() if s.startswith("k=")]
    k_tbl = mean_table(per_query, k_systems).loc[k_systems]
    k_tbl.index = [int(s[2:]) for s in k_tbl.index]
    k_tbl.index.name = "rrf_k"
    k_tbl.to_csv(OUT / "ablation_rrf_k.csv")

    # Case studies: top-5 for two contrasting queries.
    case = {}
    for qid in ("Q07", "Q13", "Q25"):
        q = next(x for x in queries if x["qid"] == qid)
        case[qid] = {"query": q["query"], "style": q["style"], "runs": {}}
        for system in MODEL_NAMES:
            top = run_cache[qid][system][:5]
            case[qid]["runs"][system] = [
                {"doc_id": d, "title": articles.set_index("doc_id").loc[d, "title"],
                 "grade": qrels[qid].get(d, 0)} for d in top]

    # ------------------------------------------------------------- D ----
    # Per-component latency (median over 10 timed passes after one warm-up pass).
    # Hybrid = BM25 + Sentence Transformer + the RRF fusion step.
    parts = {"Preprocess query": [], "TF-IDF": [], "BM25": [], "Sentence Transformer": [], "RRF fusion step": []}
    for rep in range(11):
        for qt in q_texts:
            t = time.perf_counter(); toks = preprocess(qt); t_pre = time.perf_counter() - t
            t = time.perf_counter(); rank(engine.tfidf.score(toks), DEPTH, True); t_tf = time.perf_counter() - t
            t = time.perf_counter(); bm_r = rank(engine.bm25.score(toks), DEPTH, True); t_bm = time.perf_counter() - t
            t = time.perf_counter(); v = engine.dense.encode_queries([qt])[0]
            de_r = rank(engine.dense.score_vector(v), DEPTH); t_de = time.perf_counter() - t
            t = time.perf_counter(); reciprocal_rank_fusion([bm_r, de_r], engine.n_docs, 60, DEPTH)
            t_rrf = time.perf_counter() - t
            if rep:
                for key, val in zip(parts, (t_pre, t_tf, t_bm, t_de, t_rrf)):
                    parts[key].append(val)
    med = {k: 1000 * float(np.median(v)) for k, v in parts.items()}
    latency_ms = {"TF-IDF": med["Preprocess query"] + med["TF-IDF"],
                  "BM25": med["Preprocess query"] + med["BM25"],
                  "Sentence Transformer": med["Sentence Transformer"],
                  "Hybrid RRF": med["Preprocess query"] + med["BM25"] + med["Sentence Transformer"] + med["RRF fusion step"],
                  "components": med}

    # ------------------------------------------------------------- C ----
    t0 = time.perf_counter()
    body_engine = SearchEngine(articles, text_field="body")
    ki_build = time.perf_counter() - t0
    titles = articles["title"].str.strip().str.lower()
    unique = articles[~titles.duplicated(keep=False)].reset_index(drop=True)
    ki_vecs = body_engine.dense.encode_queries(unique["title"].tolist())
    ki_rows = []
    for (_, row), q_vec in zip(unique.iterrows(), ki_vecs):
        runs = body_engine.run_all(row["title"], depth=DEPTH, q_vec=q_vec)
        rel = {row["doc_id"]: 1}
        for system in MODEL_NAMES:
            ranking = ids(body_engine, runs[system][0])
            m = evaluate_run(ranking, rel)
            pos = ranking.index(row["doc_id"]) + 1 if row["doc_id"] in ranking else None
            ki_rows.append({"doc_id": row["doc_id"], "system": system, "MRR": m["RR"],
                            "Success@1": float(pos == 1), "Recall@10": m["Recall@10"],
                            "Recall@100": m["Recall@100"], "nDCG@10": m["nDCG@10"], "rank": pos})
    ki = pd.DataFrame(ki_rows)
    ki.to_csv(OUT / "known_item_per_query.csv", index=False)
    ki_tbl = ki.groupby("system", sort=False)[["MRR", "Success@1", "Recall@10", "Recall@100", "nDCG@10"]].mean().loc[MODEL_NAMES]
    ki_tbl.to_csv(OUT / "known_item_summary.csv")
    ki_wide = ki.pivot(index="doc_id", columns="system", values="MRR")
    ki_sig = {f"Hybrid RRF vs {o}": paired_randomization_test(ki_wide["Hybrid RRF"], ki_wide[o], n_resamples=5000)
              for o in ("TF-IDF", "BM25", "Sentence Transformer")}

    # ------------------------------------------------------------ save ----
    import sentence_transformers
    import sklearn
    import torch
    stats = engine.index.stats()
    example_terms = {}
    for word in ("iPod", "Yukos", "Kenteris", "election"):
        term = preprocess(word)[0]
        docs, tfs = engine.index.postings(term)
        order = np.argsort(-tfs, kind="stable")[:5]
        example_terms[word] = {"stem": term, "df": int(docs.size), "idf_bm25": round(engine.bm25.idf(term), 3),
                               "top_postings": [[engine.doc_ids[int(docs[i])], int(tfs[i])] for i in order]}
    judged = {q: len(v) for q, v in qrels.items()}
    relevant = {q: sum(g > 0 for g in v.values()) for q, v in qrels.items()}
    pool = json.loads((DATA_DIR / "pool.json").read_text())["pool"]
    results = {
        "corpus": summary,
        "index_stats": stats,
        "top_df_terms": engine.index.top_terms(10),
        "example_postings": example_terms,
        "tfidf_vocabulary": int(engine.tfidf.doc_vectors.shape[1]),
        "dense": {"model": engine.dense.model_name, "device": engine.dense.device,
                  "passages": engine.dense.n_passages, "dim": int(engine.dense.passage_vectors.shape[1]),
                  "passages_per_doc": round(engine.dense.n_passages / engine.n_docs, 2)},
        "timings_seconds": {k: round(v, 3) for k, v in engine.timings.items()},
        "embeddings_from_cache": engine.dense.loaded_from_cache,
        "engine_build_seconds": round(build_seconds, 2),
        "latency_ms_median": {k: (round(v, 3) if not isinstance(v, dict) else {a: round(b, 3) for a, b in v.items()}) for k, v in latency_ms.items()},
        "qrels": {"queries": len(queries), "judged_pairs": sum(judged.values()),
                  "relevant": sum(relevant.values()),
                  "highly_relevant": sum(g == 2 for v in qrels.values() for g in v.values()),
                  "relevant_per_query_min": min(relevant.values()), "relevant_per_query_max": max(relevant.values()),
                  "relevant_per_query_mean": round(float(np.mean(list(relevant.values()))), 1),
                  "pool_depth": 20, "pooled_pairs": sum(len(v) for v in pool.values()),
                  "relevant_found_by_sweep": sum(1 for q in pool for d, g in qrels[q].items() if g > 0 and d not in pool[q]),
                  "relevant_per_query": relevant},
        "main": main_tbl.round(4).reset_index().to_dict(orient="records"),
        "by_style": style_tbl.round(4).to_dict(orient="records"),
        "by_category": cat_tbl.round(4).to_dict(orient="records"),
        "significance": sig.round(4).to_dict(orient="records"),
        "ablation_components": ablation.round(4).reset_index().to_dict(orient="records"),
        "ablation_rrf_k": k_tbl.round(4).reset_index().to_dict(orient="records"),
        "known_item": {"queries": int(len(unique)), "excluded_duplicate_titles": int(len(articles) - len(unique)),
                       "build_seconds": round(ki_build, 2),
                       "table": ki_tbl.round(4).reset_index().to_dict(orient="records"),
                       "p_values_mrr": {k: round(v, 4) for k, v in ki_sig.items()}},
        "case_studies": case,
        "preprocessing_example": explain(articles.loc[articles.doc_id == "tec-040", "title"].iloc[0] + ". " +
                                         articles.loc[articles.doc_id == "tec-040", "body"].iloc[0].split(". ")[0] + "."),
        "per_query_main": per_query[per_query.system.isin(MODEL_NAMES)][["qid", "system", "P@10", "Recall@100", "AP", "nDCG@10"]].round(4).to_dict(orient="records"),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "machine": platform.machine(), "torch": torch.__version__,
                        "sentence_transformers": sentence_transformers.__version__,
                        "scikit_learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=float))

    pd.set_option("display.width", 200)
    print("\n== Main topical results (25 queries) ==\n", main_tbl.round(4))
    print("\n== By query style ==\n", style_tbl.round(4))
    print("\n== Significance ==\n", sig.round(4))
    print("\n== Ablation: components ==\n", ablation.round(4))
    print("\n== Ablation: RRF k ==\n", k_tbl.round(4))
    print("\n== Known-item (headline -> article) ==\n", ki_tbl.round(4), "\n", ki_sig)
    print("\nlatency ms:", latency_ms, "\nbuild:", engine.timings)


if __name__ == "__main__":
    main()
