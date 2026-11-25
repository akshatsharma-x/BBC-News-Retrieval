"""BBC News Retrieval - Streamlit search interface ("night desk" editorial theme).

Run:  streamlit run app.py
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

from bbcir.data import ensure_articles, load_articles
from bbcir.engine import MODEL_NAMES, SearchEngine
from bbcir.metrics import evaluate_run
from bbcir.preprocess import preprocess
from ui import charts
from ui import components as ui

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
REPO_URL = "https://github.com/akshatsharma-x/BBC-News-Retrieval"
EXAMPLES = ["films that took the most money at cinemas", "why the American currency is losing value",
            "Yukos oil company tax dispute", "fraudsters tricking people into revealing bank details online",
            "Australian Open tennis"]

st.set_page_config(page_title="BBC News Retrieval", page_icon="📰", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(ui.css(), unsafe_allow_html=True)


@st.cache_resource(show_spinner="Preparing the archive: BBC dataset, inverted index, TF-IDF vectors and "
                                 "passage embeddings (about a minute on first start)…")
def get_engine() -> SearchEngine:
    ensure_articles()  # downloads and verifies the official release on a fresh deployment
    engine = SearchEngine(load_articles())
    engine.run_all("warm-up query")  # the first encoder call pays a one-off warm-up
    return engine


@st.cache_data
def load_benchmark():
    if not (OUT / "results.json").exists():
        return None
    return (json.loads((OUT / "results.json").read_text()),
            pd.read_csv(OUT / "metrics_summary.csv", index_col=0),
            pd.read_csv(OUT / "precision_recall_11pt.csv"),
            pd.read_csv(OUT / "metrics_by_query_style.csv"),
            json.loads((ROOT / "data" / "queries.json").read_text()),
            json.loads((ROOT / "data" / "qrels.json").read_text()))


def html(markup: str, target=st) -> None:
    target.markdown(markup, unsafe_allow_html=True)


def use_example() -> None:
    pick = st.session_state.get("example")
    if pick:
        st.session_state["query"] = pick
    st.session_state["example"] = None


try:
    engine = get_engine()
except Exception as exc:  # e.g. the dataset server is unreachable
    st.error(f"Could not prepare the BBC News archive: {exc}")
    st.info("The app downloads the official dataset from mlg.ucd.ie on first start. Reload in a minute, or run "
            "`python scripts/prepare_data.py data/raw/bbc` locally (see README).")
    st.stop()
html(ui.masthead(engine.n_docs, len(engine.index.vocabulary), engine.dense.n_passages))

# ---------------------------------------------------------------- controls --
st.session_state.setdefault("query", EXAMPLES[0])
query = st.text_input("Search the archive", key="query",
                      placeholder="Ask anything — e.g. why the American currency is losing value")
st.pills("Try an example", EXAMPLES, key="example", on_change=use_example)
c_model, c_k, c_desk = st.columns([1.62, 0.46, 1.58], vertical_alignment="bottom")
with c_model:
    model = st.segmented_control("Ranking model", MODEL_NAMES, default="Hybrid RRF", required=True, key="model")
with c_k:
    k = st.segmented_control("Results", [5, 10, 20], default=10, required=True, key="k")
with c_desk:
    desks = st.pills("Filter by desk", list(ui.DESK), selection_mode="multi", format_func=ui.FILTER_LABEL.get,
                     key="desks")

tab_search, tab_compare, tab_bench = st.tabs(["Search results", "Compare models", "Benchmark"])

# ------------------------------------------------------------------ search --
with tab_search:
    if not query.strip():
        html(ui.empty_state("Start with a query", "Type a topic, a name or a whole question above."))
    else:
        results, elapsed = engine.search_all(query, k=k, categories=desks or None)
        hits = results[model]
        html(ui.runbar(model, len(hits), elapsed, preprocess(query)))
        if hits:
            top = max(r["score"] for r in hits)
            html("".join(ui.result_card(r, model, top) for r in hits))
        else:
            html(ui.empty_state("No matching articles",
                                "This model found nothing for the query and desk filter — try the Sentence "
                                "Transformer or Hybrid RRF, which also match meaning rather than exact words."))

# ----------------------------------------------------------------- compare --
with tab_compare:
    if query.strip():
        top5, _ = engine.search_all(query, k=5, categories=desks or None)
        agreement = Counter(d for m in MODEL_NAMES for d in {r["doc_id"] for r in top5[m]})
        seen, all_four = set(), []
        for m in MODEL_NAMES:
            for r in top5[m]:
                if agreement[r["doc_id"]] == 4 and r["doc_id"] not in seen:
                    seen.add(r["doc_id"])
                    all_four.append(r)
        html(ui.consensus(all_four))
        for col, m in zip(st.columns(4, gap="small"), MODEL_NAMES):
            html(ui.desk_column(m, top5[m], agreement), col)

# --------------------------------------------------------------- benchmark --
with tab_bench:
    bench = load_benchmark()
    if bench is None:
        st.info("Run `python scripts/evaluate.py` to generate the evaluation results.")
    else:
        res, main, pr, style_tbl, queries, qrels = bench
        ki = pd.DataFrame(res["known_item"]["table"]).set_index("system")
        html(ui.kpi_tiles([
            (f"{main['AP'].max():.3f}", "Best MAP · 25 topics", main["AP"].idxmax()),
            (f"{main['nDCG@10'].max():.3f}", "Best nDCG@10", main["nDCG@10"].idxmax()),
            (f"{main['Recall@100'].max():.3f}", "Best Recall@100", main["Recall@100"].idxmax()),
            (f"{ki['MRR'].max():.3f}", f"Known-item MRR · {res['known_item']['queries']:,} queries", ki["MRR"].idxmax()),
        ]))
        left, right = st.columns(2, gap="medium")
        with left.container(border=True):
            html('<div class="panel-title">Performance comparison</div>'
                 '<div class="panel-sub">Macro-average over 25 graded topics · hover a bar for the exact score</div>')
            st.altair_chart(charts.metric_bars(main, [("P@10", "P@10"), ("Recall@100", "Recall@100"),
                                                      ("AP", "MAP"), ("nDCG@10", "nDCG@10")]),
                            width="stretch", theme=None)
        with right.container(border=True):
            html('<div class="panel-title">Precision–recall curves</div>'
                 '<div class="panel-sub">11-point interpolated precision · hover to compare models at a recall level</div>')
            st.altair_chart(charts.pr_curve(pr), width="stretch", theme=None)
        left, right = st.columns([1.15, 1], gap="medium")
        with left.container(border=True):
            html('<div class="panel-title">Results table</div><div class="panel-sub">Best value per column in bold</div>')
            rows = {m: main.loc[m].to_dict() for m in MODEL_NAMES}
            html(ui.metrics_table(rows, [("P@10", "P@10"), ("Recall@100", "R@100"), ("AP", "MAP"),
                                         ("nDCG@10", "nDCG@10"), ("RR", "MRR")]))
        with right.container(border=True):
            html('<div class="panel-title">Keyword vs paraphrased queries</div>'
                 '<div class="panel-sub">Lexical models lose most when the query avoids the article’s words</div>')
            st.altair_chart(charts.style_bars(style_tbl), width="stretch", theme=None)

        st.markdown("")
        html('<div class="panel-title" style="font-size:1.25rem">Topic explorer</div>'
             '<div class="panel-sub">Re-run any of the 25 test topics live and see each result’s relevance grade</div>')
        qmap = {q["qid"]: q for q in queries}
        e1, e2 = st.columns([2.2, 1.5], vertical_alignment="bottom")
        qid = e1.selectbox("Test topic", list(qmap), index=6, format_func=lambda q: f"{q} · {qmap[q]['query']}")
        e_model = e2.segmented_control("Model", MODEL_NAMES, default="Hybrid RRF", required=True, key="explorer_model")
        judg = qrels[qid]
        grades = Counter(judg.values())
        html(ui.topic_card(qmap[qid], grades[2], grades[1], len(judg)))
        runs = engine.run_all(qmap[qid]["query"], depth=1000)
        ranked = {m: [engine.doc_ids[i] for i in runs[m][0]] for m in MODEL_NAMES}
        live = {m: evaluate_run(ranked[m], judg) for m in MODEL_NAMES}
        m1, m2 = st.columns([1, 1.35], gap="medium")
        with m1.container(border=True):
            html('<div class="panel-title">Live metrics for this topic</div>'
                 '<div class="panel-sub">Computed now from the graded judgments</div>')
            html(ui.metrics_table(live, [("P@10", "P@10"), ("AP", "AP"), ("nDCG@10", "nDCG@10")], digits=3))
        with m2.container(border=True):
            html(f'<div class="panel-title">{e_model} · top 10 with judgments</div>'
                 '<div class="panel-sub">Judged blind from depth-20 pools plus a keyword sweep</div>')
            titles = dict(zip(engine.doc_ids, engine.articles["title"]))
            html(ui.judged_list([(i + 1, d, titles[d], judg.get(d)) for i, d in enumerate(ranked[e_model][:10])]))

html(ui.footer(REPO_URL))
