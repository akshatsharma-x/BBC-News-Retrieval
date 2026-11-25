"""BBC News Retrieval - Streamlit search interface.

Run:  streamlit run app.py
"""
import html
from pathlib import Path

import pandas as pd
import streamlit as st

from bbcir.data import CATEGORY_LABEL, load_articles
from bbcir.engine import MODEL_NAMES, SearchEngine
from bbcir.preprocess import preprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL_HELP = {
    "TF-IDF": "Vector-space model: sublinear TF × IDF weights, cosine similarity.",
    "BM25": "Okapi BM25 over the inverted index (k1 = 1.2, b = 0.75).",
    "Sentence Transformer": "all-MiniLM-L6-v2 passage embeddings, max-passage cosine similarity.",
    "Hybrid RRF": "Reciprocal Rank Fusion of BM25 and the Sentence Transformer (k = 60).",
}
MODEL_COLOR = {"TF-IDF": "#2a78d6", "BM25": "#eb6834", "Sentence Transformer": "#1baf7a", "Hybrid RRF": "#eda100"}
EXAMPLES = ["films that took the most money at cinemas", "Yukos oil company tax dispute",
            "fraudsters tricking people into revealing bank details online", "Australian Open tennis"]

st.set_page_config(page_title="BBC News Retrieval", page_icon="📰", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1180px;}
  .hero h1 {font-size: 2.0rem; margin-bottom: 0.1rem;}
  .hero p {color: #52514e; margin-top: 0;}
  .result {border: 1px solid #e6e6e1; border-radius: 10px; padding: 14px 18px; margin-bottom: 10px; background: #fff;}
  .result .meta {font-size: 0.8rem; color: #6f6e69; margin-bottom: 4px;}
  .result .title {font-size: 1.08rem; font-weight: 650; color: #0b0b0b; margin-bottom: 6px;}
  .result .snippet {font-size: 0.92rem; color: #2b2b2a; line-height: 1.5;}
  .result mark {background: #fde9b8; padding: 0 2px; border-radius: 3px;}
  .badge {display: inline-block; font-size: 0.72rem; font-weight: 600; padding: 1px 8px; border-radius: 999px;
          background: #f0efec; color: #3a3936; margin-right: 6px;}
  .rank {display: inline-block; min-width: 26px; font-weight: 700; color: #0b0b0b;}
  .dot {display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px;}
  .cmp {font-size: 0.86rem; line-height: 1.35; margin-bottom: 8px;}
  .cmp .n {color: #6f6e69; margin-right: 4px;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Building inverted index, TF-IDF vectors and passage embeddings…")
def get_engine() -> SearchEngine:
    return SearchEngine(load_articles())


engine = get_engine()

with st.sidebar:
    st.header("Search settings")
    model = st.radio("Ranking model", MODEL_NAMES, index=3, help="Choose the retrieval model")
    st.caption(MODEL_HELP[model])
    k = st.slider("Results to show", 5, 25, 10)
    cats = st.multiselect("Filter by category", list(CATEGORY_LABEL), format_func=CATEGORY_LABEL.get)
    st.divider()
    st.subheader("Collection")
    st.markdown(f"**{engine.n_docs:,}** unique BBC News articles (2004–05)  \n"
                f"**{engine.index.stats()['vocabulary_terms']:,}** index terms  \n"
                f"**{engine.dense.n_passages:,}** embedded passages")

st.markdown('<div class="hero"><h1>📰 BBC News Retrieval</h1>'
            '<p>Search 2,126 BBC News articles with lexical, semantic and hybrid ranking.</p></div>',
            unsafe_allow_html=True)
query = st.text_input("Search the archive", value=EXAMPLES[0], placeholder="e.g. " + EXAMPLES[1])
st.caption("Try: " + " · ".join(f"*{e}*" for e in EXAMPLES[1:]))

tab_results, tab_compare, tab_eval = st.tabs(["Search results", "Compare models", "Evaluation"])

if query.strip():
    with tab_results:
        results = engine.search(query, model=model, k=k, categories=cats or None)
        terms = preprocess(query)
        st.markdown(f'<span class="dot" style="background:{MODEL_COLOR[model]}"></span>'
                    f'**{model}** — {len(results)} results · query terms after preprocessing: '
                    f'`{" ".join(terms) if terms else "(none)"}`', unsafe_allow_html=True)
        for r in results:
            score = f"{r['score']:.4f}" if model == "Hybrid RRF" else f"{r['score']:.3f}"
            st.markdown(
                f'<div class="result"><div class="meta"><span class="rank">#{r["rank"]}</span>'
                f'<span class="badge">{CATEGORY_LABEL[r["category"]]}</span>{r["doc_id"]} · score {score}'
                f'{" · matched: " + ", ".join(r["matched_terms"]) if r["matched_terms"] else ""}</div>'
                f'<div class="title">{html.escape(r["title"])}</div>'
                f'<div class="snippet">{r["snippet_html"]}</div></div>', unsafe_allow_html=True)
            with st.expander("Read full article", expanded=False):
                st.write(r["body"])

    with tab_compare:
        st.caption("Top-5 results from each model for the same query.")
        cols = st.columns(4)
        for col, m in zip(cols, MODEL_NAMES):
            with col:
                st.markdown(f'<span class="dot" style="background:{MODEL_COLOR[m]}"></span>**{m}**',
                            unsafe_allow_html=True)
                for r in engine.search(query, model=m, k=5, categories=cats or None):
                    st.markdown(f'<div class="cmp"><span class="n">{r["rank"]}.</span>{html.escape(r["title"])}'
                                f' <span class="badge">{CATEGORY_LABEL[r["category"]]}</span></div>',
                                unsafe_allow_html=True)

with tab_eval:
    if (OUT / "metrics_summary.csv").exists():
        tbl = pd.read_csv(OUT / "metrics_summary.csv", index_col=0)
        tbl = tbl[["P@10", "Recall@100", "AP", "nDCG@10"]].rename(columns={"AP": "MAP"})
        st.subheader("Performance comparison (25 queries, graded pooled judgments)")
        st.dataframe(tbl.style.format("{:.4f}").highlight_max(axis=0, color="#fde9b8"), use_container_width=True)
        c1, c2 = st.columns(2)
        c1.image(str(OUT / "figures" / "metrics_bar_light.png"), caption="Metric comparison")
        c2.image(str(OUT / "figures" / "pr_curve_light.png"), caption="11-point interpolated precision–recall")
    else:
        st.info("Run `python scripts/evaluate.py` to generate the evaluation tables.")
