"""Build the GitHub Pages project site in docs/ (index.html + optimised assets).

Numbers come from outputs/results.json; screenshots from outputs/figures.
Usage:  python scripts/build_site.py
"""
import html
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
FIG = ROOT / "outputs" / "figures"
REPO_URL = "https://github.com/akshatsharma-x/BBC-News-Retrieval"
LIVE_APP_URL = "https://bbc-news-retrieval.streamlit.app"
MODELS = ["TF-IDF", "BM25", "Sentence Transformer", "Hybrid RRF"]
COLOR = {"TF-IDF": "#3987e5", "BM25": "#d95926", "Sentence Transformer": "#199e70", "Hybrid RRF": "#c98500"}

CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,650;1,9..144,600&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root { --bg:#0b1220; --surface:#111c30; --surface-2:#16233d; --line:#22314a; --line-2:#31446a; --ink:#f3f6fb;
  --ink-2:#b9c5d6; --muted:#8d9bb0; --dim:#6b7a92; --teal:#3cbecd; --amber:#f5b84b;
  --serif:'Fraunces',Georgia,'Times New Roman',serif; --sans:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;
  --mono:'JetBrains Mono',ui-monospace,Menlo,Consolas,monospace; }
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin: 0; color: var(--ink-2); font: 400 16px/1.6 var(--sans);
  background: radial-gradient(1100px 520px at 8% -8%, rgba(60,190,205,.10), transparent 62%),
              radial-gradient(900px 480px at 100% -4%, rgba(245,184,75,.07), transparent 58%), var(--bg); }
a { color: var(--teal); }
a:focus-visible, .btn:focus-visible { outline: 3px solid var(--amber); outline-offset: 3px; border-radius: 10px; }
.wrap { max-width: 1140px; margin: 0 auto; padding: 0 24px; }
nav.top { position: sticky; top: 0; z-index: 5; backdrop-filter: blur(10px); background: rgba(11,18,32,.78);
  border-bottom: 1px solid var(--line); }
nav.top .wrap { display: flex; align-items: center; justify-content: space-between; gap: 16px; height: 58px; }
nav.top .brand { font: 650 1.1rem var(--serif); color: var(--ink); text-decoration: none; }
nav.top .brand em { color: var(--amber); font-style: italic; }
nav.top ul { display: flex; gap: 20px; list-style: none; margin: 0; padding: 0; }
nav.top ul a { color: var(--ink-2); text-decoration: none; font-size: .9rem; font-weight: 500; }
nav.top ul a:hover { color: var(--ink); }
.hero { position: relative; overflow: hidden; margin: 28px 0 0; padding: 44px 44px 40px; border: 1px solid var(--line);
  border-radius: 26px; background:
  repeating-linear-gradient(0deg, transparent 0 31px, rgba(255,255,255,.022) 31px 32px),
  repeating-linear-gradient(90deg, transparent 0 31px, rgba(255,255,255,.022) 31px 32px),
  linear-gradient(135deg, #132039 0%, #0d1628 60%, #101a2e 100%); }
.hero::after { content: ""; position: absolute; right: -110px; top: -170px; width: 520px; height: 520px; pointer-events: none;
  background: radial-gradient(circle, rgba(60,190,205,.22), transparent 66%); }
.kicker { font: 600 .74rem/1 var(--sans); letter-spacing: .2em; text-transform: uppercase; color: var(--teal); }
.hero h1 { position: relative; z-index: 1; font: 650 clamp(2.6rem, 6vw, 4.4rem)/1.02 var(--serif); letter-spacing: -.02em;
  color: var(--ink); margin: 16px 0 14px; }
.hero h1 em { color: var(--amber); font-style: italic; font-weight: 600; }
.rule { height: 4px; border-top: 1px solid var(--line-2); border-bottom: 1px solid var(--line-2); margin: 0 0 18px; }
.lede { position: relative; z-index: 1; max-width: 760px; font-size: 1.12rem; color: var(--ink-2); margin: 0; }
.cta { position: relative; z-index: 1; display: flex; flex-wrap: wrap; gap: 12px; margin: 26px 0 10px; }
.btn { display: inline-flex; align-items: center; gap: 8px; padding: 12px 20px; border-radius: 12px; font-weight: 600;
  text-decoration: none; border: 1px solid var(--line-2); color: var(--ink); background: rgba(11,18,32,.55); }
.btn.primary { background: var(--teal); border-color: var(--teal); color: #071019; }
.btn:hover { transform: translateY(-1px); }
.note { position: relative; z-index: 1; font-size: .84rem; color: var(--muted); margin: 6px 0 0; }
.ticker { position: relative; z-index: 1; display: flex; flex-wrap: wrap; gap: 8px; margin-top: 22px; }
.ticker span { font: 500 .8rem var(--mono); padding: 6px 12px; border: 1px solid var(--line-2); border-radius: 999px;
  background: rgba(11,18,32,.55); color: var(--ink-2); }
.ticker b { color: var(--ink); font-weight: 600; }
.frame { margin: 34px 0 0; border: 1px solid var(--line-2); border-radius: 18px; overflow: hidden; background: var(--surface);
  box-shadow: 0 40px 80px -40px rgba(0,0,0,.8); }
.frame .bar { display: flex; align-items: center; gap: 7px; height: 34px; padding: 0 14px; border-bottom: 1px solid var(--line);
  background: #0e1729; }
.frame .bar i { width: 10px; height: 10px; border-radius: 50%; background: #2b3a55; display: inline-block; }
.frame .bar span { margin-left: 10px; font: 500 .76rem var(--mono); color: var(--dim); }
.frame img { display: block; width: 100%; height: auto; }
section { padding: 64px 0 8px; }
h2 { font: 650 clamp(1.7rem, 3.2vw, 2.3rem)/1.15 var(--serif); color: var(--ink); margin: 6px 0 10px; letter-spacing: -.01em; }
.sub { color: var(--muted); margin: 0 0 24px; max-width: 760px; }
.grid4 { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
.grid2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
.card { position: relative; border: 1px solid var(--line); border-radius: 18px; background: var(--surface); padding: 20px 20px 18px; }
.card .stripe { width: 44px; height: 4px; border-radius: 4px; background: var(--model); margin-bottom: 14px; }
.card h3 { font: 650 1.12rem var(--sans); color: var(--ink); margin: 0 0 4px; }
.card .k { font: 600 .68rem var(--sans); letter-spacing: .14em; text-transform: uppercase; color: var(--muted); margin: 0 0 10px; }
.card p { margin: 0; font-size: .93rem; }
.kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-bottom: 18px; }
.kpi { position: relative; overflow: hidden; border: 1px solid var(--line); border-radius: 18px; background: var(--surface); padding: 18px 20px 16px; }
.kpi .v { font: 650 2.4rem/1 var(--serif); color: var(--ink); }
.kpi .l { font: 600 .68rem var(--sans); letter-spacing: .14em; text-transform: uppercase; color: var(--muted); margin-top: 10px; }
.kpi .w { display: flex; align-items: center; gap: 7px; font-size: .86rem; margin-top: 6px; color: var(--ink-2); }
.kpi::after { content: ""; position: absolute; right: -50px; bottom: -70px; width: 150px; height: 150px;
  background: radial-gradient(circle, var(--model), transparent 70%); opacity: .16; }
.dot { width: 10px; height: 10px; border-radius: 50%; background: var(--model); display: inline-block; flex: none; }
.panel { border: 1px solid var(--line); border-radius: 18px; background: var(--surface); padding: 20px 20px; overflow-x: auto; }
.panel h3 { font: 650 1.02rem var(--sans); color: var(--ink); margin: 0 0 4px; }
.panel .s { font-size: .84rem; color: var(--muted); margin: 0 0 12px; }
table { width: 100%; border-collapse: collapse; font-size: .85rem; }
th { text-align: right; font: 600 .68rem var(--sans); letter-spacing: .12em; text-transform: uppercase; color: var(--muted);
  padding: 8px 6px; border-bottom: 1px solid var(--line-2); }
td { text-align: right; padding: 10px 6px; border-bottom: 1px solid var(--line); font-family: var(--mono); font-variant-numeric: tabular-nums; }
th:first-child, td:first-child { text-align: left; font-family: var(--sans); }
td:first-child { color: var(--ink); font-weight: 600; white-space: nowrap; }
td.best { color: var(--ink); font-weight: 700; }
tr:last-child td { border-bottom: none; }
figure { margin: 0; }
figure .frame { margin: 0; }
figcaption { font-size: .88rem; color: var(--muted); margin: 10px 2px 0; }
figcaption b { color: var(--ink); font-weight: 600; }
.steps { counter-reset: s; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 20px; }
.steps div { border: 1px solid var(--line); border-radius: 16px; background: var(--surface); padding: 16px 18px 16px 58px; position: relative; }
.steps div::before { counter-increment: s; content: counter(s, decimal-leading-zero); position: absolute; left: 18px; top: 14px;
  font: 600 1.3rem var(--serif); color: var(--dim); }
.steps b { color: var(--ink); }
.stats { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; }
.stat { border: 1px solid var(--line); border-radius: 16px; background: var(--surface); padding: 16px 16px 14px; }
.stat .v { font: 650 1.9rem/1 var(--serif); color: var(--ink); }
.stat .l { font-size: .82rem; color: var(--muted); margin-top: 8px; }
.docs a.card { text-decoration: none; display: block; }
.docs a.card:hover { border-color: var(--line-2); }
.docs .k { color: var(--teal); }
.docs a.card p:not(.k) { color: var(--ink-2); }
.docs a.card h3 { color: var(--ink); }
pre { margin: 0; padding: 18px 20px; overflow-x: auto; border: 1px solid var(--line); border-radius: 16px; background: #0e1729;
  color: #d6deea; font: 500 .86rem/1.7 var(--mono); }
pre .c { color: var(--dim); }
footer { margin-top: 72px; border-top: 1px solid var(--line); }
footer .wrap { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px 24px; padding: 22px 24px 40px;
  font-size: .84rem; color: var(--muted); }
@media (max-width: 960px) { .grid4, .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } .grid2 { grid-template-columns: 1fr; } .stats { grid-template-columns: repeat(3, 1fr); }
  nav.top ul { display: none; } }
@media (max-width: 640px) { .grid4, .kpis, .grid2, .steps { grid-template-columns: 1fr; } .stats { grid-template-columns: repeat(2, 1fr); }
  .hero { padding: 30px 22px 26px; } }
@media (prefers-reduced-motion: reduce) { html { scroll-behavior: auto; } .btn:hover { transform: none; } }
"""


def esc(x) -> str:
    return html.escape(str(x))


def webp(src: Path, name: str, width: int = 1600) -> tuple[str, int, int]:
    im = Image.open(src).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    out = ASSETS / f"{name}.webp"
    im.save(out, "WEBP", quality=84, method=6)
    return f"assets/{out.name}", im.width, im.height


def table(rows: dict, cols: list[tuple[str, str]]) -> str:
    best = {k: max(rows[m][k] for m in MODELS) for k, _ in cols}
    head = "".join(f"<th>{esc(lab)}</th>" for _, lab in cols)
    body = "".join(
        f'<tr><td><span class="dot" style="--model:{COLOR[m]};margin-right:9px"></span>{esc(m)}</td>' +
        "".join(f'<td class="{"best" if abs(rows[m][k] - best[k]) < 1e-12 else ""}">{rows[m][k]:.4f}</td>' for k, _ in cols) +
        "</tr>" for m in MODELS)
    return f"<table><thead><tr><th>Model</th>{head}</tr></thead><tbody>{body}</tbody></table>"


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    (DOCS / ".nojekyll").write_text("")
    r = json.loads((ROOT / "outputs" / "results.json").read_text())
    main_t = {x["system"]: x for x in r["main"]}
    ki = {x["system"]: x for x in r["known_item"]["table"]}
    style = {(x["style"], x["system"]): x["AP"] for x in r["by_style"]}
    C, I, Q, D, L = r["corpus"], r["index_stats"], r["qrels"], r["dense"], r["latency_ms_median"]

    shots = {n: webp(FIG / f"ui_{n}.png", f"ui_{n}") for n in ("search", "compare", "benchmark", "explorer")}
    from make_figures import THEMES, pipeline  # noqa: E402
    pipeline(THEMES["dark"], ASSETS / "pipeline_dark.png", I, D)
    pipe = webp(ASSETS / "pipeline_dark.png", "pipeline", 1500)
    (ASSETS / "pipeline_dark.png").unlink()
    (ASSETS / "favicon.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="14" fill="#111c30"/>'
        '<text x="32" y="44" font-family="Georgia,serif" font-size="36" font-style="italic" font-weight="700" '
        'fill="#f5b84b" text-anchor="middle">R</text></svg>')
    (ASSETS / "site.css").write_text(CSS.strip() + "\n")

    def img(key, alt):
        src, w, h = shots[key]
        return f'<img src="{src}" width="{w}" height="{h}" loading="lazy" alt="{esc(alt)}">'

    best = lambda t, k: max(MODELS, key=lambda m: t[m][k])  # noqa: E731
    kpis = [(f"{main_t[best(main_t, 'AP')]['AP']:.3f}", "Best MAP · 25 topics", best(main_t, "AP")),
            (f"{main_t[best(main_t, 'nDCG@10')]['nDCG@10']:.3f}", "Best nDCG@10", best(main_t, "nDCG@10")),
            (f"{main_t[best(main_t, 'Recall@100')]['Recall@100']:.3f}", "Best Recall@100", best(main_t, "Recall@100")),
            (f"{ki[best(ki, 'MRR')]['MRR']:.3f}", f"Known-item MRR · {r['known_item']['queries']:,} queries", best(ki, "MRR"))]
    kpi_html = "".join(f'<div class="kpi" style="--model:{COLOR[m]}"><div class="v">{v}</div><div class="l">{esc(l)}</div>'
                       f'<div class="w"><span class="dot"></span>{esc(m)}</div></div>' for v, l, m in kpis)
    models = [("TF-IDF", "Vector-space model", "Sublinear TF × smoothed IDF weights, L2-normalised, ranked by cosine similarity."),
              ("BM25", "Probabilistic model", "Okapi BM25 (k₁ = 1.2, b = 0.75) scored term-at-a-time from the inverted index."),
              ("Sentence Transformer", "Dense bi-encoder", f"all-MiniLM-L6-v2 embeddings of {D['passages']:,} passages; an article scores as its best passage."),
              ("Hybrid RRF", "Rank fusion", "Reciprocal Rank Fusion of BM25 and the Sentence Transformer (k = 60) — no score calibration needed.")]
    model_html = "".join(f'<div class="card" style="--model:{COLOR[m]}"><div class="stripe"></div><h3>{esc(m)}</h3>'
                         f'<p class="k">{esc(k)}</p><p>{esc(d)}</p></div>' for m, k, d in models)
    stats = [(f"{Q['queries']}", "test topics — 13 keyword, 12 paraphrased"), (f"{Q['judged_pairs']}", "graded judgments (0 / 1 / 2)"),
             (f"{Q['relevant']}", f"relevant articles, {Q['highly_relevant']} highly"), (f"{Q['relevant_found_by_sweep']}", "found only by the keyword sweep"),
             ("100%", "of every model's top 10 judged")]
    stat_html = "".join(f'<div class="stat"><div class="v">{v}</div><div class="l">{esc(l)}</div></div>' for v, l in stats)
    cols = [("P@10", "P@10"), ("Recall@100", "Recall@100"), ("AP", "MAP"), ("nDCG@10", "nDCG@10")]
    kcols = [("MRR", "MRR"), ("Success@1", "Success@1"), ("Recall@10", "Recall@10"), ("nDCG@10", "nDCG@10")]
    psrc, pw, ph = pipe

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BBC News Retrieval — lexical, semantic and hybrid search</title>
<meta name="description" content="Information-retrieval project on the BBC News corpus: TF-IDF, BM25, a Sentence Transformer and Reciprocal Rank Fusion, with a graded test collection, full evaluation and a live search app.">
<meta property="og:title" content="BBC News Retrieval">
<meta property="og:description" content="Lexical, semantic and hybrid search over {C['unique_articles']:,} BBC News articles, with a live search app.">
<meta property="og:image" content="assets/ui_search.webp">
<link rel="icon" href="assets/favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="assets/site.css">
</head>
<body>
<nav class="top" aria-label="Sections"><div class="wrap">
  <a class="brand" href="#">BBC News <em>Retrieval</em></a>
  <ul><li><a href="#models">Models</a></li><li><a href="#results">Results</a></li><li><a href="#interface">Interface</a></li>
  <li><a href="#how">How it works</a></li><li><a href="#documents">Documents</a></li><li><a href="#run">Run it</a></li></ul>
  <a class="btn primary" href="{LIVE_APP_URL}" target="_blank" rel="noopener">Live app ↗</a>
</div></nav>
<main class="wrap">
<header class="hero">
  <span class="kicker">Information Retrieval · Project 12</span>
  <h1>BBC News <em>Retrieval</em></h1>
  <div class="rule"></div>
  <p class="lede">Lexical, semantic and hybrid search over {C['unique_articles']:,} BBC News stories from 2004–2005. Four rankers
  share one inverted index, and a purpose-built, graded test collection measures which one finds the right articles.</p>
  <div class="cta">
    <a class="btn primary" href="{LIVE_APP_URL}" target="_blank" rel="noopener">Open the live search app ↗</a>
    <a class="btn" href="{REPO_URL}" target="_blank" rel="noopener">View the code on GitHub</a>
    <a class="btn" href="BBC_News_Retrieval_Report.pdf">Read the report (PDF)</a>
  </div>
  <p class="note">The live app runs on Streamlit Community Cloud. If nobody has used it for a while it sleeps, and the
  first visit takes about a minute to wake it.</p>
  <div class="ticker"><span><b>{C['unique_articles']:,}</b> articles</span><span><b>5</b> news desks</span>
  <span><b>{I['vocabulary_terms']:,}</b> index terms</span><span><b>{D['passages']:,}</b> embedded passages</span>
  <span><b>4</b> ranking models</span><span><b>≈{L['Hybrid RRF']:.0f} ms</b> per hybrid query</span></div>
</header>
<div class="frame"><div class="bar"><i></i><i></i><i></i><span>bbc-news-retrieval · search</span></div>{img("search", "The search app: Hybrid RRF results for the query films that took the most money at cinemas, with highlighted terms")}</div>

<section id="models">
  <span class="kicker">Four ways to rank</span>
  <h2>Exact words, meaning, or both</h2>
  <p class="sub">The same preprocessing pipeline (cleaning, tokenisation, stop words, Porter stemming) feeds an inverted index for the
  lexical models; the neural encoder reads the raw text.</p>
  <div class="grid4">{model_html}</div>
</section>

<section id="results">
  <span class="kicker">Results</span>
  <h2>Fusion is the most robust ranker</h2>
  <p class="sub">Hybrid RRF ties the Sentence Transformer on topical search and beats every model on finding an article from its
  headline. On paraphrased queries the lexical models fall to MAP {style[("natural", "BM25")]:.3f} (BM25) while the
  Sentence Transformer holds {style[("natural", "Sentence Transformer")]:.3f}.</p>
  <div class="kpis">{kpi_html}</div>
  <div class="grid2">
    <div class="panel"><h3>25 graded topics</h3><p class="s">Macro-average · best value per column in bold</p>{table(main_t, cols)}</div>
    <div class="panel"><h3>{r['known_item']['queries']:,} known-item queries</h3><p class="s">Find an article from its headline (headline removed from the index)</p>{table(ki, kcols)}</div>
  </div>
</section>

<section id="interface">
  <span class="kicker">The interface</span>
  <h2>A news desk, with the evaluation built in</h2>
  <p class="sub">One-click example queries, a model selector and a desk filter; result cards with a score bar, highlighted
  snippet, matched terms and the full story in place.</p>
  <div class="grid2">
    <figure><div class="frame">{img("compare", "Compare models tab: the top five results of all four models side by side with agreement badges")}</div>
      <figcaption><b>Compare models</b> — the top five of every model side by side, with a badge for how many models agree.</figcaption></figure>
    <figure><div class="frame">{img("explorer", "Topic explorer: a test topic re-scored live with each result's relevance grade")}</div>
      <figcaption><b>Topic explorer</b> — re-runs any of the 25 test topics live and grades every result against the judgments.</figcaption></figure>
  </div>
  <figure style="margin-top:18px"><div class="frame">{img("benchmark", "Benchmark tab: headline metrics, interactive comparison and precision-recall charts")}</div>
    <figcaption><b>Benchmark</b> — headline metrics and interactive charts with tooltips and a precision–recall crosshair.</figcaption></figure>
</section>

<section id="how">
  <span class="kicker">How it works</span>
  <h2>From raw text to ranked results</h2>
  <div class="frame" style="margin-top:18px"><img src="{psrc}" width="{pw}" height="{ph}" loading="lazy" alt="Architecture: preprocessing feeds an inverted index, TF-IDF vectors and passage embeddings; BM25 and the Sentence Transformer are fused with RRF"></div>
  <div class="steps">
    <div><b>Clean corpus.</b> The official release ({C['raw_articles']:,} articles) decoded as UTF-8, HTML entities resolved and
    {C['exact_duplicates_removed']} exact duplicates removed.</div>
    <div><b>Index.</b> {I['vocabulary_terms']:,} terms and {I['total_postings']:,} postings, a sparse TF-IDF matrix, and
    {D['passages']:,} passage embeddings (160-word windows).</div>
    <div><b>Rank.</b> TF-IDF, BM25 and the Sentence Transformer each return a ranking; RRF fuses BM25 with the Sentence
    Transformer by rank position.</div>
    <div><b>Evaluate.</b> P@10, Recall@100, MAP, graded nDCG@10, MRR, precision–recall curves, randomisation tests and
    ablations of passage pooling, fusion components and k.</div>
  </div>
  <h2 style="margin-top:44px;font-size:1.6rem">A purpose-built test collection</h2>
  <p class="sub">The corpus has no official queries. Topics were written TREC-style, pooled to depth 20 over five runs and judged
  blind; a keyword sweep of the whole archive filled the gaps the pool missed.</p>
  <div class="stats">{stat_html}</div>
</section>

<section id="documents" class="docs">
  <span class="kicker">Documents</span>
  <h2>Report and presentation</h2>
  <div class="grid2">
    <a class="card" href="BBC_News_Retrieval_Report.pdf" style="--model:var(--teal)"><div class="stripe"></div><h3>Project report</h3>
      <p class="k">PDF · 31 pages</p><p>Dataset, methodology with equations, test collection, results, significance tests, ablations,
      efficiency and limitations.</p></a>
    <a class="card" href="BBC_News_Retrieval_Presentation.pdf" style="--model:var(--amber)"><div class="stripe"></div><h3>Presentation</h3>
      <p class="k">PDF · 20 slides (also as <span>PPTX</span>)</p><p>The project in twenty slides, from the problem to the live benchmark,
      with speaker notes in the PPTX.</p></a>
  </div>
</section>

<section id="run">
  <span class="kicker">Run it yourself</span>
  <h2>Three commands</h2>
  <p class="sub">The app downloads and verifies the official dataset on first start; the BBC article text is not stored in the repository.</p>
<pre><span class="c"># clone and install</span>
git clone {REPO_URL}.git && cd BBC-News-Retrieval
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
<span class="c"># launch the search app (first start: ~1 minute)</span>
streamlit run app.py</pre>
</section>
</main>
<footer><div class="wrap">
  <span>Data: BBC News dataset — D. Greene &amp; P. Cunningham, ICML 2006 · articles © BBC · encoder: sentence-transformers/all-MiniLM-L6-v2</span>
  <span><a href="{REPO_URL}">Source on GitHub</a> · <a href="{LIVE_APP_URL}">Live app</a></span>
</div></footer>
</body>
</html>
"""
    (DOCS / "index.html").write_text(page)
    sizes = {p.name: p.stat().st_size // 1024 for p in sorted(ASSETS.iterdir())}
    print("site written:", DOCS / "index.html", "| assets (KB):", sizes)


if __name__ == "__main__":
    main()
