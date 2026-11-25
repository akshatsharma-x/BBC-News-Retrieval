"""HTML building blocks for the Streamlit interface.

Colour carries one meaning only - the ranking model - using the validated
categorical palette (dark steps). Categories are neutral text badges.
"""
from __future__ import annotations

import html
from pathlib import Path

MODEL_COLOR = {"TF-IDF": "#3987e5", "BM25": "#d95926", "Sentence Transformer": "#199e70",
               "Hybrid RRF": "#c98500"}
DESK = {"business": "Business", "entertainment": "Entertainment", "politics": "Politics",
        "sport": "Sport", "tech": "Technology"}
FILTER_LABEL = {**DESK, "tech": "Tech"}
SCORE_LABEL = {"TF-IDF": "cosine", "BM25": "bm25", "Sentence Transformer": "cosine", "Hybrid RRF": "rrf"}
GRADE = {2: ("g2", "Highly relevant"), 1: ("g1", "Partially relevant"), 0: ("g0", "Not relevant"),
         None: ("g0", "Unjudged")}
_CSS = Path(__file__).with_name("style.css")


def esc(value) -> str:
    return html.escape(str(value))


def css() -> str:
    return f"<style>{_CSS.read_text()}</style>"


def masthead(n_docs: int, n_terms: int, n_passages: int) -> str:
    chips = "".join(f"<span><b>{v}</b> {k}</span>" for v, k in (
        (f"{n_docs:,}", "articles"), ("5", "news desks"), (f"{n_terms:,}", "index terms"),
        (f"{n_passages:,}", "embedded passages"), ("4", "ranking models")))
    return ('<header class="mast"><div class="mast-row">'
            '<span class="kicker">Information Retrieval · Project 12</span>'
            '<span class="edition">ARCHIVE EDITION · 2004 – 2005</span></div>'
            '<h1 class="mast-title">BBC News <em>Retrieval</em></h1><div class="rule"></div>'
            f'<p class="mast-sub">Search {n_docs:,} BBC stories with lexical, semantic and hybrid ranking. '
            'Every result is scored live by TF-IDF, BM25, a Sentence Transformer and Reciprocal Rank Fusion.</p>'
            f'<div class="ticker">{chips}</div></header>')


def runbar(model: str, n: int, elapsed_ms: float, terms: list[str]) -> str:
    chips = "".join(f'<span class="chip">{esc(t)}</span>' for t in terms) or '<span class="chip">(none)</span>'
    return (f'<div class="runbar" style="--model:{MODEL_COLOR[model]}">'
            f'<span class="model"><span class="dot"></span>{esc(model)}</span><span class="sep">|</span>'
            f'<span>{n} results</span><span class="sep">|</span>'
            f'<span>all four models ranked in <span class="lat">{elapsed_ms:.1f} ms</span></span>'
            f'<span class="sep">|</span><span>index terms: {chips}</span></div>')


def result_card(r: dict, model: str, top_score: float) -> str:
    width = max(3.0, 100 * r["score"] / top_score) if top_score > 0 else 0.0
    score = f"{r['score']:.4f}" if model == "Hybrid RRF" else f"{r['score']:.3f}"
    if r["matched_terms"]:
        terms = "".join(f'<span class="chip">{esc(t)}</span>' for t in r["matched_terms"])
    else:
        terms = '<span class="chip">no shared terms · semantic match</span>'
    full = "".join(f"<p>{esc(p)}</p>" for p in r["body"].split("\n") if p.strip())
    delay = min(r["rank"], 10) * 30
    return (f'<article class="card" style="--model:{MODEL_COLOR[model]};animation-delay:{delay}ms">'
            f'<div class="rank">{r["rank"]:02d}</div><div>'
            f'<div class="meta"><span class="desk">{DESK[r["category"]]}</span><span>{esc(r["doc_id"])}</span>'
            f'<span>{SCORE_LABEL[model]} {score}</span></div>'
            f'<div class="headline">{esc(r["title"])}</div>'
            f'<p class="snippet">{r["snippet_html"]}</p>'
            f'<div class="terms">{terms}</div>'
            f'<details class="story"><summary>Read the full story</summary><div class="full">{full}</div></details>'
            f'<div class="scorebar"><span style="width:{width:.1f}%"></span></div></div></article>')


def empty_state(title: str, text: str) -> str:
    return f'<div class="empty"><b>{esc(title)}</b>{esc(text)}</div>'


def consensus(all_four: list[dict], n_models: int = 4) -> str:
    if all_four:
        titles = " · ".join(f"<b>{esc(r['title'])}</b>" for r in all_four)
        text = f"{len(all_four)} article{'s' if len(all_four) > 1 else ''} in every model's top 5: {titles}"
    else:
        text = "No article appears in all four top-5 lists — the models disagree on this query."
    return f'<div class="consensus"><span class="chip">consensus</span><span>{text}</span></div>'


def desk_column(model: str, results: list[dict], agreement: dict[str, int]) -> str:
    items = []
    for r in results:
        a = agreement[r["doc_id"]]
        items.append(f'<div class="item"><div class="n">{r["rank"]}</div><div><div class="t">{esc(r["title"])}</div>'
                     f'<div class="s"><span class="desk">{DESK[r["category"]]}</span>'
                     f'<span class="agree{" all" if a == 4 else ""}">{a}/4 models</span></div></div></div>')
    return (f'<div class="deskcol" style="--model:{MODEL_COLOR[model]}"><div class="deskhead"><span class="dot"></span>{esc(model)}</div>'
            f'<div class="bar"></div>{"".join(items)}</div>')


def kpi_tiles(items: list[tuple[str, str, str]]) -> str:
    tiles = "".join(f'<div class="kpi" style="--model:{MODEL_COLOR[m]}"><div class="v">{esc(v)}</div>'
                    f'<div class="l">{esc(label)}</div><div class="w"><span class="dot"></span>{esc(m)}</div></div>'
                    for v, label, m in items)
    return f'<div class="kpis">{tiles}</div>'


def metrics_table(rows: dict[str, dict[str, float]], columns: list[tuple[str, str]], digits: int = 4) -> str:
    best = {k: max(v[k] for v in rows.values()) for k, _ in columns}
    head = "".join(f"<th>{esc(label)}</th>" for _, label in columns)
    body = []
    for model, vals in rows.items():
        cells = "".join(f'<td class="{"best" if abs(vals[k] - best[k]) < 1e-9 else ""}">{vals[k]:.{digits}f}</td>'
                        for k, _ in columns)
        body.append(f'<tr><td><span class="dot" style="--model:{MODEL_COLOR[model]};margin-right:8px"></span>'
                    f'{esc(model)}</td>{cells}</tr>')
    return f'<table class="mtable"><thead><tr><th>Model</th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def topic_card(topic: dict, n_high: int, n_part: int, n_judged: int) -> str:
    style = "Keyword query" if topic["style"] == "keyword" else "Natural-language query"
    return (f'<div class="topic"><div class="meta"><span class="desk">{esc(topic["qid"])}</span>'
            f'<span class="desk">{DESK[topic["category"]]}</span><span class="desk">{style}</span>'
            f'<span>{n_high} highly · {n_part} partially relevant · {n_judged} judged</span></div>'
            f'<div class="q">“{esc(topic["query"])}”</div><p class="narr">{esc(topic["narrative"])}</p></div>')


def judged_list(rows: list[tuple[int, str, str, int | None]]) -> str:
    out = []
    for rank, doc_id, title, grade in rows:
        cls, label = GRADE[grade]
        out.append(f'<div class="judged"><div class="n">{rank}</div><div class="t">{esc(title)} '
                   f'<span class="chip">{esc(doc_id)}</span></div><span class="grade {cls}">{label}</span></div>')
    return f'<div>{"".join(out)}</div>'


def footer(repo_url: str) -> str:
    return ('<div class="foot"><span>Data: BBC News dataset — Greene &amp; Cunningham, ICML 2006 · '
            'articles © BBC · encoder: sentence-transformers/all-MiniLM-L6-v2</span>'
            f'<a href="{esc(repo_url)}" target="_blank">Source on GitHub ↗</a></div>')
