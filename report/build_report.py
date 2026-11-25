"""Build the project report (DOCX) from outputs/results.json and the figures.

Every number in the text is read from the evaluation outputs; statements that
compare systems are asserted against the data so the prose cannot drift from
the results.

Usage:  python report/build_report.py [output.docx]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from docx import Document  # noqa: E402
from docx.enum.section import WD_SECTION  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Cm, Pt, RGBColor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs"
FIG = OUT / "figures"
EQ = FIG / "equations"
DEFAULT_PATH = ROOT.parent / "BBC_News_Retrieval_Report.docx"

FONT, MONO = "Aptos", "Consolas"
NAVY, BLUE, INK, INK2, MUTED = "111C30", "1F5A96", "1F2328", "52514E", "6F6E69"
HEADER_FILL, ZEBRA, CALLOUT, RULE = "1B2A44", "F3F5F8", "EEF4FB", "C9D3E0"
MODELS = ["TF-IDF", "BM25", "Sentence Transformer", "Hybrid RRF"]

res = json.loads((OUT / "results.json").read_text())
queries = json.loads((ROOT / "data" / "queries.json").read_text())
qrels = json.loads((ROOT / "data" / "qrels.json").read_text())
main = {r["system"]: r for r in res["main"]}
sig = pd.DataFrame(res["significance"])
style_tbl = pd.DataFrame(res["by_style"])
abl = {r["system"]: r for r in res["ablation_components"]}
ktab = pd.DataFrame(res["ablation_rrf_k"])
ki = {r["system"]: r for r in res["known_item"]["table"]}
pq = pd.DataFrame(res["per_query_main"])
ap = pq.pivot(index="qid", columns="system", values="AP")
nd = pq.pivot(index="qid", columns="system", values="nDCG@10")
pr = pd.read_csv(OUT / "precision_recall_11pt.csv")
C, I, Q, D = res["corpus"], res["index_stats"], res["qrels"], res["dense"]
T, L = res["timings_seconds"], res["latency_ms_median"]


def f3(x):
    return f"{x:.3f}"


def f4(x):
    return f"{x:.4f}"


def pv(p):
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def best(metric, table=main):
    return max(table, key=lambda s: table[s][metric])


def p_of(metric, comp):
    row = sig[(sig.metric == metric) & (sig.comparison == comp)].iloc[0]
    return float(row.p_value), float(row.mean_diff), int(row.wins), int(row.ties), int(row.losses)


def style_ap(style, system):
    return float(style_tbl[(style_tbl["style"] == style) & (style_tbl.system == system)]["AP"].iloc[0])


# ---- claims made in the text, checked against the data -------------------
assert best("AP") == "Hybrid RRF" and best("Recall@100") == "Hybrid RRF"
assert best("P@10") == "Sentence Transformer" and best("nDCG@10") == "Sentence Transformer"
assert p_of("AP", "Hybrid RRF vs Sentence Transformer")[0] > 0.05
assert p_of("AP", "Hybrid RRF vs TF-IDF")[0] < 0.01 and p_of("AP", "Hybrid RRF vs BM25")[0] < 0.01
assert p_of("nDCG@10", "Hybrid RRF vs BM25")[0] < 0.01
assert max(MODELS, key=lambda m: style_ap("natural", m)) == "Sentence Transformer"
assert max(MODELS, key=lambda m: style_ap("keyword", m)) == "Hybrid RRF"
assert all(best(m, ki) == "Hybrid RRF" for m in ("MRR", "Success@1", "Recall@10", "nDCG@10"))
assert ki["BM25"]["MRR"] > ki["Sentence Transformer"]["MRR"]
assert all(p < 0.001 for p in res["known_item"]["p_values_mrr"].values())
assert abl["Sentence Transformer"]["AP"] > abl["ST FirstP (opening passage only)"]["AP"]
assert abl["RRF TF-IDF + BM25 + ST"]["AP"] < abl["Hybrid RRF"]["AP"]
assert main["BM25"]["AP"] > main["TF-IDF"]["AP"] and main["Sentence Transformer"]["RR"] == 1.0

# ---- document helpers ------------------------------------------------------
doc = Document()
counters = {"fig": 0, "tab": 0, "eq": 0}


def set_run_font(run, size=None, bold=None, italic=None, color=None, name=FONT):
    run.font.name = name
    rpr = run._r.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)
    return run


def style_font(style, name=FONT, size=None, bold=None, color=None, italic=None):
    style.font.name = name
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), name)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if rfonts.get(qn(a)) is not None:
            del rfonts.attrib[qn(a)]
    if size:
        style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if italic is not None:
        style.font.italic = italic
    if color:
        style.font.color.rgb = RGBColor.from_string(color)


def shade(cell, fill):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcpr.append(shd)


def cell_margins(table, top=50, bottom=50, left=90, right=90):
    tblpr = table._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for side, val in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"), str(val))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tblpr.append(mar)


def borders(table, color=RULE, inside=True, outer=True, size=4):
    tblpr = table._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        show = (edge in ("top", "bottom") and outer) or (edge == "insideH" and inside)
        el.set(qn("w:val"), "single" if show else "nil")
        if show:
            el.set(qn("w:sz"), str(size))
            el.set(qn("w:color"), color)
        b.append(el)
    tblpr.append(b)


def no_split(row):
    trpr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:cantSplit")
    trpr.append(el)


def repeat_header(row):
    trpr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    trpr.append(el)


def para(text="", size=10.5, bold=False, italic=False, color=INK, align=None, space_after=6,
         space_before=0, keep_next=False, style=None):
    p = doc.add_paragraph(style=style)
    if text:
        add_rich(p, text, size=size, bold=bold, italic=italic, color=color)
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(space_before)
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "justify":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    elif align == "right":
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    pf.keep_with_next = keep_next
    return p


def add_rich(p, text, size=10.5, bold=False, italic=False, color=INK):
    """Minimal markup: **bold**, *italic*, `code`."""
    import re
    tokens = re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", text)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("**"):
            set_run_font(p.add_run(tok[2:-2]), size, True, italic, color)
        elif tok.startswith("`"):
            set_run_font(p.add_run(tok[1:-1]), size - 1, bold, italic, "24476B", name=MONO)
        elif tok.startswith("*"):
            set_run_font(p.add_run(tok[1:-1]), size, bold, True, color)
        else:
            set_run_font(p.add_run(tok), size, bold, italic, color)
    return p


def body(text, **kw):
    kw.setdefault("align", "justify")
    return para(text, **kw)


def bullets(items, size=10.5, indent=0.6):
    for it in items:
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.left_indent = Cm(indent)
        pf.first_line_indent = Cm(-0.4)
        pf.space_after = Pt(3)
        set_run_font(p.add_run("•  "), size, color=BLUE)
        add_rich(p, it, size=size)


def numbered(items, size=10.5):
    for i, it in enumerate(items, 1):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.left_indent = Cm(0.75)
        pf.first_line_indent = Cm(-0.55)
        pf.space_after = Pt(3)
        set_run_font(p.add_run(f"{i}.  "), size, bold=True, color=BLUE)
        add_rich(p, it, size=size)


def heading(text, level=1, page_break=False):
    h = doc.add_heading(level=level)
    h.paragraph_format.page_break_before = page_break
    set_run_font(h.add_run(text), {1: 17, 2: 13, 3: 11}[level], True,
                 color={1: NAVY, 2: BLUE, 3: INK}[level])
    pf = h.paragraph_format
    pf.space_before = Pt({1: 4 if page_break else 18, 2: 14, 3: 10}[level])
    pf.space_after = Pt({1: 10, 2: 6, 3: 4}[level])
    pf.keep_with_next = True
    return h


def caption(kind, text):
    counters[kind] += 1
    label = {"fig": "Figure", "tab": "Table"}[kind]
    p = doc.add_paragraph()
    set_run_font(p.add_run(f"{label} {counters[kind]}. "), 9, True, color=INK2)
    set_run_font(p.add_run(text), 9, italic=True, color=INK2)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_after = Pt(10 if kind == "fig" else 4)
    pf.space_before = Pt(2 if kind == "fig" else 6)
    pf.keep_with_next = kind == "tab"
    return counters[kind]


def figure(path, width_cm, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(0)
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    return caption("fig", text)


def table(header, rows, widths, align=None, size=9.5, bold_cells=(), header_size=None, mono_cols=(),
          first_col_bold=False, keep_together=None):
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    borders(t)
    cell_margins(t)
    align = align or ["left"] + ["center"] * (len(header) - 1)
    amap = {"left": WD_ALIGN_PARAGRAPH.LEFT, "center": WD_ALIGN_PARAGRAPH.CENTER,
            "right": WD_ALIGN_PARAGRAPH.RIGHT}
    for j, h in enumerate(header):
        c = t.rows[0].cells[j]
        shade(c, HEADER_FILL)
        c.paragraphs[0].alignment = amap[align[j]]
        set_run_font(c.paragraphs[0].add_run(h), header_size or size, True, color="FFFFFF")
        c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    repeat_header(t.rows[0])
    for i, row in enumerate(rows, 1):
        no_split(t.rows[i])
        for j, val in enumerate(row):
            c = t.rows[i].cells[j]
            if i % 2 == 0:
                shade(c, ZEBRA)
            c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p = c.paragraphs[0]
            p.alignment = amap[align[j]]
            p.paragraph_format.space_after = Pt(0)
            is_bold = (i - 1, j) in bold_cells or (first_col_bold and j == 0)
            if j in mono_cols:
                set_run_font(p.add_run(str(val)), size - 0.5, is_bold, color="24476B", name=MONO)
            else:
                add_rich(p, str(val), size=size, bold=is_bold)
    for row in t.rows:
        for j, w in enumerate(widths):
            row.cells[j].width = Cm(w)
    if keep_together if keep_together is not None else len(rows) <= 14:
        for row in t.rows[:-1]:
            for c in row.cells:
                for q in c.paragraphs:
                    q.paragraph_format.keep_with_next = True
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def callout(title, lines, fill=CALLOUT, accent="2A78D6"):
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    tblpr = t._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "nil")
        b.append(el)
    left = OxmlElement("w:left")
    left.set(qn("w:val"), "single")
    left.set(qn("w:sz"), "24")
    left.set(qn("w:color"), accent)
    b.append(left)
    tblpr.append(b)
    cell_margins(t, 110, 110, 180, 160)
    c = t.rows[0].cells[0]
    c.width = Cm(16.2)
    shade(c, fill)
    p = c.paragraphs[0]
    set_run_font(p.add_run(title), 10.5, True, color=NAVY)
    p.paragraph_format.space_after = Pt(3)
    for ln in lines:
        q = c.add_paragraph()
        q.paragraph_format.space_after = Pt(2)
        q.paragraph_format.left_indent = Cm(0.35)
        q.paragraph_format.first_line_indent = Cm(-0.35)
        set_run_font(q.add_run("•  "), 10, color=BLUE)
        add_rich(q, ln, size=10)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def equation(tex, fontsize=13.5):
    counters["eq"] += 1
    EQ.mkdir(parents=True, exist_ok=True)
    path = EQ / f"eq{counters['eq']:02d}.png"
    fig = plt.figure(figsize=(0.01, 0.01), dpi=300)
    fig.text(0, 0, f"${tex}$", fontsize=fontsize, color="#1f2328",
             math_fontfamily="stix")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03, transparent=True)
    plt.close(fig)
    from PIL import Image
    w_px, _ = Image.open(path).size
    t = doc.add_table(rows=1, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    borders(t, inside=False, outer=False)
    t.autofit = False
    c0, c1 = t.rows[0].cells
    c0.width, c1.width = Cm(14.6), Cm(1.6)
    p0 = c0.paragraphs[0]
    p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p0.add_run().add_picture(str(path), width=Cm(min(w_px / 300 * 2.54, 14.2)))
    c0.vertical_alignment = c1.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    p1 = c1.paragraphs[0]
    p1.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    set_run_font(p1.add_run(f"({counters['eq']})"), 10, color=INK2)
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(0)
    sp.paragraph_format.line_spacing = Pt(4)
    return counters["eq"]


def add_field(paragraph, instr):
    run = paragraph.add_run()
    for kind, text in (("begin", None), ("instr", instr), ("separate", None), ("text", "1"), ("end", None)):
        if kind == "instr":
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        elif kind == "text":
            el = OxmlElement("w:t")
            el.text = text
        else:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        run._r.append(el)
    return run


def toc():
    p = doc.add_paragraph()
    run = p.add_run()
    b = OxmlElement("w:fldChar")
    b.set(qn("w:fldCharType"), "begin")
    ins = OxmlElement("w:instrText")
    ins.set(qn("xml:space"), "preserve")
    ins.text = 'TOC \\o "1-2" \\h \\z \\u'
    s = OxmlElement("w:fldChar")
    s.set(qn("w:fldCharType"), "separate")
    t = OxmlElement("w:t")
    t.text = "Table of contents - right-click and choose Update Field if it is empty."
    e = OxmlElement("w:fldChar")
    e.set(qn("w:fldCharType"), "end")
    for el in (b, ins, s, t, e):
        run._r.append(el)


# ---- page setup & styles ----------------------------------------------------
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.3)
sec.top_margin, sec.bottom_margin = Cm(2.2), Cm(2.0)
sec.header_distance, sec.footer_distance = Cm(1.0), Cm(1.0)
sec.different_first_page_header_footer = True

normal = doc.styles["Normal"]
style_font(normal, FONT, 10.5, color=INK)
normal.paragraph_format.line_spacing = 1.12
for lvl in (1, 2, 3):
    style_font(doc.styles[f"Heading {lvl}"], FONT, {1: 17, 2: 13, 3: 11}[lvl], True,
               {1: NAVY, 2: BLUE, 3: INK}[lvl])
for name in ("toc 1", "toc 2"):
    try:
        st = doc.styles[name]
    except KeyError:
        from docx.enum.style import WD_STYLE_TYPE
        st = doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        st.element.attrib.pop(qn("w:customStyle"), None)
        st.base_style = doc.styles["Normal"]
    style_font(st, FONT, 10 if name == "toc 1" else 9.5, name == "toc 1", INK)
    st.paragraph_format.space_after = Pt(1)
    st.paragraph_format.space_before = Pt(5 if name == "toc 1" else 0)
    st.paragraph_format.line_spacing = 1.0
    if name == "toc 2":
        st.paragraph_format.left_indent = Cm(0.6)

hp = sec.header.paragraphs[0]
hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
set_run_font(hp.add_run("BBC News Retrieval  ·  Information Retrieval Project 12"), 8.5, color=MUTED)
fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
set_run_font(fp.add_run("Page "), 8.5, color=MUTED)
set_run_font(add_field(fp, "PAGE"), 8.5, color=MUTED)

# =============================================================================
# Title page
# =============================================================================
para("", space_after=110)
para("INFORMATION RETRIEVAL  —  PROJECT 12", 11, True, color="1E8C9A", space_after=10)
para("BBC News Retrieval System", 30, True, color=NAVY, space_after=6)
para("Lexical, Semantic and Hybrid Ranking for a Categorised News Archive", 15, color=INK2, space_after=24)
t = doc.add_table(rows=2, cols=3)
t.alignment = WD_TABLE_ALIGNMENT.CENTER
borders(t, color="B9C4D3")
cell_margins(t, 90, 90, 120, 120)
for j, (h, v) in enumerate((("Student Name", "[Enter name]"), ("Registration Number", "[Enter registration number]"),
                            ("Institutional Email", "[Enter email address]"))):
    shade(t.rows[0].cells[j], HEADER_FILL)
    set_run_font(t.rows[0].cells[j].paragraphs[0].add_run(h), 9.5, True, color="FFFFFF")
    set_run_font(t.rows[1].cells[j].paragraphs[0].add_run(v), 10, color=INK2)
    for r in range(2):
        t.rows[r].cells[j].width = Cm(5.4)
para("", space_after=10)
for k, v in (("Course", "Information Retrieval"), ("Project", "12 — BBC News Retrieval (News Archives)"),
             ("Dataset", "BBC News, Greene & Cunningham (2006): 2,225 articles, 5 categories"),
             ("Submitted", "November 2026")):
    p = para("", space_after=2)
    set_run_font(p.add_run(f"{k}:  "), 10.5, True, color=NAVY)
    set_run_font(p.add_run(v), 10.5, color=INK)
para("", space_after=26)
callout("Results at a glance", [
    f"**Corpus:** the official BBC News dataset — {C['raw_articles']:,} articles, {C['unique_articles']:,} after removing "
    f"{C['exact_duplicates_removed']} exact duplicates; {I['vocabulary_terms']:,} index terms.",
    "**Models:** TF-IDF (cosine), Okapi BM25, Sentence Transformer (all-MiniLM-L6-v2, passage max-pooling) "
    "and Hybrid Reciprocal Rank Fusion (BM25 + Sentence Transformer, k = 60).",
    f"**Evaluation:** {Q['queries']} queries with {Q['judged_pairs']} graded relevance judgments, plus "
    f"{res['known_item']['queries']:,} automatically generated headline (known-item) queries.",
    f"**Best results:** Hybrid RRF — MAP {f3(main['Hybrid RRF']['AP'])}, Recall@100 {f3(main['Hybrid RRF']['Recall@100'])}; "
    f"Sentence Transformer — nDCG@10 {f3(main['Sentence Transformer']['nDCG@10'])}, P@10 {f3(main['Sentence Transformer']['P@10'])}; "
    f"Hybrid RRF best on known-item search (MRR {f3(ki['Hybrid RRF']['MRR'])}).",
])

# =============================================================================
# Abstract
# =============================================================================
heading("Abstract", 1, page_break=True)
s_n, b_n = style_ap("natural", "Sentence Transformer"), style_ap("natural", "BM25")
body(
    "This project builds and evaluates an information-retrieval system for the BBC News corpus, 2,225 articles "
    "published on the BBC News website in 2004–2005 across business, entertainment, politics, sport and technology. "
    f"After decoding, cleaning and exact-duplicate removal ({C['unique_articles']:,} unique articles), the text is tokenised, "
    "stop-word filtered and Porter-stemmed, stored in an inverted index, and ranked by four models: TF-IDF with cosine "
    "similarity, Okapi BM25, a Sentence Transformer bi-encoder (all-MiniLM-L6-v2 with max-passage scoring) and a hybrid "
    "model that merges BM25 and the Sentence Transformer with Reciprocal Rank Fusion (RRF).")
body(
    f"A test collection of {Q['queries']} queries — 13 keyword queries and 12 deliberately paraphrased natural-language "
    f"queries — was assessed with depth-20 pooling and graded relevance ({Q['judged_pairs']} judged pairs, "
    f"{Q['relevant']} relevant). Hybrid RRF obtained the highest Mean Average Precision (MAP {f3(main['Hybrid RRF']['AP'])}) "
    f"and Recall@100 ({f3(main['Hybrid RRF']['Recall@100'])}); the Sentence Transformer obtained the highest Precision@10 "
    f"({f3(main['Sentence Transformer']['P@10'])}) and nDCG@10 ({f3(main['Sentence Transformer']['nDCG@10'])}). The two are "
    f"statistically indistinguishable, and both significantly outperform TF-IDF (MAP {f3(main['TF-IDF']['AP'])}) and BM25 "
    f"(MAP {f3(main['BM25']['AP'])}). The lexical models lose most on paraphrased queries (BM25 MAP {f3(b_n)} versus "
    f"{f3(s_n)} for the Sentence Transformer). On a second, fully automatic test of {res['known_item']['queries']:,} "
    f"headline-to-article known-item queries, Hybrid RRF is best on every metric (MRR {f3(ki['Hybrid RRF']['MRR'])} versus "
    f"{f3(ki['BM25']['MRR'])} for BM25 and {f3(ki['Sentence Transformer']['MRR'])} for the Sentence Transformer; p < 0.001).")
body(
    f"Hybrid ranking is therefore the most robust choice: it matches the semantic model where wording differs and keeps "
    f"the precision of exact matching for names and headlines, at about {L['Hybrid RRF']:.1f} ms per query. "
    "A Streamlit search interface exposes all four models, a side-by-side comparison view and the evaluation results.")
para("Keywords", 11, True, color=NAVY, space_before=8, space_after=3)
body("Information retrieval; news search; inverted index; TF-IDF; BM25; Sentence-BERT; dense retrieval; "
     "Reciprocal Rank Fusion; test collection; pooling; MAP; nDCG; precision–recall.")
para("Abbreviations", 11, True, color=NAVY, space_before=8, space_after=4, keep_next=True)
table(["Term", "Meaning"], [
    ["IR", "Information Retrieval"], ["TF / IDF", "Term Frequency / Inverse Document Frequency"],
    ["BM25", "Okapi Best Matching 25 probabilistic ranking function"],
    ["ST / SBERT", "Sentence Transformer / Sentence-BERT bi-encoder"],
    ["MaxP / FirstP", "Document score = best passage / first passage only"],
    ["RRF", "Reciprocal Rank Fusion"], ["P@k, R@k", "Precision and Recall at rank cut-off k"],
    ["AP / MAP", "Average Precision / Mean Average Precision"],
    ["nDCG@k", "Normalised Discounted Cumulative Gain at cut-off k"], ["MRR", "Mean Reciprocal Rank"],
    ["Qrels", "Query relevance judgments"], ["MPS", "Apple Metal Performance Shaders (GPU backend)"],
], [3.4, 12.0], size=9.5, align=["left", "left"])

para("Contents", 17, True, color=NAVY, space_before=4, space_after=10).paragraph_format.page_break_before = True
toc()

# =============================================================================
# 1 Introduction
# =============================================================================
heading("1. Introduction", 1, page_break=True)
body(
    "News archives are a classic retrieval setting. They contain thousands of short, fact-dense articles; their users "
    "issue queries that range from a single name (*Yukos*) to a loosely worded information need (*why the American "
    "currency is losing value*); and users expect the most useful articles within the first few results. Exact keyword "
    "matching handles the first kind of query well but struggles with the second, where the query and the article "
    "describe the same story in different words — the *vocabulary-mismatch* problem. Neural sentence encoders address "
    "vocabulary mismatch, but they can blur rare names and exact phrases that lexical models match precisely.")
body(
    "This project implements both families of ranker over the same BBC News collection, combines them with Reciprocal "
    "Rank Fusion, and measures the trade-offs with a purpose-built test collection, a large automatic known-item test "
    "and significance testing.")
heading("1.1 Problem statement", 2)
body("Given the BBC News archive and a free-text query, return a ranked list of articles ordered by estimated "
     "relevance, and determine experimentally which ranking strategy — lexical, semantic or hybrid — produces the best "
     "ranking quality, and at what computational cost.")
heading("1.2 Objectives", 2)
numbered([
    "Prepare the BBC News corpus: decoding, headline/body separation, duplicate removal and stable document identifiers.",
    "Implement text preprocessing: cleaning, tokenisation, stop-word removal and Porter stemming.",
    "Build an inverted index and document vectors (sparse TF-IDF vectors and dense passage embeddings).",
    "Implement the ranking algorithms TF-IDF, BM25 and a Sentence Transformer, plus a hybrid model based on "
    "Reciprocal Rank Fusion (RRF).",
    "Construct a test collection (queries and graded relevance judgments) and evaluate with Precision@10, Recall@100, "
    "MAP and nDCG@10, precision–recall curves and paired significance tests.",
    "Deliver a search interface, a performance comparison table and a precision–recall graph, with reproducible code.",
])
heading("1.3 Scope", 2)
body("The system performs text-only, topical retrieval over a static English archive. Query logs, learning-to-rank, "
     "cross-encoder re-ranking, freshness and credibility are out of scope. The models use standard, untuned parameters, "
     "so the comparison reflects the methods rather than tuning effort on the test queries.")
heading("1.4 Key terminology", 2)
table(["Term", "Meaning in this project"], [
    ["Document", "One BBC article (headline + body) with an ID such as `tec-040` (technology, file 040)."],
    ["Query / topic", "A short search string plus a written narrative that defines what counts as relevant."],
    ["Term", "A normalised, stemmed token, e.g. *expanded* → `expand`."],
    ["Posting list", "For one term: the documents containing it and the term frequency in each."],
    ["Inverted index", "The mapping from every term to its posting list."],
    ["Run", "The ranked list a model returns for a query (up to 1,000 documents)."],
    ["Qrels", "Relevance judgments: graded labels (0, 1, 2) for query–document pairs."],
    ["Pooling", "Judging only the union of the top-k documents of several runs."],
    ["Embedding", "A 384-dimensional vector produced by the Sentence Transformer."],
], [3.4, 12.0], size=9.5, align=["left", "left"])

# =============================================================================
# 2 Dataset
# =============================================================================
heading("2. Dataset", 1, page_break=True)
heading("2.1 The BBC News corpus", 2)
body(
    "The collection is the BBC News dataset released by Greene and Cunningham [1] (mlg.ucd.ie/datasets/bbc.html): "
    f"{C['raw_articles']:,} articles from the BBC News website, published in 2004–2005 and labelled with one of five "
    "topical areas. Each article is a plain-text file whose first line is the headline and whose remaining lines are the "
    "body. All rights in the original articles belong to the BBC; the dataset is distributed for research and teaching.")
heading("2.2 Data preparation", 2)
bullets([
    "**Decoding.** All 2,225 files are valid UTF-8. Decoding them as Latin-1 (a common default for this dataset) "
    "corrupts the pound sign into “Â£”, so UTF-8 is used; 22 files containing HTML entities such as `&#163;` are unescaped.",
    "**Structure.** The first non-empty line becomes the `title`; the remaining lines become the `body`. Document IDs "
    "combine a category prefix and the file number (`bus-`, `ent-`, `pol-`, `spo-`, `tec-`).",
    f"**De-duplication.** {C['exact_duplicates_removed']} files are exact copies (same headline and body) of another "
    f"article and were removed, leaving **{C['unique_articles']:,} unique articles**. Duplicates would otherwise "
    "occupy two result slots with identical content.",
    f"**Shared headlines.** {C['duplicate_titles_remaining']} remaining articles share a headline with a different, "
    "non-identical article (typically updated versions). They stay in the index but are excluded from the known-item "
    "queries of Section 2.5, where a headline must identify exactly one article.",
])
rows = []
for cat, lab in (("business", "Business"), ("entertainment", "Entertainment"), ("politics", "Politics"),
                 ("sport", "Sport"), ("tech", "Technology")):
    rows.append([lab, f"{C['raw_per_category'][cat]:,}", f"{C['unique_per_category'][cat]:,}",
                 f"{100 * C['unique_per_category'][cat] / C['unique_articles']:.1f}%"])
rows.append(["**Total**", f"**{C['raw_articles']:,}**", f"**{C['unique_articles']:,}**", "**100%**"])
caption("tab", "Articles per category before and after exact-duplicate removal.")
table(["Category", "Raw articles", "Unique articles", "Share"], rows, [5.0, 3.4, 3.6, 3.0])
caption("tab", "Collection statistics (unique articles).")
table(["Statistic", "Value"], [
    ["Body length (words): mean / median", f"{C['body_words_mean']:.1f} / {C['body_words_median']:.0f}"],
    ["Body length (words): min / max", f"{C['body_words_min']} / {C['body_words_max']:,}"],
    ["Index tokens after preprocessing", f"{I['total_tokens']:,} ({I['avg_doc_length_tokens']:.1f} per article)"],
    ["Vocabulary (distinct stemmed terms)", f"{I['vocabulary_terms']:,}"],
    ["Terms occurring in a single article", f"{I['hapax_terms']:,} ({100 * I['hapax_terms'] / I['vocabulary_terms']:.0f}%)"],
    ["Passages for the Sentence Transformer", f"{D['passages']:,} ({D['passages_per_doc']:.2f} per article)"],
], [8.5, 6.5], align=["left", "left"])
figure(FIG / "categories_light.png", 12.5, "Unique articles per category. The collection is roughly balanced; "
       "sport and business are the largest sections.")
figure(FIG / "doc_length_light.png", 12.5, "Distribution of article body length. Most articles are 200–500 words; "
       "a long tail reaches 4,428 words, longer than the 256 word-piece window of the Sentence Transformer.")

heading("2.3 Query set", 2)
n_kw = sum(q["style"] == "keyword" for q in queries)
body(
    f"The dataset has no official queries, so a test collection was built in the style of TREC [12]. {Q['queries']} topics "
    "were written, five per category, each with a short query and a narrative that defines relevance. Two query styles "
    f"are mixed on purpose: **{n_kw} keyword queries** use the vocabulary of the story, often named entities "
    f"(*Kilroy-Silk UKIP Veritas*), while **{Q['queries'] - n_kw} natural-language queries** are paraphrased so that they "
    "avoid the articles' key terms (*films that took the most money at cinemas* for box-office reports; "
    "*fraudsters tricking people into revealing bank details online* for phishing). Table 3 shows examples and "
    "Appendix A lists all topics.")
ex = ["Q01", "Q02", "Q07", "Q13", "Q20", "Q25"]
caption("tab", "Example topics (full list in Appendix A). H/P = highly / partially relevant documents.")
table(["ID", "Query", "Style", "Relevant (H/P)"],
      [[q["qid"], q["query"], q["style"],
        f"{sum(g == 2 for g in qrels[q['qid']].values())} / {sum(g == 1 for g in qrels[q['qid']].values())}"]
       for q in queries if q["qid"] in ex], [1.3, 9.2, 2.3, 2.8], align=["left", "left", "center", "center"])

heading("2.4 Relevance assessment", 2)
body(
    f"**Pooling.** For each topic, the top {Q['pool_depth']} documents of five runs (TF-IDF, BM25, Sentence Transformer "
    f"MaxP and FirstP, and Hybrid RRF) were merged into a pool — {Q['pooled_pairs']} query–document pairs, 23–49 per "
    "topic. The pool was shown in document-ID order without system or rank information, so judging was blind to which "
    "model retrieved each article. Each article was judged from its headline and lead paragraph, consulting the full "
    "text for borderline cases.")
body(
    f"**Pool enrichment.** Pools built only from the evaluated systems can miss relevant documents that every system "
    f"ranks low. A keyword sweep over the whole collection (for example, every article that mentions Yukos) surfaced "
    f"unpooled candidates, which were judged with the same criteria. It added {Q['judged_pairs'] - Q['pooled_pairs']} "
    f"judged pairs and **{Q['relevant_found_by_sweep']} relevant documents**, mostly for broad topics such as Australian "
    "Open coverage, Wales' Six Nations campaign, box-office reports and malware.")
body(
    "**Grades.** 2 = highly relevant (the article is about the topic); 1 = partially relevant (the topic is a "
    "substantial secondary theme); 0 = not relevant or only mentioned in passing.")
caption("tab", "Summary of the relevance judgments.")
table(["Quantity", "Value"], [
    ["Topics", f"{Q['queries']}"], ["Judged query–document pairs", f"{Q['judged_pairs']}"],
    ["Relevant (grade ≥ 1) / highly relevant (grade 2)", f"{Q['relevant']} / {Q['highly_relevant']}"],
    ["Relevant per topic: min / mean / max", f"{Q['relevant_per_query_min']} / {Q['relevant_per_query_mean']} / "
                                             f"{Q['relevant_per_query_max']}"],
    ["Relevant documents found only by the keyword sweep", f"{Q['relevant_found_by_sweep']}"],
    ["Judged share of each run's top 10 / top 20", f"{100 * min(main[m]['Judged@10'] for m in MODELS):.0f}% / "
                                                  f"≥ {100 * min(main[m]['Judged@20'] for m in MODELS):.1f}%"],
], [9.5, 5.5], align=["left", "center"])
body("Because every document in every model's top 10 — and at least 99.8% of each top 20 — carries a judgment, "
     "the standard assumption that unjudged documents are non-relevant has almost no influence on the "
     "precision-oriented metrics.")

heading("2.5 Known-item headline queries", 2)
body(
    f"As a second, fully automatic and much larger test, every article whose headline is unique becomes a query "
    f"({res['known_item']['queries']:,} queries). Its only relevant document is the article itself. For this test the "
    "index is rebuilt over the article **bodies only**, so a query can never match its own headline verbatim. The "
    "setting simulates a user who remembers a story's headline and wants to find it again. With one relevant document "
    "per query, MAP equals the Mean Reciprocal Rank (MRR).")

# =============================================================================
# 3 Methodology
# =============================================================================
heading("3. Methodology", 1, page_break=True)
heading("3.1 System architecture", 2)
body(
    "Figure 3 shows the pipeline. Offline, each article is preprocessed once. The lexical branch builds an inverted "
    "index (used by BM25) and a sparse TF-IDF matrix. The semantic branch splits the *raw* text into passages and "
    "embeds them, because neural encoders are trained on natural text rather than stemmed tokens. At query time the "
    "query passes through the same preprocessing (lexical branch) or the same encoder (semantic branch). Each model "
    "returns a ranked list of document IDs, and RRF fuses the BM25 and Sentence-Transformer lists by rank position.")
figure(FIG / "pipeline_light.png", 16.2, "System architecture: offline indexing (left) and the four ranking models "
       "(right). Coloured bars mark the model each component feeds.")

heading("3.2 Text preprocessing", 2)
caption("tab", "Preprocessing steps (identical for documents and queries).")
table(["Step", "Implementation", "Purpose"], [
    ["Cleaning", "Unicode NFKD → ASCII; lower-case; drop possessive *'s*; non-alphanumerics → space",
     "Remove case and punctuation variation"],
    ["Tokenisation", "Regular expression `[a-z0-9]+`", "Split text into word tokens"],
    ["Stop-word removal", "NLTK English list (198 words) plus single-character tokens",
     "Drop function words with little topical value"],
    ["Stemming", "Porter stemmer [5] (NLTK), memoised", "Conflate inflections: *players* → `player`"],
], [3.0, 7.6, 5.0], align=["left", "left", "left"], size=9.2)
from bbcir.preprocess import explain  # noqa: E402
sentence = res["preprocessing_example"]["original"].split(". ", 1)[1].split("\n")[0]
pe = explain(sentence)
body(f"Worked example — the first sentence of article `tec-040`, *“{sentence}”*, becomes:", keep_next=True)


def shorten(tokens, n=18):
    return " ".join(tokens[:n]) + (" …" if len(tokens) > n else "")


table(["Stage", "Output"], [
    ["Cleaned text", pe["cleaned"]],
    ["Tokens", f"{len(pe['tokens'])} tokens: " + " ".join(pe["tokens"])],
    ["Without stop words", f"{len(pe['without_stopwords'])} tokens: " + " ".join(pe["without_stopwords"])],
    ["Stemmed terms", " ".join(pe["stemmed"])],
], [3.6, 11.8], align=["left", "left"], mono_cols=(1,), size=9.5)
body("The query *films that took the most money at cinemas* becomes the four terms `film took money cinema`. "
     "The stemmed form explains why stems such as `appl` or `yuko` appear in the index.")

heading("3.3 Inverted index", 2)
body(
    "The index maps each term to two aligned NumPy arrays: the IDs of the documents that contain it and the term "
    "frequency in each. It is built in a single pass over the token lists and also stores every document's length and "
    "the average length needed by BM25. It supports document-frequency look-ups, posting retrieval and Boolean AND/OR "
    f"(postings intersection and union). Building it for {I['documents']:,} articles takes {T['Inverted index']:.2f} s.")
caption("tab", "Inverted-index statistics.")
top = res["top_df_terms"][0]
table(["Statistic", "Value"], [
    ["Distinct terms (vocabulary)", f"{I['vocabulary_terms']:,}"],
    ["Postings (term–document pairs)", f"{I['total_postings']:,}"],
    ["Average posting-list length", f"{I['avg_postings_length']:.1f} documents"],
    ["Longest posting list", f"`{top[0]}` — {top[1]:,} of {I['documents']:,} documents"],
    ["Terms with a single posting", f"{I['hapax_terms']:,}"],
], [8.5, 6.5], align=["left", "left"])
rows = []
for word, info in res["example_postings"].items():
    rows.append([word, f"`{info['stem']}`", f"{info['df']}", f"{info['idf_bm25']:.2f}",
                 ", ".join(f"{d} ({tf})" for d, tf in info["top_postings"][:4])])
caption("tab", "Example posting lists (documents with the highest term frequency shown, tf in brackets).")
table(["Word", "Stem", "df", "BM25 idf", "Top postings"], rows, [2.2, 1.8, 1.2, 1.9, 8.4],
      align=["left", "left", "center", "center", "left"], size=9)
body("Rare, topical terms such as *Kenteris* (df = 16) receive high IDF weights, while a common term such as "
     "*election* (df = 228) is far less discriminative. The ranking models below exploit exactly this contrast.")

heading("3.4 TF-IDF vector-space model", 2)
body("Each document is a vector of term weights that combine a sublinear term frequency with a smoothed inverse "
     "document frequency, where N is the number of documents and df(t) the number of documents containing t:")
equation(r"w_{t,d} \;=\; \left(1 + \ln \mathrm{tf}_{t,d}\right)\,\times\,\left(\ln\dfrac{1+N}{1+\mathrm{df}_t} + 1\right)")
body("Vectors are L2-normalised, so the ranking score — the dot product with the equally weighted query vector — "
     "equals the cosine similarity:")
equation(r"\cos(\vec{q},\vec{d}) \;=\; \dfrac{\vec{q}\cdot\vec{d}}{\|\vec{q}\|\;\|\vec{d}\|}")
body(f"The {I['documents']:,} × {res['tfidf_vocabulary']:,} sparse document matrix is built with scikit-learn's "
     "TfidfVectorizer over the same token lists as the inverted index. Only documents with a positive score — those "
     "sharing at least one term with the query — are retrieved.")

heading("3.5 Okapi BM25", 2)
body("BM25 [3] scores a document by summing, over the query terms, an IDF weight multiplied by a saturating term-"
     "frequency component that is normalised by document length:")
equation(r"\mathrm{BM25}(q,d) \;=\; \sum_{t \in q} \mathrm{idf}(t)\cdot\dfrac{\mathrm{tf}_{t,d}\,(k_1+1)}"
         r"{\mathrm{tf}_{t,d} + k_1\left(1 - b + b\,\frac{|d|}{\mathrm{avgdl}}\right)}")
equation(r"\mathrm{idf}(t) \;=\; \ln\!\left(1 + \dfrac{N - \mathrm{df}_t + 0.5}{\mathrm{df}_t + 0.5}\right)")
body("Here k₁ = 1.2 controls term-frequency saturation and b = 0.75 the strength of length normalisation — "
     "the usual defaults, deliberately not tuned on the test topics. Scores are accumulated term-at-a-time "
     "directly from the inverted index.")

heading("3.6 Sentence Transformer (dense retrieval)", 2)
body(
    "The semantic model is *all-MiniLM-L6-v2* [6, 7], a six-layer MiniLM encoder trained with a contrastive objective on "
    f"over one billion sentence pairs. It maps text to a {D['dim']}-dimensional, unit-length embedding, so related texts "
    "are close even when they share no words. The encoder reads at most 256 word pieces, but the average article is "
    f"{C['body_words_mean']:.0f} words, so each article (headline + body) is split into overlapping 160-word passages "
    f"with a stride of 120 words — {D['passages']:,} passages in total. A document's score is its best passage (MaxP [8]):")
equation(r"s(q,d) \;=\; \max_{p\,\in\,\mathrm{passages}(d)}\; \cos\left(E(q),\,E(p)\right)")
body(f"Passage embeddings are computed once on the Apple-silicon GPU (MPS backend) in {T['Passage embeddings']:.1f} s "
     "and cached. At search time only the query is encoded; scoring is an exact dot product with the "
     f"{D['passages']:,} × {D['dim']} passage matrix, so no approximate nearest-neighbour index is needed at this scale. "
     "Scoring only the opening passage (FirstP) is evaluated as an ablation.")

heading("3.7 Hybrid model: Reciprocal Rank Fusion", 2)
body("Lexical and dense scores live on incompatible scales (a BM25 score of 12 means nothing to a cosine of 0.6). "
     "RRF [9] therefore ignores raw scores and combines rank positions only:")
equation(r"\mathrm{RRF}(d) \;=\; \sum_{m\,\in\,\{\mathrm{BM25},\,\mathrm{ST}\}} \dfrac{1}{k + \mathrm{rank}_m(d)}"
         r"\qquad k = 60")
body("The hybrid fuses one lexical ranker (BM25) and one semantic ranker (the Sentence Transformer), each "
     "contributing its top 1,000 documents; a document missing from a list receives nothing from it. Both the component "
     "choice and k = 60 (the value recommended by Cormack et al.) were fixed **before** the evaluation was run. Other "
     "combinations and values of k are reported only as ablations (Section 5.8), so the main result is not tuned on "
     "the test topics.")
caption("tab", "Illustrative RRF calculation (k = 60): consistent agreement beats a single first place.")
table(["Document", "BM25 rank", "ST rank", "RRF score", "Fused rank"], [
    ["A", "1", "9", "1/61 + 1/69 = 0.03088", "2"],
    ["B", "3", "2", "1/63 + 1/62 = 0.03200", "1"],
    ["C", "2", "—", "1/62 = 0.01613", "3"],
], [2.4, 2.4, 2.4, 5.4, 2.4])

heading("3.8 Search interface", 2)
body("A Streamlit application (`app.py`) provides the search interface, styled as a dark editorial “news desk”. "
     "The engine is built once and cached, and each query is ranked by all four models in a few milliseconds; the "
     "measured time is shown with the results. Colour is used for one thing only — the ranking model, in the same "
     "palette as the charts in this report.")
bullets([
    "Search box with one-click example queries; model selector (TF-IDF, BM25, Sentence Transformer, Hybrid RRF); "
    "result count; filter by news desk (category).",
    "Result cards showing rank, desk, document ID, the model score with a bar relative to the top result, the matched "
    "index terms, and a snippet made from the sentences with the most query-term matches (matches highlighted). The full "
    "article expands in place; results with no shared index terms are labelled as semantic matches.",
    "A **Compare models** tab with the top five of all four models side by side, a consensus strip and a badge "
    "showing how many models retrieved each article.",
    "A **Benchmark** tab with headline metrics, interactive charts (tooltips and a precision–recall crosshair) and a "
    "**topic explorer** that re-runs any of the 25 test topics live, labels every result with its relevance grade and "
    "recomputes P@10, AP and nDCG@10 from the judgments.",
])
figure(FIG / "ui_search.png", 13.2, "Search interface: Hybrid RRF results for the paraphrased query “films that took "
       "the most money at cinemas”, with highlighted query terms and live ranking time.")
figure(FIG / "ui_compare.png", 15.6, "“Compare models” view for the same query, with agreement badges. TF-IDF matches the "
       "surface words *film* and *cinema* (festival and digital-cinema stories); the Sentence Transformer finds "
       "box-office reports; only one article is in every model's top five.")
figure(FIG / "ui_explorer.png", 15.6, "Topic explorer in the Benchmark tab: topic Q07 re-scored live, with each result's "
       "relevance grade. The live values match the per-topic results in Appendix B.")

heading("3.9 Implementation", 2)
caption("tab", "Project structure.")
table(["File / module", "Responsibility"], [
    ["`bbcir/data.py`", "Read the raw corpus, decode, split headline/body, de-duplicate"],
    ["`bbcir/preprocess.py`", "Cleaning, tokenisation, stop-word removal, Porter stemming"],
    ["`bbcir/index.py`", "Inverted index, Boolean operations, index statistics"],
    ["`bbcir/models.py`", "TF-IDF, BM25, Sentence-Transformer (MaxP/FirstP) rankers and RRF"],
    ["`bbcir/metrics.py`", "P@k, Recall@k, AP, nDCG@k, RR, interpolated PR, randomisation test"],
    ["`bbcir/engine.py`", "SearchEngine facade and snippet highlighting"],
    ["`scripts/prepare_data.py`", "Raw files → `data/bbc_articles.csv`"],
    ["`scripts/build_pool.py`", "Depth-20 pooling for relevance assessment"],
    ["`scripts/write_qrels.py`", "Graded judgments → `data/qrels.json`"],
    ["`scripts/evaluate.py`", "All experiments → `outputs/*.csv`, `outputs/results.json`"],
    ["`scripts/make_figures.py`", "Charts for the report and slides"],
    ["`app.py`", "Streamlit search interface"],
], [5.6, 9.8], align=["left", "left"], size=9.2)

# =============================================================================
# 4 Experimental setup
# =============================================================================
heading("4. Experimental Setup", 1, page_break=True)
heading("4.1 Evaluation protocol", 2)
bullets([
    "Every model returns up to 1,000 documents per query; TF-IDF and BM25 return only documents with a positive score.",
    "Metrics are computed per query and macro-averaged over the 25 topics.",
    "Binary metrics (P@10, Recall@100, AP) treat grades 1 and 2 as relevant; nDCG@10 uses the grades as gains.",
    "Unjudged documents count as non-relevant (the standard pooling assumption; Section 2.4 shows its effect is negligible).",
    "The known-item test uses the body-only index, so the reported numbers measure finding an article from its headline.",
])
heading("4.2 Metrics", 2)
body("**Precision@k** and **Recall@k** count the relevant documents among the first k results, relative to k and to "
     "all known relevant documents of the query (Rel):")
equation(r"\mathrm{P@}k = \dfrac{|\,\mathrm{Rel} \cap \mathrm{Top}_k\,|}{k}\qquad\qquad"
         r"\mathrm{Recall@}k = \dfrac{|\,\mathrm{Rel} \cap \mathrm{Top}_k\,|}{|\,\mathrm{Rel}\,|}")
body("**Average Precision** averages the precision at each rank where a relevant document appears (relevant "
     "documents never retrieved contribute zero); **MAP** is its mean over the query set Q:")
equation(r"\mathrm{AP}(q) = \dfrac{1}{|\mathrm{Rel}_q|}\sum_{k=1}^{n} \mathrm{P@}k\cdot\mathrm{rel}(k)\qquad\qquad"
         r"\mathrm{MAP} = \dfrac{1}{|Q|}\sum_{q\in Q}\mathrm{AP}(q)")
body("**nDCG@k** [10] rewards highly relevant documents more than partially relevant ones and discounts lower "
     "ranks logarithmically; it is normalised by the DCG of the ideal ordering (IDCG):")
equation(r"\mathrm{DCG@}k = \sum_{i=1}^{k}\dfrac{2^{g_i}-1}{\log_2(i+1)}\qquad\qquad"
         r"\mathrm{nDCG@}k = \dfrac{\mathrm{DCG@}k}{\mathrm{IDCG@}k}")
body("**MRR** is the mean of 1/rank of the first relevant result. The **precision–recall curve** uses the standard "
     "11-point interpolated precision, averaged over queries:")
equation(r"p_{\mathrm{interp}}(r) = \max_{r' \,\geq\, r}\; p(r')\qquad r \in \{0.0,\,0.1,\,\ldots,\,1.0\}")
body("P@10 and nDCG@10 describe the first results page; Recall@100 describes how much of the relevant set a user "
     "could reach by browsing; MAP summarises the whole ranking.")
heading("4.3 Significance testing", 2)
body("Differences are tested with a paired, two-sided randomisation (sign-flip) test over per-query scores with "
     "20,000 resamples [11], at α = 0.05. Win/tie/loss counts are reported alongside p-values. With 25 topics, small "
     "differences cannot be resolved, and no correction for multiple comparisons is applied, so p-values close to 0.05 "
     "should be read with care.")
heading("4.4 Environment", 2)
E = res["environment"]
caption("tab", "Hardware and software.")
table(["Component", "Version / setting"], [
    ["Hardware", "Apple silicon (arm64) laptop; GPU via PyTorch MPS backend"],
    ["Operating system", E["platform"].split("-arm64")[0].replace("macOS-", "macOS ")],
    ["Python", E["python"]], ["PyTorch / sentence-transformers", f"{E['torch']} / {E['sentence_transformers']}"],
    ["scikit-learn / NumPy / pandas", f"{E['scikit_learn']} / {E['numpy']} / {E['pandas']}"],
    ["Other", "NLTK (Porter stemmer, stop words), Streamlit (UI), matplotlib (figures)"],
], [6.0, 9.4], align=["left", "left"])

# =============================================================================
# 5 Results
# =============================================================================
heading("5. Results", 1, page_break=True)
heading("5.1 Performance comparison", 2)
cols = [("P@10", "P@10"), ("Recall@100", "Recall@100"), ("AP", "MAP"), ("nDCG@10", "nDCG@10"), ("RR", "MRR")]
rows, bold = [], set()
for i, m in enumerate(MODELS):
    rows.append([m] + [f4(main[m][k]) for k, _ in cols])
    for j, (k, _) in enumerate(cols, 1):
        if abs(main[m][k] - max(main[x][k] for x in MODELS)) < 1e-9:
            bold.add((i, j))
caption("tab", "Performance comparison on the 25 topics (macro-average; best value per metric in bold).")
table(["Model"] + [lab for _, lab in cols], rows, [4.6, 2.1, 2.3, 2.1, 2.3, 2.1], bold_cells=bold)
figure(FIG / "metrics_bar_light.png", 15.2, "Metric comparison across the four models (best value per metric labelled).")
hs = main["Hybrid RRF"]
body(
    f"The two models that use the Sentence Transformer clearly outperform the lexical models. **Hybrid RRF achieves the "
    f"highest MAP ({f4(hs['AP'])}) and Recall@100 ({f4(hs['Recall@100'])})**; the **Sentence Transformer achieves the "
    f"highest P@10 ({f3(main['Sentence Transformer']['P@10'])}) and nDCG@10 ({f4(main['Sentence Transformer']['nDCG@10'])})**, "
    f"and its first result was relevant for every topic (MRR = 1.0). The MAP gap between the two is only "
    f"{abs(hs['AP'] - main['Sentence Transformer']['AP']):.4f}. Relative to BM25, the hybrid improves MAP by "
    f"{hs['AP'] - main['BM25']['AP']:+.3f} ({100 * (hs['AP'] / main['BM25']['AP'] - 1):.0f}%) and nDCG@10 by "
    f"{hs['nDCG@10'] - main['BM25']['nDCG@10']:+.3f}. BM25 is slightly ahead of TF-IDF on every metric, as expected "
    "from its term-frequency saturation and explicit length normalisation.")

heading("5.2 Precision–recall curve", 2)
figure(FIG / "pr_curve_light.png", 13.5, "11-point interpolated precision–recall curves, averaged over the 25 topics.")
p_at = lambda m, r: float(pr[(pr.system == m) & (pr.recall.round(1) == r)].precision.iloc[0])  # noqa: E731
body(
    "Both semantic runs dominate the lexical runs at every recall level. At recall 0.5 the Sentence Transformer and "
    f"Hybrid RRF keep precision {p_at('Sentence Transformer', 0.5):.2f} and {p_at('Hybrid RRF', 0.5):.2f}, against "
    f"{p_at('BM25', 0.5):.2f} for BM25 and {p_at('TF-IDF', 0.5):.2f} for TF-IDF. The hybrid curve is highest at high "
    f"recall (precision {p_at('Hybrid RRF', 1.0):.2f} at recall 1.0 versus {p_at('Sentence Transformer', 1.0):.2f}), "
    "because fusion lifts relevant documents that one model ranks low but the other finds. This is the same effect "
    "that gives the hybrid the best Recall@100.")

heading("5.3 Statistical significance", 2)
rows = []
for comp in ("Hybrid RRF vs TF-IDF", "Hybrid RRF vs BM25", "Hybrid RRF vs Sentence Transformer",
             "Sentence Transformer vs BM25"):
    r = [comp]
    for metric in ("AP", "nDCG@10"):
        p, d, w, t_, l_ = p_of(metric, comp)
        r += [f"{d:+.3f}", f"{w}/{t_}/{l_}", ("**" if p < 0.05 else "") + f"{p:.4f}" + ("**" if p < 0.05 else "")]
    rows.append(r)
caption("tab", "Paired randomisation tests (two-sided, 20,000 resamples). Δ = mean difference; W/T/L = per-query "
        "wins/ties/losses; p < 0.05 in bold.")
table(["Comparison", "Δ AP", "W/T/L", "p", "Δ nDCG@10", "W/T/L", "p"], rows,
      [5.4, 1.6, 1.7, 1.5, 1.9, 1.7, 1.5], size=9)
p1, p2 = p_of("AP", "Hybrid RRF vs TF-IDF")[0], p_of("AP", "Hybrid RRF vs BM25")[0]
_, _, w3, _, l3 = p_of("AP", "Hybrid RRF vs Sentence Transformer")
wl_phrase = (f"the hybrid wins and loses on {w3} topics each" if w3 == l3
             else f"the hybrid wins on {w3} topics and loses on {l3}")
p3 = p_of("AP", "Hybrid RRF vs Sentence Transformer")[0]
p4, p5 = p_of("AP", "Sentence Transformer vs BM25")[0], p_of("nDCG@10", "Sentence Transformer vs BM25")[0]
body(
    f"Hybrid RRF is significantly better than both lexical models on MAP (p {pv(p1)} vs TF-IDF, p {pv(p2)} vs BM25) "
    f"and on nDCG@10. The difference between Hybrid RRF and the Sentence Transformer is not significant "
    f"(MAP p {pv(p3)}; {wl_phrase}). The Sentence Transformer's advantage over BM25 "
    f"is significant on nDCG@10 (p {pv(p5)}) but only borderline on MAP (p {pv(p4)}), because BM25 wins clearly on "
    "several entity-centred topics (Section 5.5). Fusion turns two models with different strengths into one that is "
    "reliably better than the lexical baselines.")

heading("5.4 Keyword versus natural-language queries", 2)
rows = []
for m in MODELS:
    kw_, nl_ = style_ap("keyword", m), style_ap("natural", m)
    rows.append([m, f4(kw_), f4(nl_), f"{nl_ - kw_:+.3f}"])
bold = {(MODELS.index(max(MODELS, key=lambda m: style_ap("keyword", m))), 1),
        (MODELS.index(max(MODELS, key=lambda m: style_ap("natural", m))), 2)}
caption("tab", "MAP by query style (13 keyword, 12 natural-language topics).")
table(["Model", "Keyword MAP", "Natural-language MAP", "Change"], rows, [5.0, 3.3, 4.2, 2.6], bold_cells=bold)
figure(FIG / "style_bar_light.png", 13.8, "MAP by query style. Paraphrased queries expose the vocabulary-mismatch "
       "weakness of TF-IDF and BM25.")
body(
    f"Query style explains most of the gap. On keyword queries all four models are within "
    f"{max(style_ap('keyword', m) for m in MODELS) - min(style_ap('keyword', m) for m in MODELS):.3f} MAP of each other, "
    f"and **Hybrid RRF is best ({f3(style_ap('keyword', 'Hybrid RRF'))})**. On paraphrased natural-language queries BM25 "
    f"falls to {f3(style_ap('natural', 'BM25'))} and TF-IDF to {f3(style_ap('natural', 'TF-IDF'))}, while the "
    f"**Sentence Transformer stays at {f3(style_ap('natural', 'Sentence Transformer'))}**; the hybrid "
    f"({f3(style_ap('natural', 'Hybrid RRF'))}) recovers most of that advantage. Lexical models can only match words "
    "the query shares with the article, whereas the encoder relates “American currency” to “dollar” and “tricking "
    "people into revealing bank details” to “phishing”.")

heading("5.5 Per-query analysis", 2)
figure(FIG / "per_query_ap_light.png", 14.6, "Average precision per topic and model (darker = higher).")
gains = (ap["Sentence Transformer"] - ap["BM25"]).sort_values()
top_gain = gains.index[::-1][:4].tolist()
lex_win = gains.index[:4].tolist()
qtext = {q["qid"]: q["query"] for q in queries}
easy = [q for q in ap.index if (ap.loc[q] >= 0.99).all()]
hard = ap.mean(axis=1).sort_values().index[:3].tolist()
assert set(top_gain) == {"Q02", "Q25", "Q24", "Q11"}, top_gain
assert set(lex_win) == {"Q09", "Q06", "Q21", "Q19"}, lex_win
assert all((ap.loc[q, "Hybrid RRF"] - ap.loc[q, "Sentence Transformer"]) >=
           0.5 * (ap.loc[q, "BM25"] - ap.loc[q, "Sentence Transformer"]) for q in lex_win)
assert set(easy) == {"Q01", "Q03"} and set(hard) == {"Q09", "Q20", "Q07"}, (easy, hard)


def qref(q, cols=("BM25", "Sentence Transformer")):
    short = {"BM25": "BM25", "Sentence Transformer": "ST", "Hybrid RRF": "hybrid", "TF-IDF": "TF-IDF"}
    return f"{q} *{qtext[q]}* (" + ", ".join(f"{short[c]} {ap.loc[q, c]:.2f}" for c in cols) + ")"


para_gain = [qref(q) for q in top_gain if q != "Q11"]
body("**Largest semantic gains.** " + "; ".join(para_gain) + ". These are paraphrased queries whose key terms "
     "(dollar, phishing, file-sharing) never appear in the query. The fourth is a keyword query, " + qref("Q11") +
     ": BM25 matches the words *identity*, *card* and *bill* separately in articles about identity theft, credit "
     "cards and tax bills, while the encoder treats the phrase as one concept.")
body("**Where lexical matching wins.** " + "; ".join(qref(q, ("BM25", "Sentence Transformer", "Hybrid RRF"))
                                              for q in lex_win) +
     ". Each topic hinges on a specific name or event (Brit Awards, Bafta, iPod, London's 2012 bid), and the encoder "
     "drifts to neighbouring concepts such as other award ceremonies, other music players or other Olympic stories. "
     "In all four, the hybrid recovers at least half of the gap between the Sentence Transformer and BM25.")
body("**Easy and hard topics.** " + " and ".join(easy) + " are solved almost perfectly by every model (AP ≥ 0.99) "
     "because their vocabulary — Yukos, Deutsche Boerse — is distinctive. The hardest topics are " +
     ", ".join(f"{q} (*{qtext[q]}*, mean AP {ap.loc[q].mean():.2f})" for q in hard) +
     ". Q09 and Q20 are broad event topics where much of the relevant coverage (nomination stories, match previews, "
     "injury news) never uses the defining words. Q07 is a pure paraphrase with almost no word overlap with "
     "box-office reports.")

heading("5.6 Case study: a paraphrased query", 2)
cs = res["case_studies"]["Q07"]
mark = {2: "●", 1: "◐", 0: "○"}
rows = []
for i in range(5):
    rows.append([str(i + 1)] + [f"{mark[cs['runs'][m][i]['grade']]} {cs['runs'][m][i]['title']}" for m in MODELS])
caption("tab", f"Top-5 results for Q07 “{cs['query']}” (● highly relevant, ◐ partially relevant, ○ not relevant).")
table(["#", "TF-IDF", "BM25", "Sentence Transformer", "Hybrid RRF"], rows, [0.7, 3.7, 3.7, 3.7, 3.7],
      align=["center", "left", "left", "left", "left"], size=8.2)
rel5 = {m: sum(1 for d in cs["runs"][m] if d["grade"] > 0) for m in MODELS}
body(
    f"TF-IDF matches the surface words *film* and *cinema* and returns festival and digital-cinema stories "
    f"({rel5['TF-IDF']} of 5 relevant). BM25 does better ({rel5['BM25']}/5) but still returns a festival story and "
    f"a digital-cinema story that match the words rather than the need. The Sentence Transformer recognises that “took the most money at cinemas” means box-office "
    f"takings ({rel5['Sentence Transformer']}/5), and the hybrid keeps the best of both lists "
    f"({rel5['Hybrid RRF']}/5, with no non-relevant document in its top five).")

heading("5.7 Known-item headline search", 2)
kcols = [("MRR", "MRR (= MAP)"), ("Success@1", "Success@1"), ("Recall@10", "Recall@10"),
         ("Recall@100", "Recall@100"), ("nDCG@10", "nDCG@10")]
rows, bold = [], set()
for i, m in enumerate(MODELS):
    rows.append([m] + [f4(ki[m][k]) for k, _ in kcols])
    for j, (k, _) in enumerate(kcols, 1):
        if abs(ki[m][k] - max(ki[x][k] for x in MODELS)) < 1e-9:
            bold.add((i, j))
caption("tab", f"Known-item search: {res['known_item']['queries']:,} headline queries against the body-only index.")
table(["Model"] + [lab for _, lab in kcols], rows, [4.4, 2.5, 2.2, 2.2, 2.3, 2.1], bold_cells=bold, size=9.2)
figure(FIG / "known_item_bar_light.png", 15.2, "Known-item search results (best value per metric labelled).")
body(
    f"The ordering of the single models reverses here: headlines share many exact words with their own article, so "
    f"BM25 (MRR {f3(ki['BM25']['MRR'])}) beats the Sentence Transformer ({f3(ki['Sentence Transformer']['MRR'])}), "
    f"which tends to rank other articles about the same story just as highly. **Hybrid RRF is best on every metric** "
    f"(MRR {f3(ki['Hybrid RRF']['MRR'])}, Success@1 {f3(ki['Hybrid RRF']['Success@1'])}, Recall@10 "
    f"{f3(ki['Hybrid RRF']['Recall@10'])}); all its improvements are significant (p < 0.001 with 5,000 resamples over "
    f"{res['known_item']['queries']:,} queries). The same fused list is therefore the best option in both settings — "
    "it inherits BM25's precision on exact wording and the encoder's tolerance of paraphrase.")

heading("5.8 Ablation studies", 2)
order = ["Sentence Transformer", "ST FirstP (opening passage only)", "Hybrid RRF", "RRF BM25 + ST FirstP",
         "RRF TF-IDF + ST", "RRF TF-IDF + BM25 (lexical only)", "RRF TF-IDF + BM25 + ST"]
label = {"Sentence Transformer": "ST, MaxP (main)", "ST FirstP (opening passage only)": "ST, FirstP (opening passage)",
         "Hybrid RRF": "RRF BM25 + ST MaxP (main hybrid)", "RRF BM25 + ST FirstP": "RRF BM25 + ST FirstP",
         "RRF TF-IDF + ST": "RRF TF-IDF + ST MaxP", "RRF TF-IDF + BM25 (lexical only)": "RRF TF-IDF + BM25 (lexical only)",
         "RRF TF-IDF + BM25 + ST": "RRF TF-IDF + BM25 + ST (three-way)"}
rows = [[label[s]] + [f4(abl[s][k]) for k in ("P@10", "Recall@100", "AP", "nDCG@10")] for s in order]
caption("tab", "Ablations: passage pooling and fusion components (k = 60).")
table(["Configuration", "P@10", "Recall@100", "MAP", "nDCG@10"], rows, [6.6, 2.0, 2.4, 2.0, 2.3], size=9.2)
k_min, k_max = ktab.AP.min(), ktab.AP.max()
body(
    f"**Passage pooling.** Scoring the best passage (MaxP) beats scoring only the opening passage (FirstP) — MAP "
    f"{f3(abl['Sentence Transformer']['AP'])} versus {f3(abl['ST FirstP (opening passage only)']['AP'])} — because "
    "relevant information often appears after the first 160 words. **Fusion components.** Fusing the two lexical models "
    f"gives no gain over BM25 alone ({f3(abl['RRF TF-IDF + BM25 (lexical only)']['AP'])}), since they make the same "
    f"errors. Adding TF-IDF to the hybrid *lowers* MAP to {f3(abl['RRF TF-IDF + BM25 + ST']['AP'])}, because two highly "
    "correlated lexical lists then outvote the single semantic list. One complementary model per family is the better "
    f"design. FirstP-based fusion scored slightly higher ({f3(abl['RRF BM25 + ST FirstP']['AP'])}), but this "
    "difference was found on the test topics and is not adopted.")
rows = [[str(int(r.rrf_k)), f4(r["P@10"]), f4(r["Recall@100"]), f4(r.AP), f4(r["nDCG@10"])] for _, r in ktab.iterrows()]
caption("tab", "Sensitivity of the hybrid (BM25 + ST) to the RRF constant k.")
table(["k", "P@10", "Recall@100", "MAP", "nDCG@10"], rows, [2.0, 2.4, 2.8, 2.4, 2.6], size=9.2)
body(f"**RRF constant.** MAP varies only between {f3(k_min)} and {f3(k_max)} for k from 1 to 500, so the hybrid is "
     "insensitive to this parameter and the default k = 60 needs no tuning.")

heading("5.9 Efficiency", 2)
comp = L["components"]
caption("tab", f"Indexing time ({I['documents']:,} articles) and median query latency (25 topics × 10 runs).")
table(["Offline step", "Time", "Online (per query)", "Median latency"], [
    ["Preprocessing (all articles)", f"{T['Preprocessing']:.2f} s", "TF-IDF", f"{L['TF-IDF']:.2f} ms"],
    ["Inverted index", f"{T['Inverted index']:.2f} s", "BM25", f"{L['BM25']:.2f} ms"],
    ["TF-IDF document vectors", f"{T['TF-IDF document vectors']:.2f} s", "Sentence Transformer (encode + score)",
     f"{L['Sentence Transformer']:.2f} ms"],
    ["Passage embeddings (GPU)", f"{T['Passage embeddings']:.1f} s", "RRF fusion step only", f"{comp['RRF fusion step']:.2f} ms"],
    ["", "", "**Hybrid RRF (end to end)**", f"**{L['Hybrid RRF']:.2f} ms**"],
], [4.6, 1.9, 6.0, 2.9], align=["left", "center", "left", "center"], size=9.2)
body(f"Lexical retrieval is effectively instantaneous (BM25 needs {L['BM25'] * 1000:.0f} µs per query because it touches "
     f"only the posting lists of the query terms). The hybrid's cost is dominated by encoding the query with the "
     f"neural model; at about {L['Hybrid RRF']:.0f} ms per query it remains interactive. Indexing is dominated by the "
     "one-off passage embedding, which is cached on disk.")

# =============================================================================
# 6 Discussion
# =============================================================================
heading("6. Discussion", 1, page_break=True)
heading("6.1 Main findings", 2)
numbered([
    f"**Semantic matching is the largest single improvement.** The Sentence Transformer raises MAP from "
    f"{f3(main['BM25']['AP'])} (BM25) to {f3(main['Sentence Transformer']['AP'])}, almost entirely through paraphrased "
    "queries.",
    "**Fusion makes the system robust.** Hybrid RRF ties the best model on topical retrieval (best MAP and Recall@100) "
    "and is the best model on known-item search, where the dense model alone is weakest. No single model wins both tests.",
    "**Design details matter.** Max-passage scoring beats first-passage scoring, and fusing one lexical with one "
    "semantic ranker beats fusing all three, because the lexical models are highly correlated.",
    "**Hybrid search is cheap.** Fusion adds well under a millisecond to the encoder's cost; RRF needs no score "
    "calibration and is insensitive to k.",
])
heading("6.2 Why the hybrid helps", 2)
body(
    "Lexical and dense models fail differently. BM25 fails when the right article uses different words; the encoder "
    "fails when a precise name or phrase matters, retrieving neighbouring topics instead (other award ceremonies for "
    "*Brit Awards*, other stories about the same event for a headline). Because RRF rewards documents that rank well "
    "in either list, it acts as an OR over two kinds of evidence while rewarding agreement. Documents that both models "
    "rank highly rise to the top, and a document that only one model finds is still kept in the fused ranking, which "
    "is why the hybrid has the best Recall@100 and the best high-recall precision.")
heading("6.3 Recommendation", 2)
body("Hybrid RRF is the default model in the search interface: it is never significantly worse than the best model "
     "on either test and is significantly better than every alternative on known-item search. If queries are mostly "
     "natural-language questions, the Sentence Transformer alone is equally good. If no GPU or neural model is available, "
     "BM25 is the best lexical choice.")
heading("6.4 Limitations", 2)
bullets([
    "**Topic set size.** 25 topics are enough to detect large effects but not small ones, such as Hybrid versus ST.",
    "**Single assessor.** Each pair was judged once, mostly from the headline and lead paragraph; inter-assessor "
    "agreement was not measured.",
    "**Pool bias.** The pool comes from the evaluated systems. The keyword sweep and the 100% judged rate of every "
    "top 10 limit, but cannot remove, this bias for deep-recall metrics.",
    "**Known-item queries are headlines.** They share wording with the article and so favour lexical matching; real "
    "users' recollections would be vaguer.",
    "**No tuning.** BM25 parameters, passage length and the encoder were not tuned. A retrieval-specific encoder "
    "(for example multi-qa-MiniLM or E5) would probably strengthen the semantic side.",
    "**Corpus.** The archive is small (2,126 articles), English-only and from 2004–2005; exact search over all "
    "passages would need an approximate index at web scale.",
])
heading("6.5 Threats to validity", 2)
body(
    "The queries were written with knowledge of the collection's topics, which may make them easier than real user "
    "queries. The natural-language queries were deliberately paraphrased, so the proportion of such queries (12 of 25) "
    "influences the overall averages; results are therefore also reported per query style. All systems were evaluated "
    "with identical data, judgments and code, and the hybrid configuration was fixed before evaluation, so the "
    "comparison between models is fair even if absolute values would differ on another query set.")

# =============================================================================
# 7 Conclusion
# =============================================================================
heading("7. Conclusion and Future Work", 1, page_break=True)
body(
    "This project delivered a complete retrieval system for the BBC News archive: a cleaned and de-duplicated corpus, "
    "a preprocessing pipeline, an inverted index with TF-IDF and BM25 ranking, Sentence-Transformer passage retrieval, "
    "Reciprocal Rank Fusion, a Streamlit search interface and a reusable evaluation framework with graded relevance "
    "judgments. On 25 assessed topics, Hybrid RRF achieved the best MAP "
    f"({f3(main['Hybrid RRF']['AP'])}) and Recall@100 ({f3(main['Hybrid RRF']['Recall@100'])}) and the Sentence "
    f"Transformer the best nDCG@10 ({f3(main['Sentence Transformer']['nDCG@10'])}). Both were significantly better than "
    f"TF-IDF and BM25. On {res['known_item']['queries']:,} known-item queries, Hybrid RRF was best outright. The main "
    "lesson is that lexical and semantic evidence are complementary, and rank fusion combines them cheaply and "
    "reliably.")
heading("7.1 Future work", 2)
numbered([
    "Re-rank the fused top 100 with a cross-encoder (e.g. ms-marco-MiniLM) to raise early precision further.",
    "Replace the general-purpose encoder with a retrieval-trained one (multi-qa-MiniLM, E5, BGE) and test learned sparse "
    "retrieval (SPLADE).",
    "Add pseudo-relevance feedback (RM3) to the lexical branch to reduce vocabulary mismatch.",
    "Tune BM25 and fusion weights on a separate development query set, not on the test topics.",
    "Extend the topic set and add a second assessor to measure agreement (Cohen's κ).",
    "Use an approximate nearest-neighbour index (FAISS/HNSW) and date facets for larger, live archives.",
])

# =============================================================================
# References
# =============================================================================
heading("References", 1, page_break=True)
refs = [
    "D. Greene and P. Cunningham, “Practical solutions to the problem of diagonal dominance in kernel document "
    "clustering,” in *Proc. 23rd Int. Conf. on Machine Learning (ICML)*, 2006, pp. 377–384.",
    "C. D. Manning, P. Raghavan and H. Schütze, *Introduction to Information Retrieval*. Cambridge University Press, 2008.",
    "S. Robertson and H. Zaragoza, “The probabilistic relevance framework: BM25 and beyond,” *Foundations and Trends in "
    "Information Retrieval*, vol. 3, no. 4, pp. 333–389, 2009.",
    "G. Salton and C. Buckley, “Term-weighting approaches in automatic text retrieval,” *Information Processing & "
    "Management*, vol. 24, no. 5, pp. 513–523, 1988.",
    "M. F. Porter, “An algorithm for suffix stripping,” *Program*, vol. 14, no. 3, pp. 130–137, 1980.",
    "N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings using Siamese BERT-networks,” in *Proc. EMNLP-IJCNLP*, "
    "2019, pp. 3982–3992.",
    "W. Wang, F. Wei, L. Dong, H. Bao, N. Yang and M. Zhou, “MiniLM: Deep self-attention distillation for task-agnostic "
    "compression of pre-trained transformers,” in *Advances in NeurIPS*, 2020.",
    "Z. Dai and J. Callan, “Deeper text understanding for IR with contextual neural language modeling,” in *Proc. ACM "
    "SIGIR*, 2019, pp. 985–988.",
    "G. V. Cormack, C. L. A. Clarke and S. Büttcher, “Reciprocal rank fusion outperforms Condorcet and individual rank "
    "learning methods,” in *Proc. ACM SIGIR*, 2009, pp. 758–759.",
    "K. Järvelin and J. Kekäläinen, “Cumulated gain-based evaluation of IR techniques,” *ACM Trans. on Information "
    "Systems*, vol. 20, no. 4, pp. 422–446, 2002.",
    "M. D. Smucker, J. Allan and B. Carterette, “A comparison of statistical significance tests for information "
    "retrieval evaluation,” in *Proc. ACM CIKM*, 2007, pp. 623–632.",
    "E. M. Voorhees and D. K. Harman (eds.), *TREC: Experiment and Evaluation in Information Retrieval*. MIT Press, 2005.",
    "Sentence-Transformers, “all-MiniLM-L6-v2” model card, Hugging Face, huggingface.co/sentence-transformers/all-MiniLM-L6-v2.",
    "Information Retrieval course, *Project Assignments — Project 12: BBC News Retrieval* (course brief).",
]
for i, r in enumerate(refs, 1):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.9)
    p.paragraph_format.first_line_indent = Cm(-0.9)
    p.paragraph_format.space_after = Pt(5)
    set_run_font(p.add_run(f"[{i}]\t"), 10, True, color=BLUE)
    add_rich(p, r, size=10)

# =============================================================================
# Appendices
# =============================================================================
heading("Appendix A. Topics and relevance counts", 1, page_break=True)
rows = [[q["qid"], q["query"], q["category"].replace("tech", "technology"), q["style"],
         str(sum(g == 2 for g in qrels[q["qid"]].values())), str(sum(g == 1 for g in qrels[q["qid"]].values()))]
        for q in queries]
caption("tab", "All 25 topics with the number of highly (H) and partially (P) relevant documents.")
table(["ID", "Query", "Category", "Style", "H", "P"], rows, [1.2, 8.4, 2.4, 2.1, 0.9, 0.9],
      align=["left", "left", "left", "left", "center", "center"], size=8.8)
body("Each topic also has a narrative that defines relevance (stored in `data/queries.json`); for example, Q02: "
     f"*“{queries[1]['narrative']}”*", size=9.5)

heading("Appendix B. Per-query results", 1, page_break=True)
rows = []
for qid in ap.index:
    rows.append([qid] + [f"{ap.loc[qid, m]:.3f}" for m in MODELS] + [f"{nd.loc[qid, m]:.3f}" for m in MODELS])
rows.append(["**Mean**"] + [f"**{ap[m].mean():.3f}**" for m in MODELS] + [f"**{nd[m].mean():.3f}**" for m in MODELS])
caption("tab", "Average precision and nDCG@10 per topic (T = TF-IDF, B = BM25, S = Sentence Transformer, H = Hybrid RRF).")
table(["Topic", "AP T", "AP B", "AP S", "AP H", "nDCG T", "nDCG B", "nDCG S", "nDCG H"], rows,
      [1.6] + [1.75] * 8, size=8.6)

heading("Appendix C. Reproducing the results", 1, page_break=True)
body("All results in this report are regenerated from the raw dataset by the following commands, run from the "
     "project folder:", keep_next=True)
cmds = [
    "python -m venv .venv && source .venv/bin/activate",
    "pip install -r requirements.txt",
    "curl -LO http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip && unzip bbc-fulltext.zip -d data/raw",
    "python scripts/prepare_data.py data/raw/bbc      # corpus -> data/bbc_articles.csv",
    "python scripts/build_pool.py 20                  # depth-20 pool for judging",
    "python scripts/write_qrels.py                    # graded judgments -> data/qrels.json",
    "python scripts/evaluate.py                       # all experiments -> outputs/",
    "python scripts/make_figures.py                   # charts",
    "streamlit run app.py                             # search interface",
]
t = doc.add_table(rows=1, cols=1)
borders(t, inside=False, outer=False)
cell_margins(t, 100, 100, 160, 160)
c = t.rows[0].cells[0]
c.width = Cm(16.2)
shade(c, "F4F6F9")
c.paragraphs[0].paragraph_format.space_after = Pt(0)
for i, cmd in enumerate(cmds):
    p = c.paragraphs[0] if i == 0 else c.add_paragraph()
    p.paragraph_format.space_after = Pt(1)
    set_run_font(p.add_run(cmd), 8.6, color="24476B", name=MONO)
para("", space_after=4)
caption("tab", "Output files.")
table(["File", "Content"], [
    ["`outputs/metrics_summary.csv`", "Performance comparison table (Table 11)"],
    ["`outputs/precision_recall_11pt.csv`", "Interpolated precision–recall points"],
    ["`outputs/per_query_metrics_all_runs.csv`", "Every metric for every topic and run, including ablations"],
    ["`outputs/significance_tests.csv`", "Randomisation-test results"],
    ["`outputs/known_item_summary.csv`", "Known-item results"],
    ["`outputs/ablation_components.csv`, `ablation_rrf_k.csv`", "Ablation tables"],
    ["`outputs/results.json`", "All numbers used in this report"],
    ["`outputs/figures/`", "All charts and interface screenshots"],
], [7.6, 7.8], align=["left", "left"], size=9)
body("Embeddings are cached under `outputs/cache/`; delete that folder to re-measure the encoding time. Results are "
     "deterministic for a given corpus, model version and hardware; GPU floating-point differences can change the "
     "order of near-tied documents.", size=9.5)

out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
doc.save(out)
print("saved", out, "| figures:", counters["fig"], "tables:", counters["tab"], "equations:", counters["eq"])
