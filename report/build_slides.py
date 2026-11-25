"""Build the presentation (PPTX) from outputs/results.json and the dark-theme figures.

Usage:  python report/build_slides.py [output.pptx]
"""
import json
import sys
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = OUT / "figures"
DEFAULT_PATH = ROOT.parent / "BBC_News_Retrieval_Presentation.pptx"

BG, CARD, CARD2, LINE = "111C30", "192941", "1F314B", "2B3D5C"
TEXT, MUTED, DIM = "F8FAFD", "B2C0D3", "8D9BB0"
TEAL, AMBER, GREEN, BLUE = "3CBECD", "F5B84B", "4BB48C", "366FCD"
MODEL_COLOR = {"TF-IDF": "3987E5", "BM25": "D95926", "Sentence Transformer": "199E70", "Hybrid RRF": "C98500"}
MODELS = list(MODEL_COLOR)
FONT = "Aptos"

res = json.loads((OUT / "results.json").read_text())
queries = json.loads((ROOT / "data" / "queries.json").read_text())
main = {r["system"]: r for r in res["main"]}
ki = {r["system"]: r for r in res["known_item"]["table"]}
abl = {r["system"]: r for r in res["ablation_components"]}
ktab = pd.DataFrame(res["ablation_rrf_k"])
sig = pd.DataFrame(res["significance"])
style_tbl = pd.DataFrame(res["by_style"])
C, I, Q, D = res["corpus"], res["index_stats"], res["qrels"], res["dense"]
T, L = res["timings_seconds"], res["latency_ms_median"]


def style_ap(style, system):
    return float(style_tbl[(style_tbl["style"] == style) & (style_tbl.system == system)]["AP"].iloc[0])


def p_of(metric, comp):
    return float(sig[(sig.metric == metric) & (sig.comparison == comp)].p_value.iloc[0])


assert max(main, key=lambda m: main[m]["AP"]) == "Hybrid RRF"
assert max(main, key=lambda m: main[m]["nDCG@10"]) == "Sentence Transformer"
assert max(ki, key=lambda m: ki[m]["MRR"]) == "Hybrid RRF"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
page = {"n": 0}


def rgb(h):
    return RGBColor.from_string(h)


def text(slide, x, y, w, h, s, size=12, bold=False, color=TEXT, align="left", anchor="top", italic=False,
         font=FONT, line_spacing=None):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0)
    tf.margin_top = tf.margin_bottom = Inches(0)
    tf.vertical_anchor = {"top": MSO_ANCHOR.TOP, "middle": MSO_ANCHOR.MIDDLE, "bottom": MSO_ANCHOR.BOTTOM}[anchor]
    lines = s if isinstance(s, list) else [s]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}[align]
        if line_spacing:
            p.line_spacing = line_spacing
        rich(p, ln, size, bold, color, italic, font)
    return tb


def rich(p, s, size, bold, color, italic=False, font=FONT):
    """**bold** segments inside a line."""
    parts = s.split("**")
    for k, part in enumerate(parts):
        if not part:
            continue
        r = p.add_run()
        r.text = part
        f = r.font
        f.name, f.size, f.bold, f.italic = font, Pt(size), bold or (k % 2 == 1), italic
        f.color.rgb = rgb(TEXT if (k % 2 == 1 and color in (MUTED, DIM)) else color)


def bullets(slide, x, y, w, h, items, size=14, color=TEXT, bullet_color=TEAL, gap=8):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(0)
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        r = p.add_run()
        r.text = "●  "
        r.font.size, r.font.color.rgb, r.font.name = Pt(size - 5), rgb(bullet_color), FONT
        rich(p, it, size, False, color)
    return tb


def rect(slide, x, y, w, h, fill=CARD, radius=0.06, line=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
                                 Inches(x), Inches(y), Inches(w), Inches(h))
    if radius:
        shp.adjustments[0] = radius
    shp.fill.solid()
    shp.fill.fore_color.rgb = rgb(fill)
    if line:
        shp.line.color.rgb = rgb(line)
        shp.line.width = Pt(0.75)
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def bar(slide, x, y, w, h, color):
    return rect(slide, x, y, w, h, color, radius=0)


def new_slide(label=None, title=None, subtitle=None, notes=None):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(BG)
    page["n"] += 1
    if label:
        text(s, 0.65, 0.34, 9, 0.25, label, 10, True, TEAL)
    if title:
        text(s, 0.65, 0.62, 12.0, 0.6, title, 26, True, TEXT)
    if subtitle:
        text(s, 0.65, 1.28, 12.0, 0.35, subtitle, 12.5, False, MUTED)
    text(s, 12.25, 7.03, 0.5, 0.2, f"{page['n']:02d}", 8, True, DIM, align="right")
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def cell_style(cell, fill, txt, size, bold=False, color=TEXT, align="center"):
    cell.fill.solid()
    cell.fill.fore_color.rgb = rgb(fill)
    cell.margin_left = cell.margin_right = Inches(0.08)
    cell.margin_top = cell.margin_bottom = Inches(0.035)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf = cell.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}[align]
    r = p.add_run()
    r.text = txt
    r.font.name, r.font.size, r.font.bold = FONT, Pt(size), bold
    r.font.color.rgb = rgb(color)
    tcpr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnB", "a:lnT", "a:lnR", "a:lnL"):   # schema order: lnL, lnR, lnT, lnB, then the fill
        ln = etree.Element(qn(tag), w="6350")
        etree.SubElement(ln, qn("a:noFill"))
        tcpr.insert(0, ln)


def table(slide, x, y, w, rows, col_w, size=12, row_h=0.36, bold=(), first_align="left", head_size=None,
          dot_col0=False):
    nr, nc = len(rows), len(rows[0])
    shp = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    tbl = shp.table
    tblpr = shp._element.graphic.graphicData.tbl.tblPr
    tblpr.set("firstRow", "0")
    tblpr.set("bandRow", "0")
    style_id = tblpr.find(qn("a:tableStyleId"))
    if style_id is not None:
        style_id.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"  # "No Style, No Grid"
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i in range(nr):
        tbl.rows[i].height = Inches(row_h)
        for j in range(nc):
            head = i == 0
            fill = "22365A" if head else (CARD if i % 2 else "1C2E49")
            val = rows[i][j]
            is_bold = head or (i, j) in bold
            color = TEXT if (head or is_bold or j == 0) else "D6DEE9"
            cell_style(tbl.cell(i, j), fill, val, head_size or size if head else size, is_bold, color,
                       first_align if j == 0 else "center")
    if dot_col0:
        for i in range(1, nr):
            name = rows[i][0]
            if name in MODEL_COLOR:
                d = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x - 0.2), Inches(y + row_h * i + row_h / 2 - 0.055),
                                           Inches(0.11), Inches(0.11))
                d.fill.solid()
                d.fill.fore_color.rgb = rgb(MODEL_COLOR[name])
                d.line.fill.background()
                d.shadow.inherit = False
    return shp


def picture(slide, path, x, y, w=None, h=None):
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)


def arrow(slide, x0, y0, x1, y1, color=DIM):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x0), Inches(y0), Inches(x1), Inches(y1))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(1.5)
    ln = c.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    tail.set("w", "med")
    tail.set("len", "med")
    return c


def stat(slide, x, y, w, value, label, accent, h=1.05, vsize=30):
    rect(slide, x, y, w, h, CARD, 0.08)
    bar(slide, x, y + 0.14, 0.06, h - 0.28, accent)
    text(slide, x + 0.25, y + 0.12, w - 0.35, 0.5, value, vsize, True, TEXT)
    text(slide, x + 0.25, y + h - 0.38, w - 0.35, 0.3, label, 10.5, True, MUTED)


def f3(x):
    return f"{x:.3f}"


# ============================================================================
# 1 Title
# ============================================================================
s = new_slide(notes="BBC News Retrieval: four ranking models on the official BBC News corpus, evaluated with a "
                    "purpose-built test collection and a large automatic known-item test.")
text(s, 0.8, 0.8, 8.0, 0.3, "INFORMATION RETRIEVAL  |  PROJECT 12", 12, True, TEAL)
text(s, 0.8, 1.45, 8.0, 1.5, ["BBC News", "Retrieval System"], 40, True, TEXT, line_spacing=0.95)
text(s, 0.85, 3.25, 7.4, 0.8, "Lexical, semantic and hybrid ranking on the BBC News corpus", 18, False, MUTED)
rect(s, 0.85, 4.55, 7.3, 0.78, CARD2, 0.12)
text(s, 1.1, 4.79, 7.0, 0.3, "TF-IDF   •   BM25   •   Sentence Transformer   •   Reciprocal Rank Fusion", 13, True, TEXT)
text(s, 0.85, 6.55, 8.5, 0.25, "Prepared by: [Name]   |   Registration: [Number]   |   Course: Information Retrieval",
     10, False, MUTED)
stat(s, 9.1, 1.25, 3.4, f"{C['unique_articles']:,}", "UNIQUE BBC ARTICLES", TEAL)
stat(s, 9.1, 2.5, 3.4, f"{Q['queries']} · {Q['judged_pairs']}", "TOPICS · GRADED JUDGMENTS", AMBER)
stat(s, 9.1, 3.75, 3.4, f"{res['known_item']['queries']:,}", "KNOWN-ITEM QUERIES", GREEN)
stat(s, 9.1, 5.0, 3.4, f3(main["Hybrid RRF"]["AP"]), "BEST MAP · HYBRID RRF", BLUE)

# ============================================================================
# 2 Problem
# ============================================================================
s = new_slide("01 • PROBLEM", "Why news retrieval is hard",
              "A search system must put the useful articles first, not just expose a large archive.",
              notes="Three challenges: volume, vocabulary mismatch between query and article, and ranking quality.")
cards = [("Information overload", f"{C['unique_articles']:,} articles across five sections; users read only the first "
                                  "few results.", TEAL),
         ("Vocabulary mismatch", "“why the American currency is losing value” never says dollar — but the "
                                 "relevant articles do.", AMBER),
         ("Exact names still matter", "“Brit Awards”, “iPod”, “Yukos”: a semantic model can drift to "
                                      "neighbouring topics.", GREEN)]
for i, (h, b, a) in enumerate(cards):
    x = 0.65 + i * 4.1
    rect(s, x, 1.95, 3.85, 2.75, CARD, 0.06)
    bar(s, x + 0.3, 2.25, 0.6, 0.06, a)
    text(s, x + 0.3, 2.5, 3.3, 0.5, h, 18, True, TEXT)
    text(s, x + 0.3, 3.15, 3.3, 1.4, b, 13.5, False, MUTED)
rect(s, 0.65, 5.15, 12.05, 1.15, CARD2, 0.08)
text(s, 1.0, 5.43, 11.4, 0.6, "Aim: build TF-IDF, BM25, Sentence-Transformer and hybrid rankers over one index, "
                              "and measure which gives the best ranking — and why.", 17, True, TEXT)

# ============================================================================
# 3 Objectives & deliverables
# ============================================================================
s = new_slide("02 • OBJECTIVES", "Tasks and deliverables",
              notes="The four standardised tasks from the brief, and the three expected outputs.")
rect(s, 0.65, 1.55, 7.2, 5.05, CARD, 0.04)
text(s, 0.95, 1.8, 6.6, 0.4, "Standardised tasks", 16, True, TEAL)
bullets(s, 0.95, 2.35, 6.7, 4.2, [
    "**Text preprocessing:** cleaning, tokenisation, stop-word removal, Porter stemming",
    "**Inverted index & document vectors:** postings lists, sparse TF-IDF vectors, dense passage embeddings",
    "**Ranking algorithms:** TF-IDF (cosine), Okapi BM25, Sentence Transformer (all-MiniLM-L6-v2)",
    "**Hybrid model:** Reciprocal Rank Fusion of BM25 and the Sentence Transformer",
    "**Evaluation:** 25 graded topics + 2,066 known-item queries; significance tests",
], 13.5, gap=10)
rect(s, 8.15, 1.55, 4.55, 5.05, CARD, 0.04)
text(s, 8.45, 1.8, 4.0, 0.4, "Expected outputs", 16, True, AMBER)
items = [("Search interface", "Streamlit app, 4 models"), ("Comparison table", "P@10 · Recall · MAP · nDCG"),
         ("Precision–recall graph", "11-point interpolated"), ("Metrics", "MAP, Recall@100, nDCG@10, P@10, MRR")]
for i, (h, b) in enumerate(items):
    y = 2.4 + i * 1.0
    o = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(8.45), Inches(y + 0.05), Inches(0.36), Inches(0.36))
    o.fill.solid()
    o.fill.fore_color.rgb = rgb(GREEN)
    o.line.fill.background()
    o.shadow.inherit = False
    tf = o.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    rr = tf.paragraphs[0].add_run()
    rr.text = "✓"
    rr.font.size, rr.font.bold, rr.font.name = Pt(13), True, FONT
    rr.font.color.rgb = rgb(BG)
    text(s, 9.0, y, 3.6, 0.35, h, 14, True, TEXT)
    text(s, 9.0, y + 0.38, 3.6, 0.3, b, 11.5, False, MUTED)

# ============================================================================
# 4 Dataset
# ============================================================================
s = new_slide("03 • DATASET", "The BBC News corpus",
              "Greene & Cunningham (ICML 2006): BBC News website articles, 2004–2005, five topical sections.",
              notes="Official dataset. UTF-8 decoding fixes the pound sign; 99 exact duplicates removed.")
rows = [["Category", "Raw", "Unique"]]
for cat, lab in (("business", "Business"), ("entertainment", "Entertainment"), ("politics", "Politics"),
                 ("sport", "Sport"), ("tech", "Technology")):
    rows.append([lab, f"{C['raw_per_category'][cat]:,}", f"{C['unique_per_category'][cat]:,}"])
rows.append(["Total", f"{C['raw_articles']:,}", f"{C['unique_articles']:,}"])
table(s, 0.65, 1.95, 5.6, rows, [2.8, 1.4, 1.4], size=13, row_h=0.46, bold={(6, 0), (6, 1), (6, 2)})
stat(s, 6.75, 1.95, 2.85, f"{C['exact_duplicates_removed']}", "EXACT DUPLICATES REMOVED", AMBER, h=1.25, vsize=32)
stat(s, 9.85, 1.95, 2.85, f"{C['body_words_mean']:.0f}", "WORDS PER ARTICLE (MEAN)", TEAL, h=1.25, vsize=32)
stat(s, 6.75, 3.45, 2.85, f"{I['vocabulary_terms']:,}", "INDEX TERMS", GREEN, h=1.25, vsize=32)
stat(s, 9.85, 3.45, 2.85, f"{D['passages']:,}", "EMBEDDED PASSAGES", BLUE, h=1.25, vsize=32)
rect(s, 6.75, 4.95, 5.95, 1.35, CARD2, 0.06)
text(s, 7.05, 5.12, 5.4, 1.1, ["**Cleaning:** UTF-8 decoding (Latin-1 turns £ into “Â£”), HTML entities "
                               "unescaped, headline/body split, IDs such as tec-040."], 12, False, MUTED)

# ============================================================================
# 5 Pipeline
# ============================================================================
s = new_slide("04 • ARCHITECTURE", "From raw text to ranked results",
              "One preprocessing pipeline feeds a lexical branch and a semantic branch; RRF merges their rankings.",
              notes="Offline: preprocess once, build index, TF-IDF matrix and passage embeddings. Online: same "
                    "preprocessing or encoder for the query.")


def node(x, y, w, h, title, sub, accent=None):
    rect(s, x, y, w, h, CARD, 0.08)
    if accent:
        bar(s, x + 0.15, y + 0.12, w - 0.3, 0.05, accent)
    text(s, x + 0.1, y + 0.28, w - 0.2, 0.35, title, 14, True, TEXT, align="center")
    text(s, x + 0.1, y + 0.68, w - 0.2, 0.6, sub, 10.5, False, MUTED, align="center")


node(0.65, 3.35, 2.0, 1.3, "BBC corpus", f"{C['unique_articles']:,} articles")
node(3.05, 3.35, 2.2, 1.3, "Preprocessing", "clean · tokenise\nstop-words · stem")
node(5.75, 1.85, 2.35, 1.3, "Inverted index", f"{I['vocabulary_terms']:,} terms", MODEL_COLOR["BM25"])
node(5.75, 3.35, 2.35, 1.3, "TF-IDF vectors", "sparse, L2-normalised", MODEL_COLOR["TF-IDF"])
node(5.75, 4.85, 2.35, 1.3, "Passage embeddings", f"MiniLM · {D['passages']:,} × {D['dim']}",
     MODEL_COLOR["Sentence Transformer"])
node(8.55, 1.85, 1.85, 1.3, "BM25", "k₁ 1.2 · b 0.75", MODEL_COLOR["BM25"])
node(8.55, 3.35, 1.85, 1.3, "Cosine", "TF-IDF ranking", MODEL_COLOR["TF-IDF"])
node(8.55, 4.85, 1.85, 1.3, "MaxP cosine", "dense ranking", MODEL_COLOR["Sentence Transformer"])
node(10.85, 3.35, 1.85, 1.3, "Hybrid RRF", "BM25 + ST\nk = 60", MODEL_COLOR["Hybrid RRF"])
arrow(s, 2.65, 4.0, 3.05, 4.0)
arrow(s, 5.25, 4.0, 5.75, 2.5)
arrow(s, 5.25, 4.0, 5.75, 4.0)
arrow(s, 1.65, 4.65, 5.75, 5.5)
text(s, 2.2, 5.25, 2.6, 0.3, "raw text (no stemming)", 10, False, DIM)
arrow(s, 8.1, 2.5, 8.55, 2.5)
arrow(s, 8.1, 4.0, 8.55, 4.0)
arrow(s, 8.1, 5.5, 8.55, 5.5)
arrow(s, 10.4, 2.5, 10.85, 3.65)
arrow(s, 10.4, 5.5, 10.85, 4.35)

# ============================================================================
# 6 Preprocessing & index
# ============================================================================
from_prep = res["preprocessing_example"]["original"].split(". ", 1)[1].split("\n")[0]
sys.path.insert(0, str(ROOT))
from bbcir.preprocess import explain  # noqa: E402
pe = explain(from_prep)
s = new_slide("05 • PREPROCESSING & INDEXING", "Preprocessing and the inverted index",
              notes="Same pipeline for documents and queries; the index stores doc ids and term frequencies.")
rect(s, 0.65, 1.55, 6.6, 5.1, CARD, 0.04)
text(s, 0.95, 1.75, 6.0, 0.35, "Pipeline on a real sentence (tec-040)", 15, True, TEAL)
text(s, 0.95, 2.2, 6.0, 0.7, f"“{from_prep}”", 12, False, MUTED, italic=True)
steps = [("Clean + tokenise", f"{len(pe['tokens'])} tokens"),
         ("Remove stop words", f"{len(pe['without_stopwords'])} tokens: " + " ".join(pe["without_stopwords"][:6]) + " …"),
         ("Porter stem", " ".join(pe["stemmed"]))]
for i, (h, b) in enumerate(steps):
    y = 3.2 + i * 1.08
    rect(s, 0.95, y, 6.0, 0.92, CARD2, 0.1)
    text(s, 1.15, y + 0.12, 2.0, 0.3, h, 12.5, True, TEXT)
    text(s, 1.15, y + 0.45, 5.6, 0.45, b, 11, False, "D6DEE9", font="Consolas")
rect(s, 7.55, 1.55, 5.15, 5.1, CARD, 0.04)
text(s, 7.85, 1.75, 4.6, 0.35, "Inverted index", 15, True, AMBER)
top = res["top_df_terms"][0]
rows = [["Statistic", "Value"], ["Terms", f"{I['vocabulary_terms']:,}"], ["Postings", f"{I['total_postings']:,}"],
        ["Avg. postings / term", f"{I['avg_postings_length']:.1f}"], ["Build time", f"{T['Inverted index']:.2f} s"]]
table(s, 7.85, 2.25, 4.55, rows, [2.55, 2.0], size=12, row_h=0.4)
yk = res["example_postings"]["Yukos"]
text(s, 7.85, 4.45, 4.6, 0.3, f"Posting list for “{yk['stem']}” (df = {yk['df']}):", 12, True, TEXT)
text(s, 7.85, 4.85, 4.6, 1.2, "  ".join(f"{d}:{tf}" for d, tf in yk["top_postings"][:5]) + "  …", 11.5, False,
     "D6DEE9", font="Consolas")
text(s, 7.85, 5.6, 4.6, 0.9, f"Rare terms get high IDF (Kenteris: df {res['example_postings']['Kenteris']['df']}); "
                             f"common ones do not (election: df {res['example_postings']['election']['df']}).",
     11.5, False, MUTED)

# ============================================================================
# 7 Ranking models
# ============================================================================
s = new_slide("06 • RANKING MODELS", "Four ranking approaches",
              notes="TF-IDF and BM25 match words; the Sentence Transformer matches meaning; RRF fuses ranks.")
PROS = {"TF-IDF": ("Fast, transparent", "No synonyms or paraphrase"),
        "BM25": ("Strong lexical baseline", "Needs shared words"),
        "Sentence Transformer": ("Handles paraphrase", "Blurs exact names"),
        "Hybrid RRF": ("Robust on both query types", "Needs the encoder at query time")}
cards = [("TF-IDF", "Vector-space model", "Sublinear TF × smoothed IDF, L2-normalised; cosine similarity between "
                                          "query and document vectors."),
         ("BM25", "Probabilistic model", "Saturating term frequency with document-length normalisation "
                                         "(k₁ = 1.2, b = 0.75), scored from the inverted index."),
         ("Sentence Transformer", "Dense bi-encoder", f"all-MiniLM-L6-v2, {D['dim']}-d embeddings of 160-word passages; "
                                                      "document = best passage (MaxP)."),
         ("Hybrid RRF", "Rank fusion", "Reciprocal Rank Fusion of BM25 + Sentence Transformer, k = 60; fixed "
                                       "before evaluation.")]
for i, (h, k, b) in enumerate(cards):
    x = 0.65 + i * 3.08
    rect(s, x, 1.65, 2.88, 4.6, CARD, 0.05)
    bar(s, x + 0.25, 1.95, 0.55, 0.07, MODEL_COLOR[h])
    text(s, x + 0.25, 2.2, 2.45, 0.8, h, 18, True, TEXT)
    text(s, x + 0.25, 3.05, 2.45, 0.3, k.upper(), 10, True, MUTED)
    text(s, x + 0.25, 3.5, 2.45, 1.6, b, 12.5, False, "D6DEE9")
    plus, minus = PROS[h]
    rect(s, x + 0.2, 5.05, 2.48, 1.0, CARD2, 0.1)
    text(s, x + 0.35, 5.15, 2.25, 0.35, "+  " + plus, 11.5, True, "7FD6A9")
    text(s, x + 0.35, 5.55, 2.25, 0.35, "–  " + minus, 11.5, True, "F2A28B")
rect(s, 0.65, 6.4, 12.05, 0.5, CARD2, 0.2)
text(s, 0.9, 6.5, 11.6, 0.3, "Lexical models need shared words; the dense model needs no shared words but can blur "
                             "exact names. RRF combines both without calibrating their scores.", 12, False, MUTED)

# ============================================================================
# 8 RRF
# ============================================================================
s = new_slide("07 • HYBRID RANKING", "Reciprocal Rank Fusion",
              notes="Scores are on different scales, so fuse rank positions. Agreement beats a single first place.")
rect(s, 0.65, 1.6, 6.0, 1.45, CARD, 0.08)
text(s, 0.95, 1.9, 5.5, 0.7, "RRF(d) = Σₘ 1 / (k + rankₘ(d))", 26, True, TEXT)
text(s, 0.95, 2.6, 5.5, 0.3, "m ∈ {BM25, Sentence Transformer},  k = 60", 12, False, MUTED)
bullets(s, 0.75, 3.45, 5.9, 3.0, [
    "Uses **rank positions only** — no score calibration between BM25 and cosine.",
    "Documents ranked well by **both** models rise; documents found by only one are kept.",
    "Components and k were **fixed before evaluation**; alternatives are ablations.",
], 13.5, gap=12)
text(s, 7.1, 1.6, 5.5, 0.35, "Worked example (k = 60)", 15, True, AMBER)
rows = [["Doc", "BM25 rank", "ST rank", "RRF score", "Fused"], ["A", "1", "9", "0.03088", "2"],
        ["B", "3", "2", "0.03200", "1"], ["C", "2", "—", "0.01613", "3"]]
table(s, 7.1, 2.1, 5.6, rows, [0.9, 1.2, 1.1, 1.4, 1.0], size=13, row_h=0.5, bold={(2, 3), (2, 4)})
rect(s, 7.1, 4.4, 5.6, 1.5, CARD2, 0.08)
text(s, 7.4, 4.62, 5.0, 1.1, "B is never first in either list, but its consistent placement beats A's single "
                             "first place: 1/63 + 1/62 > 1/61 + 1/69.", 13, False, MUTED)

# ============================================================================
# 9 Evaluation design
# ============================================================================
s = new_slide("08 • EVALUATION DESIGN", "Building a test collection",
              "BBC News has no official queries, so relevance judgments were created TREC-style.",
              notes="25 topics with narratives, blind depth-20 pooling plus keyword sweep; graded 0/1/2. "
                    "Second test: headline known-item search over 2,066 queries.")
n_kw = sum(q["style"] == "keyword" for q in queries)
stat(s, 0.65, 1.9, 2.85, f"{Q['queries']}", f"TOPICS ({n_kw} KEYWORD · {Q['queries'] - n_kw} NATURAL)", TEAL, h=1.2)
stat(s, 3.7, 1.9, 2.85, f"{Q['judged_pairs']}", "GRADED JUDGMENTS (0 / 1 / 2)", AMBER, h=1.2)
stat(s, 6.75, 1.9, 2.85, f"{Q['relevant']}", f"RELEVANT ({Q['highly_relevant']} HIGHLY)", GREEN, h=1.2)
stat(s, 9.8, 1.9, 2.9, "100%", "OF EVERY TOP-10 JUDGED", BLUE, h=1.2)
rect(s, 0.65, 3.35, 5.95, 3.3, CARD, 0.05)
text(s, 0.95, 3.55, 5.4, 0.35, "Relevance assessment", 15, True, TEAL)
bullets(s, 0.95, 4.05, 5.45, 2.6, [
    f"**Depth-20 pooling** over 5 runs → {Q['pooled_pairs']} pairs, judged blind (no system or rank shown).",
    f"**Keyword sweep** of the whole archive → +{Q['relevant_found_by_sweep']} relevant documents the pool missed.",
    "**Graded:** 2 = about the topic, 1 = substantial secondary theme.",
], 12.5, gap=9)
rect(s, 6.8, 3.35, 5.9, 3.3, CARD, 0.05)
text(s, 7.1, 3.55, 5.4, 0.35, "Metrics and tests", 15, True, AMBER)
bullets(s, 7.1, 4.05, 5.4, 2.6, [
    "**P@10, nDCG@10** — the first results page.",
    "**Recall@100, MAP** — coverage and whole ranking; **11-point PR curve**.",
    f"**Known-item:** {res['known_item']['queries']:,} headlines → own article (MRR).",
    "**Paired randomisation test**, 20,000 resamples.",
], 12.5, gap=9)

# ============================================================================
# 10 Results table
# ============================================================================
s = new_slide("09 • RESULTS", "Performance comparison",
              f"Macro-average over {Q['queries']} topics; graded pooled judgments; best value per metric in bold.",
              notes="Both semantic runs beat the lexical runs by about 0.09 MAP. Hybrid best MAP and Recall@100; "
                    "ST best P@10 and nDCG@10. Hybrid vs ST is not significant.")
cols = [("P@10", "P@10"), ("Recall@100", "Recall@100"), ("AP", "MAP"), ("nDCG@10", "nDCG@10"), ("RR", "MRR")]
rows = [["Model"] + [c[1] for c in cols]]
bold = set()
for i, m in enumerate(MODELS, 1):
    rows.append([m] + [f"{main[m][k]:.4f}" for k, _ in cols])
    for j, (k, _) in enumerate(cols, 1):
        if abs(main[m][k] - max(main[x][k] for x in MODELS)) < 1e-9:
            bold.add((i, j))
table(s, 0.9, 1.95, 11.8, rows, [3.3, 1.7, 1.7, 1.7, 1.7, 1.7], size=15, row_h=0.62, bold=bold, dot_col0=True,
      head_size=14)
p_b = p_of("AP", "Hybrid RRF vs BM25")
p_s = p_of("AP", "Hybrid RRF vs Sentence Transformer")
rect(s, 0.65, 5.35, 12.05, 1.3, CARD2, 0.08)
text(s, 0.95, 5.55, 11.5, 1.0, [
    f"**Hybrid RRF: best MAP ({main['Hybrid RRF']['AP']:.3f}) and Recall@100 ({main['Hybrid RRF']['Recall@100']:.3f}).** "
    f"**Sentence Transformer: best P@10 and nDCG@10.**",
    f"Hybrid vs BM25: +{main['Hybrid RRF']['AP'] - main['BM25']['AP']:.3f} MAP (p = {p_b:.3f}); "
    f"Hybrid vs Sentence Transformer: not significant (p = {p_s:.2f})."], 14, False, MUTED)

# ============================================================================
# 11 Metric chart
# ============================================================================
s = new_slide("10 • RESULTS", "Semantic matching lifts every metric",
              notes="Chart version of the table. Lexical models are close to each other; semantic models are close "
                    "to each other.")
picture(s, FIG / "metrics_bar_dark.png", 0.65, 1.45, w=8.6)
rect(s, 9.6, 1.6, 3.1, 4.9, CARD, 0.05)
text(s, 9.85, 1.85, 2.7, 0.3, "TAKEAWAYS", 11, True, TEAL)
bullets(s, 9.85, 2.3, 2.7, 4.2, [
    f"BM25 → Hybrid: MAP **{main['BM25']['AP']:.3f} → {main['Hybrid RRF']['AP']:.3f}** "
    f"(+{100 * (main['Hybrid RRF']['AP'] / main['BM25']['AP'] - 1):.0f}%).",
    "Sentence Transformer's first result was relevant for **every** topic (MRR 1.0).",
    "BM25 edges TF-IDF on every metric.",
], 12.5, gap=12)

# ============================================================================
# 12 PR curve
# ============================================================================
pr = pd.read_csv(OUT / "precision_recall_11pt.csv")


def p_at(m, r):
    return float(pr[(pr.system == m) & (pr.recall.round(1) == r)].precision.iloc[0])


s = new_slide("11 • PRECISION–RECALL", "Precision–recall curves",
              "11-point interpolated precision, averaged over the 25 topics.",
              notes="Semantic runs dominate at every recall level; hybrid is highest at high recall.")
picture(s, FIG / "pr_curve_dark.png", 0.65, 1.75, h=5.1)
rect(s, 9.0, 1.9, 3.7, 4.6, CARD, 0.05)
text(s, 9.25, 2.15, 3.3, 0.3, "READING THE CURVE", 11, True, TEAL)
bullets(s, 9.25, 2.6, 3.3, 3.9, [
    f"At recall 0.5: **{p_at('Hybrid RRF', 0.5):.2f}** (hybrid) vs **{p_at('BM25', 0.5):.2f}** (BM25).",
    f"At recall 1.0 the hybrid keeps **{p_at('Hybrid RRF', 1.0):.2f}** — fusion rescues documents one model ranks low.",
    "TF-IDF and BM25 almost overlap.",
], 12.5, gap=12)

# ============================================================================
# 13 Query style
# ============================================================================
s = new_slide("12 • QUERY STYLE", "Lexical models fail on paraphrased queries",
              "MAP for 13 keyword queries vs 12 natural-language queries that avoid the articles' key terms.",
              notes="On keyword queries all models are close and the hybrid is best. On paraphrases BM25 drops to "
                    "0.66 while the Sentence Transformer stays at 0.85.")
picture(s, FIG / "style_bar_dark.png", 0.65, 1.8, w=7.9)
rect(s, 8.9, 1.9, 3.8, 4.6, CARD, 0.05)
text(s, 9.15, 2.15, 3.4, 0.3, "EXAMPLES (AP, BM25 → ST)", 11, True, TEAL)
qtext = {q["qid"]: q["query"] for q in queries}
pq = pd.DataFrame(res["per_query_main"]).pivot(index="qid", columns="system", values="AP")
ex = []
for qid in ("Q02", "Q25", "Q24"):
    ex.append(f"“{qtext[qid]}”  **{pq.loc[qid, 'BM25']:.2f} → {pq.loc[qid, 'Sentence Transformer']:.2f}**")
bullets(s, 9.15, 2.6, 3.4, 3.9, ex, 12, gap=12)

# ============================================================================
# 14 Known-item
# ============================================================================
s = new_slide("13 • KNOWN-ITEM SEARCH", "Finding an article from its headline",
              f"{res['known_item']['queries']:,} automatic queries; headlines removed from the index; one relevant article each.",
              notes="Here BM25 beats the Sentence Transformer, but the hybrid is best on every metric, p < 0.001.")
picture(s, FIG / "known_item_bar_dark.png", 0.65, 1.8, w=7.9)
rect(s, 8.9, 1.9, 3.8, 4.6, CARD, 0.05)
text(s, 9.15, 2.15, 3.4, 0.3, "THE ORDER FLIPS", 11, True, TEAL)
bullets(s, 9.15, 2.6, 3.4, 3.9, [
    f"BM25 (MRR **{ki['BM25']['MRR']:.3f}**) beats the Sentence Transformer (**{ki['Sentence Transformer']['MRR']:.3f}**) "
    "— headlines share exact words with their article.",
    f"Hybrid RRF is best on every metric: MRR **{ki['Hybrid RRF']['MRR']:.3f}**, Success@1 "
    f"**{ki['Hybrid RRF']['Success@1']:.3f}** (p < 0.001).",
    "Only the hybrid is a top model on both tests.",
], 12.5, gap=12)

# ============================================================================
# 15 Ablations & efficiency
# ============================================================================
s = new_slide("14 • ABLATIONS & EFFICIENCY", "What the design choices are worth",
              notes="MaxP beats FirstP; one lexical plus one semantic beats fusing all three; k barely matters; "
                    "hybrid costs about 4 ms per query.")
text(s, 0.65, 1.55, 6.5, 0.35, "Ablations (MAP on 25 topics)", 15, True, TEAL)
rows = [["Configuration", "MAP"],
        ["ST — best passage (MaxP)", f"{abl['Sentence Transformer']['AP']:.3f}"],
        ["ST — opening passage only (FirstP)", f"{abl['ST FirstP (opening passage only)']['AP']:.3f}"],
        ["RRF BM25 + ST  (main hybrid)", f"{abl['Hybrid RRF']['AP']:.3f}"],
        ["RRF TF-IDF + BM25 + ST", f"{abl['RRF TF-IDF + BM25 + ST']['AP']:.3f}"],
        ["RRF TF-IDF + BM25 (lexical only)", f"{abl['RRF TF-IDF + BM25 (lexical only)']['AP']:.3f}"],
        [f"RRF k from 1 to 500", f"{ktab.AP.min():.3f}–{ktab.AP.max():.3f}"]]
table(s, 0.65, 2.05, 6.6, rows, [4.9, 1.7], size=12.5, row_h=0.5, bold={(3, 0), (3, 1)})
text(s, 7.75, 1.55, 5.0, 0.35, "Latency (median per query)", 15, True, AMBER)
rows = [["Model", "Time"], ["BM25", f"{L['BM25']:.2f} ms"], ["TF-IDF", f"{L['TF-IDF']:.2f} ms"],
        ["Sentence Transformer", f"{L['Sentence Transformer']:.2f} ms"], ["Hybrid RRF", f"{L['Hybrid RRF']:.2f} ms"]]
table(s, 7.95, 2.05, 4.75, rows, [3.0, 1.75], size=12.5, row_h=0.5, dot_col0=True)
rect(s, 7.75, 4.75, 4.95, 1.75, CARD2, 0.06)
text(s, 8.0, 4.92, 4.5, 1.5, [f"Index build: {T['Inverted index']:.2f} s (inverted index), "
                              f"{T['Passage embeddings']:.1f} s for {D['passages']:,} passage embeddings on the "
                              "Apple GPU (cached)."], 12, False, MUTED)

# ============================================================================
# 16 Interface
# ============================================================================
s = new_slide("15 • SEARCH INTERFACE", "A news-desk search interface",
              notes="Dark editorial theme. Example queries, model selector, desk filter; every query is ranked by all "
                    "four models live and the time is shown. Cards show desk, score bar, matched terms and the full story.")
pic = picture(s, FIG / "ui_search.png", 0.65, 1.45, h=5.45)
rect(s, 6.0, 1.6, 6.7, 5.1, CARD, 0.05)
text(s, 6.3, 1.85, 6.1, 0.3, "FEATURES", 11, True, TEAL)
bullets(s, 6.3, 2.3, 6.1, 4.3, [
    "**One-click examples**, model selector and a **desk filter** — no sidebar.",
    "All four models rank every query; the **live latency** is shown (≈10 ms).",
    "Cards: desk, model score with a **relative score bar**, highlighted snippet, matched index terms.",
    "The **full story expands in place**; results with no shared terms are flagged as semantic matches.",
    "**Compare models** tab: top 5 of all four with consensus and agreement badges.",
], 14, gap=14)

# ============================================================================
# 17 Live benchmark
# ============================================================================
s = new_slide("16 • LIVE BENCHMARK", "Evaluation built into the app",
              notes="The Benchmark tab: headline metrics, interactive charts with tooltips, and a topic explorer that "
                    "re-runs any of the 25 test topics and grades each result against the judgments.")
picture(s, FIG / "ui_benchmark.png", 0.65, 1.45, w=6.4)
picture(s, FIG / "ui_explorer.png", 7.25, 1.45, w=5.45)
rect(s, 7.25, 5.05, 5.45, 1.7, CARD, 0.06)
bullets(s, 7.5, 5.25, 5.0, 1.4, [
    "**Interactive charts**: hover for exact scores, crosshair on the PR curve.",
    "**Topic explorer** re-scores any test topic live — values match the report.",
], 12.5, gap=8)

# ============================================================================
# 18 Conclusion
# ============================================================================
s = new_slide("17 • CONCLUSION", notes="Summary and future work.")
text(s, 0.65, 0.75, 12, 0.9, "Fusion gives the most robust ranking.", 30, True, TEXT)
bullets(s, 0.75, 1.85, 6.6, 4.2, [
    f"**Semantic matching** is the biggest single gain: MAP {main['BM25']['AP']:.3f} → "
    f"{main['Sentence Transformer']['AP']:.3f}.",
    "**Hybrid RRF** ties the best model on topical search and wins known-item search.",
    "**MaxP** passage scoring and fusing **one lexical + one semantic** list matter.",
    f"Hybrid search stays interactive at about **{L['Hybrid RRF']:.0f} ms per query**.",
], 15, gap=14)
rect(s, 7.75, 1.85, 4.95, 4.2, CARD, 0.05)
text(s, 8.05, 2.1, 4.4, 0.35, "Future work", 15, True, AMBER)
bullets(s, 8.05, 2.6, 4.45, 3.4, [
    "Cross-encoder re-ranking of the fused top 100",
    "Retrieval-tuned encoders (E5, BGE) and SPLADE",
    "RM3 query expansion for the lexical branch",
    "More topics and a second assessor (Cohen's κ)",
], 13, bullet_color=AMBER, gap=10)
text(s, 0.65, 6.35, 6, 0.5, "Thank you", 24, True, TEAL)

# ============================================================================
# 18 Appendix: significance
# ============================================================================
s = new_slide("APPENDIX • STATISTICS", "Significance tests",
              "Paired two-sided randomisation test, 20,000 resamples; W/T/L = per-topic wins/ties/losses.",
              notes="Hybrid significantly better than both lexical models; not distinguishable from ST.")
rows = [["Comparison", "Δ MAP", "W/T/L", "p", "Δ nDCG@10", "p"]]
bold = set()
for i, comp in enumerate(("Hybrid RRF vs TF-IDF", "Hybrid RRF vs BM25", "Hybrid RRF vs Sentence Transformer",
                          "Sentence Transformer vs BM25"), 1):
    a = sig[(sig.metric == "AP") & (sig.comparison == comp)].iloc[0]
    n = sig[(sig.metric == "nDCG@10") & (sig.comparison == comp)].iloc[0]
    rows.append([comp, f"{a.mean_diff:+.3f}", f"{a.wins}/{a.ties}/{a.losses}", f"{a.p_value:.4f}",
                 f"{n.mean_diff:+.3f}", f"{n.p_value:.4f}"])
    if a.p_value < 0.05:
        bold.add((i, 3))
    if n.p_value < 0.05:
        bold.add((i, 5))
table(s, 0.65, 2.0, 12.05, rows, [4.65, 1.45, 1.6, 1.4, 1.65, 1.3], size=13.5, row_h=0.6, bold=bold)
text(s, 0.65, 5.35, 12, 0.6, "Bold p-values are below 0.05. With 25 topics, small differences (such as Hybrid vs "
                             "Sentence Transformer) cannot be resolved; the known-item test (2,066 queries) separates "
                             "all models at p < 0.001.", 12.5, False, MUTED)

# ============================================================================
# 19 Appendix: references
# ============================================================================
s = new_slide("APPENDIX", "References and data provenance", notes="Sources.")
bullets(s, 0.75, 1.6, 11.9, 5.2, [
    "Greene, D. & Cunningham, P. (2006). Practical solutions to the problem of diagonal dominance in kernel document "
    "clustering. ICML. — BBC News dataset, mlg.ucd.ie/datasets/bbc.html",
    "Robertson, S. & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. FnTIR 3(4).",
    "Reimers, N. & Gurevych, I. (2019). Sentence-BERT. EMNLP-IJCNLP. — model: sentence-transformers/all-MiniLM-L6-v2",
    "Dai, Z. & Callan, J. (2019). Deeper text understanding for IR with contextual neural language modeling. SIGIR.",
    "Cormack, G. V., Clarke, C. L. A. & Büttcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and "
    "individual rank learning methods. SIGIR.",
    "Järvelin, K. & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques. ACM TOIS 20(4).",
    "Smucker, M. D., Allan, J. & Carterette, B. (2007). A comparison of statistical significance tests for IR "
    "evaluation. CIKM.",
    "Manning, C. D., Raghavan, P. & Schütze, H. (2008). Introduction to Information Retrieval. CUP.",
], 14, gap=13)

out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
prs.save(out)
print("saved", out, "| slides:", len(prs.slides))
