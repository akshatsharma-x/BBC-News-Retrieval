"""Render every figure twice: light (report, white page) and dark (slides).

Palette: validated categorical order (blue, orange, aqua, yellow) - one fixed
colour per retrieval model in every chart. Light steps were validated on
#ffffff, dark steps on the slide background #111c30.

Usage:  python scripts/make_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bbcir.data import load_articles  # noqa: E402

OUT = ROOT / "outputs"
FIG = OUT / "figures"
MODELS = ["TF-IDF", "BM25", "Sentence Transformer", "Hybrid RRF"]
MARKERS = ["o", "s", "^", "D"]

THEMES = {
    "light": dict(surface="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#6f6e69", grid="#e7e6e1",
                  axis="#c3c2b7", series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
                  seq=["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]),
    "dark": dict(surface="#111c30", ink="#f8fafd", ink2="#b2c0d3", muted="#8d9bb0", grid="#22314a",
                 axis="#3a4b68", series=["#3987e5", "#d95926", "#199e70", "#c98500"],
                 seq=["#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#6da7ec", "#9ec5f4"]),
}


def setup_fonts() -> str:
    font_dir = Path("/Applications/Microsoft Word.app/Contents/Resources/DFonts")
    for name in ("Aptos.ttf", "Aptos-Bold.ttf", "Aptos-SemiBold.ttf"):
        if (font_dir / name).exists():
            font_manager.fontManager.addfont(str(font_dir / name))
    names = {f.name for f in font_manager.fontManager.ttflist}
    return "Aptos" if "Aptos" in names else "Helvetica"


FONT = setup_fonts()


def style(ax, th, ygrid=True, xgrid=False):
    ax.set_facecolor(th["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(th["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=th["ink2"], labelsize=9, length=0, pad=6)
    if ygrid:
        ax.yaxis.grid(True, color=th["grid"], linewidth=0.8, linestyle="-")
    if xgrid:
        ax.xaxis.grid(True, color=th["grid"], linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def new_fig(th, w, h):
    plt.rcParams.update({"font.family": FONT, "font.size": 10})
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(th["surface"])
    return fig, ax


def rounded_bar(ax, x, width, height, color, radius_px=4):
    """Column with 4px rounded data end and a square baseline."""
    if height <= 0:
        return
    fig = ax.figure
    fig.canvas.draw_idle()
    bbox = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    px_x = (x1 - x0) / bbox.width
    px_y = (y1 - y0) / bbox.height
    r = min(radius_px * px_x * (fig.dpi / 100), width / 2)
    aspect = px_y / px_x
    r_y = r * aspect
    ax.add_patch(FancyBboxPatch((x - width / 2, 0), width, height,
                                boxstyle=f"round,pad=0,rounding_size={r}", mutation_aspect=aspect,
                                linewidth=0, facecolor=color, zorder=3))
    ax.add_patch(Rectangle((x - width / 2, 0), width, max(height - r_y, 0), linewidth=0,
                           facecolor=color, zorder=3))


def legend(ax, th, handles_labels, loc="lower center", ncol=4, anchor=(0.5, 1.03)):
    handles, labels = handles_labels
    leg = ax.legend(handles, labels, loc=loc, ncol=ncol, frameon=False, bbox_to_anchor=anchor,
                    fontsize=9, handlelength=1.1, handleheight=0.9, columnspacing=1.4)
    for t in leg.get_texts():
        t.set_color(th["ink2"])
    return leg


def grouped_bars(th, groups, values, title_note, path, ylim=(0, 1.0), w=7.2, h=3.6, label_best=True,
                 group_labels=None):
    """values: (n_groups, n_models) array."""
    fig, ax = new_fig(th, w, h)
    style(ax, th)
    n_g, n_m = values.shape
    gap = 0.04
    bw = 0.17
    ax.set_xlim(-0.6, n_g - 0.4)
    ax.set_ylim(ylim[0], ylim[1] + 0.07)
    ax.spines["left"].set_bounds(ylim[0], ylim[1])
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    handles = []
    for j in range(n_m):
        for i in range(n_g):
            x = i + (j - (n_m - 1) / 2) * (bw + gap)
            rounded_bar(ax, x, bw, values[i, j], th["series"][j])
        handles.append(Rectangle((0, 0), 1, 1, facecolor=th["series"][j], linewidth=0))
    if label_best:
        for i in range(n_g):
            j = int(np.argmax(values[i]))
            x = i + (j - (n_m - 1) / 2) * (bw + gap)
            ax.text(x, values[i, j] + 0.015, f"{values[i, j]:.3f}", ha="center", va="bottom",
                    fontsize=8.5, color=th["ink"], fontweight="bold", zorder=4)
    ax.set_xticks(range(n_g))
    ax.set_xticklabels(group_labels or groups, fontsize=10, color=th["ink"])
    ax.set_yticks(np.linspace(ylim[0], ylim[1], 6))
    ax.set_yticklabels([f"{v:.1f}" for v in np.linspace(ylim[0], ylim[1], 6)])
    legend(ax, th, (handles, MODELS))
    if title_note:
        ax.text(1.0, -0.17, title_note, transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
                color=th["muted"])
    fig.savefig(path, facecolor=th["surface"], bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def pr_curve(th, pr: pd.DataFrame, path, w=6.4, h=4.2):
    fig, ax = new_fig(th, w, h)
    style(ax, th, ygrid=True, xgrid=True)
    handles = []
    for j, m in enumerate(MODELS):
        d = pr[pr.system == m].sort_values("recall")
        (ln,) = ax.plot(d.recall, d.precision, color=th["series"][j], linewidth=2, marker=MARKERS[j],
                        markersize=6.5 if MARKERS[j] != "D" else 5.5, markeredgecolor=th["surface"],
                        markeredgewidth=1.4, solid_capstyle="round", solid_joinstyle="round", zorder=3 + j)
        handles.append(ln)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.04)
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_xticklabels([f"{v:.1f}" for v in np.linspace(0, 1, 11)])
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{v:.1f}" for v in np.linspace(0, 1, 6)])
    ax.set_xlabel("Recall", color=th["ink2"], fontsize=10, labelpad=6)
    ax.set_ylabel("Interpolated precision", color=th["ink2"], fontsize=10, labelpad=6)
    leg = ax.legend(handles, MODELS, loc="lower left", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(th["ink2"])
    fig.tight_layout()
    fig.savefig(path, facecolor=th["surface"], bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def heatmap(th, per_query: pd.DataFrame, queries, path, metric="AP", w=6.6, h=7.6):
    tab = per_query.pivot(index="qid", columns="system", values=metric)[MODELS]
    fig, ax = new_fig(th, w, h)
    ax.set_facecolor(th["surface"])
    seq = th["seq"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", seq)
    vals = tab.values
    ax.imshow(vals, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(vals):
        rgb = np.array(matplotlib.colors.to_rgb(cmap(v)))
        lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="#0b0b0b" if lum > 0.5 else "#ffffff")
    qtext = {q["qid"]: q["query"] for q in queries}
    style_ = {q["qid"]: q["style"] for q in queries}
    labels = [f"{qid}  {qtext[qid][:44] + ('…' if len(qtext[qid]) > 44 else '')}" + ("  ◦" if style_[qid] == "natural" else "")
              for qid in tab.index]
    ax.set_yticks(range(len(tab.index)))
    ax.set_yticklabels(labels, fontsize=7.6, color=th["ink"])
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(["TF-IDF", "BM25", "Sentence\nTransformer", "Hybrid\nRRF"], fontsize=8.5, color=th["ink"])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # 2px surface gaps between cells
    for i in range(len(tab.index) + 1):
        ax.axhline(i - 0.5, color=th["surface"], linewidth=2)
    for j in range(len(MODELS) + 1):
        ax.axvline(j - 0.5, color=th["surface"], linewidth=2)
    ax.text(0, 1.0, "", transform=ax.transAxes)
    fig.text(0.01, 0.005, "◦ = natural-language (paraphrased) query.  Cell value = average precision.",
             fontsize=7.5, color=th["muted"])
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, facecolor=th["surface"], bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def category_bars(th, counts: dict, path, w=6.2, h=2.6):
    fig, ax = new_fig(th, w, h)
    style(ax, th, ygrid=False, xgrid=True)
    cats = list(counts)[::-1]
    vals = [counts[c] for c in cats]
    ax.barh(cats, vals, height=0.42, color=th["series"][0], zorder=3)
    for y, v in enumerate(vals):
        ax.text(v + 8, y, f"{v:,}", va="center", fontsize=9, color=th["ink"])
    ax.set_xlim(0, max(vals) * 1.15)
    ax.tick_params(axis="y", labelsize=10, labelcolor=th["ink"])
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor=th["surface"], bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def length_hist(th, words: pd.Series, path, w=6.2, h=2.6):
    fig, ax = new_fig(th, w, h)
    style(ax, th)
    clipped = words.clip(upper=1000)
    ax.hist(clipped, bins=np.arange(0, 1050, 50), color=th["series"][0], edgecolor=th["surface"],
            linewidth=1.2, zorder=3)
    ax.axvline(words.median(), color=th["ink2"], linewidth=1, zorder=4)
    ax.text(words.median() + 12, ax.get_ylim()[1] * 0.92, f"median {int(words.median())} words",
            fontsize=8.5, color=th["ink2"])
    ax.set_xlabel("Article body length (words; last bin = 1,000+)", color=th["ink2"], fontsize=9)
    ax.set_ylabel("Articles", color=th["ink2"], fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=th["surface"], bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def pipeline(th, path, stats, dense, w=7.2, h=3.4):
    """Architecture diagram for the report."""
    plt.rcParams.update({"font.family": FONT})
    fig = plt.figure(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(th["surface"])
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 47)
    ax.axis("off")
    card = "#f4f6f9" if th is THEMES["light"] else "#192941"
    edge = "#d9dee6" if th is THEMES["light"] else "#2b3d5c"

    def box(x, y, bw, bh, title, sub, accent=None):
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0,rounding_size=1.2",
                                    facecolor=card, edgecolor=edge, linewidth=0.8))
        if accent:
            ax.add_patch(FancyBboxPatch((x, y + bh - 1.1), bw, 1.1, boxstyle="round,pad=0,rounding_size=0.5",
                                        facecolor=accent, linewidth=0))
        ax.text(x + bw / 2, y + bh * 0.62, title, ha="center", va="center", fontsize=8.6,
                fontweight="bold", color=th["ink"])
        ax.text(x + bw / 2, y + bh * 0.28, sub, ha="center", va="center", fontsize=6.9, color=th["ink2"],
                linespacing=1.25)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=th["muted"], lw=1.0, shrinkA=0, shrinkB=0,
                                    mutation_scale=9))

    s = th["series"]
    box(1, 18, 15, 11, "BBC corpus", f"2,225 articles\n→ {stats['documents']:,} unique")
    box(20, 18, 17, 11, "Preprocessing", "clean · tokenise\nstop-words · Porter stem")
    box(41, 33, 20, 11, "Inverted index", f"{stats['vocabulary_terms']:,} terms\n{stats['total_postings']:,} postings", s[1])
    box(41, 18.5, 20, 11, "TF-IDF vectors", "sublinear TF × IDF\nL2-normalised", s[0])
    box(41, 4, 20, 11, "Passage embeddings", f"all-MiniLM-L6-v2\n{dense['passages']:,} passages × {dense['dim']}-d", s[2])
    box(65, 33, 14, 11, "BM25", "k₁ = 1.2, b = 0.75", s[1])
    box(65, 18.5, 14, 11, "Cosine", "TF-IDF ranking", s[0])
    box(65, 4, 14, 11, "MaxP cosine", "dense ranking", s[2])
    box(83, 18.5, 16, 11, "Hybrid RRF", "BM25 + ST, k = 60\n→ UI / evaluation", s[3])
    arrow(16, 23.5, 20, 23.5)
    arrow(37, 23.5, 41, 38.5)
    arrow(37, 23.5, 41, 24)
    ax.annotate("", xy=(41, 9.5), xytext=(8.5, 18),
                arrowprops=dict(arrowstyle="-|>", color=th["muted"], lw=1.0, mutation_scale=9,
                                connectionstyle="angle,angleA=-90,angleB=180,rad=0"))
    ax.text(10, 11.5, "raw text\n(no stemming)", fontsize=6.6, color=th["muted"], ha="left")
    arrow(61, 38.5, 65, 38.5)
    arrow(61, 24, 65, 24)
    arrow(61, 9.5, 65, 9.5)
    arrow(79, 38.5, 83, 26.5)
    arrow(79, 9.5, 83, 21)
    fig.savefig(path, facecolor=th["surface"], bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    res = json.loads((OUT / "results.json").read_text())
    queries = json.loads((ROOT / "data" / "queries.json").read_text())
    main_tbl = pd.read_csv(OUT / "metrics_summary.csv", index_col=0).loc[MODELS]
    pr = pd.read_csv(OUT / "precision_recall_11pt.csv")
    style_tbl = pd.read_csv(OUT / "metrics_by_query_style.csv")
    ki = pd.read_csv(OUT / "known_item_summary.csv", index_col=0).loc[MODELS]
    per_query = pd.DataFrame(res["per_query_main"])
    articles = load_articles()
    words = articles["body"].str.split().str.len()
    counts = {"Business": 503, "Entertainment": 369, "Politics": 403, "Sport": 504, "Technology": 347}
    counts = {k: res["corpus"]["unique_per_category"][v] for k, v in
              zip(counts, ["business", "entertainment", "politics", "sport", "tech"])}

    for mode, th in THEMES.items():
        metrics = ["P@10", "Recall@100", "AP", "nDCG@10"]
        grouped_bars(th, metrics, main_tbl[metrics].T.values, "", FIG / f"metrics_bar_{mode}.png",
                     group_labels=["Precision@10", "Recall@100", "MAP", "nDCG@10"])
        pr_curve(th, pr, FIG / f"pr_curve_{mode}.png")
        sv = np.array([[style_tbl[(style_tbl["style"] == s) & (style_tbl.system == m)]["AP"].iloc[0] for m in MODELS]
                       for s in ("keyword", "natural")])
        grouped_bars(th, ["keyword", "natural"], sv, "", FIG / f"style_bar_{mode}.png", w=6.4, h=3.3,
                     group_labels=["Keyword queries (13)", "Natural-language queries (12)"])
        kim = ["MRR", "Success@1", "Recall@10", "nDCG@10"]
        grouped_bars(th, kim, ki[kim].T.values, "", FIG / f"known_item_bar_{mode}.png",
                     group_labels=["MRR (= MAP)", "Success@1", "Recall@10", "nDCG@10"])
        category_bars(th, counts, FIG / f"categories_{mode}.png")
        length_hist(th, words, FIG / f"doc_length_{mode}.png")
    heatmap(THEMES["light"], per_query, queries, FIG / "per_query_ap_light.png")
    pipeline(THEMES["light"], FIG / "pipeline_light.png", res["index_stats"], res["dense"])
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
