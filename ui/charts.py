"""Interactive Altair charts for the Benchmark tab (dark theme, validated palette).

Bars: thin, 4px rounded data ends, best value per group labelled, tooltip per bar.
Lines: 2px, marker shapes as a second identity channel, crosshair tooltip.
"""
from __future__ import annotations

import altair as alt
import pandas as pd

from .components import MODEL_COLOR

MODELS = list(MODEL_COLOR)
COLORS = [MODEL_COLOR[m] for m in MODELS]
INK, INK2, MUTED, GRID, AXIS = "#F3F6FB", "#B9C5D6", "#8D9BB0", "#22314A", "#3A4B68"


def _style(chart: alt.TopLevelMixin, height: int = 300) -> alt.TopLevelMixin:
    return (chart.properties(height=height, background="transparent", padding={"left": 4, "right": 8, "top": 6, "bottom": 4})
            .configure_view(stroke=None)
            .configure_axis(labelColor=INK2, titleColor=MUTED, gridColor=GRID, gridWidth=1, domainColor=AXIS,
                            tickColor=AXIS, labelFont="Inter", titleFont="Inter", labelFontSize=11.5,
                            titleFontSize=11.5, titleFontWeight=500, labelPadding=6)
            .configure_legend(orient="top", title=None, labelColor=INK2, labelFont="Inter", labelFontSize=12,
                              symbolType="square", symbolSize=110, columnPadding=18, padding=0)
            .configure_text(font="Inter"))


def _color(field: str = "model:N") -> alt.Color:
    return alt.Color(field, sort=MODELS, scale=alt.Scale(domain=MODELS, range=COLORS), legend=alt.Legend(title=None))


def metric_bars(table: pd.DataFrame, metrics: list[tuple[str, str]], height: int = 300) -> alt.TopLevelMixin:
    """table: index = model, columns include the metric keys."""
    rows = [{"model": m, "metric": label, "value": float(table.loc[m, key])}
            for m in MODELS for key, label in metrics]
    df = pd.DataFrame(rows)
    order = [label for _, label in metrics]
    best = df.loc[df.groupby("metric")["value"].idxmax()]
    x = alt.X("metric:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelColor=INK, labelFontSize=12.5,
                                                                domain=True, ticks=False, grid=False))
    off = alt.XOffset("model:N", sort=MODELS)
    y = alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1.08]),
              axis=alt.Axis(values=[0, 0.2, 0.4, 0.6, 0.8, 1.0], format=".1f", domain=False, ticks=False))
    bars = alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=15).encode(
        x=x, xOffset=off, y=y, color=_color(),
        tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("metric:N", title="Metric"),
                 alt.Tooltip("value:Q", title="Score", format=".4f")])
    labels = alt.Chart(best).mark_text(dy=-9, fontSize=11, fontWeight=600, color=INK).encode(
        x=x, xOffset=off, y=y, text=alt.Text("value:Q", format=".3f"))
    return _style(alt.layer(bars, labels), height)


def pr_curve(pr: pd.DataFrame, height: int = 300) -> alt.TopLevelMixin:
    """pr: columns system, recall, precision (11-point interpolated)."""
    df = pr.rename(columns={"system": "model"})
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["recall"], empty=False, clear="pointerout")
    x = alt.X("recall:Q", title="Recall", scale=alt.Scale(domain=[0, 1]),
              axis=alt.Axis(values=[i / 10 for i in range(11)], format=".1f"))
    y = alt.Y("precision:Q", title="Interpolated precision", scale=alt.Scale(domain=[0, 1.02]),
              axis=alt.Axis(values=[0, 0.2, 0.4, 0.6, 0.8, 1.0], format=".1f", domain=False, ticks=False))
    base = alt.Chart(df).encode(x=x, y=y, color=_color())
    lines = base.mark_line(strokeWidth=2, strokeCap="round", strokeJoin="round")
    points = base.mark_point(filled=True, size=58, stroke="#111C30", strokeWidth=1.6, opacity=1).encode(
        shape=alt.Shape("model:N", sort=MODELS, legend=None,
                        scale=alt.Scale(domain=MODELS, range=["circle", "square", "triangle-up", "diamond"])))
    rule = (alt.Chart(df).transform_pivot("model", value="precision", groupby=["recall"])
            .mark_rule(color=MUTED, strokeWidth=1)
            .encode(x="recall:Q", opacity=alt.condition(nearest, alt.value(0.9), alt.value(0)),
                    tooltip=[alt.Tooltip("recall:Q", title="Recall", format=".1f")] +
                            [alt.Tooltip(f"{m}:Q", title=m, format=".3f") for m in MODELS])
            .add_params(nearest))
    return _style(alt.layer(lines, points, rule), height)


def style_bars(style_tbl: pd.DataFrame, height: int = 260) -> alt.TopLevelMixin:
    """MAP per query style (keyword / natural)."""
    df = style_tbl.rename(columns={"system": "model"}).copy()
    df["style_label"] = df["style"].map({"keyword": "Keyword queries (13)", "natural": "Natural-language queries (12)"})
    best = df.loc[df.groupby("style_label")["AP"].idxmax()]
    x = alt.X("style_label:N", title=None, sort=["Keyword queries (13)", "Natural-language queries (12)"],
              axis=alt.Axis(labelAngle=0, labelColor=INK, labelFontSize=12.5, ticks=False, grid=False))
    off = alt.XOffset("model:N", sort=MODELS)
    y = alt.Y("AP:Q", title="MAP", scale=alt.Scale(domain=[0, 1.08]),
              axis=alt.Axis(values=[0, 0.2, 0.4, 0.6, 0.8, 1.0], format=".1f", domain=False, ticks=False))
    bars = alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=18).encode(
        x=x, xOffset=off, y=y, color=_color(),
        tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("style_label:N", title="Queries"),
                 alt.Tooltip("AP:Q", title="MAP", format=".4f")])
    labels = alt.Chart(best).mark_text(dy=-9, fontSize=11, fontWeight=600, color=INK).encode(
        x=x, xOffset=off, y=y, text=alt.Text("AP:Q", format=".3f"))
    return _style(alt.layer(bars, labels), height)
