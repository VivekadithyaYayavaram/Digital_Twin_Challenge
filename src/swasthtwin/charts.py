"""Plotly figures for the dashboard (one y-axis per chart, observed vs projected
clearly separated)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

NAVY = "#0B1F3A"
TEAL = "#0F9D8F"
TEAL_FILL = "rgba(15,157,143,0.10)"
AMBER = "#C77700"
RED = "#C0392B"
GRID = "#E8EDF3"
MUTED = "#6B7A90"
BAND_COLORS = {"Low": "#1E8E5A", "Moderate": "#C77700", "Elevated": "#C0392B"}


def _layout(fig: go.Figure, title: str, ytitle: str, height: int = 340) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=NAVY)),
        height=height, margin=dict(l=10, r=10, t=48, b=10),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter, Segoe UI, sans-serif", color=NAVY, size=12),
        hovermode="x unified", legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, linecolor="#C9D1DC")
    fig.update_yaxes(title_text=ytitle, showgrid=True, gridcolor=GRID, linecolor="#C9D1DC", zeroline=False)
    return fig


def cgm_timeline(observed: pd.DataFrame, cursor: pd.Timestamp, horizon_min: int, threshold: float,
                 probability: float | None, band: str | None, lookback_h: int = 12,
                 revealed: pd.DataFrame | None = None) -> go.Figure:
    start = cursor - pd.Timedelta(hours=lookback_h)
    obs = observed[observed.timestamp >= start]
    end = cursor + pd.Timedelta(minutes=horizon_min)
    fig = go.Figure()
    # forecast window: a shaded REGION, never a line pretending to be data
    fig.add_vrect(x0=cursor, x1=end, fillcolor=TEAL_FILL, line_width=0, layer="below")
    fig.add_vline(x=cursor, line=dict(color=NAVY, width=1.5, dash="dot"))
    label = "Forecast window (next 120 min)"
    if probability is not None:
        label = f"Forecast window: P(≥{threshold:g} mg/dL) = {probability:.0%} · {band}"
    fig.add_annotation(x=end, y=0.98, yref="paper", text=f"<b>{label}</b>", showarrow=False,
                       xanchor="right", yanchor="top", bgcolor="rgba(255,255,255,0.92)",
                       bordercolor=BAND_COLORS.get(band, TEAL), borderwidth=1, borderpad=4,
                       font=dict(size=11, color=BAND_COLORS.get(band, TEAL)))
    fig.add_hline(y=threshold, line=dict(color=AMBER, width=1.2, dash="dash"),
                  annotation_text=f"{threshold:g} mg/dL event threshold", annotation_position="top left",
                  annotation_font=dict(size=10, color=AMBER))
    fig.add_hline(y=70, line=dict(color="#B8C2CF", width=1, dash="dot"))
    fig.add_trace(go.Scatter(x=obs.timestamp, y=obs.cgm_mg_dl, mode="lines", name="Observed CGM",
                             line=dict(color=NAVY, width=2), connectgaps=False,
                             hovertemplate="%{y:.0f} mg/dL<extra>Observed</extra>"))
    if revealed is not None and len(revealed):
        fig.add_trace(go.Scatter(x=revealed.timestamp, y=revealed.cgm_mg_dl, mode="lines",
                                 name="Replay: what happened next (hidden from model)",
                                 line=dict(color="#8A94A6", width=1.6, dash="dot"),
                                 hovertemplate="%{y:.0f} mg/dL<extra>Revealed (not a forecast)</extra>"))
    fig.update_xaxes(range=[start, end + pd.Timedelta(minutes=10)])
    fig.update_yaxes(range=[40, max(320, float(obs.cgm_mg_dl.max() or 0) + 20)])
    return _layout(fig, "CGM glucose — observed history and 2-hour forecast window", "mg/dL", 380)


def single_series(df: pd.DataFrame, col: str, cursor: pd.Timestamp, title: str, ytitle: str,
                  color: str, lookback_h: int = 12, kind: str = "line") -> go.Figure:
    d = df[df.timestamp >= cursor - pd.Timedelta(hours=lookback_h)]
    fig = go.Figure()
    if kind == "bar":
        fig.add_trace(go.Bar(x=d.timestamp, y=d[col], marker_color=color, name=title,
                             hovertemplate=f"%{{y:.0f}} {ytitle}<extra></extra>"))
    else:
        fig.add_trace(go.Scatter(x=d.timestamp, y=d[col], mode="lines", line=dict(color=color, width=1.8),
                                 name=title, connectgaps=False,
                                 hovertemplate=f"%{{y:.0f}} {ytitle}<extra></extra>"))
    fig.add_vline(x=cursor, line=dict(color=NAVY, width=1, dash="dot"))
    fig.update_layout(showlegend=False)
    return _layout(fig, title, ytitle, 230)


def sleep_bars(sleep: pd.DataFrame) -> go.Figure:
    d = sleep.tail(10)
    fig = go.Figure(go.Bar(x=d.night_of.astype(str), y=d.sleep_duration_h, marker_color="#2F5D9E",
                           text=[f"{h:.1f} h · Q{q:.0f}" for h, q in zip(d.sleep_duration_h,
                                                                        d.sleep_quality_score)],
                           textposition="outside", hovertemplate="%{y:.2f} h<extra></extra>"))
    fig.add_hline(y=7, line=dict(color="#B8C2CF", dash="dot"), annotation_text="7 h",
                  annotation_font=dict(size=10, color=MUTED))
    fig.update_yaxes(range=[0, 11])
    fig.update_layout(showlegend=False)
    return _layout(fig, "Sleep (completed nights, wearable summary)", "hours", 260)


def risk_history(hist: pd.DataFrame, alert_threshold: float, bands: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=bands["low_max"], fillcolor="rgba(30,142,90,0.06)", line_width=0)
    fig.add_hrect(y0=bands["low_max"], y1=bands["moderate_max"], fillcolor="rgba(199,119,0,0.06)", line_width=0)
    fig.add_hrect(y0=bands["moderate_max"], y1=1, fillcolor="rgba(192,57,43,0.06)", line_width=0)
    fig.add_hline(y=alert_threshold, line=dict(color=RED, dash="dash", width=1),
                  annotation_text=f"model alert threshold {alert_threshold:.2f}",
                  annotation_font=dict(size=10, color=RED), annotation_position="bottom left")
    if len(hist):
        rep = hist[hist.source.isin(["replay", "manual"])]
        fig.add_trace(go.Scatter(x=rep.as_of, y=rep.probability, mode="lines+markers", name="Twin forecast",
                                 line=dict(color=TEAL, width=2), marker=dict(size=8, color=TEAL,
                                                                            line=dict(color="white", width=2)),
                                 hovertemplate="%{y:.1%}<extra>forecast at %{x|%H:%M}</extra>"))
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
    fig.update_layout(showlegend=False)
    return _layout(fig, "How the twin's 120-min risk estimate has changed", "P(event)", 280)


def contribution_bars(expl: pd.DataFrame, top: int = 10) -> go.Figure:
    d = expl.head(top).iloc[::-1]
    colors = [RED if c > 0 else "#1E8E5A" for c in d.contribution]
    fig = go.Figure(go.Bar(x=d.contribution, y=[f"{l}  [{s.split(' ')[0]}]" for l, s in zip(d.label, d.stream)],
                           orientation="h", marker_color=colors,
                           hovertemplate="%{x:+.1%}<extra></extra>"))
    fig.update_xaxes(tickformat="+.0%", title_text="Change in probability vs typical training patient")
    fig.update_layout(showlegend=False)
    return _layout(fig, "Most influential inputs for THIS forecast (association, not cause)", "", 400)
