"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import kpb_constants as C
import kpb_dynamics as K


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.2), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _thin(a, k=3):
    return a[::k]


def build_potential(a: dict):
    """Potential V(x) bei dem eingestellten h; Gleichgewichte: stabil gefüllt, instabil hohl."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.round(a["xs"], 4), y=np.round(a["V"], 5), mode="lines", line=dict(color=C.COLOR_LINE, width=2), name="V(x)", hovertemplate="x = %{x:.3f}: V = %{y:.4f}<extra></extra>"))
    for x, rt in zip(a["roots"], a["rates"]):
        v = float(K.potential(x, a["h"]))
        fig.add_trace(go.Scatter(x=[x], y=[v], mode="markers", marker=dict(size=12, color=C.COLOR_DOWN if rt > 0 else "rgba(0,0,0,0)", line=dict(color=C.COLOR_DOWN, width=2)),
                                 name="stabil" if rt > 0 else "instabil", showlegend=False, hovertemplate=f"x = {x:.4f}: {'stabil' if rt > 0 else 'instabil'}, Rate {rt:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="Potential V(x)", range=[float(np.min(a["V"])) - 0.1, 0.6])
    return _base(fig, 330)


def _scurve_traces(fig, s: dict):
    st = s["stable"]
    fig.add_trace(go.Scatter(x=np.round(s["h"][st], 4), y=np.round(s["x"][st], 4), mode="markers", marker=dict(size=3, color=C.COLOR_GREY), name="stabile Gleichgewichte", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=np.round(s["h"][~st], 4), y=np.round(s["x"][~st], 4), mode="markers", marker=dict(size=2, color="rgba(150,150,150,0.4)"), name="instabile Gleichgewichte", hoverinfo="skip"))
    fx = 1.0 / math.sqrt(3.0)
    fig.add_trace(go.Scatter(x=[K.HC, -K.HC], y=[-fx, fx], mode="markers", marker=dict(size=10, symbol="diamond", color=C.COLOR_UP), name="Faltung ±h_c",
                             hovertemplate="Faltung bei h = %{x:.4f}<extra></extra>"))


def build_scurve(s: dict, h_now: float):
    """Gleichgewichtsdiagramm (S-Kurve) mit Markierung der Faltungen und dem eingestellten h."""
    fig = go.Figure()
    _scurve_traces(fig, s)
    fig.add_vline(x=h_now, line=dict(color=C.COLOR_LINE, dash="dot"))
    fig.update_xaxes(title_text="h")
    fig.update_yaxes(title_text="Gleichgewicht x", range=[-1.7, 1.7])
    return _base(fig, 330)


def build_loop(s: dict, loop: dict):
    """S-Kurve und die Bahn x(h) hinauf und hinunter; Kipp-Orte markiert."""
    fig = go.Figure()
    _scurve_traces(fig, s)
    for key, color, name in (("trace_up", C.COLOR_UP, "Rampe hinauf"), ("trace_down", C.COLOR_DOWN, "Rampe hinunter")):
        tr = loop[key]
        if tr is not None:
            hs, xs = tr
            fig.add_trace(go.Scatter(x=np.round(hs, 4), y=np.round(xs, 4), mode="lines", line=dict(color=color, width=2.5), name=name, hovertemplate="h = %{x:.4f}: x = %{y:.3f}<extra></extra>"))
    for tip, color in ((loop["up"], C.COLOR_UP), (loop["down"], C.COLOR_DOWN)):
        if tip is not None:
            fig.add_vline(x=tip, line=dict(color=color, dash="dash"))
    fig.update_xaxes(title_text="h")
    fig.update_yaxes(title_text="x", range=[-1.7, 1.7])
    return _base(fig, 380)


def build_delay(rows: list[dict]):
    rs = [x["r"] for x in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rs, y=[x["formula"] for x in rows], mode="lines", line=dict(color=C.COLOR_GREY, dash="dash"), name="Formel C·r^(2/3)", hovertemplate="r = %{x:.0e}: %{y:.3e}<extra></extra>"))
    fig.add_trace(go.Scatter(x=rs, y=[x["delta"] for x in rows], mode="markers", marker=dict(size=9, color=C.COLOR_UP), name="gemessen", hovertemplate="r = %{x:.0e}: δ = %{y:.3e}<extra></extra>"))
    fig.update_xaxes(type="log", title_text="Rampenrate r")
    fig.update_yaxes(type="log", title_text="Verzögerung δ = h_Kipp − h_c")
    return _base(fig, 360)


def build_rate(h_now: float):
    """Rate des Topfes nahe der Faltung gegen h/h_c: exakt (3x² − 1) und Näherung √(8/3)·√(1 − h/h_c)."""
    fr = np.linspace(0.0, 0.995, 200)
    exact = [K.rate(K.lower_root(f * K.HC)) for f in fr]
    approx = [K.rate_approx(f * K.HC) for f in fr]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.round(fr, 4), y=np.round(exact, 5), mode="lines", line=dict(color=C.COLOR_LINE, width=2.5), name="exakt 3x² − 1", hovertemplate="h/h_c = %{x:.3f}: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=np.round(fr, 4), y=np.round(approx, 5), mode="lines", line=dict(color=C.COLOR_GREY, dash="dash"), name="Näherung √(8/3)·√(1 − h/h_c)", hovertemplate="h/h_c = %{x:.3f}: %{y:.4f}<extra></extra>"))
    if abs(h_now) < K.HC:
        fig.add_vline(x=abs(h_now) / K.HC, line=dict(color=C.COLOR_UP, dash="dot"))
    fig.update_xaxes(title_text="|h| / h_c")
    fig.update_yaxes(title_text="Rate des Topfes")
    return _base(fig, 330)


def build_series(x, dtm: float, title: str):
    x = np.asarray(x)
    k = max(1, len(x) // 700)
    t = np.arange(len(x)) * dtm
    fig = go.Figure(go.Scatter(x=np.round(t[::k], 2), y=np.round(x[::k], 4), mode="lines", line=dict(color=C.COLOR_LINE, width=1), hovertemplate="t = %{x:.1f}: x = %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Zeit")
    fig.update_yaxes(title_text="x")
    fig.update_layout(title=dict(text=title, font=dict(size=13)))
    return _base(fig, 300)


def build_warning(w: dict, thr: float):
    """Oben: ein Beispiellauf (Lauf 1); Mitte: Fenster-Autokorrelation dieses Laufs; unten: Trendmaße aller Läufe mit der Alarmschwelle."""
    x = w["sample"]
    fig = make_subplots(rows=3, cols=1, vertical_spacing=0.1, row_heights=[0.34, 0.33, 0.33],
                        subplot_titles=("Beispiellauf (Lauf 1) bis zum Kippen", "Autokorrelation je Fenster (200 Messungen, Schritt 50)", "Trendmaß aller Läufe (Korrelation Fensternummer gegen Autokorrelation)"))
    k = max(1, len(x) // 600)
    fig.add_trace(go.Scatter(x=np.arange(len(x))[::k], y=np.round(x[::k], 4), mode="lines", line=dict(color=C.COLOR_LINE, width=1), name="x", showlegend=False,
                             hovertemplate="Messung %{x}: x = %{y:.3f}<extra></extra>"), row=1, col=1)
    if len(x) >= K.MIN_LEN:
        st, acs = K.window_ac1(x)
        fig.add_trace(go.Scatter(x=st, y=np.round(acs, 4), mode="lines+markers", line=dict(color=C.COLOR_ORANGE), marker=dict(size=4), name="AC1", showlegend=False,
                                 hovertemplate="Fenster ab %{x}: AC1 = %{y:.3f}<extra></extra>"), row=2, col=1)
    tr = w["trends"][~np.isnan(w["trends"])]
    fig.add_trace(go.Histogram(x=np.round(tr, 3), xbins=dict(start=-1, end=1, size=0.1), marker_color=C.COLOR_LINE, name="Läufe", showlegend=False,
                               hovertemplate="Trendmaß %{x}: %{y} Läufe<extra></extra>"), row=3, col=1)
    fig.add_vline(x=thr, line=dict(color=C.COLOR_UP, dash="dash"), row=3, col=1)
    fig.update_xaxes(title_text="Messung (Abstand 0,1)", row=1, col=1)
    fig.update_xaxes(range=[-1, 1], title_text="Trendmaß (rot gestrichelt: Alarmschwelle)", row=3, col=1)
    return _base(fig, 640)


def build_tempo(t: dict):
    fig = go.Figure()
    colors = [C.COLOR_GREY, C.COLOR_LINE, C.COLOR_UP]
    for j, ss in enumerate(C.GRID_SIGMA):
        rs = list(C.GRID_R)
        means = [t["cells"][i][j]["mean"] for i in range(len(rs))]
        stds = [t["cells"][i][j]["std"] for i in range(len(rs))]
        fig.add_trace(go.Scatter(x=rs, y=means, mode="lines+markers", line=dict(color=colors[j]), error_y=dict(type="data", array=stds, visible=True), name=f"σ = {ss:.2f}".replace(".", ","),
                                 hovertemplate="r = %{x:.4f}: h = %{y:.4f}<extra></extra>"))
    o = t["own"]
    if o["n_tipped"]:
        fig.add_trace(go.Scatter(x=[o["r"]], y=[o["mean"]], mode="markers", marker=dict(size=13, symbol="star", color=C.COLOR_ORANGE), name="eigene Zelle",
                                 hovertemplate="r = %{x:.4f}: h = %{y:.4f}<extra></extra>"))
    fig.add_hline(y=K.HC, line=dict(color=C.COLOR_UP, dash="dash"), annotation_text="Faltung h_c", annotation_position="bottom right")
    fig.update_xaxes(type="log", title_text="Rampenrate r")
    fig.update_yaxes(title_text="mittleres Kipp-h")
    return _base(fig, 380)
