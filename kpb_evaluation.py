"""Auswertungen der Abschnitte der App und ihre Verdikte (reine Funktionen: (Art, Text)).

Jede Auswertung ruft nur den Kern `kpb_dynamics` auf; die App legt sie hinter `st.cache_data`. Zahlen in den Verdikten kommen aus der Rechnung."""

import math
from dataclasses import dataclass

import numpy as np

import kpb_constants as C
import kpb_dynamics as K


@dataclass(frozen=True)
class Settings:
    h: float = C.DEFAULT_H
    r: float = C.DEFAULT_R
    sigma: float = C.DEFAULT_SIGMA
    runs: int = C.DEFAULT_RUNS
    seed: int = C.DEFAULT_SEED
    dtm: float = C.DEFAULT_DTM
    thr: float = C.DEFAULT_THR
    cond: str = C.DEFAULT_COND
    sigma_cell: float = C.DEFAULT_SIGMA_CELL


def de(x: float, nd: int = 2, sign: bool = False) -> str:
    """Zahl mit Dezimalkomma (Anzeige)."""
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace(".", ",")


def pct(x: float, nd: int = 1) -> str:
    return de(100.0 * x, nd) + " %"


def sci(x: float, nd: int = 2) -> str:
    """Zahl in Zehnerpotenz-Schreibweise mit Komma, z. B. 5,5·10⁻⁴."""
    m, e = f"{x:.{nd}e}".split("e")
    sup = str.maketrans("-+0123456789", "⁻⁺⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{m.replace('.', ',')}·10{str(int(e)).translate(sup)}"


# --- Gleichgewichte und Faltung ----------------------------------------------------------------------------------------------------------

def s_curve(n: int = 241) -> dict:
    """Gleichgewichte über h in [-0,6; 0,6] (S-Kurve): stabile und instabile Punkte."""
    hs = np.linspace(C.H_MIN, C.H_MAX, n)
    H, X, S = K.equilibrium_branches(hs)
    return {"h": H, "x": X, "stable": S}


def equilibria_analysis(h: float) -> dict:
    roots = K.real_roots(h)
    return {"h": h, "roots": roots, "rates": [K.rate(x) for x in roots], "n": len(roots), "disc": K.discriminant(h), "well": K.well_root(h) if len(roots) == 3 else None,
            "xs": np.linspace(-1.7, 1.7, 341), "V": K.potential(np.linspace(-1.7, 1.7, 341), h)}


def verdict_equilibria(a: dict) -> tuple[str, str]:
    h, roots, rates = a["h"], a["roots"], a["rates"]
    hc = K.HC
    if abs(abs(h) - hc) < 1e-9:
        return "warning", f"h = {de(h, 4)} liegt genau auf der Faltung: Doppelwurzel, die Rate dort ist 0; die Faltung liegt bei ±{de(hc, 4)}."
    if a["n"] == 1:
        return "info", (f"Bei h = {de(h, 3)} gibt es nur **ein Gleichgewicht** (x = {de(roots[0], 3)}, stabil, Rate {de(rates[0], 3)}): h liegt hinter der Faltung bei ±{de(hc, 4)}, "
                        "der zweite Topf ist verschwunden.")
    parts = ", ".join(f"{de(x, 3)} ({'stabil' if rt > 0 else 'instabil'})" for x, rt in zip(roots, rates))
    well = a["well"]
    rate_w = K.rate(well)
    approx = K.rate_approx(h)
    return "success", (f"Bei h = {de(h, 3)} gibt es **drei Gleichgewichte**: {parts}; zwei sind stabil. Der Topf nahe der Faltung ({'unten' if h >= 0 else 'oben'}) hat die Rate {de(rate_w, 4)} "
                       f"(Näherung √(8/3)·√(1 − |h|/h_c) = {de(approx, 4)}, {pct(1 - approx / rate_w, 0)} darunter). Die Faltung liegt bei ±{de(hc, 4)}, Abstand {de(hc - abs(h), 4)}.")


# --- Hysterese-Schleife ----------------------------------------------------------------------------------------------------------------------

def verdict_loop(a: dict) -> tuple[str, str]:
    r = a["r"]
    if a["up"] is None or a["down"] is None:
        return "info", f"Rampenrate r = {de(r, 4)}: Die Rampe erreicht ihre Grenze ±{de(a['hmax'], 1)} ohne zu kippen."
    delay = a["up"] - K.HC
    f = a["width_formula"]
    rel = f / a["width"] - 1.0
    return "info", (f"Rampenrate r = {de(r, 4)}: Das System kippt hinauf bei h = {de(a['up'], 4)} und hinunter bei h = {de(a['down'], 4)}; die **Schleife ist {de(a['width'], 4)} breit** "
                    f"(2·h_c = {de(2 * K.HC, 4)} bei unendlich langsamer Rampe), das Kippen kommt {de(delay, 4)} nach der Faltung. Die Formel 2·h_c + 2·C·r^(2/3) gibt {de(f, 4)} "
                    f"({pct(abs(rel))} {'darüber' if rel > 0 else 'darunter'}).")


# --- Verzögerung -----------------------------------------------------------------------------------------------------------------------------

def delay_table() -> list[dict]:
    return [K.delay_measurement(r) for r in C.DELAY_RS]


def delay_slopes(rows: list[dict]) -> tuple[float, float]:
    """Steigung im Log-Log-Bild: alle sechs Werte und r <= 1e-4 (Theorie 2/3)."""
    rs, ds = np.log([x["r"] for x in rows]), np.log([x["delta"] for x in rows])
    return float(np.polyfit(rs, ds, 1)[0]), float(np.polyfit(rs[2:], ds[2:], 1)[0])


def verdict_delay(rows: list[dict], now: dict) -> tuple[str, str]:
    all_s, small_s = delay_slopes(rows)
    first, last = rows[0], rows[-1]
    mid = next(x for x in rows if x["r"] == 1e-5)
    return "info", (f"Bei r = {de(now['r'], 4)}: gemessene Verzögerung {sci(now['delta'], 3)}, Formel C·r^(2/3) = {sci(now['formula'], 3)}, Verhältnis **{de(now['ratio'], 4)}**. "
                    f"Über sechs Raten: Verhältnis {de(first['ratio'], 3)} bei r = 0,01, {de(mid['ratio'], 4)} bei r = 10⁻⁵ und {de(last['ratio'], 4)} bei r = 10⁻⁷; Steigung im Log-Log-Bild "
                    f"{de(all_s, 4)} (alle sechs) und {de(small_s, 4)} (r ≤ 10⁻⁴), Erwartung 2/3. Die Formel gilt für kleine r.")


# --- Kritisches Verlangsamen -----------------------------------------------------------------------------------------------------------------

def slowing_analysis(h: float, sigma: float, dtm: float, seed: int, nch: int = 100) -> dict | None:
    if abs(h) >= K.HC - 1e-9:
        return None
    return K.stationary_stats(h, sigma, dtm, seed, nch=nch)


def verdict_slowing(a: dict | None, h: float) -> tuple[str, str]:
    if a is None:
        return "info", f"Bei h = {de(h, 3)} gibt es keinen zweiten Topf (|h| ≥ h_c = {de(K.HC, 4)}): hier gibt es kein kritisches Verlangsamen vor einem Kippen."
    if a["kept"] == 0:
        return "warning", (f"Alle {a['nch']} Ketten haben den Topf verlassen: das Rauschen {de(a['sigma'], 3)} ist bei diesem Abstand zur Faltung zu stark, die Simulation hat keine Werte. "
                           f"Theorie: Rate {de(a['rate'], 4)}, Autokorrelation e^(−Rate·Abstand) = {de(a['ac_theory'], 4)}, Varianz Boltzmann {sci(a['var_boltzmann'])}, linear {sci(a['var_linear'])}.")
    rel = a["var_linear"] / a["var_boltzmann"] - 1.0
    kept = "" if a["kept"] == a["nch"] else f" Nur {a['kept']} von {a['nch']} Ketten sind im Topf geblieben (die übrigen ausgeschlossen)."
    return "info", (f"Rate {de(a['rate'], 4)}; Autokorrelation im Abstand {de(a['dtm'], 2)}: Theorie e^(−Rate·Abstand) = {de(a['ac_theory'], 4)}, Simulation {de(a['ac_sim'], 4)}. "
                    f"Varianz: Boltzmann-Quadratur {sci(a['var_boltzmann'])}, lineare Formel σ²/(2·Rate) {sci(a['var_linear'])} ({pct(abs(rel))} {'darüber' if rel > 0 else 'darunter'}), "
                    f"Simulation {sci(a['var_sim'])} ({a['nch']} Ketten).{kept}")


# --- Frühwarnung --------------------------------------------------------------------------------------------------------------------------------

def warning_analysis(cond: str, sigma: float, runs: int, seed: int) -> dict:
    c = K.CONDITIONS[cond]
    sg = c.get("sigma", sigma)
    w = K.warning_runs(c["a"], c["b"], sg, runs, seed)
    w["cond"] = cond
    w["label"] = c["label"]
    w["sigma_used"] = sg
    return w


def verdict_warning(w: dict, thr: float) -> tuple[str, str]:
    tr = w["trends"]
    ok = tr[~np.isnan(tr)]
    n_ok = int(ok.size)
    n_al = int((ok > thr).sum())
    share = (n_al / n_ok) if n_ok else float("nan")
    mean = float(ok.mean()) if n_ok else float("nan")
    voraus = f"in diesem Modell, bei Rauschen {de(w['sigma_used'], 2)}, Fenster 200 Messungen"
    zahl = f"Alarm (Trendmaß über {de(thr, 2)}) in **{n_al} von {n_ok} Läufen ({pct(share)})**, mittleres Trendmaß {de(mean, 2, True)}"
    cond = w["cond"]
    if cond == "A":
        return "info", f"{w['label']}: {zahl}; {w['n_tipped']} von {w['n_runs']} Läufen kippen. Das ist die Fehlalarmrate des Maßes ohne Trend ({voraus})."
    if cond == "B":
        return "warning", (f"{w['label']}: {zahl}; {w['n_tipped']} von {w['n_runs']} Läufen kippen. Jeder Alarm ist hier ein **Fehlalarm**: das System kommt dem Kipppunkt zwar näher, kippt aber nie "
                           f"({voraus}).")
    if cond == "C":
        tip = de(w["mean_tip_frac"], 3) if w["n_tipped"] else "-"
        return "success", (f"{w['label']}: {zahl}; {w['n_tipped']} von {w['n_runs']} Läufen kippen, im Mittel bei h/h_c = {tip}. Die Alarme sind hier berechtigt ({voraus}).")
    return "info", (f"{w['label']}: {w['n_tipped']} von {w['n_runs']} Läufen kippen allein durch das Rauschen; {zahl} (Läufe mit mindestens 600 Messungen vor dem Kippen). "
                    f"Der Trend kündigt dieses Kippen nicht an ({voraus}).")


def warning_table(sigma: float, runs: int, seed: int, thr: float) -> dict:
    """Alle Bedingungen im kubischen Modell, A bis C im Erntemodell (zweites Vehikel), AUC (Kipp-Läufe gegen Läufe ohne Kippen)."""
    rows = []
    store = {}
    for model in ("cubic", "harvest"):
        for key, c in K.CONDITIONS.items():
            if model == "harvest" and key == "D":
                continue
            sg = c.get("sigma", sigma)
            w = K.warning_runs(c["a"], c["b"], sg, runs, seed, model=model)
            store[(model, key)] = w
            tr = w["trends"]
            ok = tr[~np.isnan(tr)]
            rows.append({"model": model, "cond": key, "label": c["label"], "n_tipped": w["n_tipped"], "n_runs": runs, "n_valid": int(ok.size), "mean": float(ok.mean()) if ok.size else float("nan"),
                         "alarm": K.alarm_share(tr, thr), "tip": w["mean_tip_frac"]})
    out = {"rows": rows, "thr": thr, "sigma": sigma, "runs": runs}
    for model in ("cubic", "harvest"):
        neg = np.concatenate([store[(model, "A")]["trends"], store[(model, "B")]["trends"]])
        out["auc_" + model] = K.auc(store[(model, "C")]["trends"], neg)
    return out


def verdict_table(t: dict) -> tuple[str, str]:
    g = {(r["model"], r["cond"]): r for r in t["rows"]}
    cu = " / ".join(pct(g[("cubic", k)]["alarm"], 1) for k in "ABC")
    ha = " / ".join(pct(g[("harvest", k)]["alarm"], 1) for k in "ABC")
    return "info", (f"Alarmanteil bei den Bedingungen A / B / C (Schwelle {de(t['thr'], 2)}, {t['runs']} Läufe, Rauschen {de(t['sigma'], 2)}): kubisches Modell {cu}, Erntemodell {ha}. "
                    f"Trennung ohne Schwelle (Wahrscheinlichkeit, dass ein Kipp-Lauf ein größeres Trendmaß hat als ein Lauf ohne Kippen): {de(t['auc_cubic'], 3)} kubisch, {de(t['auc_harvest'], 3)} Ernte. "
                    "Die Güte der Warnung hängt am Modell, am Rauschen und am Fenster.")


# --- Tempo gegen Rauschen ----------------------------------------------------------------------------------------------------------------

def tempo_analysis(runs: int, seed: int, r: float, sigma_cell: float) -> dict:
    cells = [[K.tempo_noise_cell(rr, ss, runs, seed) for ss in C.GRID_SIGMA] for rr in C.GRID_R]
    own = K.tempo_noise_cell(r, sigma_cell, runs, seed)
    return {"cells": cells, "own": own, "runs": runs}


def verdict_tempo(t: dict) -> tuple[str, str]:
    o = t["own"]
    hc = K.HC
    if o["n_tipped"] == 0:
        return "info", f"Bei r = {de(o['r'], 4)}, σ = {de(o['sigma'], 2)}: kein Lauf kippt bis h = 1."
    tips = o["tips"][~np.isnan(o["tips"])]
    before = int((tips < hc).sum())
    ort = "**vor** der Faltung" if o["mean"] < hc else "hinter der Faltung"
    return "warning" if o["mean"] < hc else "info", (f"Bei r = {de(o['r'], 4)}, σ = {de(o['sigma'], 2)} kippt das System im Mittel bei h = {de(o['mean'], 4)} (Streuung {de(o['std'], 4)}, {o['n_tipped']} von {o['n_runs']} Läufen): "
                                                      f"{ort} bei h_c = {de(hc, 4)}; {before} von {tips.size} Läufen kippen vor h_c. Ohne Rauschen nach der Formel h_c + C·r^(2/3) bei {de(o['formula'], 4)}. "
                                                      "Schnelle Rampe: spät; Rauschen: früh.")
