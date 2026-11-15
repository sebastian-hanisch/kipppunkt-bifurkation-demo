"""Kern: Normalform der Faltung x' = x - x^3 + h, Rampen (Hysterese, Verzögerung), Rauschen (kritisches Verlangsamen, Frühwarnung, Tempo gegen Rauschen).

Modell: x' = f(x, h) = x - x^3 + h; zwei stabile Gleichgewichte für |h| < h_c = 2/(3 sqrt 3) = 0,3849, bei +-h_c fällt je eines mit dem Sattel zusammen (Faltung).
Symmetrie: f(-x, -h) = -f(x, h). Relaxationsrate eines Gleichgewichts: rate = 3 x^2 - 1 (positiv = stabil).
Rampe h(t) = h_start + Richtung * r * t, Integration mit eigenem RK4 (Rampe) und Euler-Maruyama (Rauschen); „Kippen“ heißt: x kreuzt 0.
Verzögerung des Kippens: delta = h_Kipp - h_c = C r^(2/3) mit C = |z1| / 3^(1/6) und z1 = erste Nullstelle der Airy-Funktion Ai (eigene Herleitung aus der Normalform
s' = r t + sqrt(3) s^2 mit s = x + 1/sqrt 3, siehe README; keine Quelle gelesen).
Zufall nur über SplitMix64 (`kpb_rng`); die Messlogik ist neu geschrieben (die Wegwerf-Skripte der Planung sind nicht importiert)."""

import math

import numpy as np
from scipy.special import ai_zeros

from kpb_rng import normal_array

HC = 2.0 / (3.0 * math.sqrt(3.0))                      # Faltung bei h = +-h_c
AIRY_C = float(-ai_zeros(1)[0][0] / 3.0 ** (1.0 / 6.0))  # 1,946903
DT_RAMP = 0.02                                          # Zeitschritt (nicht 1/64); langsame Rampen (r < 1e-5) mit 0,05
DT_SLOW = 0.05
START_ABSTAND = 12.0                                    # Start bei h_c - 12 r^(2/3) auf dem Gleichgewicht (dort folgt die Bahn ihm bis auf O(r))


def f(x, h):
    return x - x ** 3 + h


def rate(x):
    """Relaxationsrate des Gleichgewichts x: -f'(x) = 3 x^2 - 1 (positiv: stabil, negativ: instabil)."""
    return 3.0 * x * x - 1.0


def discriminant(h: float) -> float:
    """Diskriminante von x^3 - x - h: 4 - 27 h^2; positiv heißt drei reelle Gleichgewichte, negativ eines."""
    return 4.0 - 27.0 * h * h


def real_roots(h: float) -> list[float]:
    """Reelle Gleichgewichte von x - x^3 + h = 0, aufsteigend."""
    rts = np.roots([1.0, 0.0, -1.0, -h])
    return sorted(float(v.real) for v in rts if abs(v.imag) < 1e-7)


def lower_root(h: float) -> float:
    return real_roots(h)[0]


def upper_root(h: float) -> float:
    return real_roots(h)[-1]


def well_root(h: float) -> float:
    """Gleichgewicht des Topfes, der bei wachsendem |h| zuerst verschwindet: unten für h >= 0, oben für h < 0."""
    return lower_root(h) if h >= 0 else upper_root(h)


def rate_approx(h: float) -> float:
    """Näherung der Rate nahe der Faltung: sqrt(8/3) sqrt(1 - |h|/h_c) (zu klein: 18 % bei h = 0, 5 % bei 0,95 h_c)."""
    return math.sqrt(8.0 / 3.0) * math.sqrt(max(0.0, 1.0 - abs(h) / HC))


def potential(x, h):
    """Potential V mit x' = -V'(x): V = -(x^2/2 - x^4/4 + h x)."""
    return -(x ** 2 / 2.0 - x ** 4 / 4.0 + h * x)


def equilibrium_branches(hs):
    """Gleichgewichte über einem Raster von h: (h-Werte, x-Werte, Stabilität) als flache Felder für die S-Kurve."""
    H, X, S = [], [], []
    for h in hs:
        for x in real_roots(float(h)):
            H.append(float(h))
            X.append(x)
            S.append(rate(x) > 0)
    return np.array(H), np.array(X), np.array(S)


def dt_for(r: float) -> float:
    return DT_RAMP if r >= 1e-5 else DT_SLOW


def delay_formula(r: float) -> float:
    """Verzögerung des Kippens nach der Faltung: C r^(2/3)."""
    return AIRY_C * r ** (2.0 / 3.0)


def rk4(x: float, h: float, dt: float) -> float:
    k1 = x - x ** 3 + h
    x2 = x + 0.5 * dt * k1
    k2 = x2 - x2 ** 3 + h
    x3 = x + 0.5 * dt * k2
    k3 = x3 - x3 ** 3 + h
    x4 = x + dt * k3
    k4 = x4 - x4 ** 3 + h
    return x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def ramp_tip(r: float, h_start: float, x_start: float, direction: int, dt: float, h_stop: float, trace_every: int = 0):
    """Rampe h = h_start + direction * r * t ab dem Gleichgewicht x_start; Kippen = x überschreitet 0 (gegen die Startseite).

    Gibt (h beim Kippen oder None, Spur (h, x) oder None) zurück. h steht über den Schritt bei seinem Mittelwert. Die Kippzeit wird innerhalb des Schritts linear
    zwischen den Werten x vor und nach dem Nulldurchgang interpoliert (Fehler O(dt^2) statt O(r dt)). None, wenn h_stop vorher erreicht wird."""
    x = x_start
    t = 0.0
    s0 = math.copysign(1.0, x_start)
    hs, xs = ([h_start], [x_start]) if trace_every else (None, None)
    n = 0
    while True:
        x_prev = x
        x = rk4(x, h_start + direction * r * (t + 0.5 * dt), dt)
        t += dt
        n += 1
        h_now = h_start + direction * r * t
        if trace_every and n % trace_every == 0:
            hs.append(h_now)
            xs.append(x)
        if x * s0 < 0.0:
            frac = abs(x_prev) / (abs(x_prev) + abs(x))          # Anteil des Schritts bis zum Nulldurchgang
            return h_start + direction * r * (t - dt + frac * dt), ((np.array(hs), np.array(xs)) if trace_every else None)
        if direction * (h_now - h_stop) > 0:
            return None, ((np.array(hs), np.array(xs)) if trace_every else None)


def hysteresis_loop(r: float, hmax: float = 0.8, trace_every: int = 25) -> dict:
    """Schleife: Rampe hinauf (unteres Gleichgewicht) und hinunter (oberes) mit Rate r, Start je bei h_c -+ 12 r^(2/3), begrenzt auf +-hmax.

    Breite = Kipp-h hinauf minus Kipp-h hinunter; Vergleich mit 2 h_c (r -> 0) und 2 h_c + 2 C r^(2/3) (Formel)."""
    w = r ** (2.0 / 3.0)
    dt = dt_for(r)
    ha = max(-hmax, HC - START_ABSTAND * w)
    hb = min(hmax, -HC + START_ABSTAND * w)
    up, tr_up = ramp_tip(r, ha, lower_root(ha), +1, dt, hmax, trace_every)
    down, tr_down = ramp_tip(r, hb, upper_root(hb), -1, dt, -hmax, trace_every)
    width = None if up is None or down is None else up - down
    return {"r": r, "up": up, "down": down, "width": width, "width_formula": 2 * HC + 2 * delay_formula(r), "trace_up": tr_up, "trace_down": tr_down, "hmax": hmax}


def delay_measurement(r: float) -> dict:
    """Verzögerung delta = h_Kipp - h_c beim Hinauffahren ab h_c - 12 r^(2/3); Verhältnis zur Formel C r^(2/3)."""
    w = r ** (2.0 / 3.0)
    hs = HC - START_ABSTAND * w
    tip, _ = ramp_tip(r, hs, lower_root(hs), +1, dt_for(r), HC + 100.0 * w)
    delta = None if tip is None else tip - HC
    return {"r": r, "delta": delta, "formula": delay_formula(r), "ratio": None if delta is None else delta / delay_formula(r)}


# --- Rauschen: Euler-Maruyama, vektorisiert über die Ketten ----------------------------------------------------------------------------

def _normals(seed: int, start: int, count: int, nch: int, sub: int) -> np.ndarray:
    """Standardnormalwerte für die Messschritte start ... start + count - 1: Form (count, sub, nch); Schritt i nutzt die Folge ab 2 i sub nch."""
    return normal_array(seed, count * sub * nch, offset=2 * start * sub * nch).reshape(count, sub, nch)


def simulate_chains(h_of_step, sigma: float, x0, dtm: float, sub: int, n_meas: int, seed: int, nch: int, stop_when_tipped: bool = False, drift=f, tipped=lambda x: x > 0.0):
    """Euler-Maruyama für nch Ketten: n_meas Messungen im Abstand dtm, je sub Schritte der Größe dtm/sub; `h_of_step(i)` = h während der Messung i.

    Gibt (xs, tip_index) zurück: xs hat die Form (n_meas, nch) (nach dem Abbruch der Rest mit NaN), tip_index[j] = erste Messung, in der `tipped(x)` gilt (Standard x > 0; n_meas, wenn nie).
    `drift(x, h)`: Standard x - x^3 + h.
    `stop_when_tipped`: bricht ab, sobald alle Ketten gekippt sind."""
    dts = dtm / sub
    sq = sigma * math.sqrt(dts)
    x = np.full(nch, float(x0)) if np.isscalar(x0) else np.array(x0, float)
    xs = np.full((n_meas, nch), np.nan)
    tip = np.full(nch, n_meas)
    alive = np.ones(nch, bool)
    chunk = 100
    for c0 in range(0, n_meas, chunk):
        cnt = min(chunk, n_meas - c0)
        Z = _normals(seed, c0, cnt, nch, sub) if sigma > 0 else None
        for k in range(cnt):
            i = c0 + k
            h = h_of_step(i)
            for s in range(sub):
                x = np.where(alive, x + drift(x, h) * dts + (sq * Z[k, s] if sigma > 0 else 0.0), x)      # gekippte Ketten bleiben stehen (kein Überlauf im Erntemodell)
            xs[i] = x
            newly = alive & tipped(x)
            tip[newly] = i
            alive &= ~newly
        if stop_when_tipped and not alive.any():
            break
    return xs, tip


def boltzmann_variance(h: float, sigma: float) -> float:
    """Varianz im Topf nach der stationären Dichte exp(-2 V / sigma^2) (Quadratur, unabhängig vom linearen Näherungsansatz sigma^2 / (2 Rate))."""
    xl = well_root(h)
    mid = real_roots(h)[1]
    if h >= 0:
        g = np.linspace(xl - 1.0, mid, 200001)
    else:
        g = np.linspace(mid, xl + 1.0, 200001)
    V = potential(g, h)
    w = np.exp(-2.0 * (V - V.min()) / sigma ** 2)
    w /= np.trapezoid(w, g)
    m1 = np.trapezoid(g * w, g)
    return float(np.trapezoid((g - m1) ** 2 * w, g))


def stationary_stats(h: float, sigma: float, dtm: float, seed: int, nch: int = 100, steps_total: int = 30000, dts: float = 0.01) -> dict:
    """Kritisches Verlangsamen: Autokorrelation (Lag = Messabstand dtm) und Varianz im Topf, Theorie gegen Simulation.

    Ketten, die den Topf verlassen haben (jenseits des Sattels), werden ausgeschlossen und gezählt. h muss zwei Gleichgewichte haben (|h| < h_c)."""
    xl = well_root(h)
    mid = real_roots(h)[1]
    sub = max(1, int(round(dtm / dts)))
    n_meas = max(200, steps_total // sub)
    xs, _ = simulate_chains(lambda i: h, sigma, xl, dtm, sub, n_meas, seed, nch, tipped=(lambda x: x > 0.0) if h >= 0 else (lambda x: x < 0.0))
    keep = np.all(xs < mid, axis=0) if h >= 0 else np.all(xs > mid, axis=0)
    z = xs[n_meas // 5:, keep]
    if z.shape[1] > 0:
        a = z - z.mean(axis=0)
        ac = float(np.mean(np.sum(a[:-1] * a[1:], axis=0) / np.sum(a * a, axis=0)))
        var_sim = float(np.mean(np.var(z, axis=0)))
    else:                                                  # alle Ketten haben den Topf verlassen: keine Simulationswerte
        ac = var_sim = float("nan")
    rt = rate(xl)
    return {"h": h, "sigma": sigma, "dtm": dtm, "rate": rt, "rate_approx": rate_approx(h), "ac_theory": math.exp(-rt * dtm), "ac_sim": ac,
            "var_boltzmann": boltzmann_variance(h, sigma), "var_linear": sigma ** 2 / (2.0 * rt), "var_sim": var_sim,
            "kept": int(keep.sum()), "nch": nch, "sample": xs[:, 0].copy(), "n_meas": n_meas}


# --- Frühwarnung ------------------------------------------------------------------------------------------------------------------------

WINDOW = 200
WINDOW_STEP = 50
MIN_LEN = 600


def window_ac1(x: np.ndarray):
    """Autokorrelation (Lag 1) je Fenster (Länge 200, Schritt 50, linearer Trend entfernt) einer Zeitreihe; Fensteranfänge zurück."""
    starts = list(range(0, len(x) - WINDOW, WINDOW_STEP))
    idx = np.arange(WINDOW)
    A = np.vstack([idx, np.ones(WINDOW)]).T
    pinv = np.linalg.pinv(A)
    acs = []
    for st in starts:
        w = x[st:st + WINDOW]
        w = w - A @ (pinv @ w)
        acs.append(float(np.sum(w[:-1] * w[1:]) / np.sum(w * w)))
    return np.array(starts), np.array(acs)


def trend_measure(x: np.ndarray) -> float:
    """Trendmaß: Korrelation (Pearson) zwischen Fensternummer und Fenster-Autokorrelation; NaN, wenn die Reihe unter 600 Messungen kurz ist."""
    if len(x) < MIN_LEN:
        return float("nan")
    _, acs = window_ac1(x)
    if len(acs) < 6:
        return float("nan")
    k = np.arange(len(acs), dtype=float)
    kc, ac = k - k.mean(), acs - acs.mean()
    den = math.sqrt(float(np.sum(kc ** 2)) * float(np.sum(ac ** 2)))
    return float("nan") if den == 0 else float(kc @ ac / den)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Wahrscheinlichkeit, dass ein Wert aus pos größer ist als einer aus neg (Gleichstand zählt halb)."""
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    return float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))


CONDITIONS = {
    "A": {"label": "A: konstantes h = 0,5 h_c", "a": 0.5, "b": 0.5},
    "B": {"label": "B: Anstieg 0,5 → 0,8 h_c (kein Kippen)", "a": 0.5, "b": 0.8},
    "C": {"label": "C: Anstieg 0,5 → 1,2 h_c (kippt)", "a": 0.5, "b": 1.2},
    "D": {"label": "D: konstantes h = 0,8 h_c, Rauschen 0,12 (Kontrolle: Kippen durch Rauschen)", "a": 0.8, "b": 0.8, "sigma": 0.12},
}


HARVEST_HMAX = 0.25      # Erntemodell x' = x (1 - x) - h (r = K = 1): Faltung bei h_max = r K / 4
HARVEST_TIPPED = 0.1     # Kippen im Erntemodell: x < 0,1


def harvest_drift(x, h):
    return x * (1.0 - x) - h


def warning_runs(a: float, b: float, sigma: float, n_runs: int, seed: int, n_meas: int = 4000, dtm: float = 0.1, sub: int = 5, model: str = "cubic") -> dict:
    """Läufe mit h/h_c (kubisch) bzw. h/h_max (Erntemodell) linear von a nach b über n_meas Messungen (Abstand dtm, je sub Schritte), Start auf dem Gleichgewicht bei a.

    Trendmaß je Lauf aus den Messungen bis zum Kippen (kubisch x > 0, Ernte x < 0,1); Läufe mit weniger als 600 Messungen bis dahin bekommen NaN.
    Das Erntemodell ist das zweite Vehikel für den Vergleich (r = K = 1, Zufalls-Seed und Rauschen wie im kubischen Modell)."""
    if model == "harvest":
        h_of = lambda i: (a + (b - a) * i / n_meas) * HARVEST_HMAX  # noqa: E731
        x0 = 0.5 * (1.0 + math.sqrt(1.0 - a))
        xs, tip = simulate_chains(h_of, sigma, x0, dtm, sub, n_meas, seed, n_runs, stop_when_tipped=True, drift=harvest_drift, tipped=lambda x: x < HARVEST_TIPPED)
    else:
        h_of = lambda i: (a + (b - a) * i / n_meas) * HC  # noqa: E731
        xs, tip = simulate_chains(h_of, sigma, lower_root(a * HC), dtm, sub, n_meas, seed, n_runs, stop_when_tipped=True)
    trends = np.array([trend_measure(xs[:tip[j], j]) for j in range(n_runs)])
    tipped = tip < n_meas
    tip_frac = np.array([a + (b - a) * tip[j] / n_meas for j in range(n_runs) if tipped[j]])
    return {"a": a, "b": b, "sigma": sigma, "n_runs": n_runs, "n_tipped": int(tipped.sum()), "trends": trends, "tip_frac": tip_frac, "tip": tip, "n_meas": n_meas, "model": model,
            "mean_tip_frac": float(tip_frac.mean()) if tip_frac.size else float("nan"), "sample": xs[:tip[0], 0].copy() if tip[0] > 0 else xs[:1, 0].copy()}


def alarm_share(trends: np.ndarray, threshold: float) -> float:
    ok = trends[~np.isnan(trends)]
    return float("nan") if ok.size == 0 else float(np.mean(ok > threshold))


# --- Tempo gegen Rauschen -----------------------------------------------------------------------------------------------------------------

def tempo_noise_cell(r: float, sigma: float, n_runs: int, seed: int, dt: float = DT_RAMP) -> dict:
    """Start unten bei h = 0 auf dem Gleichgewicht, Anstieg h = r t, Kippen = x > 0 (Euler-Maruyama, n_runs Ketten; sigma = 0: eine Kette).

    Mittleres Kipp-h, Streuung, Zahl der gekippten Läufe; Vergleich mit h_c + C r^(2/3) (Formel ohne Rauschen)."""
    nch = 1 if sigma == 0 else n_runs
    x = np.full(nch, lower_root(0.0))
    tip = np.full(nch, np.nan)
    sq = sigma * math.sqrt(dt)
    nsteps = int(1.0 / r / dt)
    chunk = 400
    t = 0.0
    done = False
    for c0 in range(0, nsteps, chunk):
        cnt = min(chunk, nsteps - c0)
        Z = normal_array(seed, cnt * nch, offset=2 * c0 * nch).reshape(cnt, nch) if sigma > 0 else None
        for k in range(cnt):
            h = r * (t + 0.5 * dt)
            x = x + (x - x ** 3 + h) * dt + (sq * Z[k] if sigma > 0 else 0.0)
            t += dt
            newly = np.isnan(tip) & (x > 0.0)
            tip[newly] = r * t
        if not np.isnan(tip).any():
            done = True
            break
    ok = ~np.isnan(tip)
    return {"r": r, "sigma": sigma, "n_runs": nch, "n_tipped": int(ok.sum()), "tips": tip, "mean": float(np.nanmean(tip)) if ok.any() else float("nan"),
            "std": float(np.nanstd(tip)) if ok.any() else float("nan"), "formula": HC + delay_formula(r), "all_tipped": done}
