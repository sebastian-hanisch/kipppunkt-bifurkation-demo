"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Keine Reglerwerte, die Zweierpotenzen sind (kein h = 0,5 oder 0,25, keine Rate 2^-k, kein Zeitschritt 1/64): bei Iterationen in Gleitkomma rechnet der Rechner dann in Dualbrüchen
exakt. Das Handbeispiel h = 3/8 steht nur als Preset (kein Zweierpotenz-Wert, aber ein Dualbruch); die Messreihen laufen auf nicht-dyadischen Werten."""

H_MIN, H_MAX, DEFAULT_H, H_STEP = -0.6, 0.6, 0.375, 0.005
R_OPTIONS = (0.0001, 0.0002, 0.0005, 0.0008, 0.001, 0.002, 0.004, 0.005, 0.01, 0.02, 0.03)
DEFAULT_R = 0.005
SIGMA_MIN, SIGMA_MAX, DEFAULT_SIGMA, SIGMA_STEP = 0.005, 0.05, 0.02, 0.005
RUNS_MIN, RUNS_MAX, DEFAULT_RUNS, RUNS_STEP = 20, 200, 100, 10
SEED_MAX, DEFAULT_SEED = 999999, 35
DTM_OPTIONS = (0.05, 0.1, 0.2, 0.5)
DEFAULT_DTM = 0.1
THR_MIN, THR_MAX, DEFAULT_THR, THR_STEP = 0.3, 0.8, 0.5, 0.05
COND_OPTIONS = ("A", "B", "C", "D")
DEFAULT_COND = "C"
SIGMA_CELL_MIN, SIGMA_CELL_MAX, DEFAULT_SIGMA_CELL, SIGMA_CELL_STEP = 0.0, 0.12, 0.05, 0.01
GRID_R = (0.02, 0.004, 0.0008)           # Raster Tempo gegen Rauschen (feste Zellen)
GRID_SIGMA = (0.0, 0.05, 0.1)
DELAY_RS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
HMAX_LOOP = 0.8                          # Rampengrenze der Schleife

COLOR_UP = "#c0392b"
COLOR_DOWN = "#2F6B65"
COLOR_LINE = "#4c78a8"
COLOR_GREY = "rgba(150,150,150,0.8)"
COLOR_ORANGE = "#c4782b"

_BASE = {"h": DEFAULT_H, "r": DEFAULT_R, "sigma": DEFAULT_SIGMA, "runs": DEFAULT_RUNS, "seed": DEFAULT_SEED, "dtm": DEFAULT_DTM, "thr": DEFAULT_THR,
         "cond": DEFAULT_COND, "sigma_cell": DEFAULT_SIGMA_CELL}
PRESETS = {
    "Handbeispiel h = 0,375": dict(_BASE),
    "Langsame Schleife": {**_BASE, "r": 0.001},
    "Schnelle Schleife": {**_BASE, "r": 0.02},
    "Nahe der Faltung": {**_BASE, "h": 0.365},
    "Anstieg ohne Kippen": {**_BASE, "cond": "B"},
    "Rauschen und langsame Rampe": {**_BASE, "r": 0.0008, "sigma_cell": 0.1},
}
PRESET_HELP = {
    "Handbeispiel h = 0,375": "Das Handbeispiel: bei h = 3/8 gibt es die Gleichgewichte −0,651; −0,5 und 1,151; die Rate auf dem unteren Ast ist 0,2729, die Autokorrelation 0,973.",
    "Langsame Schleife": "Rampenrate 0,001: die Schleife ist nur 0,8077 breit (2 h_c = 0,7698 im Grenzfall unendlich langsamer Rampe).",
    "Schnelle Schleife": "Rampenrate 0,02: das System kippt erst bei h = 0,5127, die Schleife ist 1,025 breit.",
    "Nahe der Faltung": "h = 0,365, knapp unter der Faltung 0,3849: die Rate des unteren Astes ist klein, die Autokorrelation hoch, die lineare Varianzformel versagt.",
    "Anstieg ohne Kippen": "Frühwarnung bei einem Anstieg von 0,5 auf 0,8 h_c: das System kippt nie, trotzdem steigt die Autokorrelation und löst gelegentlich Alarm aus.",
    "Rauschen und langsame Rampe": "Rampenrate 0,0008 mit Rauschen 0,1 im Abschnitt Tempo gegen Rauschen: das System kippt im Mittel schon vor der Faltung.",
}
