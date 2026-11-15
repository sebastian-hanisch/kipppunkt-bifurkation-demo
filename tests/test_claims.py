"""Claims-Tests: jede Zahl der README steht hier als Band (Zufallsläufe: Standardfehler der Anteile bis 3 Punkte bei 200 Läufen, Bänder 8 Punkte).

Die Spalte „Test“ der README-Tabelle nennt die Funktionen dieser Datei. Voraussetzungen überall: Modell x' = x - x^3 + h, Zeitschritt RK4 0,02 (r < 1e-5: 0,05),
Euler-Maruyama mit 5 Schritten je Messung (Messabstand 0,1, Rauschen 0,02, Fenster 200 Messungen), Zufalls-Seed 35 (SplitMix64) wo Zufall gebraucht wird."""

import math

import numpy as np
import pytest

import kpb_constants as C
import kpb_dynamics as K
import kpb_evaluation as E


def _near(value, expected, tol):
    assert abs(value - expected) <= tol, (value, expected, tol)


# --- H1/H2: Faltung, Schleife, Verzögerung -----------------------------------------------------------------------------------------------

def test_loop_widths_and_tip_places_at_the_readme_rates():
    for r, up, width, formula in ((0.02, 0.5125, 1.0249, 1.0567), (0.005, 0.4384, 0.8767, 0.8837), (0.001, 0.4038, 0.8077, 0.8087), (1e-4, 0.3891, 0.7781, 0.7782)):
        lp = K.hysteresis_loop(r)
        _near(lp["up"], up, 6e-5)
        _near(lp["width"], width, 1.2e-4)
        _near(lp["width_formula"], formula, 6e-5)
        assert lp["up"] + lp["down"] == pytest.approx(0.0, abs=1e-12)
    _near(2 * K.HC, 0.7698, 5e-5)


def test_delay_ratios_and_slopes_at_the_six_rates():
    rows = E.delay_table()
    for x, ratio in zip(rows, (0.917, 0.973, 0.9938, 1.0001, 1.0014, 1.0013)):
        _near(x["ratio"], ratio, 6e-4)
    assert 0.99 <= rows[3]["ratio"] <= 1.01 and 0.99 <= rows[4]["ratio"] <= 1.01 and 0.99 <= rows[5]["ratio"] <= 1.01       # r <= 1e-5: Band 0,99 bis 1,01
    all_s, small_s = E.delay_slopes(rows)
    _near(all_s, 0.6601, 0.001)
    _near(small_s, 0.6656, 0.001)


def test_symmetry_of_the_loop_tip_places():
    lp = K.hysteresis_loop(1e-3)
    assert lp["up"] + lp["down"] == pytest.approx(0.0, abs=1e-12)
    _near(lp["up"], 0.4038, 6e-5)


# --- H3: kritisches Verlangsamen ------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("frac,rate,ac_theory,var_boltzmann", [(0.0, 2.0, 0.8187, 1.001e-4), (0.5, 1.3473, 0.8740, 1.487e-4), (0.8, 0.8111, 0.9221, 2.484e-4),
                                                              (0.9, 0.5579, 0.9457, 3.660e-4), (0.95, 0.3863, 0.9621, 5.515e-4)])
def test_rate_autocorrelation_and_variance_at_five_distances(frac, rate, ac_theory, var_boltzmann):
    s = K.stationary_stats(frac * K.HC, 0.02, 0.1, 35)
    _near(s["rate"], rate, 5e-5)
    _near(s["ac_theory"], ac_theory, 5e-5)
    _near(s["var_boltzmann"] / var_boltzmann, 1.0, 1e-3)
    assert s["kept"] == s["nch"] == 100
    _near(s["ac_sim"], ac_theory, 0.005)                                  # Band 0,5 %
    _near(s["var_sim"] / var_boltzmann, 1.0, 0.04)


def test_the_linear_variance_formula_loses_near_the_fold():
    near = K.stationary_stats(0.95 * K.HC, 0.02, 0.1, 35)
    _near(near["var_linear"] / near["var_boltzmann"], 0.939, 0.002)       # 6 % darunter bei 0,95 h_c
    far = K.stationary_stats(0.5 * K.HC, 0.02, 0.1, 35)
    _near(far["var_linear"] / far["var_boltzmann"], 0.998, 0.002)
    hand = K.stationary_stats(0.375, 0.02, 0.1, 35)
    _near(hand["var_linear"] / hand["var_boltzmann"], 0.784, 0.002)       # 21,6 % darunter bei h = 0,375
    _near(hand["var_boltzmann"] / hand["var_linear"] - 1, 0.276, 0.003)   # die Boltzmann-Varianz liegt 27,6 % über der linearen
    assert hand["kept"] < hand["nch"]                                     # nahe der Faltung verlassen Ketten den Topf (90 von 100)
    assert 80 <= hand["kept"] <= 98


# --- H4: Frühwarnung und Fehlalarme ---------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def table():
    return E.warning_table(0.02, 200, 35, 0.5)


def _row(t, model, cond):
    return next(r for r in t["rows"] if r["model"] == model and r["cond"] == cond)


def test_alarm_shares_cubic_model(table):
    a, b, c, d = (_row(table, "cubic", k) for k in "ABCD")
    assert (a["n_tipped"], b["n_tipped"]) == (0, 0) and c["n_tipped"] == 200
    _near(a["alarm"], 0.010, 0.03)                  # 1,0 %
    _near(b["alarm"], 0.195, 0.08)                  # 19,5 %: Fehlalarme bei Anstieg ohne Kippen
    _near(c["alarm"], 0.800, 0.08)                  # 80,0 %
    _near(a["mean"], 0.018, 0.05)
    _near(b["mean"], 0.319, 0.06)
    _near(c["mean"], 0.615, 0.06)
    _near(c["tip"], 1.033, 0.005)                   # mittleres h/h_c beim Kippen
    assert a["n_valid"] == b["n_valid"] == c["n_valid"] == 200


def test_control_d_noise_induced_tipping_is_not_announced(table):
    d = _row(table, "cubic", "D")
    assert 165 <= d["n_tipped"] <= 185 and d["n_tipped"] == 172
    assert d["n_valid"] < d["n_tipped"] and 150 <= d["n_valid"] <= 165          # Läufe mit mindestens 600 Messungen vor dem Kippen
    _near(d["alarm"], 0.057, 0.04)                  # kaum mehr als bei A und weit unter C
    assert d["alarm"] < 0.15 < _row(table, "cubic", "C")["alarm"]
    _near(d["mean"], 0.025, 0.05)


def test_alarm_shares_harvest_model_and_auc(table):
    a, b, c = (_row(table, "harvest", k) for k in "ABC")
    assert c["n_tipped"] == 200 and a["n_tipped"] == b["n_tipped"] == 0
    _near(a["alarm"], 0.005, 0.03)
    _near(b["alarm"], 0.045, 0.06)
    _near(c["alarm"], 0.290, 0.08)
    _near(c["tip"], 1.044, 0.005)
    _near(table["auc_cubic"], 0.941, 0.03)
    _near(table["auc_harvest"], 0.828, 0.04)
    assert table["auc_cubic"] > table["auc_harvest"]
    assert _row(table, "cubic", "C")["alarm"] > _row(table, "harvest", "C")["alarm"] + 0.3   # das kubische Modell trennt besser


def test_alarm_shares_are_stable_over_other_seeds():
    """Rasterzahl auf Stabilität: Zufalls-Seeds 36 und 37 (je 200 Läufe) ergeben dieselben Bänder wie der Seed 35."""
    for seed in (36, 37):
        al = {k: K.alarm_share(K.warning_runs(K.CONDITIONS[k]["a"], K.CONDITIONS[k]["b"], 0.02, 200, seed)["trends"], 0.5) for k in "ABC"}
        assert al["A"] <= 0.03 and 0.09 <= al["B"] <= 0.22 and 0.72 <= al["C"] <= 0.88, (seed, al)
        d = K.warning_runs(0.8, 0.8, 0.12, 200, seed)
        assert 165 <= d["n_tipped"] <= 190 and K.alarm_share(d["trends"], 0.5) <= 0.12, seed


def test_the_condition_labels_name_the_runs():
    assert K.CONDITIONS["A"]["a"] == K.CONDITIONS["A"]["b"] == 0.5 and K.CONDITIONS["B"]["b"] == 0.8 and K.CONDITIONS["C"]["b"] == 1.2
    assert K.CONDITIONS["D"]["sigma"] == 0.12 and K.CONDITIONS["D"]["a"] == K.CONDITIONS["D"]["b"] == 0.8


# --- H5: Tempo gegen Rauschen -----------------------------------------------------------------------------------------------------------------

def test_tempo_against_noise_grid():
    t = E.tempo_analysis(100, 35, 0.0008, 0.1)
    expect = {  # r: ((Mittel, Streuung) für sigma = 0, 0,05, 0,1)
        0.02: ((0.5128, 0.0), (0.511, 0.019), (0.5019, 0.0371)),
        0.004: ((0.4314, 0.0), (0.4256, 0.013), (0.4043, 0.0281)),
        0.0008: ((0.4013, 0.0), (0.3901, 0.011), (0.3607, 0.0205)),
    }
    for i, r in enumerate(C.GRID_R):
        for j, (mean, std) in enumerate(expect[r]):
            cell = t["cells"][i][j]
            _near(cell["mean"], mean, 0.003 if j else 3e-4)
            _near(cell["std"], std, 0.006 if j else 1e-12)
            assert cell["n_tipped"] == (1 if j == 0 else 100)
        _near(t["cells"][i][0]["formula"], {0.02: 0.5283, 0.004: 0.4340, 0.0008: 0.4017}[r], 5e-5)


def test_without_noise_the_tip_is_behind_the_fold_and_only_one_noisy_cell_is_before_it():
    t = E.tempo_analysis(100, 35, 0.0008, 0.1)
    means = np.array([[t["cells"][i][j]["mean"] for j in range(3)] for i in range(3)])
    assert (means[:, 0] > K.HC).all() and means[:, 0].min() == pytest.approx(0.4013, abs=3e-4)          # ohne Rauschen nie vor h_c
    assert int((means[:, 1:] < K.HC).sum()) == 1 and means[2, 2] < K.HC                                   # 1 von 6 Zellen mit Rauschen
    assert t["own"]["mean"] == pytest.approx(t["cells"][2][2]["mean"], abs=1e-12)                          # eigene Zelle = Zelle (0,0008; 0,1)
    before = int((t["own"]["tips"] < K.HC).sum())
    assert 80 <= before <= 95                                                                               # 88 von 100 Läufen kippen vor h_c
    assert E.verdict_tempo(t)[0] == "warning" and "**vor** der Faltung" in E.verdict_tempo(t)[1]


def test_tempo_mean_before_the_fold_holds_for_other_seeds():
    for seed in (36, 37):
        c = K.tempo_noise_cell(0.0008, 0.1, 100, seed)
        assert 0.352 < c["mean"] < K.HC, (seed, c["mean"])
    for seed in (36, 37):
        assert K.tempo_noise_cell(0.0008, 0.05, 100, seed)["mean"] > K.HC - 0.002          # bei σ = 0,05 im Mittel nicht klar davor


# --- Presets und Handbeispiel --------------------------------------------------------------------------------------------------------------------

PRESET_EXPECT = {
    "Handbeispiel h = 0,375": ("success", ["drei Gleichgewichte", "-0,651 (stabil)", "-0,500 (instabil)", "1,151 (stabil)", "Rate 0,2729", "0,2619"]),
    "Nahe der Faltung": ("success", ["drei Gleichgewichte", "h = 0,365"]),
}


@pytest.mark.parametrize("name", list(PRESET_EXPECT))
def test_equilibria_verdict_of_the_presets(name):
    kind, text = E.verdict_equilibria(E.equilibria_analysis(C.PRESETS[name]["h"]))
    assert kind == PRESET_EXPECT[name][0]
    for part in PRESET_EXPECT[name][1]:
        assert part in text, (name, part, text)


@pytest.mark.parametrize("name,up,width", [("Handbeispiel h = 0,375", 0.4384, 0.8767), ("Langsame Schleife", 0.4038, 0.8077), ("Schnelle Schleife", 0.5125, 1.0249)])
def test_loop_verdict_of_the_presets(name, up, width):
    lp = K.hysteresis_loop(C.PRESETS[name]["r"])
    kind, text = E.verdict_loop(lp)
    assert kind == "info" and f"hinauf bei h = {E.de(up, 4)}" in text and f"{E.de(width, 4)} breit" in text


def test_the_alarm_verdict_of_the_preset_without_tipping_reports_false_alarms_not_justified_alarms():
    p = C.PRESETS["Anstieg ohne Kippen"]
    w = E.warning_analysis(p["cond"], p["sigma"], 200, p["seed"])
    kind, text = E.verdict_warning(w, p["thr"])
    assert w["n_tipped"] == 0 and kind == "warning" and "Fehlalarm" in text and "berechtigt" not in text
    w2 = E.warning_analysis("C", p["sigma"], 200, p["seed"])
    assert E.verdict_warning(w2, p["thr"])[0] == "success" and "berechtigt" in E.verdict_warning(w2, p["thr"])[1]


def test_the_verdict_of_the_slow_loop_preset_names_two_h_c_and_the_formula():
    text = E.verdict_loop(K.hysteresis_loop(0.001))[1]
    assert "2·h_c = 0,7698" in text and "0,8087" in text and "0,1 % darüber" in text


def test_the_delay_verdict_names_ratios_and_slopes():
    rows = E.delay_table()
    text = E.verdict_delay(rows, K.delay_measurement(0.005))[1]
    assert "Verhältnis **0,9392**" in text
    assert "0,917 bei r = 0,01" in text and "1,0001 bei r = 10⁻⁵" in text and "0,6601" in text and "0,6656" in text


def test_erntemodell_x0_and_tip_criterion():
    """x0 = (1 + sqrt(1 - 0,5)) / 2 = 0,8536 bei a = 0,5; Kippen bei x < 0,1; h_max = 0,25."""
    assert 0.5 * (1.0 + math.sqrt(1.0 - 0.5)) == pytest.approx(0.853553, abs=1e-6)
    assert K.HARVEST_HMAX == 0.25 and K.HARVEST_TIPPED == 0.1
    assert K.harvest_drift(0.853553, 0.125) == pytest.approx(0.0, abs=1e-5)          # Gleichgewicht bei h = 0,5 h_max


def test_rates_quoted_for_the_two_models_before_the_alarm_comparison():
    """Kubisch: Rate 1,35 bei 0,5 h_c und 0,81 bei 0,8 h_c; Ernte: r sqrt(1 - h/h_max) = 0,71 bei 0,5 und 0,45 bei 0,8 h_max."""
    _near(K.rate(K.lower_root(0.5 * K.HC)), 1.3473, 5e-5)
    _near(K.rate(K.lower_root(0.8 * K.HC)), 0.8111, 5e-5)
    for frac, rate in ((0.5, 0.7071), (0.8, 0.4472)):
        x = 0.5 * (1.0 + math.sqrt(1.0 - frac))                               # Gleichgewicht des Erntemodells
        _near(1.0 - 2.0 * x, -math.sqrt(1.0 - frac), 1e-12)                  # f'(x) = 1 - 2x
        _near(math.sqrt(1.0 - frac), rate, 5e-5)
