"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zeilen, nicht Zufalls-Seeds."""

import numpy as np
import pytest

import kpb_dynamics as K
import kpb_evaluation as ev


def test_number_formatting_uses_decimal_commas_and_powers_of_ten():
    assert ev.de(1.5, 2) == "1,50" and ev.de(-0.05, 4, True) == "-0,0500" and ev.pct(0.508, 1) == "50,8 %"
    assert ev.sci(5.515e-4) == "5,52·10⁻⁴" and ev.sci(1.0e-5, 0) == "1·10⁻⁵" and ev.sci(4.2e-5, 3) == "4,200·10⁻⁵"


def test_verdict_equilibria_three_one_and_fold():
    kind, text = ev.verdict_equilibria(ev.equilibria_analysis(0.375))
    assert kind == "success" and "h = 0,375" in text and "drei Gleichgewichte" in text and "-0,651 (stabil)" in text and "-0,500 (instabil)" in text and "1,151 (stabil)" in text
    assert "(unten)" in text and "Rate 0,2729" in text and "0,2619" in text and "4 % darunter" in text and "Abstand 0,0099" in text
    kind, text = ev.verdict_equilibria(ev.equilibria_analysis(-0.375))
    assert "(oben)" in text and "Rate 0,2729" in text                                  # Spiegelung: oberer Topf
    kind, text = ev.verdict_equilibria(ev.equilibria_analysis(0.45))
    assert kind == "info" and "nur **ein Gleichgewicht**" in text and "hinter der Faltung bei ±0,3849" in text
    kind, text = ev.verdict_equilibria(ev.equilibria_analysis(K.HC))
    assert kind == "warning" and "genau auf der Faltung" in text and "Rate dort ist 0" in text


def _loop(r=0.001, up=0.4038, down=-0.4038, width=0.8077, formula=0.8087, hmax=0.8):
    return {"r": r, "up": up, "down": down, "width": width, "width_formula": formula, "hmax": hmax}


def test_verdict_loop_names_tips_width_two_hc_delay_and_formula():
    kind, text = ev.verdict_loop(_loop())
    assert kind == "info" and "r = 0,0010" in text and "hinauf bei h = 0,4038" in text and "hinunter bei h = -0,4038" in text
    assert "0,8077 breit" in text and "2·h_c = 0,7698" in text and "0,0189 nach der Faltung" in text and "0,8087" in text and "0,1 % darüber" in text
    kind, text = ev.verdict_loop(_loop(width=1.0249, formula=1.0100))
    assert "1,5 % darunter" in text                                                      # Formel unter der Messung
    kind, text = ev.verdict_loop(_loop(up=None, down=None, width=None, hmax=0.3))
    assert "Grenze ±0,3 ohne zu kippen" in text


def _rows():
    return [{"r": r, "delta": d, "formula": f, "ratio": d / f} for r, d, f in
            ((1e-2, 0.0829, 0.0904), (1e-3, 0.0189, 0.0195), (1e-4, 0.00417, 0.00419), (1e-5, 0.0009037, 0.0009037), (1e-6, 0.000195, 0.0001947), (1e-7, 4.2e-5, 4.194e-5))]


def test_verdict_delay_names_ratio_now_and_the_spread_over_six_rates():
    rows = _rows()
    now = {"r": 0.005, "delta": 0.0535, "formula": 0.0569, "ratio": 0.94}
    kind, text = ev.verdict_delay(rows, now)
    assert kind == "info" and "r = 0,0050" in text and "5,350·10⁻²" in text and "**0,9400**" in text
    assert "0,917 bei r = 0,01" in text and "1,0000 bei r = 10⁻⁵" in text and "r = 10⁻⁷" in text and "Erwartung 2/3" in text


def _slow(kept=100):
    return {"rate": 0.2729, "dtm": 0.1, "ac_theory": 0.9731, "ac_sim": 0.9736, "var_boltzmann": 9.35e-4, "var_linear": 7.33e-4, "var_sim": 8.13e-4, "kept": kept, "nch": 100}


def test_verdict_slowing_all_numbers_and_kept_chains():
    kind, text = ev.verdict_slowing(_slow(), 0.375)
    assert kind == "info" and "Rate 0,2729" in text and "0,9731" in text and "0,9736" in text and "9,35·10⁻⁴" in text and "7,33·10⁻⁴" in text and "21,6 % darunter" in text and "8,13·10⁻⁴" in text
    assert "Ketten sind im Topf geblieben" not in text
    assert "Nur 90 von 100 Ketten sind im Topf geblieben" in ev.verdict_slowing(_slow(kept=90), 0.375)[1]
    kind, text = ev.verdict_slowing(None, 0.45)
    assert "keinen zweiten Topf" in text and "h = 0,450" in text


def _warn(cond, trends, n_tipped, sigma=0.02, tip=1.033):
    return {"cond": cond, "label": K.CONDITIONS[cond]["label"], "trends": np.array(trends, float), "n_tipped": n_tipped, "n_runs": 4, "sigma_used": sigma, "mean_tip_frac": tip}


def test_verdict_warning_counts_alarms_per_condition():
    trends = [0.7, 0.6, 0.2, np.nan]                                                     # 2 von 3 Läufen mit Trendmaß über 0,5
    kind, text = ev.verdict_warning(_warn("A", trends, 0), 0.5)
    assert kind == "info" and "2 von 3 Läufen (66,7 %)" in text and "+0,50" in text and "0 von 4 Läufen kippen" in text and "Fehlalarmrate des Maßes" in text
    kind, text = ev.verdict_warning(_warn("B", trends, 0), 0.5)
    assert kind == "warning" and "Fehlalarm" in text and "kippt aber nie" in text
    kind, text = ev.verdict_warning(_warn("C", trends, 4), 0.5)
    assert kind == "success" and "4 von 4 Läufen kippen, im Mittel bei h/h_c = 1,033" in text and "berechtigt" in text
    kind, text = ev.verdict_warning(_warn("D", trends, 3, sigma=0.12), 0.5)
    assert kind == "info" and "3 von 4 Läufen kippen allein durch das Rauschen" in text and "kündigt dieses Kippen nicht an" in text and "Rauschen 0,12" in text


def test_verdict_warning_threshold_is_exclusive_and_all_nan_is_safe():
    assert "0 von 1 Läufen" in ev.verdict_warning(_warn("A", [0.5], 0), 0.5)[1]            # genau auf der Schwelle: kein Alarm
    assert "1 von 1 Läufen" in ev.verdict_warning(_warn("A", [0.51], 0), 0.5)[1]
    text = ev.verdict_warning(_warn("A", [np.nan, np.nan], 0), 0.5)[1]
    assert "0 von 0 Läufen" in text


def test_verdict_table_names_both_models_and_auc():
    rows = [{"model": m, "cond": k, "alarm": a} for m, vals in (("cubic", (0.01, 0.195, 0.8)), ("harvest", (0.005, 0.045, 0.29))) for k, a in zip("ABC", vals)]
    kind, text = ev.verdict_table({"rows": rows, "thr": 0.5, "runs": 200, "sigma": 0.02, "auc_cubic": 0.941, "auc_harvest": 0.828})
    assert kind == "info" and "1,0 % / 19,5 % / 80,0 %" in text and "0,5 % / 4,5 % / 29,0 %" in text and "0,941 kubisch, 0,828 Ernte" in text and "200 Läufe" in text


def _tempo(mean, before_tips, n_tipped=4):
    tips = np.array(before_tips, float)
    return {"own": {"r": 0.0008, "sigma": 0.1, "n_runs": 4, "n_tipped": n_tipped, "mean": mean, "std": 0.02, "formula": 0.4017, "tips": tips}}


def test_verdict_tempo_before_and_behind_the_fold_and_no_tipping():
    kind, text = ev.verdict_tempo(_tempo(0.3607, [0.35, 0.36, 0.37, 0.39]))
    assert kind == "warning" and "**vor** der Faltung" in text and "h = 0,3607" in text and "3 von 4 Läufen kippen vor h_c" in text and "0,4017" in text
    kind, text = ev.verdict_tempo(_tempo(0.4300, [0.41, 0.43, 0.44, 0.44]))
    assert kind == "info" and "hinter der Faltung" in text and "0 von 4 Läufen kippen vor h_c" in text
    kind, text = ev.verdict_tempo(_tempo(float("nan"), [np.nan] * 4, n_tipped=0))
    assert kind == "info" and "kein Lauf kippt" in text


def test_tempo_boundary_run_exactly_at_the_fold_does_not_count_as_before():
    kind, text = ev.verdict_tempo(_tempo(K.HC, [K.HC, K.HC, K.HC, K.HC]))
    assert kind == "info" and "0 von 4 Läufen kippen vor h_c" in text


def test_verdict_slowing_when_every_chain_left_the_well_gives_theory_only():
    a = {**_slow(kept=0), "sigma": 0.05, "ac_sim": float("nan"), "var_sim": float("nan")}
    kind, text = ev.verdict_slowing(a, 0.375)
    assert kind == "warning" and "Alle 100 Ketten haben den Topf verlassen" in text and "Rauschen 0,050" in text and "Rate 0,2729" in text and "0,9731" in text and "nan" not in text


def test_slowing_analysis_for_negative_h_mirrors_the_positive_one():
    pos = K.stationary_stats(0.3, 0.02, 0.1, 5, nch=40)
    neg = K.stationary_stats(-0.3, 0.02, 0.1, 5, nch=40)
    assert neg["kept"] == neg["nch"] == 40
    assert neg["rate"] == pytest.approx(pos["rate"]) and neg["var_boltzmann"] == pytest.approx(pos["var_boltzmann"], rel=1e-6)
    assert neg["var_sim"] == pytest.approx(pos["var_sim"], rel=0.25) and neg["ac_sim"] == pytest.approx(pos["ac_sim"], abs=0.01)
