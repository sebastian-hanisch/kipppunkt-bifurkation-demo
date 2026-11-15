"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Aussagen, die für das Modell gelten müssen, ohne dass man das richtige Ergebnis kennt, und Randfälle.

Keine Reglerwerte, die Zweierpotenzen sind (Raten aus R_OPTIONS, h-Raster mit nicht-dyadischem Versatz)."""

import math

import numpy as np
import pytest

import kpb_constants as C
import kpb_dynamics as K
import kpb_evaluation as E
import kpb_rng as R


def test_loop_is_mirror_symmetric_for_every_rate():
    """Spiegelsymmetrie x -> -x, h -> -h: Kipp-h hinauf plus Kipp-h hinunter ist 0 (bis 1e-12)."""
    for r in C.R_OPTIONS:
        lp = K.hysteresis_loop(r)
        assert lp["up"] + lp["down"] == pytest.approx(0.0, abs=1e-12), r


def test_loop_width_grows_monotonically_with_the_rate_and_tends_to_two_h_c():
    widths = [K.hysteresis_loop(r)["width"] for r in sorted(C.R_OPTIONS)]
    assert all(b > a for a, b in zip(widths, widths[1:]))
    assert widths[0] > 2 * K.HC                                           # jede endliche Rate gibt eine Verspätung
    assert widths[0] - 2 * K.HC < 2 * K.delay_formula(min(C.R_OPTIONS)) * 1.01 + 1e-4
    assert K.hysteresis_loop(1e-6)["width"] == pytest.approx(2 * K.HC, abs=2 * K.delay_formula(1e-6) * 1.05)


def test_without_noise_the_tip_is_never_before_the_fold_on_a_grid_of_rates_and_start_values():
    """Start auf dem unteren Gleichgewicht bei beliebigem h_start < h_c: Bei steigendem h steigt das Gleichgewicht, die Bahn bleibt unter ihm und kreuzt 0 nicht früher als bei h_c."""
    rng = R.SplitMix64(4)
    for _ in range(30):
        r = 10.0 ** rng.uniform(-4.0, -1.5)
        hs = rng.uniform(-0.8, 0.37)
        tip = K.ramp_tip(r, hs, K.lower_root(hs), +1, 0.02, 1.0)[0]
        assert tip is not None and tip >= K.HC - 1e-9, (r, hs)


def test_same_seed_same_output_other_seed_other_output():
    a = K.warning_runs(0.5, 0.8, 0.02, 20, 35, n_meas=900)
    b = K.warning_runs(0.5, 0.8, 0.02, 20, 35, n_meas=900)
    c = K.warning_runs(0.5, 0.8, 0.02, 20, 36, n_meas=900)
    assert np.array_equal(a["trends"], b["trends"]) and not np.array_equal(a["trends"], c["trends"])
    t1, t2 = K.tempo_noise_cell(0.004, 0.05, 20, 35), K.tempo_noise_cell(0.004, 0.05, 20, 35)
    assert np.array_equal(t1["tips"], t2["tips"])


def test_chains_are_independent_of_each_other():
    """Ein anderer Startwert der Kette 1 ändert Kette 0 nicht (gleicher Zufalls-Seed)."""
    x0 = np.array([-1.0, -1.0, -1.0])
    a = K.simulate_chains(lambda i: 0.0, 0.05, x0, 0.1, 5, 60, 9, 3)[0]
    x0b = np.array([-1.0, -0.8, -1.0])
    b = K.simulate_chains(lambda i: 0.0, 0.05, x0b, 0.1, 5, 60, 9, 3)[0]
    assert np.array_equal(a[:, 0], b[:, 0]) and np.array_equal(a[:, 2], b[:, 2]) and not np.array_equal(a[:, 1], b[:, 1])


def test_stronger_noise_means_a_larger_variance():
    v = [K.stationary_stats(0.5 * K.HC, s, 0.1, 5, nch=60)["var_sim"] for s in (0.01, 0.02, 0.04)]
    assert v[0] < v[1] < v[2]
    assert v[2] / v[0] == pytest.approx(16.0, rel=0.25)                    # Varianz ~ sigma^2


def test_rate_falls_monotonically_towards_the_fold_and_is_zero_there():
    fr = np.linspace(0.0, 0.99, 50)
    rates = [K.rate(K.lower_root(float(f) * K.HC)) for f in fr]
    assert all(b < a for a, b in zip(rates, rates[1:]))
    assert K.rate(-1 / math.sqrt(3)) == pytest.approx(0.0, abs=1e-15)      # Faltungsort x = -1/sqrt 3


def test_exactly_on_the_fold_is_a_double_root_with_rate_zero():
    a = E.equilibria_analysis(K.HC)
    assert E.verdict_equilibria(a)[0] == "warning" and "genau auf der Faltung" in E.verdict_equilibria(a)[1]
    roots = K.real_roots(K.HC)
    assert roots[0] == pytest.approx(-1 / math.sqrt(3), abs=1e-6)


def test_a_ramp_that_never_reaches_the_fold_does_not_tip_and_the_loop_reports_it():
    lp = K.hysteresis_loop(0.0005, hmax=0.3)
    assert lp["up"] is None and lp["down"] is None and lp["width"] is None
    assert E.verdict_loop(lp)[1].endswith("ohne zu kippen.")


def test_start_behind_the_fold_has_only_one_equilibrium_and_no_tipping_ramp():
    assert len(K.real_roots(0.45)) == 1
    assert K.ramp_tip(1e-3, 0.45, K.lower_root(0.45), +1, 0.02, 0.8)[0] is None


def test_zero_noise_cell_is_one_deterministic_run_and_noise_moves_the_mean_earlier():
    t0 = K.tempo_noise_cell(0.004, 0.0, 50, 35)
    assert t0["n_runs"] == 1 and t0["std"] == 0.0
    t1 = K.tempo_noise_cell(0.004, 0.1, 50, 35)
    assert t1["mean"] < t0["mean"] and t1["std"] > 0.01


def test_tempo_cell_without_tipping_inside_the_ramp_returns_nan():
    c = K.tempo_noise_cell(0.5, 0.0, 10, 1)                                  # Rate so hoch, dass 1/r/dt = 100 Schritte nicht reichen
    assert c["n_tipped"] == 0 and math.isnan(c["mean"])
    assert E.verdict_tempo({"own": c})[1].startswith("Bei r = 0,5000")


def test_all_presets_and_default_values_are_not_powers_of_two_except_none():
    def pow2(x):
        return x > 0 and math.frexp(x)[0] == 0.5
    for name, p in C.PRESETS.items():
        for key in ("r", "sigma", "dtm", "h"):
            assert not pow2(abs(float(p[key]))), (name, key)
    assert not any(pow2(r) for r in C.R_OPTIONS)
