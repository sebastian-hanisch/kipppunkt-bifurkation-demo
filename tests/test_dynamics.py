"""Einheiten des Kerns gegen Handrechnung (DEMO-PLAYBOOK §1b): jede Einheit an einer Mini-Instanz, deren Ergebnis von Hand gerechnet ist.

Handbeispiel (Quelle: dynamische-systeme-planung/handbeispiele_04.py, hier neu nachgerechnet): x' = x - x^3 + h, h = 3/8."""

import math
from fractions import Fraction as Fr

import numpy as np
import pytest

import kpb_dynamics as K
import kpb_rng as R


def test_fold_by_hand():
    """dh/dx = 3x^2 - 1 = 0 bei x = -+1/sqrt 3; dort h = x^3 - x = +-2/(3 sqrt 3) = 0,384900; Diskriminante 4 - 27 h^2 = 0 bei demselben h."""
    assert K.HC == pytest.approx(0.3849001795, abs=1e-10)
    x = -1 / math.sqrt(3)
    assert x ** 3 - x == pytest.approx(K.HC, abs=1e-15)
    assert K.discriminant(K.HC) == pytest.approx(0.0, abs=1e-12)
    assert K.discriminant(0.0) == 4.0


def test_equilibria_at_h_three_eighths_by_hand():
    """x^3 - x - 3/8 = (x + 1/2)(x^2 - x/2 - 3/4): Wurzeln -1/2 und (1 -+ sqrt 13)/4 = -0,651388 und 1,151388. Das Produkt wird in Brüchen nachgerechnet."""
    # (x + 1/2)(x^2 - x/2 - 3/4) = x^3 - x/2 * x ... Koeffizienten: x^3 + (1/2 - 1/2) x^2 + (-3/4 - 1/4) x - 3/8
    prod = [Fr(1), Fr(1, 2) - Fr(1, 2), Fr(-3, 4) + Fr(1, 2) * Fr(-1, 2), Fr(1, 2) * Fr(-3, 4)]
    assert prod == [1, 0, -1, Fr(-3, 8)]
    roots = K.real_roots(0.375)
    assert roots == pytest.approx([(1 - math.sqrt(13)) / 4, -0.5, (1 + math.sqrt(13)) / 4], abs=1e-12)
    assert roots == pytest.approx([-0.651388, -0.5, 1.151388], abs=1e-6)
    rates = [K.rate(x) for x in roots]
    assert rates == pytest.approx([0.272918, -0.25, 2.977], abs=1e-3)           # f'(x) = 1 - 3x^2 = -rate


def test_rate_approximation_by_hand():
    """sqrt(8/3) * sqrt(1 - 0,375/0,3849) = 1,63299 * 0,16040 = 0,2619 (4 % unter 0,2729)."""
    assert K.rate_approx(0.375) == pytest.approx(0.261898, abs=2e-6)
    assert K.rate_approx(0.375) / K.rate(K.lower_root(0.375)) - 1 == pytest.approx(-0.0404, abs=5e-4)
    assert K.rate_approx(K.HC) == 0.0 and K.rate_approx(1.0) == 0.0


def test_equilibria_at_h_zero():
    assert K.real_roots(0.0) == pytest.approx([-1.0, 0.0, 1.0], abs=1e-9)
    assert [K.rate(x) for x in (-1.0, 0.0, 1.0)] == [2.0, -1.0, 2.0]


def test_symmetry_x_h():
    """f(-x, -h) = -f(x, h) bis auf die letzte Stelle: Addition und Vorzeichenwechsel sind in Gleitkomma symmetrisch, `x ** 3` aber nicht
    plattformunabhängig (die Linux-CI weicht von Windows im letzten Bit ab), darum ein Band von wenigen Einheiten der letzten Stelle."""
    rng = np.random.default_rng(1)
    x, h = rng.uniform(-2, 2, 200), rng.uniform(-1, 1, 200)
    np.testing.assert_allclose(K.f(-x, -h), -K.f(x, h), rtol=1e-14, atol=1e-14)
    assert K.well_root(-0.3) == pytest.approx(-K.well_root(0.3), abs=1e-12)


def test_autocorrelation_and_variance_at_h_three_eighths_by_hand():
    """dt = 0,1, sigma = 0,02: AC1 = exp(-0,2729 * 0,1) = 0,9731; lineare Varianz sigma^2 / (2 Rate) = 7,33e-4; Boltzmann-Quadratur 9,35e-4."""
    rate = K.rate(K.lower_root(0.375))
    assert math.exp(-rate * 0.1) == pytest.approx(0.9731, abs=5e-5)
    assert 0.02 ** 2 / (2 * rate) == pytest.approx(7.33e-4, abs=5e-7)
    assert K.boltzmann_variance(0.375, 0.02) == pytest.approx(9.35e-4, abs=1e-6)


def test_rk4_step_equals_the_formula_in_fractions():
    """Ein RK4-Schritt von x = 0, h = 1, dt = 1/10 in Brüchen nach der Formel x + dt/6 (k1 + 2 k2 + 2 k3 + k4)."""
    f = lambda x, h: x - x ** 3 + h  # noqa: E731
    x, h, dt = Fr(0), Fr(1), Fr(1, 10)
    k1 = f(x, h)
    k2 = f(x + dt / 2 * k1, h)
    k3 = f(x + dt / 2 * k2, h)
    k4 = f(x + dt * k3, h)
    expected = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    assert K.rk4(0.0, 1.0, 0.1) == pytest.approx(float(expected), abs=1e-14)
    assert float(expected) == pytest.approx(0.105142, abs=1e-6)


def test_delay_constant_and_formulas_by_hand():
    """C = |z1| / 3^(1/6) mit z1 = -2,338107: 1,946903; r = 0,01: delta = 0,0904 (Kippen bei 0,4753); r = 1e-4: 4,19e-3; Schleifenbreite bei r = 0,001: 2 h_c + 2 C 1e-2 = 0,8087."""
    assert K.AIRY_C == pytest.approx(1.946903, abs=1e-6)
    assert K.delay_formula(0.01) == pytest.approx(0.0904, abs=5e-5) and K.HC + K.delay_formula(0.01) == pytest.approx(0.4753, abs=5e-5)
    assert K.delay_formula(1e-4) == pytest.approx(4.19e-3, abs=1e-5)
    assert 2 * K.HC + 2 * K.delay_formula(1e-3) == pytest.approx(0.8087, abs=5e-5)
    assert 2 * K.HC == pytest.approx(0.7698, abs=5e-5)


def test_ramp_tip_returns_none_when_the_ramp_ends_before_the_fold_or_on_the_wrong_branch():
    assert K.ramp_tip(1e-3, 0.0, K.lower_root(0.0), +1, 0.02, 0.1)[0] is None            # h_stop 0,1 liegt vor der Faltung
    assert K.ramp_tip(1e-3, 0.0, K.upper_root(0.0), +1, 0.02, 0.6)[0] is None           # oberer Ast kippt beim Hinauffahren nie


def test_ramp_tip_trace_is_thinned_and_starts_at_the_equilibrium():
    tip, (hs, xs) = K.ramp_tip(1e-2, 0.0, K.lower_root(0.0), +1, 0.02, 0.8, trace_every=10)
    assert tip is not None and hs[0] == 0.0 and xs[0] == pytest.approx(-1.0) and len(hs) == len(xs) > 5
    assert hs[-1] <= tip + 1e-12 and tip - hs[-1] < 10 * 1e-2 * 0.02 + 1e-9          # letzte Spur höchstens 10 Schritte vor dem Kippen


def test_discriminant_sign_equals_number_of_equilibria_around_the_fold():
    for h in (0.3848992, -0.3848992, 0.3, 0.0):
        assert K.discriminant(h) > 0 and len(K.real_roots(h)) == 3
    for h in (0.3849012, -0.3849012, 0.5, 0.6):
        assert K.discriminant(h) < 0 and len(K.real_roots(h)) == 1


def test_window_ac1_and_trend_on_constructed_series():
    n = 1600
    freq = np.linspace(2.8, 0.05, n)                      # AC1 ~ cos(frequenz): wächst mit sinkender Frequenz
    x = np.sin(np.cumsum(freq))
    assert K.trend_measure(x) > 0.9
    assert K.trend_measure(x[::-1].copy()) < -0.9
    alt = (-1.0) ** np.arange(400) + 0.0
    _, acs = K.window_ac1(alt)
    assert np.all(acs < -0.98)
    assert math.isnan(K.trend_measure(np.arange(500.0)))     # unter 600 Messungen kein Trendmaß


def test_auc_and_alarm_share_on_hand_numbers():
    assert K.auc(np.array([1.0, 2.0]), np.array([0.0, 1.0])) == pytest.approx(0.875)        # (1 + 0,5 + 1 + 1) / 4
    assert K.alarm_share(np.array([0.1, 0.6, np.nan, 0.7]), 0.5) == pytest.approx(2 / 3)
    assert math.isnan(K.alarm_share(np.array([np.nan]), 0.5))


def test_noise_blocks_are_consistent_across_chunk_boundaries():
    a = K._normals(5, 0, 4, 3, 2)
    b = K._normals(5, 2, 2, 3, 2)
    assert np.array_equal(a[2:], b)
    full = R.normal_array(9, 40)
    assert np.array_equal(R.normal_array(9, 30, offset=20), full[10:])           # Wert j nutzt die Ausgaben 2j + 1 und 2j + 2


def test_splitmix_reference_values_and_array_equals_class():
    g = R.SplitMix64(0)
    assert [g.next_u64() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]
    cls = R.SplitMix64(123456789)
    assert [int(v) for v in R.splitmix_array(123456789, 20)] == [cls.next_u64() for _ in range(20)]
    cls = R.SplitMix64(123456789)
    assert list(R.normal_array(123456789, 15)) == pytest.approx([cls.normal() for _ in range(15)], abs=1e-12)


def test_simulate_chains_without_noise_is_the_scalar_euler_scheme():
    xs, tip = K.simulate_chains(lambda i: 0.1, 0.0, -1.0, 0.1, 5, 40, 3, 2)
    x = -1.0
    for _ in range(40 * 5):
        x = x + (x - x ** 3 + 0.1) * 0.02
    assert xs[-1] == pytest.approx([x, x], abs=1e-12) and (tip == 40).all()


def test_equilibrium_branches_marks_stability_by_the_rate():
    H, X, S = K.equilibrium_branches([0.0, 0.5])
    assert list(H) == [0.0, 0.0, 0.0, 0.5] and list(S) == [True, False, True, True]
    assert X[3] == pytest.approx(K.real_roots(0.5)[0])
