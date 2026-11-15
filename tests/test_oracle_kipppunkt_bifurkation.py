"""Orakel-Tests des Kerns: jede Kernroutine gegen einen anderen Rechenweg (ORACLE-PLAYBOOK §2, §5 und §6).

Orakel:
 1. Gleichgewichte gegen die trigonometrische bzw. hyperbolische Cardano-Formel (nicht numpy.roots, das der Kern benutzt) an 200 Werten von h.
 2. Zahl der Gleichgewichte gegen das Vorzeichen der Diskriminante auf einem Raster um +-h_c.
 3. Kipp-h der Rampe (eigener RK4) gegen `scipy.integrate.solve_ivp` (DOP853, rtol 1e-12) mit Ereignis x = 0, auf 1e-4, bei r = 0,02; 0,005; 0,001.
 4. Verzögerungskonstante: erste Nullstelle der Airy-Funktion (scipy.special.airy) und der Literaturwert -2,338107410459767; Verhältnis Messung zu Formel bei r = 1e-5.
 5. Varianz im Topf: Boltzmann-Quadratur des Kerns gegen `scipy.integrate.quad`; Autokorrelation gegen e^(-Rate Delta) (Band 0,5 %); Mittel der Ketten gegen das Gleichgewicht.
 6. Euler-Maruyama gegen die exakte Varianz des Ornstein-Uhlenbeck-Prozesses (h = 0, Rate 2).
Gegenprobe für Absolutaussagen (ORACLE-PLAYBOOK §6): „ohne Rauschen kippt keine Rampe vor h_c“ an 120 log-gleichverteilten Raten (SplitMix64); der Scratch-Lauf mit 1000 Raten
steht in der README (nicht im Repo)."""

import math

import numpy as np
import pytest
from scipy import integrate, special

import kpb_dynamics as K
import kpb_rng as R


def _cardano(h: float) -> list[float]:
    """Reelle Wurzeln von x^3 - x - h = 0 nach Cardano (p = -1, q = -h)."""
    if abs(h) <= K.HC:
        arg = (3.0 * math.sqrt(3.0) / 2.0) * h
        return sorted(2.0 / math.sqrt(3.0) * math.cos(math.acos(arg) / 3.0 - 2.0 * math.pi * k / 3.0) for k in range(3))
    return [math.copysign(2.0 / math.sqrt(3.0) * math.cosh(math.acosh((3.0 * math.sqrt(3.0) / 2.0) * abs(h)) / 3.0), h)]


def test_equilibria_agree_with_the_cardano_formula_at_200_values_of_h():
    rng = np.random.default_rng(2)
    hs = np.concatenate([rng.uniform(-0.384, 0.384, 100), rng.uniform(0.3856, 0.6, 50), rng.uniform(-0.6, -0.3856, 50)])
    for h in hs:
        mine, ref = K.real_roots(float(h)), _cardano(float(h))
        assert len(mine) == len(ref), h
        assert mine == pytest.approx(ref, abs=1e-9), h


def test_number_of_equilibria_follows_the_sign_of_the_discriminant():
    for h in np.linspace(-0.6, 0.6, 241) + 1.234e-4:           # nicht dyadisch, nie genau auf der Faltung
        n = len(K.real_roots(float(h)))
        assert n == (3 if 4 - 27 * h * h > 0 else 1), h


@pytest.mark.parametrize("r", [0.02, 0.005, 0.001])
def test_tip_of_the_own_rk4_agrees_with_solve_ivp(r):
    """Rampe h = -0,8 + r t ab dem unteren Gleichgewicht; Kippen = x kreuzt 0. Die Kern-Schleife startet 12 r^(2/3) vor der Faltung: dasselbe Kipp-h auf 1e-4."""
    def rhs(t, y):
        h = -0.8 + r * t
        return [y[0] - y[0] ** 3 + h]

    def event(t, y):
        return y[0]
    event.terminal = True
    event.direction = 1
    sol = solve_ivp = integrate.solve_ivp(rhs, (0.0, 1.8 / r), [K.lower_root(-0.8)], method="DOP853", rtol=1e-12, atol=1e-14, events=event)
    h_ref = -0.8 + r * float(sol.t_events[0][0])
    assert K.hysteresis_loop(r)["up"] == pytest.approx(h_ref, abs=1e-4)
    # und von der Rampe aus dem Start -0,8 mit dem eigenen RK4 (volle Strecke)
    own = K.ramp_tip(r, -0.8, K.lower_root(-0.8), +1, 0.02, 0.8)[0]
    assert own == pytest.approx(h_ref, abs=1e-4)


def test_delay_constant_agrees_with_the_first_airy_zero():
    z1 = -2.338107410459767                                    # erste Nullstelle von Ai (Literaturwert, Abramowitz/Stegun)
    assert float(special.airy(z1)[0]) == pytest.approx(0.0, abs=1e-12)
    assert K.AIRY_C == pytest.approx(abs(z1) / 3 ** (1 / 6), abs=1e-12)
    assert special.ai_zeros(1)[0][0] == pytest.approx(z1, abs=1e-12)
    m = K.delay_measurement(1e-5)
    assert m["ratio"] == pytest.approx(1.0, abs=0.01)


def test_boltzmann_variance_agrees_with_scipy_quad():
    for h, sigma in ((0.0, 0.02), (0.375, 0.02), (0.3, 0.03), (-0.3, 0.03)):
        xl = K.well_root(h)
        mid = K.real_roots(h)[1]
        lo, hi = (xl - 1.0, mid) if h >= 0 else (mid, xl + 1.0)
        w = lambda x: math.exp(-2.0 * (float(K.potential(x, h)) - float(K.potential(xl, h))) / sigma ** 2)  # noqa: E731
        z = integrate.quad(w, lo, hi, limit=400, epsabs=0, epsrel=1e-12)[0]
        m1 = integrate.quad(lambda x: x * w(x), lo, hi, limit=400, epsabs=0, epsrel=1e-12)[0] / z
        var = integrate.quad(lambda x: (x - m1) ** 2 * w(x), lo, hi, limit=400, epsabs=0, epsrel=1e-12)[0] / z
        assert K.boltzmann_variance(h, sigma) == pytest.approx(var, rel=1e-4), (h, sigma)


def test_stationary_simulation_agrees_with_theory_at_four_distances_from_the_fold():
    for frac in (0.0, 0.5, 0.8, 0.9):
        h = frac * K.HC
        s = K.stationary_stats(h, 0.02, 0.1, 5)
        assert s["kept"] == s["nch"], frac
        assert s["ac_sim"] == pytest.approx(s["ac_theory"], abs=0.005), frac                        # Band 0,5 %
        assert s["var_sim"] == pytest.approx(s["var_boltzmann"], rel=0.05), frac
        xs = K.simulate_chains(lambda i: h, 0.02, K.well_root(h), 0.1, 10, 300, 7, 40)[0]
        assert abs(float(xs[100:].mean()) - K.well_root(h)) < 1e-3, frac                           # Mittel der Ketten gegen das Gleichgewicht


def test_euler_maruyama_gives_the_ornstein_uhlenbeck_variance_for_small_noise():
    """h = 0, Topf bei x = -1 mit Rate 2: Varianz sigma^2 / (2 * 2) = 1e-4 (linearisiert; Anharmonizität < 1 %)."""
    s = K.stationary_stats(0.0, 0.02, 0.1, 11, nch=200)
    assert s["var_sim"] == pytest.approx(1e-4, rel=0.05)
    assert s["ac_sim"] == pytest.approx(math.exp(-0.2), abs=0.005)


def test_counter_check_no_ramp_tips_before_the_fold_without_noise():
    """ORACLE-PLAYBOOK §6: 120 log-gleichverteilte Raten in [1e-5; 0,1] (SplitMix64), Start auf dem unteren Gleichgewicht: kein Kipp-h unter h_c - 1e-9."""
    u = R.uniform_array(2026, 120)
    rs = 10.0 ** (-5.0 + 4.0 * u)
    tips = [K.hysteresis_loop(float(r))["up"] for r in rs]
    assert all(t is not None for t in tips)
    assert min(tips) >= K.HC - 1e-9
    assert min(tips) > K.HC                                                       # und in jedem Lauf strikt dahinter
