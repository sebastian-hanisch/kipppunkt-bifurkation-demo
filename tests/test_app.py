"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


# --- Zusätze für diese Demo -------------------------------------------------------------------------------------------------------------

import pytest  # noqa: E402

import kpb_constants as C  # noqa: E402
import kpb_presets as P  # noqa: E402


def _texts(at) -> str:
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption] + [s.value for s in at.success] + [i.value for i in at.info] + [w.value for w in at.warning])


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_runs_and_sets_its_values(name):
    at = _fresh_app()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.sidebar.slider(key="h_slider").value == pytest.approx(C.PRESETS[name]["h"])
    assert at.sidebar.select_slider(key="r_select").value == C.PRESETS[name]["r"]


def test_all_minima_and_all_maxima_together_do_not_raise():
    for edge in ("min", "max"):
        at = _fresh_app()
        for key, spec in P.SETTING_SPECS.items():
            if spec.options is not None:
                options = spec.options
                value = options[0] if edge == "min" else options[-1]
                for bereich in (at.sidebar, at.main):
                    for art in ("select_slider", "selectbox"):
                        try:
                            getattr(bereich, art)(key=key).set_value(value)
                        except KeyError:
                            continue
                continue
            for bereich in (at.sidebar, at.main):
                for art in ("slider", "number_input"):
                    try:
                        w = getattr(bereich, art)(key=key)
                    except KeyError:
                        continue
                    v = spec.lo if edge == "min" else spec.hi
                    w.set_value(int(v) if spec.caster is int else float(v))
        at.run(timeout=TIMEOUT)
        assert not at.exception, (edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("key,options", [("cond_select", C.COND_OPTIONS), ("dtm_select", C.DTM_OPTIONS), ("r_select", C.R_OPTIONS)])
def test_every_option_of_the_selection_widgets_runs(key, options):
    for option in options:
        at = _fresh_app()
        for bereich in (at.sidebar, at.main):
            for art in ("select_slider", "selectbox"):
                try:
                    getattr(bereich, art)(key=key).set_value(option)
                except KeyError:
                    continue
        at.run(timeout=TIMEOUT)
        assert not at.exception, (key, option, [str(e) for e in at.exception])


def test_h_exactly_on_the_fold_and_behind_it_run_without_exception():
    for h in (0.385, 0.45, 0.6, -0.385, -0.6):
        at = _fresh_app()
        at.sidebar.slider(key="h_slider").set_value(h).run(timeout=TIMEOUT)
        assert not at.exception, (h, [str(e) for e in at.exception])
    assert "genau auf der Faltung" not in _texts(at)


def test_comparison_table_button_computes_both_models():
    at = _fresh_app()
    at.sidebar.slider(key="runs_slider").set_value(40).run(timeout=TIMEOUT)
    at.button[[b.label for b in at.button].index("Vergleichstabelle berechnen")].click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    text = _texts(at)
    assert "kubisches Modell" in text and "Erntemodell" in text and "Trennung ohne Schwelle" in text


def test_handbeispiel_default_shows_three_equilibria_and_the_loop():
    at = _fresh_app()
    texts = _texts(at)
    assert "drei Gleichgewichte" in texts and "-0,651 (stabil)" in texts and "Rate 0,2729" in texts
    assert "0,8767 breit" in texts and "hinauf bei h = 0,4384" in texts


def test_preset_without_tipping_selects_condition_b_and_reports_false_alarms():
    at = _fresh_app()
    at.sidebar.button(key="preset_Anstieg ohne Kippen").click().run(timeout=TIMEOUT)
    assert at.selectbox(key="cond_select").value == "B"
    assert "Fehlalarm" in _texts(at)
