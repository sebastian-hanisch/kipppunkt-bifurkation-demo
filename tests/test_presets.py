"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, keine Zweierpotenzen, jedes Preset verschieden."""

import kpb_constants as C
import kpb_evaluation as ev
import kpb_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_widget_bounds_and_options():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            value = preset[key]
            if spec.options is not None:
                assert value in spec.options, (name, key)
            else:
                assert spec.lo <= value <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Handbeispiel h = 0,375"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_presets_are_pairwise_different():
    seen = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(seen)) == len(seen)


def test_presets_set_what_their_help_text_names():
    assert C.PRESETS["Langsame Schleife"]["r"] == 0.001 and C.PRESETS["Schnelle Schleife"]["r"] == 0.02
    assert abs(C.PRESETS["Nahe der Faltung"]["h"]) < 0.3849 and abs(C.PRESETS["Nahe der Faltung"]["h"]) > 0.9 * 0.3849
    assert C.PRESETS["Anstieg ohne Kippen"]["cond"] == "B"
    assert C.PRESETS["Rauschen und langsame Rampe"]["r"] == 0.0008 and C.PRESETS["Rauschen und langsame Rampe"]["sigma_cell"] == 0.1
