# Kipppunkt (Bifurkation) – Hysterese, verspätetes Kippen und was die Frühwarnung wert ist – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-kipppunkt-bifurkation-demo.streamlit.app/)**

Stück der Linie **Dynamische Systeme** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. **„Kipppunkt“ heißt hier eine Bifurkation einer Differentialgleichung**, nicht die Netzschwelle (Anteil aktiver Nachbarn) wie in
`kaskaden-demo` und `komplexe-ansteckung-demo` und nicht der Wechsel des besten Plans bei verschobenem Gewicht wie in `gewichte-kippen-demo`; deshalb steht „(Bifurkation)“ im Titel. Das Stück hängt an
[`logistische-abbildung-demo`](https://github.com/sebastian-hanisch/logistische-abbildung-demo): Dort ist die Bifurkation ein Diagramm über einem Parameter, der **unendlich langsam** verstellt wird und von keinem Rauschen gestört ist.
Hier wird der Parameter mit endlichem Tempo und mit Rauschen gefahren, und die Fragen lauten: Wo kippt das System dann wirklich, kommt es auf demselben Weg zurück, und kündigt es sich an?

## Warum dieses Problem

Im Modell **x′ = x − x³ + h** gibt es für |h| < h_c = 2/(3√3) = 0,3849 zwei stabile Gleichgewichte; bei h = ±h_c fällt eines mit dem Sattel zusammen und verschwindet (**Faltung**). Beim langsamen Verstellen kippt das System ohne Rauschen **erst hinter der Faltung**
und kehrt erst hinter der Gegenfaltung zurück: eine **Schleife**, die mit dem Tempo breiter wird (mit Rauschen kann es davor kippen, siehe Tempo gegen Rauschen). Nahe der Faltung klingen Störungen langsamer ab; das lässt sich als **Frühwarnung** messen (Autokorrelation steigt). Die Demo zeigt, wie gut das **in diesem Modell, bei diesem Rauschen und diesem Fenster** funktioniert,
und wo es Fehlalarme gibt. Ein zweites Modell (Ernte) dient als Gegenprobe der Aussage: Die Güte der Warnung ist keine Konstante: Sie ist eine Eigenschaft des Modells (hier an zwei Modellen gezeigt) und hängt auch am Rauschen und am Fenster (das Fenster ist nicht gemessen).

## Modell

- **Gleichgewichte:** x − x³ + h = 0, also h = x³ − x; Diskriminante 4 − 27h² (positiv: drei Gleichgewichte). Rate eines Gleichgewichts: 3x² − 1 (positiv: stabil). Symmetrie: f(−x, −h) = −f(x, h).
- **Rampe:** h(t) = h₀ ± r·t; „Kippen“ heißt: x kreuzt 0 (andere Kriterien verschieben die Kipp-Orte um Größen der Ordnung r). Eigener RK4, Zeitschritt 0,02 (für r < 10⁻⁵: 0,05), die Kippzeit linear im Schritt interpoliert.
- **Verzögerung:** δ = h_Kipp − h_c ≈ C·r^(2/3) mit C = |z₁|/3^(1/6) = 1,9469 und z₁ = −2,3381 (erste Nullstelle der Airy-Funktion). **Eigene Herleitung** aus der Normalform s′ = r·t + √3·s² (s = x + 1/√3) durch die Substitution s = −u′/(√3·u) auf die Airy-Gleichung u″ = −√3·r·t·u; die Lösung, die dem stabilen Ast folgt, hat bei der ersten Nullstelle von Ai ihre Polstelle.
  Es ist **keine Quelle gelesen**; die Messung ist die Probe. Die Formeln stehen im Expander 📐 der App.
- **Rauschen:** Euler–Maruyama, x ← x + (x − x³ + h)·Δt + σ·√Δt·ξ; Zufall aus einem eigenen **SplitMix64** (`kpb_rng.py`, vektorisiert gegen die Klasse geprüft). Autokorrelation im Topf e^(−Rate·Δ), Varianz linear σ²/(2·Rate) und exakt aus der stationären Dichte exp(−2V/σ²).
- **Frühwarnung:** Fenster von 200 Messungen (Abstand 0,1), Schritt 50, linearer Trend entfernt, Autokorrelation (Lag 1) je Fenster; das **Trendmaß** ist die Korrelation zwischen Fensternummer und Autokorrelation, **Alarm** heißt Trendmaß über der Schwelle (Voreinstellung 0,5), Läufe mit weniger als 600 Messungen vor dem Kippen zählen nicht.
  Bedingungen (je 4000 Messungen, Rauschen 0,02): **A** konstantes h = 0,5 h_c; **B** Anstieg 0,5 → 0,8 h_c (kein Kippen); **C** Anstieg 0,5 → 1,2 h_c (kippt); **D** (Kontrolle) konstantes h = 0,8 h_c mit Rauschen 0,12 (kippt durch das Rauschen). Zweites Vehikel: Ernte x′ = x(1 − x) − h (r = K = 1, Faltung bei h_max = 0,25, Kippen bei x < 0,1), dieselben Bedingungen A bis C.

## Methodik

| Verfahren | Wie es arbeitet | Grenze |
|---|---|---|
| **Rampe mit RK4** (`kpb_dynamics.hysteresis_loop`, `delay_measurement`) | Start auf dem Gleichgewicht bei h_c ∓ 12·r^(2/3) (dort folgt die Bahn ihm bis auf O(r); die Probe gegen `solve_ivp` ab h = −0,8 steht im Orakel), Rampe bis zum Nulldurchgang | Kippen = x kreuzt 0 |
| **Euler–Maruyama** (`simulate_chains`) | vektorisiert über die Ketten, Normalwerte aus SplitMix64 in Blöcken | Schrittfehler der Autokorrelation (bis 0,004 bei Rate 2, Schritt 0,01) |
| **Boltzmann-Quadratur** (`boltzmann_variance`) | exakte Varianz im Topf nach exp(−2V/σ²), Trapezregel, unabhängig vom linearen Ansatz | nur im Topf (Sattel als Grenze) |
| **Trendmaß** (`trend_measure`) | wie oben; Alarmanteil und AUC (Wahrscheinlichkeit, dass ein Kipp-Lauf ein größeres Trendmaß hat als ein Lauf ohne Kippen) | ein Maß, ein Fenster, eine Schwelle |

Die Live-Rechnung der App braucht je Eingabe 0,02 bis 1,7 Sekunden (Messung auf dem Entwicklungsrechner, nicht festgeschrieben): Schleife 0,02 s, Verzögerung 0,1 s, kritisches Verlangsamen 0,3 s, Frühwarnung 0,2 s (100 Läufe) bis 0,4 s (200 Läufe), Tempo gegen Rauschen 1,0 s (100 Läufe) bis 1,7 s (200 Läufe);
die Vergleichstabelle (alle Bedingungen, beide Modelle) liegt hinter einem Knopf (1,6 s bei 100, 4,9 s bei 200 Läufen). Nichts davon ist vorgerechnet; alles läuft live.

## Befunde (gemessen, keine Behauptungen)

Zufall: Zufalls-Seed 35 (SplitMix64) und 200 Läufe, wo nichts anderes steht; die App rechnet mit 100 Läufen voreingestellt, ihre Alarmanteile streuen deshalb bis etwa ±5 Punkte. Die Tests prüfen Bänder.

| Frage | Befund | Test |
|---|---|---|
| Faltung (exakt) | h_c = 0,384900; drei Gleichgewichte bei h = 0,3848992 (Diskriminante +0,000020), eines bei 0,3849012, spiegelbildlich bei −h_c. Handbeispiel h = 3/8: Gleichgewichte −0,651388; −0,5 (exakte Wurzel); 1,151388, Raten 0,2729 (stabil), −0,25 (instabil), 2,977 (stabil); Näherung √(8/3)·√(1 − h/h_c) = 0,2619 (4 % darunter) | `test_fold_by_hand`, `test_equilibria_at_h_three_eighths_by_hand`, `test_discriminant_sign_equals_number_of_equilibria_around_the_fold`, `test_equilibria_agree_with_the_cardano_formula_at_200_values_of_h` |
| Schleife bei endlicher Rampenrate | Rampe von −0,8 nach +0,8 und zurück: **r = 0,02 / 0,005 / 0,001**: Kippen hinauf bei h = 0,5125 / 0,4384 / 0,4038 (hinunter jeweils der negative Wert, Summe 0 auf 10⁻¹²), **Breite 1,0249 / 0,8767 / 0,8077**; Formel 2·h_c + 2·C·r^(2/3): 1,0567 / 0,8837 / 0,8087. Für r = 10⁻⁴: 0,7781 gegen Formel 0,7782; Grenzwert r → 0: 2·h_c = 0,7698 | `test_loop_widths_and_tip_places_at_the_readme_rates`, `test_symmetry_of_the_loop_tip_places`, `test_tip_of_the_own_rk4_agrees_with_solve_ivp` |
| Verzögerung gegen das r^(2/3)-Gesetz | Verhältnis Messung zu C·r^(2/3) bei r = 10⁻², 10⁻³, 10⁻⁴, 10⁻⁵, 10⁻⁶, 10⁻⁷: **0,917 / 0,973 / 0,9938 / 1,0001 / 1,0014 / 1,0013**; Steigung im Log-Log-Bild 0,6601 (alle sechs) und 0,6656 (r ≤ 10⁻⁴), Erwartung 2/3. Die Formel gilt für kleine r (8 % daneben bei r = 0,01). Kein Rampenlauf kippt ohne Rauschen vor h_c: 1000 log-gleichverteilte Raten von 10⁻⁶ bis 10⁻¹ (Zufalls-Seed 777, **Scratch-Lauf, nicht im Repo**; im Test 120 Raten von 10⁻⁵ bis 10⁻¹): alle kippen, der kleinste Abstand hinter h_c ist 1,97·10⁻⁴ | `test_delay_ratios_and_slopes_at_the_six_rates`, `test_delay_constant_agrees_with_the_first_airy_zero`, `test_counter_check_no_ramp_tips_before_the_fold_without_noise` |
| Kritisches Verlangsamen | Je Abstand h/h_c = 0 / 0,5 / 0,8 / 0,9 / 0,95 (σ = 0,02, Messabstand 0,1, 100 Ketten): Rate 2,0000 / 1,3473 / 0,8111 / 0,5579 / 0,3863; Autokorrelation Theorie e^(−Rate·0,1) 0,8187 / 0,8740 / 0,9221 / 0,9457 / 0,9621, Simulation 0,8154 / 0,8717 / 0,9205 / 0,9445 / 0,9616 (Abweichung höchstens 0,0033, Band des Tests 0,005, ein Teil davon Schrittfehler der Euler–Maruyama-Schritte); Varianz (Boltzmann) 1,001 / 1,487 / 2,484 / 3,660 / 5,515 ·10⁻⁴, Simulation 1,007 / 1,492 / 2,476 / 3,618 / 5,359 ·10⁻⁴ (höchstens 3 % daneben); die lineare Formel σ²/(2·Rate) liegt bei 0,95 h_c **6 % unter der exakten** (5,178·10⁻⁴) | `test_rate_autocorrelation_and_variance_at_five_distances`, `test_the_linear_variance_formula_loses_near_the_fold`, `test_boltzmann_variance_agrees_with_scipy_quad` |
| Dasselbe beim Handbeispiel h = 0,375 | Rate 0,2729, Autokorrelation 0,9731 (Simulation 0,9736); Varianz exakt 9,35·10⁻⁴, lineare Formel 7,33·10⁻⁴ (**21,6 % darunter**; die exakte liegt 27,6 % darüber), Simulation 8,13·10⁻⁴: von 100 Ketten bleiben nur 90 im Topf (die übrigen verlassen ihn durch das Rauschen und werden ausgeschlossen, das drückt vermutlich die Simulationsvarianz; nicht gesondert gemessen) | `test_autocorrelation_and_variance_at_h_three_eighths_by_hand`, `test_the_linear_variance_formula_loses_near_the_fold` |
| Frühwarnung und Fehlalarme (kubisch, Rauschen 0,02, Schwelle 0,5, 200 Läufe) | Alarmanteil **A 1,0 %** (0 von 200 kippen), **B 19,5 %** (0 kippen: Fehlalarme bei Anstieg ohne Kippen), **C 80,0 %** (200 von 200 kippen, im Mittel bei h/h_c = 1,033); mittleres Trendmaß +0,02 / +0,32 / +0,62. Trennung ohne Schwelle (AUC): **0,941**. Andere Zufalls-Seeds (36, 37; Seed 38 im Scratch): A 0,5 bis 1,0 %, B 11,5 bis 19,5 %, C 79,0 bis 82,5 % | `test_alarm_shares_cubic_model`, `test_alarm_shares_are_stable_over_other_seeds` |
| Kontrolle D: Kippen durch Rauschen bei konstantem h | h = 0,8 h_c, Rauschen 0,12: **172 von 200 Läufen kippen** durch das Rauschen (andere Seeds 180 und 181 von 200), 157 davon nach mindestens 600 Messungen; **Alarm in 5,7 % dieser Läufe** (Seeds 36, 37: 7,7 und 6,0 %), mittleres Trendmaß +0,03: der Trend ordnet D-Läufe kaum anders als A-Läufe (AUC 0,5 bis 0,6), die Alarme liegen aber über denen bei A (0,5 bis 1,0 %) und weit unter denen bei C (etwa 80 %); Seed 38 im Scratch: 8,3 % | `test_control_d_noise_induced_tipping_is_not_announced`, `test_alarm_shares_are_stable_over_other_seeds` |
| Gegenprobe Erntemodell (x′ = x(1 − x) − h, Rauschen 0,02, 200 Läufe) | Alarmanteil **A 0,5 %, B 4,5 %, C 29,0 %** (200 von 200 kippen bei h/h_max = 1,044), AUC **0,828**: Die Warnung trennt im Erntemodell deutlich schwächer als im kubischen (C: 29,0 gegen 80,0 %). Die Rate fällt dort von 0,71 bei 0,5 auf 0,45 bei 0,8 h_max, im kubischen Modell von 1,35 auf 0,81; was davon die Ursache ist, ist hier nicht gemessen | `test_alarm_shares_harvest_model_and_auc`, `test_rates_quoted_for_the_two_models_before_the_alarm_comparison` |
| Tempo gegen Rauschen (Start unten bei h = 0, Anstieg mit Rate r, Euler–Maruyama, Schritt 0,02, 100 Läufe, Zufalls-Seed 35) | Mittleres Kipp-h (Streuung), σ = 0 / 0,05 / 0,1: **r = 0,02**: 0,5128 / 0,5110 (0,019) / 0,5019 (0,037); **r = 0,004**: 0,4314 / 0,4256 (0,013) / 0,4043 (0,028); **r = 0,0008**: 0,4013 / 0,3901 (0,011) / 0,3607 (0,021); Formel ohne Rauschen h_c + C·r^(2/3): 0,5283 / 0,4340 / 0,4017. Ohne Rauschen kippt das System in diesen Zellen nicht vor h_c (kleinster Wert 0,4013); **mit Rauschen kippt es im Mittel nur in 1 von 6 Zellen vor der Faltung** (r = 0,0008, σ = 0,1: 0,3607, 88 von 100 Läufen vor h_c; Seeds 36, 37: 0,3598 und 0,3580), einzelne Läufe aber auch bei r = 0,0008, σ = 0,05 (34 von 100) und r = 0,004, σ = 0,1 (26 von 100). Schnelle Rampe: spät; Rauschen: früh | `test_tempo_against_noise_grid`, `test_without_noise_the_tip_is_behind_the_fold_and_only_one_noisy_cell_is_before_it`, `test_tempo_mean_before_the_fold_holds_for_other_seeds` |

Ergebnis je Preset (Verdikt der App, Zufalls-Seed 35): Handbeispiel (h = 0,375): drei Gleichgewichte, Schleife bei r = 0,005 0,8767 breit; Langsame Schleife (r = 0,001): 0,8077; Schnelle Schleife (r = 0,02): Kippen bei 0,5125, Breite 1,0249; Nahe der Faltung (h = 0,365): drei Gleichgewichte;
Anstieg ohne Kippen: Verdikt „Fehlalarm“ (kein Lauf kippt), Bedingung C: „berechtigt“ (`test_equilibria_verdict_of_the_presets`, `test_loop_verdict_of_the_presets`, `test_the_alarm_verdict_of_the_preset_without_tipping_reports_false_alarms_not_justified_alarms`).

### Befunde und Korrekturen gegenüber dem Plan

- **Kippzeit interpoliert:** Der Plan stellte das Kippen am Ende des Schritts fest (Fehler bis r·Δt, bei r = 0,02 4·10⁻⁴). Das Orakel `solve_ivp` (DOP853, rtol 10⁻¹²) fand 2,6·10⁻⁴ Unterschied; die Kippzeit wird jetzt linear im Schritt interpoliert. Folge: Schleife bei r = 0,02 1,0249 statt 1,0256, Verhältnisse der Verzögerung 0,917 statt 0,9186 (r = 0,01) und 1,0001 statt 1,0002 (r = 10⁻⁵).
- **Alarmanteile mit anderen Zufallsfolgen:** Der Plan nannte A 1,0 %, B 13,5 %, C 74,5 % und AUC 0,944 aus einer Zufallsfolge des numpy-Generators; mit SplitMix64 und Seed 35 sind es 1,0 / 19,5 / 80,0 % und 0,941. B schwankt zwischen den Seeds am stärksten (11,5 bis 19,5 %, Standardfehler etwa 3 Punkte); deshalb Bänder von ±8 Punkten, keine Einzelzahl als Regel.
- **Erntemodell neu gemessen:** Die Planung hatte 5 / 7 / 28 % aus 60 Läufen (`out_rk_v9.txt`); mit 200 Läufen und denselben Bedingungen sind es 0,5 / 4,5 / 29,0 %. Das Urteil (schwächere Trennung als im kubischen Modell) bleibt.
- **Kontrolle D gemessen** (im Plan unter Z5 als offen): Kippen durch Rauschen bei konstantem h wird vom Trendmaß nicht angekündigt (5,7 % Alarm). Boettiger und Hastings (arXiv 1307.4415) schreiben für ein Ornstein–Uhlenbeck-Modell mit einem stabilen Punkt, steigende Varianz und Autokorrelation erschienen bei **jeder** großen Abweichung; das ist **keine** Aussage über unser Maß, Fenster und Modell, und die Messung widerlegt sie nicht. Sie zeigt nur, dass dieses Trendmaß hier keinen Vorlauf findet.
- **Start der Schleife:** Die Bahn startet bei h_c ∓ 12·r^(2/3) statt bei ∓0,8, damit langsame Raten schnell rechnen; gegen die Rampe ab −0,8 stimmt das Kipp-h auf 10⁻⁴ (`test_tip_of_the_own_rk4_agrees_with_solve_ivp`).
- **Eckfall gefunden:** Bei h = 0,375 und Rauschen 0,05 verlassen alle Ketten den Topf; die Simulation hat dann keine Werte (die erste Fassung brach ab). Die App meldet jetzt „Alle Ketten haben den Topf verlassen“ und zeigt nur die Theorie. Bei negativem h wird der obere Topf gerechnet (Spiegelung).
- **Zeitschritt der Rauschläufe:** Euler–Maruyama mit Schritt 0,01 (Verlangsamen) und 0,02 (Frühwarnung, Tempo) statt 0,005 der Vormessung; der Autokorrelationsunterschied (bis 0,0033) liegt im Band des Tests.

## Quellen

| Quelle | Stufe | Was hier verwendet wird |
|---|---|---|
| Boettiger, Hastings: No early warning signals for stochastic transitions: insights from large deviation theory, arXiv 1307.4415 (2013) | Wortlaut laut Planung gelesen (eine Antwort auf Drake 2013, Ornstein–Uhlenbeck-Modell mit einem stabilen Punkt); die Seite der Arbeit hier nur als Abstract erneut geprüft | steigende Varianz und Autokorrelation erscheinen bei jeder großen Abweichung, nicht nur vor einem Übergang; Bedingungen: genug Messpunkte, lange Bahn |
| Ritchie, Sieber (ratenbedingtes Kippen) | **nicht gelesen** | keine Aussage der README stützt sich darauf; das Airy-Gesetz ist eigene Herleitung mit eigener Messung |

## Ehrliche Grenzen

- **Ein Modell als Vehikel**, ein zweites (Ernte) nur für die Frühwarnung; die Hysterese ist nur im symmetrischen kubischen Modell gemessen.
- **Additives weißes Rauschen**, ein Rauschniveau je Tabelle (σ = 0,02 für Verlangsamen und Frühwarnung, 0,05 und 0,1 für das Tempo, 0,12 für Kontrolle D), **ein Fenster (200 Messungen) und ein Maß** (Autokorrelation; keine Varianz, kein Kendall-τ), eine feste Schwelle (Regler); „Alarm“ gilt nur für diese Wahl. Die Güte der Warnung ist nur für diese Wahl belegt.
- **Alarmanteile streuen mit den Läufen**: Standardfehler bis etwa 3 Punkte bei 200 und 5 Punkte bei 100 Läufen; B am stärksten.
- **Kippen heißt x > 0 (bzw. x < 0)**; andere Kriterien verschieben die Zahlen um Größen der Ordnung r.
- **Airy-Gesetz:** eigene Herleitung ohne Quellenbeleg; gilt für kleine Raten, Abweichung 8 % bei r = 0,01.
- **Fehlalarm und Alarm:** B zeigt, dass ein Anstieg ohne Kippen den Alarm auslöst, weil das System dem Kipppunkt tatsächlich näherkommt; nicht gemessen ist, wie sich das mit einem Maß oder Fenster ändert, das Anstieg und Kippen besser trennt.
- **Nicht mit echtem Mausklick getestet:** App im Browser über die Presets und mit AppTest geprüft; die Plotly-Diagramme sind gesperrt (keine Auswahl).

## Verifikation

Die Suite hat **124 Tests** (`pytest --collect-only -q tests`); sie läuft lokal in etwa einer Minute mit `-n 2`.

- **Kern (`tests/test_dynamics.py`):** Faltung und Handbeispiel h = 3/8 (Gleichgewichte, Raten, Näherung, Autokorrelation 0,9731, Varianzen), ein RK4-Schritt in Brüchen, Verzögerungskonstante und Schleifenbreite von Hand, Diskriminante gegen Zahl der Gleichgewichte, Trendmaß an konstruierten Reihen, AUC und Alarmanteil an Handzahlen, Zufallsblöcke über Blockgrenzen, SplitMix64 gegen Referenzwerte.
- **Orakel (`tests/test_oracle_kipppunkt_bifurkation.py`):** Gleichgewichte gegen die Cardano-Formel an 200 Werten, Kipp-h des eigenen RK4 gegen `scipy.integrate.solve_ivp` (DOP853) bei drei Raten, Verzögerungskonstante gegen die Airy-Nullstelle (`scipy.special`), Boltzmann-Varianz gegen `scipy.integrate.quad`, Simulation gegen Theorie an vier Abständen zur Faltung (Autokorrelation auf 0,5 %, Mittel gegen das Gleichgewicht), Ornstein–Uhlenbeck-Varianz,
  Gegenprobe „ohne Rauschen nie vor h_c“ an 120 Raten.
- **Eigenschaften (`tests/test_eigenschaften_kipppunkt_bifurkation.py`):** Spiegelsymmetrie der Schleife, Breite wächst monoton mit r und strebt gegen 2·h_c, kein Kippen vor der Faltung ohne Rauschen (30 Zufallsstarts), gleicher Zufalls-Seed gleiche Ausgabe, unabhängige Ketten, Varianz ∝ σ², Rate fällt zur Faltung, Randfälle (h genau auf der Faltung, Rampe erreicht h_c nicht, Start hinter der Faltung, σ = 0, Rate ohne Kippen).
- **Claims (`tests/test_claims.py`):** jede Zahl der Tabelle als Band, je Preset die erwartete Verdikt-Zeile. **Verdikte (`tests/test_verdicts.py`):** jede Schwelle und jede Zahl im Text an handgesetzten Werten.
- **Presets und Oberfläche:** Presets vollständig, paarweise verschieden, keine Zweierpotenz; AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum, alle Minima und alle Maxima zusammen, jedes Preset, jede Auswahl, h auf der Faltung und dahinter), Layout, Footer, Laufzeit-Importe.
- **Nicht im Test:** der Scratch-Lauf mit 1000 Raten und der Seed 38 stehen nur in dieser README; die Messungen aus Befunde und Korrekturen zu Seeds 36 bis 38 sind als Bänder im Test.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Gleichgewichte und Faltung, Kernfragen, fünf Reiter (Schleife, Verzögerung, Verlangsamen, Frühwarnung, Tempo gegen Rauschen) |
| `kpb_dynamics.py` | Kern: Gleichgewichte, Rampen (RK4), Euler–Maruyama, Boltzmann-Varianz, Trendmaß, Alarm, Tempo gegen Rauschen |
| `kpb_rng.py` | SplitMix64 (Kopie aus `lager-chaos-demo`, dort aus `restaurant-tischmix-demo`; vektorisiert, mit `offset`) |
| `kpb_evaluation.py` | Auswertungen der Abschnitte und ihre Verdikte (reine Funktionen) |
| `kpb_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `kpb_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `kpb_layout.py` | Fester Kopf (Stylesheet) |
| `kpb_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt; scipy für die Airy-Konstante) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `tests/` | Kern, Orakel, Eigenschaften, Claims, Verdikte, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Zweites Hysterese-Modell** (das Erntemodell hat nur die Frühwarnung gemessen) und **rotes Rauschen**, andere Fenster, Varianz oder Kendall-τ als Maß.
- **Ratenbedingtes Kippen nach Ritchie und Sieber:** die Arbeit ist nicht gelesen; das Airy-Gesetz bleibt eigene Herleitung.
- **Der Exponent des Verlangsamens als Schätzung aus Daten** (Zeitreihe ohne Modell): das ist die Aufgabe von [`exponent-aus-daten-demo`](https://github.com/sebastian-hanisch/exponent-aus-daten-demo).
- **PDF-Export:** nicht eingebaut.
- **Netzschwellen und Planwechsel** (`kaskaden-demo`, `komplexe-ansteckung-demo`, `gewichte-kippen-demo`): anderer Begriff von „Kipppunkt“.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Dynamische Systeme: Ruhepunkt, Kreislauf, Chaos](https://sebastianhanisch.net/konzepte-dynamische-systeme.html).
