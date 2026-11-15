"""Kipppunkt (Bifurkation) - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Normalform der Faltung x' = x - x^3 + h: Gleichgewichte und Faltung, Hysterese-Schleife, Verzögerung des Kippens nach dem r^(2/3)-Gesetz, kritisches
Verlangsamen, Frühwarnung und Fehlalarme, Tempo gegen Rauschen. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import kpb_constants as C
import kpb_dynamics as K
import kpb_evaluation as E
from kpb_evaluation import de
from kpb_layout import dunkles_design, kopf_css
from kpb_presets import (
    SETTING_SPECS,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from kpb_visualization import build_delay, build_loop, build_potential, build_rate, build_scurve, build_series, build_tempo, build_warning

st.set_page_config(page_title="Kipppunkt (Bifurkation) – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _s_curve():
    return E.s_curve()


@st.cache_data(show_spinner=False)
def _equilibria(h):
    return E.equilibria_analysis(h)


@st.cache_data(show_spinner=False)
def _loop(r):
    return K.hysteresis_loop(r, hmax=C.HMAX_LOOP)


@st.cache_data(show_spinner="Messe die Verzögerung bei sechs Raten …")
def _delay_rows():
    return E.delay_table()


@st.cache_data(show_spinner=False)
def _delay_now(r):
    return K.delay_measurement(r)


@st.cache_data(show_spinner="Simuliere die Ketten (rund eine Sekunde) …")
def _slowing(h, sigma, dtm, seed):
    return E.slowing_analysis(h, sigma, dtm, seed)


@st.cache_data(show_spinner="Simuliere die Läufe (rund eine Sekunde) …")
def _warning(cond, sigma, runs, seed):
    return E.warning_analysis(cond, sigma, runs, seed)


@st.cache_data(show_spinner="Simuliere alle Bedingungen in beiden Modellen (einige Sekunden) …")
def _warning_table(sigma, runs, seed, thr):
    return E.warning_table(sigma, runs, seed, thr)


@st.cache_data(show_spinner="Simuliere das Raster aus Tempo und Rauschen (rund eine Sekunde) …")
def _tempo(runs, seed, r, sigma_cell):
    return E.tempo_analysis(runs, seed, r, sigma_cell)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in kpb_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🌋 Kipppunkt (Bifurkation)")
st.markdown(
    """
**„Kipppunkt“ heißt hier eine Bifurkation einer Differentialgleichung**, nicht die Netzschwelle (Anteil aktiver Nachbarn) und nicht der Wechsel des besten Plans bei verschobenem Gewicht, wie in anderen Demos der Seite.
Im Modell **x′ = x − x³ + h** gibt es für |h| < 0,3849 zwei stabile Gleichgewichte; bei **h = ±0,3849** fällt eines mit dem Sattel zusammen (**Faltung**) und verschwindet.
Wird h mit endlichem Tempo verstellt, kippt das System ohne Rauschen **erst hinter der Faltung** und kommt auf einem anderen Weg zurück (**Hysterese**); mit Rauschen kann es früher kippen. Die Frage ist, ob es sich vorher ankündigt: durch **kritisches Verlangsamen** und seine Fehlalarme.

Weiter unten: **❓ Kernfragen**, die Reiter **🔬 Experimente** (Schleife, Verzögerung, kritisches Verlangsamen, Frühwarnung, Tempo gegen Rauschen), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/kipppunkt-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Stellen Sie sich eine Schale mit zwei Mulden vor und eine Kugel in der linken. Neigt man die Schale ganz langsam, wird die linke Mulde flacher, und bei einer bestimmten Neigung verschwindet sie: Die Kugel rollt in die rechte Mulde. Das ist ein **Kipppunkt**. Zurück kommt sie nicht auf demselben Weg, sondern erst, wenn man die Schale weit über die Ausgangsneigung hinaus zur anderen Seite neigt (**Hysterese**). Nahe am Kipppunkt erholt sich die Kugel nach einem Anstoß langsamer; das lässt sich als **Frühwarnung** messen, warnt aber nur teilweise und löst auch Fehlalarme aus. Die Demo rechnet dieses Modell nach, auch mit Zufallsstößen, und misst, wie gut die Warnung hier funktioniert.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/kipppunkt-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert der Kipppunkt", expanded=True):
    st.markdown(
        """
1. **Gleichgewichte:** x′ = 0 heißt h = x³ − x. Für |h| < h_c = 2/(3√3) = 0,3849 gibt es drei Lösungen (zwei stabil, eine instabil als Sattel dazwischen), für |h| > h_c nur eine.
2. **Rampe:** h wird mit der Rate r hinauf- oder hinuntergefahren. Das System folgt dem unteren Gleichgewicht, bis es bei der Faltung verschwindet; es kippt aber erst **später** (Verzögerung ≈ 1,9469 · r^(2/3)), und beim Rückweg kippt es erst hinter −h_c: eine **Schleife**.
3. **Rauschen:** Je näher h an der Faltung liegt, desto kleiner ist die Rate, mit der das System zum Gleichgewicht zurückkehrt: Störungen klingen langsamer ab (Autokorrelation steigt, Varianz wächst). Das kann man als **Frühwarnung** messen.
4. **Grenzen:** Rauschen kann das System auch **vor** der Faltung kippen lassen, und ein Anstieg ohne Kippen hebt die Autokorrelation ebenfalls (schwächer als vor einem Kippen): Fehlalarme.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Modell**")
    h = st.slider("Parameter h", *bounds("h_slider"), step=C.H_STEP, key="h_slider", help="Eingestellter Wert von h für Gleichgewichte und kritisches Verlangsamen; die Faltung liegt bei ±0,3849.")
    st.caption(f"Faltung bei ±{de(K.HC, 4)}; h ist {de(abs(h) / K.HC, 3)} h_c.")
    r = st.select_slider("Rampenrate r", options=C.R_OPTIONS, key="r_select", format_func=lambda v: de(v, 4), help="Wie schnell h hinauf- oder hinuntergefahren wird (je Zeiteinheit).")
    st.markdown("**Rauschen und Läufe**")
    sigma = st.slider("Rauschen σ", *bounds("sigma_slider"), step=C.SIGMA_STEP, key="sigma_slider", help="Standardabweichung des additiven weißen Rauschens (kritisches Verlangsamen, Frühwarnung).")
    runs = st.slider("Läufe je Bedingung", *bounds("runs_slider"), step=C.RUNS_STEP, key="runs_slider", help="Zahl der Zufallsläufe in Frühwarnung und Tempo gegen Rauschen; mehr Läufe sind genauer, aber langsamer.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Gleicher Zufalls-Seed, gleiche Zufallsfolgen.")
    st.button("🎲 Neuer Zufalls-Seed", width="stretch", on_click=randomize_seed)

h, r, sigma, runs, seed = float(h), float(r), float(sigma), int(runs), int(seed)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis: Gleichgewichte und Faltung")
eq = _equilibria(h)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Gleichgewichte", str(eq["n"]), help="Zahl der reellen Lösungen von x − x³ + h = 0 (Vorzeichen der Diskriminante 4 − 27 h²).")
m2.metric("Diskriminante 4 − 27 h²", de(eq["disc"], 3), help="Positiv: drei Gleichgewichte, negativ: eines.")
m3.metric("Abstand zur Faltung", de(K.HC - abs(h), 4, True), help="h_c − |h|; negativ hinter der Faltung.")
m4.metric("Rate des Topfes", de(K.rate(eq["well"]), 4) if eq["well"] is not None else "–", help="3x² − 1 am Gleichgewicht des Topfes, der bei wachsendem |h| zuerst verschwindet.")
c1, c2 = st.columns(2)
c1.plotly_chart(build_potential(eq), width="stretch", key="potential_chart")
c2.plotly_chart(build_scurve(_s_curve(), h), width="stretch", key="scurve_chart")
st.caption("Links: Potential V(x), die Kugel rollt bergab; gefüllt: stabiles, hohl: instabiles Gleichgewicht. Rechts: die Gleichgewichte über h (S-Kurve) mit den Faltungen (rote Rauten) und dem eingestellten h (gepunktet).")

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
st.markdown("**Wie viele Gleichgewichte gibt es bei diesem h, und wie weit ist die Faltung entfernt?**")
kind, text = E.verdict_equilibria(eq)
getattr(st, kind)(text)
st.markdown("**Wo kippt das System bei Rampenrate r, und kommt es auf demselben Weg zurück?**")
lp = _loop(r)
kind, text = E.verdict_loop(lp)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experimente")
tab_loop, tab_delay, tab_slow, tab_warn, tab_tempo = st.tabs(
    ["🔁 Hysterese-Schleife", "⏳ Verzögerung (r^(2/3))", "🐌 Kritisches Verlangsamen", "🚨 Frühwarnung und Fehlalarme", "⚡ Tempo gegen Rauschen"]
)

with tab_loop:
    st.markdown(
        "Die Rampe fährt h von −0,8 nach +0,8 (rot) und zurück (grün); gemessen wird, wo x die Null kreuzt (**Kippen**). Die Bahn startet 12 · r^(2/3) vor der Faltung auf dem Gleichgewicht, dort folgt sie ihm bis auf einen Fehler der Ordnung r. "
        "Eigener RK4, Zeitschritt 0,02. Gilt für die Rampenrate r der Seitenleiste."
    )
    st.plotly_chart(build_loop(_s_curve(), lp), width="stretch", key="loop_chart")
    kind, text = E.verdict_loop(lp)
    getattr(st, kind)(text)
    st.caption("Die Schleife ist die Breite zwischen den beiden Kipp-Orten; sie wird mit sinkender Rate schmaler und nähert sich 2·h_c. Die Spiegelsymmetrie x → −x, h → −h macht beide Kipp-Orte betragsgleich.")

with tab_delay:
    st.markdown(
        "Die **Verzögerung** δ = h_Kipp − h_c wächst nach der Formel **C · r^(2/3)** mit C = |z₁| / 3^(1/6) = 1,9469 (z₁ = −2,3381: erste Nullstelle der Airy-Funktion). Das ist **eigene Herleitung** aus der Normalform nahe der Faltung "
        "(s′ = r·t + √3·s², Substitution zur Airy-Gleichung), keine Quelle gelesen; die Messung ist die Probe."
    )
    rows = _delay_rows()
    now = _delay_now(r)
    st.plotly_chart(build_delay(rows), width="stretch", key="delay_chart")
    tbl = ["| r | gemessen δ | Formel C·r^(2/3) | Verhältnis |", "|---|---|---|---|"]
    for x in rows:
        tbl.append(f"| {E.sci(x['r'], 0)} | {E.sci(x['delta'], 3)} | {E.sci(x['formula'], 3)} | {de(x['ratio'], 4)} |")
    st.markdown("\n".join(tbl))
    kind, text = E.verdict_delay(rows, now)
    getattr(st, kind)(text)

with tab_slow:
    st.markdown(
        "Je näher h an der Faltung liegt, desto langsamer kehrt das System nach einer Störung zurück: Rate = 3x² − 1 am Gleichgewicht. Mit Rauschen σ gilt für die Autokorrelation im Messabstand Δ: **e^(−Rate·Δ)**, für die Varianz "
        "σ²/(2·Rate) (linear) und exakt die Boltzmann-Dichte exp(−2V/σ²). Simuliert werden 100 Ketten (Euler–Maruyama, Schritt 0,01) im Topf; gilt für h, σ und den Zufalls-Seed der Seitenleiste."
    )
    dtm = st.select_slider("Messabstand Δ", options=C.DTM_OPTIONS, key="dtm_select", format_func=lambda v: de(v, 2), help="Zeit zwischen zwei Messungen, aus denen die Autokorrelation bestimmt wird.")
    sl = _slowing(h, sigma, float(dtm), seed)
    s1, s2 = st.columns(2)
    s1.plotly_chart(build_rate(h), width="stretch", key="rate_chart")
    if sl is not None:
        s2.plotly_chart(build_series(sl["sample"], float(dtm), "Beispielkette im Topf"), width="stretch", key="slow_series_chart")
    kind, text = E.verdict_slowing(sl, h)
    getattr(st, kind)(text)

with tab_warn:
    st.markdown(
        "**Frühwarnung:** Aus den Messungen wird die Autokorrelation je Fenster (200 Messungen, Schritt 50, linearer Trend entfernt) berechnet; das **Trendmaß** ist die Korrelation zwischen Fensternummer und Autokorrelation. "
        "**Alarm** heißt Trendmaß über der Schwelle. Bedingungen: **A** konstantes h = 0,5 h_c; **B** Anstieg von 0,5 auf 0,8 h_c, das System kippt nie; **C** Anstieg von 0,5 auf 1,2 h_c, es kippt; "
        "**D** (Kontrolle) konstantes h = 0,8 h_c mit starkem Rauschen 0,12, es kippt durch das Rauschen. 4000 Messungen im Abstand 0,1, Rauschen σ der Seitenleiste (D: 0,12)."
    )
    w1, w2 = st.columns(2)
    cond = w1.selectbox("Bedingung", C.COND_OPTIONS, key="cond_select", format_func=lambda k: K.CONDITIONS[k]["label"], help="Welche Bedingung im Einzelbild unten gezeigt wird.")
    thr = w2.slider("Alarmschwelle (Trendmaß)", *bounds("thr_slider"), step=C.THR_STEP, key="thr_slider", help="Trendmaß, ab dem ein Lauf als Alarm zählt.")
    wr = _warning(cond, sigma, runs, seed)
    st.plotly_chart(build_warning(wr, float(thr)), width="stretch", key="warning_chart")
    kind, text = E.verdict_warning(wr, float(thr))
    getattr(st, kind)(text)
    st.markdown("**Alle Bedingungen und zwei Modelle nebeneinander** (kubisch gegen Erntemodell x′ = x(1 − x) − h, r = K = 1, Faltung bei h_max = 0,25):")
    if st.button("Vergleichstabelle berechnen", help="Rechnet A bis D im kubischen Modell und A bis C im Erntemodell mit den eingestellten Läufen; einige Sekunden."):
        st.session_state["kpb_table"] = ((sigma, runs, seed, float(thr)), _warning_table(sigma, runs, seed, float(thr)))
    saved = st.session_state.get("kpb_table")
    if saved is not None and saved[0] == (sigma, runs, seed, float(thr)):
        t = saved[1]
        lines = ["| Modell | Bedingung | Läufe mit Kippen | Alarm | mittleres Trendmaß |", "|---|---|---|---|---|"]
        for row in t["rows"]:
            lines.append(f"| {'kubisch' if row['model'] == 'cubic' else 'Ernte'} | {row['label'].split(':')[0]} | {row['n_tipped']} von {row['n_runs']} | "
                         f"{E.pct(row['alarm'])} ({row['n_valid']} Läufe mit Trendmaß) | {de(row['mean'], 2, True)} |")
        st.markdown("\n".join(lines))
        kind, text = E.verdict_table(t)
        getattr(st, kind)(text)
    else:
        st.caption("Die Tabelle gilt für die Einstellungen beim Berechnen; nach einer Änderung der Regler erneut berechnen.")

with tab_tempo:
    st.markdown(
        "Das System startet unten bei h = 0 und h steigt mit der Rate r; gemessen wird das mittlere Kipp-h über die Läufe der Seitenleiste (Euler–Maruyama, Schritt 0,02; ohne Rauschen ein Lauf). "
        "Das Raster zeigt r = 0,02; 0,004; 0,0008 und σ = 0; 0,05; 0,1; der Stern ist die eigene Zelle (r der Seitenleiste, σ unten)."
    )
    sc = st.slider("Rauschen der eigenen Zelle", *bounds("sigma_cell_slider"), step=C.SIGMA_CELL_STEP, key="sigma_cell_slider", help="0: ohne Rauschen (ein Lauf).")
    tp = _tempo(runs, seed, r, float(sc))
    st.plotly_chart(build_tempo(tp), width="stretch", key="tempo_chart")
    tbl = ["| r | σ = 0 | σ = 0,05 | σ = 0,1 | Formel ohne Rauschen h_c + C·r^(2/3) |", "|---|---|---|---|---|"]
    for i, rr in enumerate(C.GRID_R):
        cells = " | ".join(f"{de(tp['cells'][i][j]['mean'], 4)} ({de(tp['cells'][i][j]['std'], 4)})" for j in range(len(C.GRID_SIGMA)))
        tbl.append(f"| {de(rr, 4)} | {cells} | {de(tp['cells'][i][0]['formula'], 4)} |")
    st.markdown("\n".join(tbl))
    st.caption("Mittleres Kipp-h, in Klammern die Streuung über die Läufe.")
    kind, text = E.verdict_tempo(tp)
    getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Gemessen |
|---|---|---|
| **Ein Modell** (kubische Normalform, symmetrisch) | Das Erntemodell trennt die Frühwarnung schwächer (Reiter Frühwarnung); die Hysterese ist nur im kubischen Modell gemessen | Reiter Frühwarnung |
| **Additives weißes Rauschen, ein Fenster, ein Maß** | Rotes Rauschen, andere Fenster, Varianz oder Kendall-τ können die Güte der Warnung verschieben | nicht gemessen |
| **Kippen heißt: x kreuzt 0** | Andere Kriterien verschieben die Kipp-Orte um Größen der Ordnung r | README |
| **Feste Alarmschwelle** | Die Alarmanteile hängen an der Schwelle (Regler) und an der Zahl der Läufe (Streuung bis etwa ±5 Punkte) | Reiter Frühwarnung |
| **Airy-Gesetz** | Gilt für kleine Raten (Verhältnis 0,917 bei r = 0,01, 1,0001 bei 10⁻⁵); eigene Herleitung, nicht aus einer Quelle | Reiter Verzögerung |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/kipppunkt-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben ist die Gleichung $x' = x - x^3 + h$ mit einem Parameter $h$, der mit der Rate $r$ langsam verändert wird, und zufälligen Stößen der Stärke $\sigma$. Gesucht ist, bei welchem $h$ das System kippt, auf welchem Weg es zurückkehrt und ob sich das Kippen aus den Messungen vorher ankündigt.

| Symbol | Bedeutung |
|---|---|
| $x$ | Zustand des Systems (die Lage der Kugel) |
| $h$ | Parameter, der verstellt wird (die Neigung der Schale) |
| $h_c$ | Faltung: der Wert von $h$, bei dem ein Gleichgewicht verschwindet, $h_c = 2/(3\sqrt{3}) = 0{,}3849$ |
| $r$ | Rampenrate: wie schnell $h$ verstellt wird |
| $\sigma$ | Stärke des zufälligen Rauschens |
| $\kappa$ | Rate eines Gleichgewichts, $\kappa = 3x^2 - 1$ (positiv: stabil) |
| $\delta$ | Verzögerung des Kippens gegenüber der Faltung |
| $\rho$ | Autokorrelation: wie ähnlich sich zwei Messungen im Abstand $\Delta$ sind |
| $T$ | Trendmaß der Frühwarnung: Korrelation zwischen Fensternummer und Autokorrelation |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/kipppunkt-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(
        r"""
**Lesehilfe.** Gegeben ist die Gleichung $x' = x - x^3 + h$ mit einem Parameter $h$, der langsam verändert wird; gesucht ist, bei welchem $h$ das System kippt und ob sich das ankündigt.

| Symbol | Bedeutung |
|---|---|
| $x$ | Zustand |
| $h$ | Parameter (Rampe $h(t) = h_0 \pm r\,t$) |
| $h_c$ | Faltung, $h_c = 2/(3\sqrt{3}) = 0{,}3849$ |
| $r$ | Rampenrate |
| $\sigma$ | Stärke des Rauschens |
| $\kappa$ | Rate des Gleichgewichts, $\kappa = 3x^2 - 1$ |

**Gleichgewichte.** $x - x^3 + h = 0$, also $h = x^3 - x$. Die Ableitung $\mathrm{d}h/\mathrm{d}x = 3x^2 - 1$ verschwindet bei $x = \pm 1/\sqrt{3}$, dort liegt $h = \mp 2/(3\sqrt{3})$. Die Diskriminante von $x^3 - x - h$ ist

$$\Delta = 4 - 27 h^2$$

mit drei reellen Lösungen für $\Delta > 0$ und einer für $\Delta < 0$. Ein Gleichgewicht ist stabil, wenn $\kappa = 3x^2 - 1 > 0$.

**Rate nahe der Faltung** (Näherung, 18 % zu klein bei $h = 0$, 5 % bei $0{,}95\,h_c$):

$$\kappa \approx \sqrt{8/3}\,\sqrt{1 - h/h_c}$$

**Verzögerung des Kippens** (eigene Herleitung aus der Normalform $s' = r t + \sqrt{3}\,s^2$ mit $s = x + 1/\sqrt{3}$; die Substitution $s = -u'/(\sqrt{3}\,u)$ führt auf die Airy-Gleichung $u'' = -\sqrt{3}\,r\,t\,u$, deren Lösung bei der ersten Nullstelle $z_1$ von $\mathrm{Ai}$ ihre Polstelle hat):

$$\delta = h_{\mathrm{Kipp}} - h_c \approx \frac{|z_1|}{3^{1/6}}\, r^{2/3} = 1{,}9469\, r^{2/3}$$

**Schleife.** Breite $\approx 2 h_c + 2 \cdot 1{,}9469\, r^{2/3}$; für $r \to 0$ gilt $2 h_c = 0{,}7698$. Symmetrie: $f(-x, -h) = -f(x, h)$.

**Rauschen.** Euler–Maruyama mit Schritt $\Delta t$ und Normalverteilung $\xi_n$:

$$x_{n+1} = x_n + (x_n - x_n^3 + h)\,\Delta t + \sigma \sqrt{\Delta t}\;\xi_n$$

Im Topf gilt linear für Autokorrelation und Varianz

$$\rho(\Delta) = e^{-\kappa \Delta}, \qquad \operatorname{Var} = \frac{\sigma^2}{2\kappa}$$

und exakt die stationäre Dichte

$$p(x) \propto \exp\!\left(-\frac{2 V(x)}{\sigma^2}\right), \qquad V(x) = -\left(\frac{x^2}{2} - \frac{x^4}{4} + h x\right)$$

**Trendmaß.** Aus den Fenstern $k = 1, \dots, m$ der Zeitreihe (Länge 200, Schritt 50, linearer Trend entfernt) mit Autokorrelationen $\rho_k$:

$$T = \operatorname{corr}(k, \rho_k)$$

Alarm heißt $T$ über der Schwelle. Das Modell ist vollständig bis auf das Erntemodell des Vergleichs, $x' = x(1 - x) - h$ mit $x_0 = \tfrac{1}{2}\left(1 + \sqrt{1 - h/h_{\max}}\right)$, $h_{\max} = 1/4$, Kippen bei $x < 0{,}1$.

**Literatur.** Boettiger, C., Hastings, A. (2013): No early warning signals for stochastic transitions: insights from large deviation theory, arXiv 1307.4415 (eine Antwort auf Drake 2013; Wortlaut gelesen: Ornstein–Uhlenbeck-Modell mit einem stabilen Punkt; steigende Varianz und Autokorrelation erscheinen bei jeder großen Abweichung, nicht nur vor einem Übergang; Bedingungen: genug Messpunkte, lange Bahn). Ritchie und Sieber (ratenbedingtes Kippen) wurden **nicht gelesen**; das Airy-Gesetz oben ist eigene Herleitung mit eigener Messung.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)

sync_query_params({key: st.session_state[key] for key in SETTING_SPECS})
