"""
Presentation Streamlit — Prediction de la pluie en Australie
Classification binaire avec XGBoost, metrique principale F2-score.
Source des donnees live : Open-Meteo API (gratuit, sans cle).
"""

import json
import os
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Prediction Pluie Australie",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR     = Path(__file__).parent
FIG_DIR     = APP_DIR / ".." / "reports" / "figures"
MODEL_PATH  = APP_DIR / ".." / "models" / "final_model.joblib"
RESULTS_PATH = APP_DIR / ".." / "reports" / "final_results.json"

# ---------------------------------------------------------------------------
# CSS : theme professionnel australien (ocean + sable + nuit)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #0d1b2e; }
[data-testid="stSidebar"] * { color: #cdd8e3 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.88rem; }
[data-testid="stSidebar"] hr { border-color: #2a3f5f; }
h1 { color: #0d1b2e; font-weight: 700; margin-bottom: 0.2rem; }
h2 { color: #1a5276; border-bottom: 2px solid #1a5276; padding-bottom: 0.25rem; }
h3 { color: #0d1b2e; }
.kpi-row { display: flex; gap: 1rem; margin: 1rem 0; }
.kpi { background: #eaf3fb; border-left: 4px solid #1a5276;
       padding: 0.8rem 1.2rem; border-radius: 4px; flex: 1; }
.kpi strong { display: block; font-size: 1.6rem; color: #0d1b2e; }
.kpi span   { font-size: 0.82rem; color: #555; }
.alert-rain    { background: #dceefb; border-left: 5px solid #1565c0;
                 padding: 1.2rem 1.5rem; border-radius: 4px; margin-top: 1rem; }
.alert-no-rain { background: #fef9e7; border-left: 5px solid #d4ac0d;
                 padding: 1.2rem 1.5rem; border-radius: 4px; margin-top: 1rem; }
.step { display: flex; align-items: flex-start; gap: 0.8rem; margin: 0.35rem 0; }
.step-num { background: #1a5276; color: white; border-radius: 4px;
            padding: 0.2rem 0.55rem; font-weight: 700; font-size: 0.85rem;
            white-space: nowrap; margin-top: 0.15rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Donnees de reference
# ---------------------------------------------------------------------------
STATIONS = {
    "Adelaide":(-34.93,138.60),"Albany":(-35.02,117.88),"Albury":(-36.08,146.92),
    "AliceSprings":(-23.70,133.88),"BadgerysCreek":(-33.88,150.73),
    "Ballarat":(-37.55,143.85),"Bendigo":(-36.76,144.28),"Brisbane":(-27.47,153.02),
    "Cairns":(-16.92,145.77),"Canberra":(-35.28,149.13),"Cobar":(-31.50,145.83),
    "CoffsHarbour":(-30.30,153.12),"Dartmoor":(-37.92,141.27),"Darwin":(-12.46,130.84),
    "GoldCoast":(-28.00,153.43),"Hobart":(-42.88,147.33),"Jabiru":(-12.66,132.89),
    "Katherine":(-14.47,132.27),"Launceston":(-41.43,147.14),"Melbourne":(-37.81,144.96),
    "MelbourneAirport":(-37.67,144.83),"Mildura":(-34.19,142.15),"Moree":(-29.47,149.83),
    "MountGambier":(-37.83,140.78),"MountGinini":(-35.53,148.77),"Newcastle":(-32.92,151.78),
    "Nhil":(-36.33,141.65),"NorahHead":(-33.28,151.57),"NorfolkIsland":(-29.04,167.96),
    "Nuriootpa":(-34.47,138.99),"PearceRAAF":(-31.67,116.03),"Penrith":(-33.75,150.70),
    "Perth":(-31.95,115.86),"PerthAirport":(-31.94,115.97),"Portland":(-38.34,141.60),
    "Richmond":(-33.60,150.75),"Sale":(-38.10,147.07),"SalmonGums":(-32.98,121.63),
    "Sydney":(-33.87,151.21),"SydneyAirport":(-33.94,151.18),"Townsville":(-19.25,146.82),
    "Tuggeranong":(-35.42,149.09),"Uluru":(-25.35,131.04),"WaggaWagga":(-35.16,147.47),
    "Walpole":(-34.98,116.73),"Watsonia":(-37.71,145.08),"Williamtown":(-32.80,151.84),
    "Witchcliffe":(-34.03,115.10),"Wollongong":(-34.42,150.89),"Woomera":(-31.15,136.82),
}

WIND_DIR_MAP = {
    "N":0,"NNE":22.5,"NE":45,"ENE":67.5,"E":90,"ESE":112.5,
    "SE":135,"SSE":157.5,"S":180,"SSW":202.5,"SW":225,"WSW":247.5,
    "W":270,"WNW":292.5,"NW":315,"NNW":337.5,
}

SEASON_MAP = {
    12:"Summer",1:"Summer",2:"Summer",
    3:"Autumn",4:"Autumn",5:"Autumn",
    6:"Winter",7:"Winter",8:"Winter",
    9:"Spring",10:"Spring",11:"Spring",
}

# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)

def fig_path(name: str) -> str:
    return str(FIG_DIR / name)

def deg_to_compass(degrees) -> str:
    if pd.isna(degrees):
        return np.nan
    dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
            "S","SSW","SW","WSW","W","WNW","NW","NNW"]
    return dirs[round(float(degrees) / 22.5) % 16]

def compass_sincos(compass_str):
    if pd.isna(compass_str):
        return np.nan, np.nan
    deg = WIND_DIR_MAP.get(str(compass_str), np.nan)
    if pd.isna(deg):
        return np.nan, np.nan
    rad = np.radians(deg)
    return float(np.sin(rad)), float(np.cos(rad))

@st.cache_data(ttl=3600)
def fetch_open_meteo(lat: float, lon: float) -> dict:
    """Appel Open-Meteo : 4 derniers jours en daily + hourly pour extraire 9h et 15h."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,"
                 "windspeed_10m_max,windgusts_10m_max,winddirection_10m_dominant",
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,"
                  "cloudcover,windspeed_10m,winddirection_10m",
        "past_days": 4,
        "timezone": "auto",
        "timeformat": "iso8601",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()

def build_feature_row(location: str, api_data: dict) -> pd.DataFrame:
    """
    Construit le vecteur de features pour le jour courant a partir des donnees Open-Meteo.
    Reproduit fidelement le feature engineering de ml_pipeline.py (df2).
    Retourne un DataFrame a une ligne pret pour model.predict_proba().
    """
    daily  = pd.DataFrame(api_data["daily"])
    hourly = pd.DataFrame(api_data["hourly"])
    daily["time"]  = pd.to_datetime(daily["time"])
    hourly["time"] = pd.to_datetime(hourly["time"])
    hourly["date"] = hourly["time"].dt.date
    hourly["hour"] = hourly["time"].dt.hour

    def get_hourly(day_date, hour):
        mask = (hourly["date"] == day_date) & (hourly["hour"] == hour)
        subset = hourly[mask]
        return subset.iloc[0] if len(subset) else None

    rows = []
    for _, day in daily.tail(4).iterrows():
        d   = day["time"].date()
        h9  = get_hourly(d, 9)
        h15 = get_hourly(d, 15)

        def safe(series, col):
            return float(series[col]) if series is not None and col in series.index else np.nan

        rain_mm = float(day.get("precipitation_sum") or 0)
        rows.append({
            "Date":          pd.Timestamp(d),
            "Location":      location,
            "MinTemp":       float(day["temperature_2m_min"]) if pd.notna(day["temperature_2m_min"]) else np.nan,
            "MaxTemp":       float(day["temperature_2m_max"]) if pd.notna(day["temperature_2m_max"]) else np.nan,
            "Rainfall":      rain_mm,
            "Evaporation":   np.nan,
            "Sunshine":      np.nan,
            "WindGustSpeed": float(day["windgusts_10m_max"]) if pd.notna(day.get("windgusts_10m_max")) else np.nan,
            "WindGustDir":   deg_to_compass(day.get("winddirection_10m_dominant")),
            "WindSpeed9am":  safe(h9,  "windspeed_10m"),
            "WindSpeed3pm":  safe(h15, "windspeed_10m"),
            "WindDir9am":    deg_to_compass(safe(h9,  "winddirection_10m")),
            "WindDir3pm":    deg_to_compass(safe(h15, "winddirection_10m")),
            "Humidity9am":   safe(h9,  "relative_humidity_2m"),
            "Humidity3pm":   safe(h15, "relative_humidity_2m"),
            "Pressure9am":   safe(h9,  "surface_pressure"),
            "Pressure3pm":   safe(h15, "surface_pressure"),
            "Cloud9am":      safe(h9,  "cloudcover") / 100 * 9 if h9 is not None else np.nan,
            "Cloud3pm":      safe(h15, "cloudcover") / 100 * 9 if h15 is not None else np.nan,
            "Temp9am":       safe(h9,  "temperature_2m"),
            "Temp3pm":       safe(h15, "temperature_2m"),
            "RainToday":     1 if rain_mm > 1.0 else 0,
        })

    df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)

    df["Month"]     = df["Date"].dt.month
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["Season"]    = df["Month"].map(SEASON_MAP)

    df["Delta_Pressure"] = df["Pressure3pm"] - df["Pressure9am"]
    df["Delta_Humidity"] = df["Humidity3pm"]  - df["Humidity9am"]
    df["Delta_Wind"]     = df["WindSpeed3pm"] - df["WindSpeed9am"]
    df["TempRange"]      = df["MaxTemp"]      - df["MinTemp"]
    df["Wind_x_Humidity"] = (
        df["WindGustSpeed"].fillna(0) * df["Humidity3pm"].fillna(df["Humidity3pm"].median())
    )

    df["Pressure_drop_flag"] = (df["Delta_Pressure"] < -2).astype(int)
    df["HighHumidity_flag"]  = (df["Humidity3pm"] > 85).astype(int)
    df["StrongWind_flag"]    = (df["WindGustSpeed"] > 60).astype(int)
    df["PressureLow_flag"]   = (df["Pressure9am"] < 1010).astype(int)

    for col in ["WindGustDir", "WindDir9am", "WindDir3pm"]:
        sc = df[col].apply(compass_sincos)
        df[f"{col}_sin"] = sc.apply(lambda x: x[0])
        df[f"{col}_cos"] = sc.apply(lambda x: x[1])

    df["Rainfall_roll3"]      = df["Rainfall"].shift(1).rolling(3).mean()
    df["Humidity3pm_roll3"]   = df["Humidity3pm"].shift(1).rolling(3).mean()
    df["WindGustSpeed_roll3"] = df["WindGustSpeed"].shift(1).rolling(3).max()
    df["Pressure9am_roll3"]   = df["Pressure9am"].shift(1).rolling(3).mean()
    df["TempRange_roll3"]     = df["TempRange"].shift(1).rolling(3).mean()

    df = df.drop(columns=["WindGustDir", "WindDir9am", "WindDir3pm", "Date"])
    return df.tail(1).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Slides
# ---------------------------------------------------------------------------

def slide_introduction():
    st.markdown("# Prediction de la pluie en Australie")
    st.markdown("### Classification binaire avec Machine Learning — Bureau of Meteorology 2007–2017")
    st.markdown("---")
    st.markdown("""
<div class="kpi-row">
  <div class="kpi"><strong>145 000</strong><span>observations</span></div>
  <div class="kpi"><strong>49</strong><span>stations</span></div>
  <div class="kpi"><strong>10 ans</strong><span>2007–2017</span></div>
  <div class="kpi"><strong>F2-score</strong><span>metrique principale</span></div>
</div>
""", unsafe_allow_html=True)
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("""
**Question**

Etant donnees les mesures meteorologiques d'aujourd'hui,
va-t-il pleuvoir demain ?

**Approche**

Machine Learning classique (pas de Deep Learning).
Metrique principale : **F2-score**, car manquer de la pluie
coute bien plus qu'une fausse alarme.

**Donnees**

Bureau of Meteorology australien (BOM).
49 stations reparties sur l'ensemble du continent.
Serie temporelle de novembre 2007 a juin 2017.
""")
    with c2:
        st.image(fig_path("08_australia_map.png"),
                 caption="Les 49 stations du dataset (humidite, pression, pluie)")


def slide_probleme():
    st.markdown("# Le Probleme")
    st.markdown("## Deux defis structurels")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("""
**1. Desequilibre de classes**

Seulement **22 %** des jours sont pluvieux.
Un modele naif qui predit toujours "pas de pluie"
atteint 78 % d'accuracy — mais est totalement inutile.
L'accuracy n'est pas la bonne metrique.

**2. Bruit de label**

Les capteurs enregistrent parfois 0.1 mm sur un ciel
sans nuage, ou ratent une petite averse.
Cette imprecision dans la labelisation doit etre prise
en compte explicitement via la ponderation.

**Pourquoi le F2-score ?**

| Erreur | Situation reelle | Cout |
|--------|-----------------|------|
| Faux Negatif | Predit beau, il pleut | Eleve |
| Faux Positif | Predit pluie, beau temps | Faible |

Le F2 penalise **deux fois plus** les Faux Negatifs.
""")
    with c2:
        st.image(fig_path("01_target_distribution.png"))
        st.latex(r"F_2 = \frac{5 \cdot P \cdot R}{4 \cdot P + R}")


def slide_dataset():
    st.markdown("# Le Dataset")
    st.markdown("## weatherAUS.csv — Bureau of Meteorology")
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.markdown("""
| Caracteristique | Valeur |
|----------------|--------|
| Source | BOM (Commonwealth of Australia) |
| Periode | Nov 2007 — Juin 2017 |
| Localites | 49 villes |
| Observations | ~145 000 jours |
| Variables | 23 (16 num., 7 categ.) |
| Target | RainTomorrow (Yes / No) |
| Ratio No/Yes | 3.5 pour 1 |

**Variables les plus importantes (SHAP) :**
- Humidity3pm, Wind_x_Humidity
- Pressure3pm, Sunshine, Delta_Pressure

**Valeurs manquantes notables :**
- Sunshine : 48 % de NaN
- Evaporation : 43 % de NaN

Conservees car informatives — gerees
par SimpleImputer (mediane) dans le pipeline.
""")
    with c2:
        st.image(fig_path("02_correlation_target.png"),
                 caption="Correlations des variables numeriques avec RainTomorrow")


def slide_demarche():
    st.markdown("# La Demarche — Pipeline 13 Etapes")
    etapes = [
        ("01", "EDA",
         "Distributions, correlations, saisonnalite, valeurs manquantes."),
        ("02", "Cleaning",
         "Aberrations physiques uniquement : pression < 900 hPa, "
         "humidite > 100 %, vitesses negatives."),
        ("03", "Feature Engineering",
         "Derivees temporelles, fenetres glissantes 3 j, encodage cyclique "
         "sin/cos, interactions."),
        ("04", "Split chronologique",
         "80/20 sur les dates triees. Interdit de voir le futur pendant "
         "l'entrainement."),
        ("05", "Preprocessing",
         "Pipeline sklearn : RobustScaler + SimpleImputer + OneHotEncoder."),
        ("06", "Ponderation anti-bruit",
         "sample_weight 0.6 / 0.8 / 1.5 selon la fiabilite du label "
         "(ambiguite pluviometrique)."),
        ("07", "Feature Selection",
         "SelectKBest (Mutual Information, random_state=42). k optimise "
         "en cross-validation."),
        ("08", "Pas de SMOTE",
         "SMOTE ignore la temporalite et genere des observations "
         "meteorologiquement incoherentes."),
        ("09", "Comparaison modeles",
         "RandomForest, XGBoost, LightGBM evalues en CV 5-fold (F2-score)."),
        ("10", "Hyperparameter Tuning",
         "RandomizedSearchCV 30 iterations sur XGBoost et LightGBM."),
        ("11", "Entrainement final",
         "Meilleur modele reentrainee sur 100 % du train."),
        ("12", "Evaluation",
         "F2, Recall, Precision, ROC-AUC, Brier Score + seuil optimal."),
        ("13", "Interpretabilite",
         "SHAP TreeExplainer : importance, beeswarm, dependance, "
         "saisonnalite, carte climatologique."),
    ]
    for num, titre, desc in etapes:
        st.markdown(
            f'<div class="step">'
            f'<span class="step-num">{num}</span>'
            f'<span><strong>{titre}</strong> — {desc}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def slide_features():
    st.markdown("# Feature Engineering")
    st.markdown("## Encoder la physique meteorologique dans les donnees")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**Derivees temporelles intra-journalieres**
```python
Delta_Pressure = Pressure3pm - Pressure9am
Delta_Humidity = Humidity3pm - Humidity9am
```
Une chute de pression dans la journee signale
l'approche d'un front pluvieux.

**Encodage cyclique sin/cos**
```python
Month_sin = sin(2*pi * Month / 12)
Month_cos = cos(2*pi * Month / 12)
```
Janvier et decembre sont adjacents dans le calendrier.
Un encodage lineaire ne le voit pas.

**Fenetres glissantes 3 jours**
```python
Rainfall_roll3 = shift(1).rolling(3).mean()
```
`shift(1)` obligatoire : on ne regarde que le passe,
jamais le jour courant (evite la fuite de donnees).

**Indicateurs binaires**
```python
Pressure_drop_flag = (Delta_Pressure < -2)
HighHumidity_flag  = (Humidity3pm > 85 %)
StrongWind_flag    = (WindGustSpeed > 60 km/h)
```
""")
    with c2:
        st.image(fig_path("03_seasonality.png"),
                 caption="Saisonnalite de la pluie — hiver austral (juin-aout) dans le sud")


def slide_resultats():
    r = load_results()
    st.markdown("# Modelisation et Resultats")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.markdown("""
### Split chronologique : regle fondamentale

Un split aleatoire sur des donnees temporelles
constitue une **fuite de donnees** (data leakage).
Le modele peut apprendre sur 2016 et etre evalue
sur 2009 — les metriques sont alors trompeuses.

```
Split aleatoire  →  metriques artificiellement gonflees
Split chrono     →  metriques honnetes sur le futur
2007 ——— 2015 | 2015 ——— 2017
   TRAIN (80%)      TEST (20%)
```

### Modele final : XGBoost
""")
        st.markdown(f"""
| Metrique | Valeur |
|----------|--------|
| F2-score (seuil opt.) | **{r['f2_test']:.3f}** |
| Recall pluie | **{r['recall_rain']:.3f}** |
| Precision pluie | {r['precision_rain']:.3f} |
| ROC-AUC | {r['roc_auc']:.3f} |
| Brier Score | {r['brier_score']:.3f} |
| Seuil optimal | {r['optimal_threshold']:.3f} |
| Faux Negatifs | {r['false_negatives']} sur {r['test_size']:,} |
""")
    with c2:
        st.image(fig_path("04_evaluation.png"))


def slide_shap():
    st.markdown("# Interpretabilite — SHAP")
    st.markdown("""
SHAP (SHapley Additive exPlanations) decompose chaque prediction en contributions individuelles.
Pour les modeles arbres, **TreeExplainer** calcule les valeurs de Shapley exactement, en temps polynomial.
""")
    tab1, tab2, tab3, tab4 = st.tabs([
        "Importance globale", "Beeswarm", "Dependance", "Saisonnalite"
    ])
    with tab1:
        st.image(fig_path("05a_shap_importance.png"),
                 caption="Mean |SHAP| — Top 20 features par importance globale")
        st.markdown(
            "Humidity3pm et Wind_x_Humidity dominent. "
            "Delta_Pressure capture l'approche des fronts."
        )
    with tab2:
        st.image(fig_path("05b_shap_beeswarm.png"),
                 caption="Chaque point = une observation. Rouge = valeur haute, Bleu = valeur basse.")
        st.markdown("""
**Lecture :**
- Humidity3pm rouge a droite : humidite elevee pousse vers la pluie.
- Sunshine rouge a gauche : ensoleillement eleve pousse vers pas de pluie.
- Delta_Pressure bleu a droite : chute de pression signale un front pluvieux.
""")
    with tab3:
        st.image(fig_path("06_shap_dependence.png"),
                 caption="Effet marginal de chaque feature sur la prediction (+ interaction automatique)")
    with tab4:
        st.image(fig_path("07_shap_seasonal.png"),
                 caption="Importance SHAP par saison — le modele a capture les patterns climatiques australiens")


def slide_carte():
    st.markdown("# Analyse Spatiale")
    st.markdown("## Les profils climatiques varient-ils selon la localite ?")
    st.image(
        fig_path("08_australia_map.png"),
        caption="Moyennes 2007–2017 : humidite (gauche), pression + vents (centre), pluviometrie (droite)",
        use_container_width=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**Observations :**
- Nord tropical (Darwin, Cairns) : humidite elevee, pression basse, pluviometrie dominante
- Interieur aride (Alice Springs, Uluru) : sec, peu de pluie
- Sud-Est (Melbourne, Sydney) : profils intermediaires temperes
""")
    with c2:
        st.markdown("""
**Conclusion SHAP :**
Location n'est pas dans le top 20 des features SHAP.
Le modele capture implicitement la geographie via
Humidity3pm, Pressure9am et WindGustSpeed — qui encodent
deja le profil climatique de chaque station.
""")


def slide_demo():
    st.markdown("# Demonstration — Prediction en temps reel")
    st.markdown(
        "Donnees meteorologiques des dernieres 48 h via **Open-Meteo API** "
        "(gratuit, open-source, sans cle API)."
    )
    st.markdown("---")

    location = st.selectbox(
        "Station meteorologique",
        sorted(STATIONS.keys()),
        index=sorted(STATIONS.keys()).index("Sydney"),
    )
    lat, lon = STATIONS[location]
    st.caption(f"Coordonnees : {lat:.2f} N, {lon:.2f} E")

    if st.button("Lancer la prediction", type="primary"):
        with st.spinner(f"Recuperation Open-Meteo pour {location}..."):
            try:
                api_data = fetch_open_meteo(lat, lon)
            except requests.exceptions.RequestException as e:
                st.error(f"Erreur API Open-Meteo : {e}")
                return

        with st.spinner("Construction des features et inference..."):
            try:
                row     = build_feature_row(location, api_data)
                model   = load_model()
                results = load_results()
                thresh  = results["optimal_threshold"]
                proba   = float(model.predict_proba(row)[:, 1][0])
                rain    = proba >= thresh
            except Exception as e:
                st.error(f"Erreur lors de la prediction : {e}")
                st.exception(e)
                return

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Probabilite de pluie", f"{proba:.1%}")
        col2.metric("Seuil optimal (F2)", f"{thresh:.3f}")
        col3.metric("Prediction", "Pluie demain" if rain else "Pas de pluie")

        today = date.today()
        if rain:
            st.markdown(
                f'<div class="alert-rain">'
                f'<strong>Pluie prevue le {today} a {location}</strong><br>'
                f'Probabilite : {proba:.1%} — seuil F2 : {thresh:.3f}<br>'
                f'Le modele recommande de se preparer a la pluie pour demain.'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="alert-no-rain">'
                f'<strong>Pas de pluie prevue le {today} a {location}</strong><br>'
                f'Probabilite : {proba:.1%} — seuil F2 : {thresh:.3f}'
                f'</div>',
                unsafe_allow_html=True,
            )

        with st.expander("Donnees brutes et features calculees"):
            st.dataframe(
                row.T.rename(columns={0: "Valeur"}).style.format("{:.3f}", na_rep="NaN"),
                use_container_width=True,
            )

# ---------------------------------------------------------------------------
# Sidebar et navigation
# ---------------------------------------------------------------------------

SLIDES = {
    "Introduction":            slide_introduction,
    "Le Probleme":             slide_probleme,
    "Le Dataset":              slide_dataset,
    "La Demarche (13 etapes)": slide_demarche,
    "Feature Engineering":     slide_features,
    "Modelisation et Resultats": slide_resultats,
    "Interpretabilite SHAP":   slide_shap,
    "Analyse Spatiale":        slide_carte,
    "Demonstration":           slide_demo,
}

with st.sidebar:
    st.markdown("## Prediction de la pluie")
    st.markdown("*Australie — XGBoost — F2-score*")
    st.markdown("---")
    selection = st.radio(
        "Navigation",
        list(SLIDES.keys()),
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Bureau of Meteorology\n2007–2017 — 49 stations")

SLIDES[selection]()
