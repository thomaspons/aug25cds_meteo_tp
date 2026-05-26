"""
Présentation Streamlit, Prédiction de la pluie en Australie
Classification binaire avec XGBoost, métrique principale F2-score.
Source des données live : Open-Meteo API (gratuit, sans clé).
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
    page_title="Prédiction Pluie Australie",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR     = Path(__file__).parent
FIG_DIR     = APP_DIR / ".." / "reports" / "figures"
MODEL_PATH  = APP_DIR / ".." / "models" / "final_model.joblib"
RESULTS_PATH = APP_DIR / ".." / "reports" / "final_results.json"

# ---------------------------------------------------------------------------
# CSS : thème professionnel australien (océan + sable + nuit)
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
# Données de référence
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
    Construit le vecteur de features pour le jour courant à partir des données Open-Meteo.
    Reproduit fidèlement le feature engineering de ml_pipeline.py (df2).
    Retourne un DataFrame à une ligne prêt pour model.predict_proba().
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

    df["Month"]         = df["Date"].dt.month
    df["DayOfYear"]     = df["Date"].dt.dayofyear
    df["Month_sin"]     = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"]     = np.cos(2 * np.pi * df["Month"] / 12)
    df["DayOfYear_sin"] = np.sin(2 * np.pi * df["DayOfYear"] / 365)
    df["DayOfYear_cos"] = np.cos(2 * np.pi * df["DayOfYear"] / 365)
    df["Season"]        = df["Month"].map(SEASON_MAP)

    df["Delta_Pressure"] = df["Pressure3pm"] - df["Pressure9am"]
    df["Delta_Humidity"] = df["Humidity3pm"] - df["Humidity9am"]
    df["Delta_Temp"]     = df["Temp3pm"]     - df["Temp9am"]
    df["Delta_Wind"]     = df["WindSpeed3pm"] - df["WindSpeed9am"]

    for col in ["Pressure9am", "Humidity3pm", "MaxTemp"]:
        df[f"{col}_diff1"] = df[col] - df[col].shift(1)

    for col in ["Rainfall", "Humidity3pm", "Pressure9am"]:
        df[f"{col}_roll3_mean"] = df[col].shift(1).rolling(3, min_periods=1).mean()
        df[f"{col}_roll3_max"]  = df[col].shift(1).rolling(3, min_periods=1).max()

    df["Pressure_drop_flag"]  = (df["Delta_Pressure"] < -2).astype(int)
    df["HighHumidity_flag"]   = (df["Humidity3pm"] > 85).astype(int)
    df["StrongWind_flag"]     = (df["WindGustSpeed"] > 60).astype(int)
    df["HumidityRising_flag"] = (df["Delta_Humidity"] > 10).astype(int)
    df["PressureLow_flag"]    = (df["Pressure9am"] < 1010).astype(int)

    med_humidity = df["Humidity3pm"].median()
    if pd.isna(med_humidity):
        med_humidity = 50.0
    df["Rain_x_Humidity"] = df["RainToday"].fillna(0) * df["Humidity3pm"].fillna(med_humidity)
    df["Wind_x_Humidity"] = df["WindGustSpeed"].fillna(0) * df["Humidity3pm"].fillna(med_humidity)
    df["TempRange"]       = df["MaxTemp"] - df["MinTemp"]

    for col in ["WindGustDir", "WindDir9am", "WindDir3pm"]:
        sc = df[col].apply(compass_sincos)
        df[f"{col}_sin"] = sc.apply(lambda x: x[0])
        df[f"{col}_cos"] = sc.apply(lambda x: x[1])

    df = df.drop(columns=["WindGustDir", "WindDir9am", "WindDir3pm", "Date"])
    return df.tail(1).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Slides
# ---------------------------------------------------------------------------

def slide_introduction():
    st.markdown("# Prédiction de la pluie en Australie")
    st.markdown("### Classification binaire avec Machine Learning, Bureau of Meteorology 2007,2017")
    st.markdown("---")
    st.markdown("""
<div class="kpi-row">
  <div class="kpi"><strong>145 000</strong><span>observations</span></div>
  <div class="kpi"><strong>49</strong><span>stations</span></div>
  <div class="kpi"><strong>10 ans</strong><span>2007,2017</span></div>
  <div class="kpi"><strong>F2-score</strong><span>métrique principale</span></div>
</div>
""", unsafe_allow_html=True)
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("""
**Question**

Étant données les mesures météorologiques d'aujourd'hui,
va-t-il pleuvoir demain ?

**Approche**

Machine Learning classique (pas de Deep Learning).
Métrique principale : **F2-score**, car manquer de la pluie
coûte bien plus qu'une fausse alarme.

**Données**

Bureau of Meteorology australien (BOM).
49 stations réparties sur l'ensemble du continent.
Série temporelle de novembre 2007 à juin 2017.
""")
    with c2:
        st.image(fig_path("08_australia_map.png"),
                 caption="Les 49 stations du dataset (humidité, pression, pluie)")


def slide_probleme():
    st.markdown("# Le Problème")
    st.markdown("## Deux défis structurels")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("""
**1. Déséquilibre de classes**

Seulement **22 %** des jours sont pluvieux.
Un modèle naïf qui prédit toujours « pas de pluie »
atteint 78 % d'accuracy, mais est totalement inutile.
L'accuracy n'est pas la bonne métrique.

**2. Bruit de label**

Les capteurs enregistrent parfois 0.1 mm sur un ciel
sans nuage, ou ratent une petite averse.
Cette imprécision dans la labélisation doit être prise
en compte explicitement via la pondération.

**Pourquoi le F2-score ?**

| Erreur | Situation réelle | Coût |
|--------|-----------------|------|
| Faux Négatif | Prédit beau, il pleut | Élevé |
| Faux Positif | Prédit pluie, beau temps | Faible |

Le F2 pénalise **deux fois plus** les Faux Négatifs.

**Contexte australien**

L'asymétrie des coûts est particulièrement forte sur ce territoire.
Les sécheresses prolongées rendent chaque épisode pluvieux précieux
pour la planification de l'irrigation et des restrictions d'eau.
Le risque de **bushfires** dépend directement de l'humidité prévue :
un faux négatif fait sous-estimer le danger feu par les services d'alerte.
Les **crues éclair** en saison cyclonique (Queensland, NSW) imposent
une anticipation fiable. L'agriculture (semis, récoltes, traitements)
subit aussi un coût élevé quand une pluie est ratée.
Rater une pluie réelle coûte bien plus cher qu'une fausse alerte,
d'où le choix du F2.
""")
    with c2:
        st.image(fig_path("01_target_distribution.png"))
        st.latex(r"F_2 = \frac{5 \cdot P \cdot R}{4 \cdot P + R}")
        st.markdown("""
**Lecture de la formule**

Le F-bêta généralisé pondère précision et rappel selon β.
Pour β = 2, le **rappel pèse 4 fois plus** que la précision
au dénominateur (β² = 4). Cela rend la métrique très sensible
aux Faux Négatifs : un modèle qui rate des pluies est lourdement
pénalisé, alors qu'un modèle qui fait des fausses alertes
reste correctement noté.

- F2 = 1 → modèle parfait (aucun FN, aucun FP)
- F2 = 0 → modèle qui rate toutes les pluies (recall = 0)
- F2 ≈ recall si la précision est bien plus faible
""")


def slide_dataset():
    st.markdown("# Le Dataset")
    st.markdown("## weatherAUS.csv, Bureau of Meteorology")
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.markdown("""
| Caractéristique | Valeur |
|----------------|--------|
| Source | BOM (Commonwealth of Australia) |
| Période | Nov 2007, Juin 2017 |
| Localités | 49 villes |
| Observations | ~145 000 jours |
| Variables | 23 (16 num., 7 catég.) |
| Target | RainTomorrow (Yes / No) |
| Ratio No/Yes | 3.5 pour 1 |

**Variables les plus importantes (SHAP) :**
- Humidity3pm, Wind_x_Humidity
- Pressure3pm, Sunshine, Delta_Pressure

**Valeurs manquantes notables :**
- Sunshine : 48 % de NaN
- Evaporation : 43 % de NaN
- Cloud3pm : 40 % de NaN
- Cloud9am : 38 % de NaN

Ces variables sont **conservées** car physiquement
informatives (ensoleillement, évaporation : indicateurs
forts de pluie). Imputation par **médiane** dans
le pipeline pour ne pas dégrader le signal.
""")
    with c2:
        st.image(fig_path("02_correlation_target.png"),
                 caption="Corrélations des variables numériques avec RainTomorrow")


def slide_cleaning():
    st.markdown("# Cleaning, Aberrations physiques uniquement")
    st.markdown("## Principe : ne jamais imputer ce qui est physiquement impossible")
    st.markdown("""
Le cleaning est **minimaliste et justifié**. On ne retire que les valeurs
physiquement impossibles (capteur défectueux ou erreur de saisie).
On n'imputera **jamais** une valeur aberrante par la médiane, ce serait
introduire un faux signal.
""")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("""
| Filtre | Justification physique |
|--------|-----------------------|
| `Humidity9am > 100` retiré | L'humidité relative est par définition bornée à 100 % |
| `Humidity3pm > 100` retiré | Idem |
| `Rainfall < 0` retiré | Une pluviométrie négative n'existe pas |
| `Pressure9am < 900 hPa` retiré | Pression au sol jamais observée sous 900 hPa (record mondial = 870 hPa en cyclone) |
| `Pressure3pm < 900 hPa` retiré | Idem |
| `WindSpeed9am < 0` retiré | Une vitesse de vent négative est impossible |
| `WindSpeed3pm < 0` retiré | Idem |
| `RainTomorrow` NaN retiré | Pas de target = ligne inutilisable |
| Doublons retirés | `drop_duplicates()` |

**Résultat : 3 267 lignes retirées sur 145 460 (2.2 %).**

**Encoding des targets binaires :**
```python
RainTomorrow = (RainTomorrow == 'Yes').astype(int)
RainToday    = RainToday.map({'Yes': 1, 'No': 0})
```
""")
    with c2:
        st.markdown("""
### Ce qu'on ne fait PAS

- **Pas de suppression des NaN** sur les features
  (Sunshine, Cloud3pm, etc. en ont >40 %).
  Les arbres de boosting gèrent bien l'imputation par médiane.

- **Pas de détection statistique d'outliers** (z-score,
  IQR, Isolation Forest). Une valeur extrême mais
  physiquement valide (pic de chaleur à 45 °C, rafale à
  120 km/h) est un **vrai signal météo**, pas une erreur.

- **Pas de winsorization**. On garde la queue de
  distribution naturelle des phénomènes météo.

### Pourquoi c'est important

Sur des données météo, **les extrêmes portent le signal**.
Lisser ou retirer les outliers détruirait justement
les jours intéressants (orages, tempêtes, fronts).
""")


def slide_demarche():
    st.markdown("# La Démarche, Pipeline 13 Étapes")
    etapes = [
        ("01", "EDA",
         "Distributions, corrélations, saisonnalité, valeurs manquantes."),
        ("02", "Cleaning",
         "Aberrations physiques uniquement : pression < 900 hPa, "
         "humidité > 100 %, vitesses négatives."),
        ("03", "Feature Engineering",
         "Dérivées temporelles, fenêtres glissantes 3 j, encodage cyclique "
         "sin/cos, interactions."),
        ("04", "Split chronologique",
         "80/20 sur les dates triées. Interdit de voir le futur pendant "
         "l'entraînement."),
        ("05", "Preprocessing",
         "Pipeline sklearn : RobustScaler + SimpleImputer + OneHotEncoder."),
        ("06", "Pondération anti-bruit",
         "sample_weight 0.6 / 0.8 / 1.5 selon la fiabilité du label "
         "(ambiguïté pluviométrique)."),
        ("07", "Feature Selection",
         "SelectKBest (Mutual Information, random_state=42). k optimisé "
         "en cross-validation."),
        ("08", "Pas de SMOTE",
         "SMOTE ignore la temporalité et génère des observations "
         "météorologiquement incohérentes."),
        ("09", "Comparaison modèles",
         "RandomForest, XGBoost, LightGBM évalués en CV 5-fold (F2-score)."),
        ("10", "Hyperparameter Tuning",
         "RandomizedSearchCV 30 itérations sur XGBoost et LightGBM."),
        ("11", "Entraînement final",
         "Meilleur modèle réentraîné sur 100 % du train."),
        ("12", "Évaluation",
         "F2, Recall, Precision, ROC-AUC, Brier Score + seuil optimal."),
        ("13", "Interprétabilité",
         "SHAP TreeExplainer : importance, beeswarm, dépendance, "
         "saisonnalité, carte climatologique."),
    ]
    for num, titre, desc in etapes:
        st.markdown(
            f'<div class="step">'
            f'<span class="step-num">{num}</span>'
            f'<span><strong>{titre}</strong>, {desc}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def slide_features():
    st.markdown("# Feature Engineering")
    st.markdown("## Encoder la physique météorologique dans les données")
    st.markdown(
        "**34 features créées** à partir des 23 colonnes brutes. "
        "Chaque feature a une justification physique ou statistique. "
        "Récap par catégorie : 7 temporelles, 4 deltas, 9 lags/rolling, "
        "5 flags binaires, 3 interactions, 6 vent cyclique."
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Temporelles (7)", "Deltas intra-jour (4)", "Lags & fenêtres (9)",
        "Flags binaires (5)", "Interactions & vent (3+6)"
    ])

    with tab1:
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.markdown("""
### 7 features temporelles
```python
# Variables intermédiaires (gardées dans le modèle)
Month         = Date.dt.month         # 1..12
DayOfYear     = Date.dt.dayofyear     # 1..365
Season        = Month.map(...)        # Summer/Autumn/Winter/Spring

# Encodage cyclique dérivé
Month_sin     = sin(2π × Month / 12)
Month_cos     = cos(2π × Month / 12)
DayOfYear_sin = sin(2π × DayOfYear / 365)
DayOfYear_cos = cos(2π × DayOfYear / 365)
```

**Statut de Month et DayOfYear** : ce sont des intermédiaires
nécessaires au calcul des sin/cos, mais on les garde aussi
en features. L'arbre peut s'en servir pour des splits simples
(par exemple « MaxTemp élevée seulement si Month entre 11 et 3 »).

**Season** : encodage catégoriel (4 modalités), redondant avec
Month_sin/cos mais lisible pour l'analyse SHAP.

### Pourquoi sin/cos pour le temps ?

Avec un encodage linéaire, janvier vaut 1 et décembre vaut 12 :
ils sont **distants de 11**, alors qu'ils sont **adjacents**
dans le cycle des saisons. Un arbre qui split sur Month=6
sépare arbitrairement juin de juillet.

On projette donc chaque mois sur le **cercle unité** :
chaque mois devient un angle θ = 2π × Month / 12, et on prend
ses coordonnées (cos θ, sin θ) comme features.

```
            Month_sin
                |
       Avr  Mar |  Fév  Jan
         \\    \\|/    /
   Mai ----+----+----+---- Déc   → Month_cos
         /    /|\\    \\
       Juin Juil| Aoû  Sep
                |
                Nov, Oct
```

Sur ce cercle, décembre et janvier sont voisins (distance ≈ 0,52),
juin et décembre sont opposés (distance = 2). L'arbre peut alors
faire un split du type « Month_cos > 0,5 » pour isoler la fenêtre
décembre-février (été austral) en **une seule coupure**.

**DayOfYear_sin/cos** suivent la même logique avec une granularité
plus fine : ils capturent les variations intra-mois (début vs fin
d'hiver austral).
""")
        with c2:
            st.image(fig_path("03_seasonality.png"),
                     caption="Saisonnalité de la pluie par région")

    with tab2:
        st.markdown("""
### 4 deltas intra-journaliers (9 h → 15 h)
```python
Delta_Pressure = Pressure3pm  - Pressure9am
Delta_Humidity = Humidity3pm  - Humidity9am
Delta_Temp     = Temp3pm      - Temp9am
Delta_Wind     = WindSpeed3pm - WindSpeed9am
```

**Justification physique** : un front pluvieux qui arrive
provoque dans la journée :

- **Delta_Pressure** très négatif (chute de pression) : air ascendant
- **Delta_Humidity** > 0 (humidité monte) : apport de vapeur d'eau
- **Delta_Temp** : variation thermique, signal d'inversion
- **Delta_Wind** : accélération du vent à l'approche du front

Ces variations sont **plus informatives** que les valeurs
absolues, car elles capturent la **dynamique** du système.
C'est ce qui fait que **Delta_Pressure** apparaît dans le top SHAP,
plus important que Pressure3pm seul.
""")

    with tab3:
        st.markdown("""
### 9 features de mémoire temporelle (3 diff + 6 rolling)
```python
# Diff avec la veille (3 features)
for col in ['Pressure9am', 'Humidity3pm', 'MaxTemp']:
    df[f'{col}_diff1'] = df.groupby('Location')[col].diff()

# Fenêtres 3 jours sur le passé uniquement (6 features : 3 × {mean, max})
for col in ['Rainfall', 'Humidity3pm', 'Pressure9am']:
    df[f'{col}_roll3_mean'] = groupby('Location')[col].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df[f'{col}_roll3_max']  = groupby('Location')[col].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).max())
```

**Les 3 diff1** mesurent la variation jour-à-jour. Pression qui
chute par rapport à hier, humidité qui grimpe, MaxTemp qui baisse :
trois signaux d'arrivée de système dépressionnaire.

**Les 6 rolling** donnent un état moyen et un pic récents :
- `mean` : tendance générale (humidité installée sur 3 jours)
- `max` : pic récent (orage la veille, instabilité résiduelle)

**Le `shift(1)` est CRITIQUE** : sans lui, on inclut le jour
courant dans la moyenne, donc **fuite de données** (data leakage).
Le modèle « voit » ce qu'il doit prédire.

**Le `groupby(Location)`** : on ne mélange pas les stations.
Le rolling de Sydney ne doit pas intégrer la veille de Darwin.

**Pourquoi ces 3 colonnes seulement (Rainfall, Humidity, Pressure) ?**
Ce sont les meilleurs prédicteurs persistants. Ajouter du rolling
sur Temp ou Wind dégrade le modèle (bruit, pas de persistance).
""")

    with tab4:
        st.markdown("""
### 5 indicateurs binaires (seuils physiques)
```python
Pressure_drop_flag  = (Delta_Pressure  < -2)    # chute > 2 hPa
HighHumidity_flag   = (Humidity3pm     > 85)    # air saturé
StrongWind_flag     = (WindGustSpeed   > 60)    # km/h rafales
HumidityRising_flag = (Delta_Humidity  > 10)    # apport vapeur
PressureLow_flag    = (Pressure9am     < 1010)  # dépression
```

**Pourquoi binariser ce qu'on a déjà en continu ?**

Les arbres peuvent découvrir les seuils eux-mêmes, mais en
les fournissant explicitement on :

1. **Accélère l'apprentissage** (split trivial sur le flag)
2. **Aide l'interprétabilité** (SHAP plus lisible)
3. **Capture les non-linéarités** (effet de seuil météo)

**Choix des seuils** : valeurs **issues de la littérature
météorologique**, pas optimisées sur le train (sinon overfitting).

- 85 % humidité : seuil de saturation typique
- 60 km/h rafales : coup de vent (Beaufort 8)
- 1010 hPa : seuil empirique des dépressions australes
- 2 hPa de chute : signal de front significatif
- 10 points d'humidité en hausse : apport de vapeur marqué
""")

    with tab5:
        st.markdown("""
### 3 features d'interaction
```python
Rain_x_Humidity = RainToday    × Humidity3pm
Wind_x_Humidity = WindGustSpeed × Humidity3pm
TempRange       = MaxTemp      - MinTemp
```

**Rain_x_Humidity** : il pleut aujourd'hui + humidité élevée =
système pluvieux installé (la pluie de demain est très probable).

**Wind_x_Humidity** : ressort **#2 en SHAP**. Vent fort +
humidité forte = front actif (transport horizontal d'air humide).

**TempRange** : amplitude thermique journalière. Faible amplitude
= ciel couvert = pluie probable. Forte amplitude = ciel clair.

### 6 features de direction de vent (cyclique)
```python
for col in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
    angles = df[col].map({N: 0, NNE: 22.5, ..., NNW: 337.5})
    df[f'{col}_sin'] = sin(radians(angles))
    df[f'{col}_cos'] = cos(radians(angles))
```

3 colonnes texte × 2 (sin et cos) = 6 features numériques.

**Même logique que Month_sin/cos** : N (0°) et NNW (337,5°)
sont voisins sur la rose des vents mais distants numériquement.
En projetant sur le cercle unité, N et NNW deviennent proches
dans le plan (cos, sin), donc l'arbre traite les directions
comme un continuum et non comme 16 catégories disjointes.

Après encodage on **drop** les colonnes texte (WindGustDir, etc.)
sinon elles seraient OneHotEncoded en 48 colonnes binaires
inutiles (3 × 16 directions).
""")


def slide_choix():
    st.markdown("# Choix Méthodologiques")
    st.markdown("## Les 6 décisions structurantes du projet")

    choix = [
        ("F2-score plutôt que F1 ou Accuracy",
         "Le coût d'un Faux Négatif (prédire beau alors qu'il pleut) est "
         "**supérieur** à celui d'un Faux Positif. Le F2 pondère le recall "
         "deux fois plus que la précision. L'accuracy est trompeuse "
         "(78 % en prédisant toujours « pas de pluie »)."),
        ("Split CHRONOLOGIQUE, jamais aléatoire",
         "Le dataset est temporel (2007,2017). Un split aléatoire fait fuiter "
         "le futur dans le train. **Coupure : 80 % des dates uniques** → "
         "Train 2007 → 08-2015, Test 08-2015 → 2017. C'est aussi la réalité "
         "opérationnelle : on prédit le futur, pas le passé."),
        ("Pas de SMOTE pour le déséquilibre",
         "SMOTE génère des observations synthétiques par interpolation. "
         "Sur des données météo, cela crée des journées **physiquement "
         "impossibles** (humidité 95 % + ensoleillement 12 h). On préfère : "
         "`scale_pos_weight` (XGBoost) et `class_weight='balanced'` (LGBM)."),
        ("Pondération anti-bruit (sample_weight)",
         "Les labels eux-mêmes sont bruités : 0.5 mm un jour « sec », "
         "rien un jour « humide ». On affecte un poids :\n"
         "- **0.6** si Rainfall < 0.5 mm mais labélisé « pluie » (ambigu)\n"
         "- **0.8** si Humidity entre 60,75 % labélisé « pas pluie » (frontière)\n"
         "- **1.5** si Rainfall > 5 mm et labélisé « pluie » (signal clair)"),
        ("Feature selection par Mutual Information",
         "**SelectKBest(mutual_info_classif, k=35)** intégré dans le pipeline "
         "(pas en pré-processing). La MI capture les dépendances **non-linéaires** "
         "entre feature et target, contrairement à la corrélation de Pearson. "
         "Le `k` est optimisé dans le RandomizedSearchCV."),
        ("Seuil optimisé sur la courbe Précision-Rappel",
         "Le seuil 0.5 par défaut maximise l'accuracy, pas le F2. "
         "On parcourt tous les seuils, on calcule F2 à chaque pas, "
         "on garde **le seuil qui maximise F2** (~0.33 dans notre cas). "
         "C'est ce qui fait passer le recall de 0.55 à 0.87."),
    ]
    for titre, desc in choix:
        st.markdown(f"### {titre}")
        st.markdown(desc)
        st.markdown("---")


def slide_resultats():
    r = load_results()
    st.markdown("# Modélisation et Résultats")

    st.markdown("## Comparaison des 3 modèles (5-fold CV, F2-score)")
    st.markdown("""
| Modèle | F2 CV (default) | F2 CV (tuné) | Test F2 | Recall | Précision | ROC-AUC |
|--------|----------------:|-------------:|--------:|-------:|----------:|--------:|
| RandomForest | 0.667 ± 0.004 |, |, |, |, |, |
| LightGBM | 0.729 ± 0.004 | **0.732** |, |, |, |, |
| **XGBoost (final)** | 0.729 ± 0.003 | **0.734** | **0.730** | **0.867** | 0.447 | **0.879** |

**RandomForest** : 6 points de F2 sous les boosters → écarté d'office.
**LightGBM** et **XGBoost** ex-æquo en CV. XGBoost légèrement meilleur
après tuning (RandomizedSearchCV 30 iter) → retenu comme modèle final.
""")

    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.markdown("""
### Split chronologique : règle fondamentale

Un split aléatoire sur des données temporelles
constitue une **fuite de données** (data leakage).
Le modèle peut apprendre sur 2016 et être évalué
sur 2009, les métriques sont alors trompeuses.

```
Split aléatoire  →  métriques gonflées
Split chrono     →  métriques honnêtes
2007 ─── 2015 │ 2015 ─── 2017
   TRAIN (80%)    TEST (20%)
```

### Modèle final : XGBoost tuné (RandomizedSearchCV)
""")
        st.markdown(f"""
| Métrique | Valeur |
|----------|--------|
| F2-score test (seuil opt.) | **{r['f2_test']:.3f}** |
| F1-score test | {r['f1_test']:.3f} |
| Recall pluie | **{r['recall_rain']:.3f}** |
| Précision pluie | {r['precision_rain']:.3f} |
| ROC-AUC | {r['roc_auc']:.3f} |
| Brier Score | {r['brier_score']:.3f} |
| Log Loss | {r['log_loss']:.3f} |
| Seuil optimal | {r['optimal_threshold']:.3f} |
| Faux Négatifs | {r['false_negatives']} sur {r['test_size']:,} ({r['false_negatives']/r['test_size']*100:.1f} %) |
| Faux Positifs | {r['false_positives']} sur {r['test_size']:,} ({r['false_positives']/r['test_size']*100:.1f} %) |
""")
    with c2:
        st.image(fig_path("04_evaluation.png"),
                 caption="Précision-Rappel + Matrice de confusion + Calibration (Brier)")
        st.markdown("""
**Lecture de la matrice de confusion** : sur 7 211 jours
pluvieux dans le test, le modèle en attrape **6 255** (recall 86.7 %).
Il manque seulement **956 jours de pluie** (2.9 % du test), c'est ce que
le F2 visait. En contrepartie, **7 743 fausses alertes** (23.5 %) : on
préfère prévenir pour rien plutôt que de rater une pluie.
""")


def slide_shap():
    st.markdown("# Interprétabilité, SHAP")
    st.markdown("""
SHAP (SHapley Additive exPlanations) décompose chaque prédiction en contributions individuelles.
Pour les modèles arbres, **TreeExplainer** calcule les valeurs de Shapley exactement, en temps polynomial.
""")
    tab1, tab2, tab3, tab4 = st.tabs([
        "Importance globale", "Beeswarm", "Dépendance", "Saisonnalité"
    ])
    with tab1:
        st.image(fig_path("05a_shap_importance.png"),
                 caption="Mean |SHAP|, Top 20 features par importance globale")
        st.markdown(
            "Humidity3pm et Wind_x_Humidity dominent. "
            "Delta_Pressure capture l'approche des fronts."
        )
    with tab2:
        st.image(fig_path("05b_shap_beeswarm.png"),
                 caption="Chaque point = une observation. Rouge = valeur haute, Bleu = valeur basse.")
        st.markdown("""
**Lecture :**
- Humidity3pm rouge à droite : humidité élevée pousse vers la pluie.
- Sunshine rouge à gauche : ensoleillement élevé pousse vers pas de pluie.
- Delta_Pressure bleu à droite : chute de pression signale un front pluvieux.
""")
    with tab3:
        st.image(fig_path("06_shap_dependence.png"),
                 caption="Effet marginal de chaque feature sur la prédiction (+ interaction automatique)")
        st.markdown("""
**Lecture des dependence plots**

Chaque point est une observation. L'axe X est la valeur de la feature,
l'axe Y sa contribution SHAP à la prédiction (positif = pousse vers
la pluie, négatif = pousse vers pas de pluie). La couleur encode
la feature en interaction automatique (celle qui module le plus l'effet).

- **Humidity3pm** : effet **monotone et fort**. En dessous de 50 %
  d'humidité, contribution clairement négative (pas de pluie). Au-dessus
  de 70 %, contribution positive qui s'accélère vers 90,100 %.
  La courbe est **non-linéaire**, c'est pour cela qu'on utilise un booster.

- **Wind_x_Humidity** : effet croissant. Plus le produit vent × humidité
  est élevé, plus le modèle prédit la pluie. Capture le **transport
  horizontal d'humidité** (front actif).

- **Pressure3pm** : effet **inversé**. Pression basse → contribution
  positive (dépression = pluie). Pression haute → contribution négative
  (anticyclone = beau temps).

- **Sunshine** : effet inversé également. Beaucoup d'ensoleillement
  aujourd'hui → faible probabilité de pluie demain.

L'interaction colorée montre que les effets se **combinent** : par exemple,
Humidity3pm seul ne suffit pas, il faut aussi une chute de pression
pour que le modèle prédise franchement la pluie.
""")
    with tab4:
        st.image(fig_path("07_shap_seasonal.png"),
                 caption="Importance SHAP par saison, le modèle a capturé les patterns climatiques australiens")
        st.markdown("""
**Lecture de la heatmap (gauche)**

Pour chaque saison (Summer, Autumn, Winter, Spring), on calcule
la **moyenne |SHAP|** des top 10 features. Plus la case est rouge,
plus la feature compte pour cette saison-là.

- **Hiver austral (Winter)** : Humidity3pm et Pressure3pm dominent.
  Les fronts dépressionnaires venus de l'océan Antarctique gouvernent
  la météo du sud de l'Australie.

- **Été (Summer)** : Wind_x_Humidity prend de l'importance, la mousson
  tropicale du nord (Darwin, Cairns) est dominée par les vents humides
  et les orages convectifs.

- **Printemps / Automne** : profils intermédiaires, plus équilibrés.

**Lecture du boxplot (droite)**

Distribution de la valeur SHAP de la **feature #1** (typiquement Humidity3pm)
par saison. Une boîte étalée signifie que cette feature **discrimine
fortement** les jours de pluie/pas pluie pour cette saison.

- En hiver, la boîte est large et la médiane proche de 0, la feature
  bascule franchement d'un côté ou de l'autre selon les jours.
- En été, distribution plus resserrée, la pluie est plus systématique
  (saison humide), Humidity3pm explique une part stable.

**Conclusion** : le modèle a appris des **règles différentes selon
la saison**, ce qui valide l'utilité des features temporelles cycliques
(Month_sin/cos, DayOfYear_sin/cos).
""")


def slide_carte():
    st.markdown("# Analyse Spatiale")
    st.markdown("## Les profils climatiques varient-ils selon la localité ?")
    st.image(
        fig_path("08_australia_map.png"),
        caption="Moyennes 2007,2017 : humidité (gauche), pression + vents (centre), pluviométrie (droite)",
        use_container_width=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**Observations :**
- Nord tropical (Darwin, Cairns) : humidité élevée, pression basse, pluviométrie dominante
- Intérieur aride (Alice Springs, Uluru) : sec, peu de pluie
- Sud-Est (Melbourne, Sydney) : profils intermédiaires tempérés
""")
    with c2:
        st.markdown("""
**Conclusion SHAP :**
Location n'est pas dans le top 20 des features SHAP.
Le modèle capture implicitement la géographie via
Humidity3pm, Pressure9am et WindGustSpeed, qui encodent
déjà le profil climatique de chaque station.
""")


def slide_demo():
    st.markdown("# Démonstration, Prédiction en temps réel")
    st.markdown(
        "Données météorologiques des dernières 48 h via **Open-Meteo API** "
        "(gratuit, open-source, sans clé API)."
    )
    st.markdown("---")

    location = st.selectbox(
        "Station météorologique",
        sorted(STATIONS.keys()),
        index=sorted(STATIONS.keys()).index("Sydney"),
    )
    lat, lon = STATIONS[location]
    st.caption(f"Coordonnées : {lat:.2f} N, {lon:.2f} E")

    if st.button("Lancer la prédiction", type="primary"):
        with st.spinner(f"Récupération Open-Meteo pour {location}..."):
            try:
                api_data = fetch_open_meteo(lat, lon)
            except requests.exceptions.RequestException as e:
                st.error(f"Erreur API Open-Meteo : {e}")
                return

        with st.spinner("Construction des features et inférence..."):
            try:
                row     = build_feature_row(location, api_data)
                model   = load_model()
                results = load_results()
                thresh  = results["optimal_threshold"]
                proba   = float(model.predict_proba(row)[:, 1][0])
                rain    = proba >= thresh
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")
                st.exception(e)
                return

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Probabilité de pluie", f"{proba:.1%}")
        col2.metric("Seuil optimal (F2)", f"{thresh:.3f}")
        col3.metric("Prédiction", "Pluie demain" if rain else "Pas de pluie")

        today = date.today()
        if rain:
            st.markdown(
                f'<div class="alert-rain">'
                f'<strong>Pluie prévue le {today} à {location}</strong><br>'
                f'Probabilité : {proba:.1%}, seuil F2 : {thresh:.3f}<br>'
                f'Le modèle recommande de se préparer à la pluie pour demain.'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="alert-no-rain">'
                f'<strong>Pas de pluie prévue le {today} à {location}</strong><br>'
                f'Probabilité : {proba:.1%}, seuil F2 : {thresh:.3f}'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.info(
            "**Note méthodologique.** Les probabilités affichées sont "
            "calibrées pour maximiser la détection des épisodes pluvieux "
            "(F2-score). Elles surestiment volontairement la fréquence "
            "réelle de pluie pour minimiser les faux négatifs, choix "
            "justifié par le coût élevé d'une pluie ratée en contexte "
            "australien (sécheresses, incendies, agriculture)."
        )

        with st.expander("Données brutes et features calculées"):
            display_df = row.T.rename(columns={0: "Valeur"})
            display_df["Valeur"] = display_df["Valeur"].apply(
                lambda v: f"{v:.3f}" if isinstance(v, (int, float)) and pd.notna(v)
                else ("NaN" if pd.isna(v) else str(v))
            )
            st.dataframe(display_df, use_container_width=True)

# ---------------------------------------------------------------------------
# Sidebar et navigation
# ---------------------------------------------------------------------------

SLIDES = {
    "Introduction":              slide_introduction,
    "Le Problème":               slide_probleme,
    "Le Dataset":                slide_dataset,
    "La Démarche (13 étapes)":   slide_demarche,
    "Cleaning":                  slide_cleaning,
    "Feature Engineering":       slide_features,
    "Choix Méthodologiques":     slide_choix,
    "Modélisation et Résultats": slide_resultats,
    "Interprétabilité SHAP":     slide_shap,
    "Analyse Spatiale":          slide_carte,
    "Démonstration":             slide_demo,
}

with st.sidebar:
    st.markdown("## Prédiction de la pluie")
    st.markdown("*Australie, XGBoost, F2-score*")
    st.markdown("---")
    selection = st.radio(
        "Navigation",
        list(SLIDES.keys()),
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Bureau of Meteorology\n2007,2017, 49 stations")

SLIDES[selection]()
