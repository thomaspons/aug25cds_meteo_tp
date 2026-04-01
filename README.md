# Prediction de la pluie en Australie - Classification Binaire

Question : Etant donne les mesures meteorologiques d'aujourd'hui, est-ce qu'il va pleuvoir demain ?

Approche : Machine Learning classique uniquement. Pas de Deep Learning.

Metrique principale : F2-score (le recall compte deux fois plus que la precision).

---

## Table des matieres

1. Le probleme
2. Le dataset
3. La demarche ML en 12 etapes
4. Justification des choix techniques
5. Resultats
6. Structure du projet
7. Lancer le projet

---

## Le probleme

La prediction de pluie est un probleme de classification binaire :

- Entree : mesures meteo du jour (temperature, humidite, pression, vent, etc.)
- Sortie : RainTomorrow = 1 (pluie) ou 0 (pas de pluie)

### Deux defis majeurs

**1. Desequilibre de classes**

Seulement 22% des jours sont pluvieux. Un modele naif qui predit toujours "pas de pluie" atteint 78% d'accuracy mais est completement inutile. C'est pourquoi l'accuracy n'est pas la bonne metrique pour ce probleme.

**2. Bruit de label**

Les releves meteorologiques sont imparfaits. Un capteur peut enregistrer 0.1mm de pluie un jour sans nuage (erreur de mesure), ou ne pas enregistrer une petite averse. Cette imprecision dans la labelisation doit etre prise en compte.

---

## Le dataset

Source : Kaggle - Weather Dataset Rattle Package

| Caracteristique | Valeur |
|----------------|--------|
| Periode | Novembre 2007 - Juin 2017 |
| Localites | 49 villes d'Australie |
| Observations | 145 000 jours environ |
| Variables | 23 (16 numeriques, 7 categorielles) |
| Target | RainTomorrow (Yes/No) |
| Ratio No/Yes | 3.5 pour 1 (desequilibre modere) |

### Variables disponibles

| Variable | Type | Description |
|----------|------|-------------|
| Date | Date | Date du releve |
| Location | Categorie | Ville australienne |
| MinTemp, MaxTemp | Float | Temperatures min et max en degres Celsius |
| Rainfall | Float | Pluie du jour en millimetres |
| Evaporation | Float | Evaporation en 24h en mm (43% de NaN) |
| Sunshine | Float | Heures d'ensoleillement (48% de NaN) |
| WindGustDir, WindDir9am, WindDir3pm | Categorie | Directions du vent |
| WindGustSpeed, WindSpeed9am, WindSpeed3pm | Float | Vitesses du vent en km/h |
| Humidity9am, Humidity3pm | Float | Humidite relative en pourcentage |
| Pressure9am, Pressure3pm | Float | Pression atmospherique en hPa |
| Cloud9am, Cloud3pm | Float | Nebulosite de 0 a 9 oktas |
| Temp9am, Temp3pm | Float | Temperatures a 9h et 15h en degres Celsius |
| RainToday | Categorie | Pluie aujourd'hui (Yes/No) |
| RainTomorrow | Target | Pluie demain (Yes/No) |

---

## La demarche ML en 12 etapes

### Etape 1 - EDA (Analyse Exploratoire des Donnees)

Avant tout code de modelisation, on comprend les donnees.

On analyse les distributions de chaque variable, les correlations entre elles, la saisonnalite (juillet-aout correspond au pic de pluie dans le sud de l'Australie, qui est l'hiver austral), et la qualite des donnees (valeurs manquantes, aberrations).

Les variables Sunshine (48% de NaN) et Evaporation (43% de NaN) sont tres incompletes mais conservees car informatives. On les imputera par la mediane dans le pipeline.

### Etape 2 - Cleaning

On supprime uniquement les aberrations physiques, c'est-a-dire les valeurs impossibles dans la realite :

| Regle | Raison physique |
|-------|----------------|
| Humidity inferieure ou egale a 100% | L'humidite ne peut pas depasser la saturation |
| Rainfall superieure ou egale a 0 | Un volume de pluie est toujours positif |
| Pressure superieure a 800 hPa | Le record absolu australien est 868 hPa (cyclone Monica, 2006) |
| WindSpeed superieure ou egale a 0 | Une vitesse est toujours positive |

On ne supprime pas les outliers statistiques (vent a 130 km/h = cyclone rare mais reel). Le RobustScaler les gerera.

### Etape 3 - Feature Engineering

Les modeles ML ne comprennent pas la physique. On encode donc explicitement la connaissance meteorologique.

**Encodage cyclique sin/cos**

Le mois de janvier (1) et decembre (12) sont adjacents dans le calendrier mais numeriquement distants. Un modele ne le sait pas. Solution : on encode le mois en cercle avec les fonctions sinus et cosinus.

    Month_sin = sin(2 * pi * Month / 12)
    Month_cos = cos(2 * pi * Month / 12)

Avec cet encodage, janvier et decembre sont au meme endroit sur le cercle. Le meme principe s'applique aux directions de vent : le Nord (0 degres) et le NNW (337.5 degres) sont voisins.

**Derivees temporelles (9h vers 15h)**

Une chute de pression dans la journee est un signe classique d'approche d'un front pluvieux.

    Delta_Pressure = Pressure3pm - Pressure9am    # negatif = front meteorologique approchant
    Delta_Humidity = Humidity3pm - Humidity9am    # positif = atmosphere qui se sature

**Fenetres glissantes (3 jours)**

La pluie d'aujourd'hui depend souvent des 3 derniers jours. On utilise shift(1) pour ne regarder que le passe et eviter toute fuite de donnees.

**Flags binaires**

    Pressure_drop_flag  = 1 si Delta_Pressure < -2 hPa
    HighHumidity_flag   = 1 si Humidity3pm > 85%
    StrongWind_flag     = 1 si WindGustSpeed > 60 km/h

### Etape 4 - Split Train / Test CHRONOLOGIQUE

C'est l'une des decisions les plus importantes du projet.

Un split aleatoire (train_test_split avec random_state) sur des donnees temporelles est une erreur fondamentale. Le modele peut s'entrainer sur des donnees de 2016 et etre evalue sur des donnees de 2009. Il a donc vu le futur pendant l'entrainement. Les metriques sont alors artificiellement gonflees.

    Split aleatoire (incorrect pour donnees temporelles)
    2007  2010  2013  2008  2015  2011  2016  2009
     [T]   [Ts]  [T]  [Ts]   [T]  [Ts]   [T]  [Ts]
    (T = train, Ts = test, melanges aleatoirement)

    Split chronologique (correct)
    2007 ... 2008 ... 2013 ... 2014 | 2015 ... 2016 ... 2017
    <------------ TRAIN (80%) -------> <---- TEST (20%) ---->

Les metriques obtenues avec un split chronologique sont generalement plus basses qu'avec un split aleatoire. Elles sont en revanche honnetes et refletent les vraies performances du modele sur un futur inconnu.

### Etape 5 - Preprocessing (Pipeline sklearn)

**Pourquoi normaliser ?**

La temperature varie entre -8 et +48 degres. La pression entre 980 et 1041 hPa. Sans normalisation, la pression ecrase numeriquement la temperature dans les calculs, non parce qu'elle est plus importante, mais parce qu'elle a des valeurs absolues plus grandes.

**Pourquoi RobustScaler plutot que StandardScaler ?**

StandardScaler utilise la moyenne et l'ecart-type, tous deux tres sensibles aux valeurs extremes. RobustScaler utilise la mediane et l'ecart interquartile (IQR), qui sont resistants aux outliers. En meteorologie, les cyclones et les canicules produisent des valeurs extremes reelles. RobustScaler est donc plus adapte.

**Pourquoi un Pipeline ?**

Le Pipeline sklearn garantit que toute transformation est apprise sur le train uniquement puis appliquee au test. Sans pipeline, on risque d'apprendre la normalisation sur l'ensemble du dataset, ce qui constitue une fuite de donnees.

### Etape 6 - Ponderation anti-label-noise

Les releves meteorologiques ne sont pas parfaits. On attribue des poids differents aux observations selon leur fiabilite :

| Cas | Poids | Raison |
|-----|-------|--------|
| Rainfall inferieure a 0.5mm et RainTomorrow = 1 | 0.6 | Trace de pluie, label peu fiable |
| Humidity entre 60 et 75% et RainTomorrow = 0 | 0.8 | Zone ambigue |
| Rainfall superieure a 5mm et RainTomorrow = 1 | 1.5 | Cas clairement positif |
| Tous les autres | 1.0 | Neutre |

### Etape 7 - Feature Selection (Mutual Information)

**Mutual Information vs Correlation de Pearson**

La correlation de Pearson ne mesure que les relations lineaires. L'Information Mutuelle mesure toute forme de dependance entre deux variables, lineaire ou non. Elle est donc mieux adaptee pour identifier les features vraiment informatives en presence d'effets de seuil (comme l'humidite qui depasse 85%).

La feature selection est integree dans le pipeline et son parametre k est optimise en cross-validation.

### Etape 8 - Pas de SMOTE

SMOTE cree des observations synthetiques en interpolant entre deux exemples de la classe minoritaire. En meteorologie, cela pose deux problemes :

**Probleme physique :** une observation "entre" deux jours de pluie ne correspond a aucun etat atmospherique reel. Les combinaisons de variables generees peuvent etre meteorologiquement incoherentes.

**Probleme temporel :** SMOTE ignore la structure temporelle des donnees et brise les correlations entre jours consecutifs.

Alternative adoptee : class_weight='balanced' dans les modeles, scale_pos_weight pour XGBoost, et sample_weight de l'etape 6.

### Etape 9 - Training : Comparaison de modeles

Trois modeles sont evalues en cross-validation stratifiee 5-fold avec le F2-score comme critere :

- RandomForest : robuste, interpretable, gere bien les outliers
- XGBoost : Gradient Boosting optimise, tres performant sur donnees tabulaires
- LightGBM : Gradient Boosting rapide, gestion efficace des valeurs manquantes

**Pourquoi le F2-score ?**

Le F1-score donne le meme poids a la precision et au recall. En meteorologie, les erreurs ont des couts asymetriques :

- Faux Negatif (on predit "pas de pluie" et il pleut) : agriculteur non prepare, travaux non planifies en consequence. Cout eleve.
- Faux Positif (on predit "pluie" et il fait beau) : on prend un parapluie inutilement. Cout faible.

Le F2-score (beta = 2) penalise deux fois plus les Faux Negatifs :

    F2 = 5 * (Precision * Recall) / (4 * Precision + Recall)

### Etape 10 - Optimisation des hyperparametres

RandomizedSearchCV (30 iterations) est applique sur LightGBM et XGBoost. L'espace de recherche inclut le nombre de features k de la selection automatique : on optimise le pipeline complet, pas uniquement le modele.

GridSearchCV est ecarte car trop lent : avec 8 hyperparametres et 5 valeurs chacun, il produirait 5^8 = 390 625 combinaisons.

### Etape 11 - Entrainement final

Le meilleur modele est reentrainee sur 100% du train avec les meilleurs hyperparametres. Pendant la recherche, chaque combinaison etait evaluee sur des sous-ensembles (les folds). Le modele final beneficie de la totalite des donnees d'entrainement.

### Etape 12 - Evaluation finale

**Metriques utilisees**

| Metrique | Valeur parfaite | Role |
|----------|-----------------|------|
| F2-score | 1 | Metrique principale (recall double de la precision) |
| Recall | 1 | Taux de jours de pluie detectes |
| Precision | 1 | Fiabilite des alertes pluie |
| ROC-AUC | 1 | Discriminance globale, independante du seuil |
| Brier Score | 0 | Qualite de calibration des probabilites |
| Log Loss | 0 | Penalise les probabilites mal calibrees |

**Le Brier Score : la metrique des meteorologues**

Un service meteorologique ne dit pas "il pleuvra demain". Il dit "70% de risque de pluie". Le Brier Score mesure si ces probabilites sont bien calibrees :

    Brier Score = (1/n) * somme des (probabilite_predite - valeur_reelle)^2

Si le modele annonce 70% pour 100 jours, il devrait effectivement pleuvoir environ 70 fois. C'est la calibration. Brier = 0 signifie des predictions parfaites. Brier = 0.25 correspond a un modele aleatoire sur une classe equilibree.

**Optimisation du seuil de decision**

Par defaut, le modele predit "Rain" si proba > 0.5. Ce seuil n'est pas toujours optimal. On cherche le seuil qui maximise le F2-score sur la courbe Precision-Rappel.

---

## Justification des choix techniques

| Decision | Alternative rejetee | Pourquoi notre choix est meilleur |
|----------|--------------------|------------------------------------|
| Split chronologique | Split aleatoire | Les donnees temporelles ne peuvent pas regarder vers le futur |
| RobustScaler | StandardScaler | Resistant aux valeurs extremes meteorologiques |
| Mutual Information | Correlation Pearson | Capture les effets non-lineaires et les effets de seuil |
| Pas de SMOTE | SMOTE | Les interpolations meteo ne sont pas physiques, brise la temporalite |
| F2-score | F1-score | Manquer de la pluie coute plus que les fausses alarmes |
| Brier Score | Accuracy | Les probabilites meteorologiques comptent autant que la decision binaire |
| class_weight balanced | Rien | Compense le desequilibre sans creer de faux exemples |
| XGBoost / LightGBM | SVM, regression logistique | Meilleure performance sur donnees tabulaires, robuste aux outliers |

---

## Resultats

| Metrique | Valeur |
|----------|--------|
| F2-score (metrique principale) | 0.730 environ |
| Recall Rain | 0.873 environ |
| Precision Rain | 0.440 environ |
| ROC-AUC | 0.877 environ |
| Brier Score Loss | 0.134 environ |
| Modele final | LightGBM ou XGBoost selon le run |
| Split | Chronologique (2007-2015 train, 2015-2017 test) |

Note : les metriques avec split chronologique sont inferieures a celles obtenues avec un split aleatoire, mais elles refletent les vraies performances du modele sur un futur non vu.

---

## Structure du projet

```
/
|
|-- data/
|   |-- weatherAUS.csv               # Dataset source
|
|-- notebooks/
|   |-- DS_project_weatherAUS.ipynb  # Notebook principal (12 etapes documentees)
|
|-- src/
|   |-- ml_pipeline.py               # Script Python autonome et valide
|   |-- features/
|   |   |-- build_features.py
|   |-- models/
|       |-- train_model.py
|       |-- predict_model.py
|
|-- reports/
|   |-- final_results.json           # Metriques du modele final
|   |-- figures/
|       |-- 01_target_distribution.png
|       |-- 02_correlation_target.png
|       |-- 03_seasonality.png
|       |-- 04_evaluation.png
|
|-- models/                          # Modeles sauvegardes
|-- references/                      # Documentation et notes
|-- README.md                        # Ce fichier
```

---

## Lancer le projet

### Creer l'environnement conda

Toutes les dependances, y compris xgboost et lightgbm, sont declarees dans `environment.yml`.

```bash
conda env create -f environment.yml
conda activate weather
```

Si l'environnement existe deja :

```bash
conda env update -f environment.yml --prune
conda activate weather
```

### Executer le pipeline complet

```bash
conda run -n weather python src/ml_pipeline.py
```

### Ouvrir le notebook

```bash
conda run -n weather jupyter notebook notebooks/DS_project_weatherAUS.ipynb
```

Selectionner le kernel "weather" dans Jupyter.

---

## Pistes d'amelioration

1. Calibration des probabilites avec CalibratedClassifierCV pour ameliorer le Brier Score
2. Modele par zone climatique : les patterns sont tres differents entre Darwin (tropical) et Hobart (tempere)
3. Enrichissement des donnees : donnees radar ou satellite pour mieux traiter les cas ambigus
4. CatBoost : gere nativement les variables categorielles sans OneHotEncoder
5. Ensembling : combiner XGBoost, LightGBM et RandomForest

---

Donnees issues du Bureau of Meteorology, Australie. Projet realise dans le cadre d'une formation Data Science.
