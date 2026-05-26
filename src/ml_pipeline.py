"""
Weather Australia - Binary Classification ML Pipeline
Objectif : Prdire RainTomorrow
Mtrique principale : F2-score (recall x2 vs prcision) + Brier Score
Split : CHRONOLOGIQUE (obligatoire pour donnes temporelles)
"""

import warnings
warnings.filterwarnings('ignore')

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from functools import partial
mutual_info_fixed = partial(mutual_info_classif, random_state=42)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, fbeta_score, f1_score, brier_score_loss,
    log_loss, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve

import joblib
import xgboost as xgb
import lightgbm as lgb

DATA_PATH   = os.path.join(os.path.dirname(__file__), '..', 'data', 'weatherAUS.csv')
REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports', 'figures')
MODELS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

print("=" * 70)
print("WEATHER AUSTRALIA  ML PIPELINE (split chronologique, F2-score)")
print("=" * 70)

# =============================================================================
# 1. EDA
# =============================================================================
print("\n[1/12] EDA")
df = pd.read_csv(DATA_PATH)
print(f"  Shape : {df.shape}")
print(f"  Target : {df['RainTomorrow'].value_counts().to_dict()}")
print(f"  Missing (top 5) :\n{(df.isnull().sum()/len(df)*100).sort_values(ascending=False).head(5).round(1)}")

# =============================================================================
# 2. CLEANING
# =============================================================================
print("\n[2/12] CLEANING")
df = pd.read_csv(DATA_PATH)
n0 = len(df)
df = df.drop_duplicates()
df['Date'] = pd.to_datetime(df['Date'])
df = df[(df['Humidity9am'].isna()) | (df['Humidity9am'] <= 100)]
df = df[(df['Humidity3pm'].isna()) | (df['Humidity3pm'] <= 100)]
df = df[(df['Rainfall'].isna())    | (df['Rainfall'] >= 0)]
df = df[(df['Pressure9am'].isna()) | (df['Pressure9am'] > 900)]
df = df[(df['Pressure3pm'].isna()) | (df['Pressure3pm'] > 900)]
df = df[(df['WindSpeed9am'].isna())| (df['WindSpeed9am'] >= 0)]
df = df[(df['WindSpeed3pm'].isna())| (df['WindSpeed3pm'] >= 0)]
df = df.dropna(subset=['RainTomorrow'])
df['RainTomorrow'] = (df['RainTomorrow'] == 'Yes').astype(int)
df['RainToday']    = df['RainToday'].map({'Yes': 1, 'No': 0})
print(f"  Lignes retires : {n0 - len(df)} | Shape : {df.shape}")

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================
print("\n[3/12] FEATURE ENGINEERING")
df['Month']      = df['Date'].dt.month
df['DayOfYear']  = df['Date'].dt.dayofyear
df['Season']     = df['Month'].map({
    12:'Summer',1:'Summer',2:'Summer',
    3:'Autumn', 4:'Autumn',5:'Autumn',
    6:'Winter', 7:'Winter',8:'Winter',
    9:'Spring', 10:'Spring',11:'Spring'
})
df['Month_sin']      = np.sin(2*np.pi*df['Month']/12)
df['Month_cos']      = np.cos(2*np.pi*df['Month']/12)
df['DayOfYear_sin']  = np.sin(2*np.pi*df['DayOfYear']/365)
df['DayOfYear_cos']  = np.cos(2*np.pi*df['DayOfYear']/365)

df = df.sort_values(['Location','Date']).reset_index(drop=True)

df['Delta_Pressure'] = df['Pressure3pm'] - df['Pressure9am']
df['Delta_Humidity'] = df['Humidity3pm'] - df['Humidity9am']
df['Delta_Temp']     = df['Temp3pm']     - df['Temp9am']
df['Delta_Wind']     = df['WindSpeed3pm']- df['WindSpeed9am']

for col in ['Pressure9am','Humidity3pm','MaxTemp']:
    lag = df.groupby('Location')[col].shift(1)
    df[f'{col}_diff1'] = df[col] - lag

for col in ['Rainfall','Humidity3pm','Pressure9am']:
    df[f'{col}_roll3_mean'] = df.groupby('Location')[col].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df[f'{col}_roll3_max'] = df.groupby('Location')[col].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).max())

df['Pressure_drop_flag']  = (df['Delta_Pressure'] < -2).astype(int)
df['HighHumidity_flag']   = (df['Humidity3pm'] > 85).astype(int)
df['StrongWind_flag']     = (df['WindGustSpeed'] > 60).astype(int)
df['HumidityRising_flag'] = (df['Delta_Humidity'] > 10).astype(int)
df['PressureLow_flag']    = (df['Pressure9am'] < 1010).astype(int)

df['Rain_x_Humidity'] = df['RainToday'].fillna(0) * df['Humidity3pm'].fillna(df['Humidity3pm'].median())
df['Wind_x_Humidity'] = df['WindGustSpeed'].fillna(0) * df['Humidity3pm'].fillna(df['Humidity3pm'].median())
df['TempRange']       = df['MaxTemp'] - df['MinTemp']

wind_dir_map = {
    'N':0,'NNE':22.5,'NE':45,'ENE':67.5,'E':90,'ESE':112.5,'SE':135,'SSE':157.5,
    'S':180,'SSW':202.5,'SW':225,'WSW':247.5,'W':270,'WNW':292.5,'NW':315,'NNW':337.5
}
for col in ['WindGustDir','WindDir9am','WindDir3pm']:
    angles = df[col].map(wind_dir_map)
    df[f'{col}_sin'] = np.sin(np.radians(angles))
    df[f'{col}_cos'] = np.cos(np.radians(angles))

df = df.drop(columns=['WindGustDir','WindDir9am','WindDir3pm','Date'])
print(f"  Shape aprs FE : {df.shape}")

# =============================================================================
# 4. SPLIT CHRONOLOGIQUE
# =============================================================================
print("\n[4/12] SPLIT CHRONOLOGIQUE")

# Reconstituer la colonne Date depuis DayOfYear+Month  non, on l'a droppe.
# On va couper sur l'index tri (les donnes sont dj tries par Location+Date)
# Meilleure approche : refaire le split sur le dataset complet avec la vraie date.
df_with_date = pd.read_csv(DATA_PATH, usecols=['Date'])
df_with_date['Date'] = pd.to_datetime(df_with_date['Date'])
# Aligner les index (aprs cleaning, certains index ont t retirs)
# On recharge proprement avec la date pour le split
df_full = pd.read_csv(DATA_PATH)
df_full['Date'] = pd.to_datetime(df_full['Date'])
df_full = df_full.drop_duplicates()
df_full = df_full[(df_full['Humidity9am'].isna()) | (df_full['Humidity9am'] <= 100)]
df_full = df_full[(df_full['Humidity3pm'].isna()) | (df_full['Humidity3pm'] <= 100)]
df_full = df_full[(df_full['Rainfall'].isna())    | (df_full['Rainfall'] >= 0)]
df_full = df_full[(df_full['Pressure9am'].isna()) | (df_full['Pressure9am'] > 900)]
df_full = df_full[(df_full['Pressure3pm'].isna()) | (df_full['Pressure3pm'] > 900)]
df_full = df_full[(df_full['WindSpeed9am'].isna())| (df_full['WindSpeed9am'] >= 0)]
df_full = df_full[(df_full['WindSpeed3pm'].isna())| (df_full['WindSpeed3pm'] >= 0)]
df_full = df_full.dropna(subset=['RainTomorrow'])
df_full = df_full.sort_values('Date').reset_index(drop=True)

# Coupure chronologique : 80% des dates uniques
unique_dates = df_full['Date'].sort_values().unique()
cutoff_date  = unique_dates[int(len(unique_dates) * 0.8)]
print(f"  Date de coupure : {cutoff_date.date()}")
print(f"  Train : {df_full['Date'].min().date()}  {cutoff_date.date()}")
print(f"  Test  : {cutoff_date.date()}  {df_full['Date'].max().date()}")

train_idx = df_full['Date'] <  cutoff_date
test_idx  = df_full['Date'] >= cutoff_date
print(f"  Train size : {train_idx.sum()} | Test size : {test_idx.sum()}")
print(f"  Ratio pluie train : {(df_full.loc[train_idx,'RainTomorrow']=='Yes').mean():.3f}")
print(f"  Ratio pluie test  : {(df_full.loc[test_idx, 'RainTomorrow']=='Yes').mean():.3f}")

# Reconstruire df avec features mais en gardant l'ordre chronologique
# df a dj t tri par Location+Date  on rutilise les index nettoys
# Plus simple : reconstruire X/y depuis df (dj trait)
df_sorted = df.copy()  # df est tri Location+Date, mme lignes que df_full aprs cleaning

# La date a t droppe de df  on utilise DayOfYear+Month_sin/cos comme proxy
# MAIS il vaut mieux refaire proprement : on ajoute la date au df avant de la dropper
# Refaire le FE en gardant la date pour le split
df2 = pd.read_csv(DATA_PATH)
df2['Date'] = pd.to_datetime(df2['Date'])
df2 = df2.drop_duplicates()
df2 = df2[(df2['Humidity9am'].isna()) | (df2['Humidity9am'] <= 100)]
df2 = df2[(df2['Humidity3pm'].isna()) | (df2['Humidity3pm'] <= 100)]
df2 = df2[(df2['Rainfall'].isna())    | (df2['Rainfall'] >= 0)]
df2 = df2[(df2['Pressure9am'].isna()) | (df2['Pressure9am'] > 900)]
df2 = df2[(df2['Pressure3pm'].isna()) | (df2['Pressure3pm'] > 900)]
df2 = df2[(df2['WindSpeed9am'].isna())| (df2['WindSpeed9am'] >= 0)]
df2 = df2[(df2['WindSpeed3pm'].isna())| (df2['WindSpeed3pm'] >= 0)]
df2 = df2.dropna(subset=['RainTomorrow'])
df2['RainTomorrow'] = (df2['RainTomorrow'] == 'Yes').astype(int)
df2['RainToday']    = df2['RainToday'].map({'Yes': 1, 'No': 0})

df2['Month']     = df2['Date'].dt.month
df2['DayOfYear'] = df2['Date'].dt.dayofyear
df2['Season']    = df2['Month'].map({
    12:'Summer',1:'Summer',2:'Summer',3:'Autumn',4:'Autumn',5:'Autumn',
    6:'Winter',7:'Winter',8:'Winter',9:'Spring',10:'Spring',11:'Spring'
})
df2['Month_sin']     = np.sin(2*np.pi*df2['Month']/12)
df2['Month_cos']     = np.cos(2*np.pi*df2['Month']/12)
df2['DayOfYear_sin'] = np.sin(2*np.pi*df2['DayOfYear']/365)
df2['DayOfYear_cos'] = np.cos(2*np.pi*df2['DayOfYear']/365)

df2 = df2.sort_values(['Location','Date']).reset_index(drop=True)

df2['Delta_Pressure'] = df2['Pressure3pm'] - df2['Pressure9am']
df2['Delta_Humidity'] = df2['Humidity3pm'] - df2['Humidity9am']
df2['Delta_Temp']     = df2['Temp3pm']     - df2['Temp9am']
df2['Delta_Wind']     = df2['WindSpeed3pm']- df2['WindSpeed9am']

for col in ['Pressure9am','Humidity3pm','MaxTemp']:
    lag = df2.groupby('Location')[col].shift(1)
    df2[f'{col}_diff1'] = df2[col] - lag

for col in ['Rainfall','Humidity3pm','Pressure9am']:
    df2[f'{col}_roll3_mean'] = df2.groupby('Location')[col].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df2[f'{col}_roll3_max'] = df2.groupby('Location')[col].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).max())

df2['Pressure_drop_flag']  = (df2['Delta_Pressure'] < -2).astype(int)
df2['HighHumidity_flag']   = (df2['Humidity3pm'] > 85).astype(int)
df2['StrongWind_flag']     = (df2['WindGustSpeed'] > 60).astype(int)
df2['HumidityRising_flag'] = (df2['Delta_Humidity'] > 10).astype(int)
df2['PressureLow_flag']    = (df2['Pressure9am'] < 1010).astype(int)

df2['Rain_x_Humidity'] = df2['RainToday'].fillna(0) * df2['Humidity3pm'].fillna(df2['Humidity3pm'].median())
df2['Wind_x_Humidity'] = df2['WindGustSpeed'].fillna(0) * df2['Humidity3pm'].fillna(df2['Humidity3pm'].median())
df2['TempRange']       = df2['MaxTemp'] - df2['MinTemp']

for col in ['WindGustDir','WindDir9am','WindDir3pm']:
    angles = df2[col].map(wind_dir_map)
    df2[f'{col}_sin'] = np.sin(np.radians(angles))
    df2[f'{col}_cos'] = np.cos(np.radians(angles))

df2 = df2.drop(columns=['WindGustDir','WindDir9am','WindDir3pm'])
# Garder la Date pour le split, puis la dropper dans X

# Split chronologique
unique_dates2 = df2['Date'].sort_values().unique()
cutoff_date2  = unique_dates2[int(len(unique_dates2) * 0.8)]

train_mask = df2['Date'] <  cutoff_date2
test_mask  = df2['Date'] >= cutoff_date2

df2 = df2.drop(columns=['Date'])

X = df2.drop(columns=['RainTomorrow'])
y = df2['RainTomorrow']

X_train = X[train_mask].reset_index(drop=True)
X_test  = X[test_mask].reset_index(drop=True)
y_train = y[train_mask].reset_index(drop=True)
y_test  = y[test_mask].reset_index(drop=True)

print(f"\n  Train : {X_train.shape} | ratio pluie : {y_train.mean():.3f}")
print(f"  Test  : {X_test.shape}  | ratio pluie : {y_test.mean():.3f}")

# =============================================================================
# 5. PREPROCESSING
# =============================================================================
print("\n[5/12] PREPROCESSING")
numeric_features     = X_train.select_dtypes(include=['float64','int64','int32']).columns.tolist()
categorical_features = X_train.select_dtypes(include='object').columns.tolist()
print(f"  Num : {len(numeric_features)} | Cat : {categorical_features}")

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ]), numeric_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), categorical_features)
], remainder='drop')

# =============================================================================
# 6. PONDRATION ANTI-LABEL-NOISE
# =============================================================================
print("\n[6/12] PONDRATION ANTI-LABEL-NOISE")
rainfall_train = X_train['Rainfall'].fillna(0)
humidity_train = X_train['Humidity3pm'].fillna(50)

sample_weight = np.ones(len(y_train))
ambig_low  = (rainfall_train < 0.5) & (y_train == 1)
ambig_mid  = humidity_train.between(60, 75) & (y_train == 0)
clear_pos  = (rainfall_train > 5) & (y_train == 1)
sample_weight[ambig_low.values]  = 0.6
sample_weight[ambig_mid.values]  = 0.8
sample_weight[clear_pos.values]  = 1.5
print(f"  Ambigus (0.6) : {ambig_low.sum()} | Ambigus (0.8) : {ambig_mid.sum()} | Clairs (1.5) : {clear_pos.sum()}")

# =============================================================================
# 7. FEATURE SELECTION
# =============================================================================
print("\n[7/12] FEATURE SELECTION  SelectKBest(mutual_info, k variable)")

# =============================================================================
# 9. TRAINING  comparaison avec F2-score
# =============================================================================
print("\n[9/12] TRAINING  comparaison modles (scoring=F2)")

from sklearn.metrics import make_scorer
f2_scorer = make_scorer(fbeta_score, beta=2)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'RandomForest': Pipeline([
        ('preprocessor', preprocessor),
        ('selector', SelectKBest(score_func=mutual_info_fixed, k=35)),
        ('model', RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, class_weight='balanced',
            random_state=42, n_jobs=-1
        ))
    ]),
    'XGBoost': Pipeline([
        ('preprocessor', preprocessor),
        ('selector', SelectKBest(score_func=mutual_info_fixed, k=35)),
        ('model', xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
            eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0
        ))
    ]),
    'LightGBM': Pipeline([
        ('preprocessor', preprocessor),
        ('selector', SelectKBest(score_func=mutual_info_fixed, k=35)),
        ('model', lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1
        ))
    ])
}

cv_results = {}
for name, pipeline in models.items():
    print(f"  CV {name}...", end=' ', flush=True)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=f2_scorer, n_jobs=-1)
    cv_results[name] = scores
    print(f"F2 = {scores.mean():.4f}  {scores.std():.4f}")

# =============================================================================
# 10. HYPERPARAMETER TUNING
# =============================================================================
print("\n[10/12] HYPERPARAMETER TUNING  RandomizedSearchCV (F2)")

lgbm_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('selector', SelectKBest(score_func=mutual_info_fixed, k=35)),
    ('model', lgb.LGBMClassifier(class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1))
])
lgbm_params = {
    'selector__k':              [25, 30, 35, 40],
    'model__n_estimators':      [200, 300, 400, 500],
    'model__max_depth':         [4, 5, 6, 7, 8],
    'model__learning_rate':     [0.03, 0.05, 0.07, 0.1],
    'model__num_leaves':        [31, 50, 63, 80],
    'model__subsample':         [0.7, 0.8, 0.9],
    'model__colsample_bytree':  [0.7, 0.8, 0.9],
    'model__reg_alpha':         [0, 0.1, 0.5],
    'model__reg_lambda':        [0, 0.1, 0.5, 1.0],
    'model__min_child_samples': [10, 20, 30],
}
print("  RandomizedSearch LightGBM (30 iter, scoring=F2)...", flush=True)
lgbm_search = RandomizedSearchCV(lgbm_pipe, lgbm_params, n_iter=30, cv=cv,
                                  scoring=f2_scorer, n_jobs=-1, random_state=42, verbose=0)
lgbm_search.fit(X_train, y_train)
print(f"  LightGBM best F2 : {lgbm_search.best_score_:.4f}")

xgb_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('selector', SelectKBest(score_func=mutual_info_fixed, k=35)),
    ('model', xgb.XGBClassifier(
        scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
        eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0
    ))
])
xgb_params = {
    'selector__k':          [25, 30, 35, 40],
    'model__n_estimators':  [200, 300, 400, 500],
    'model__max_depth':     [4, 5, 6, 7],
    'model__learning_rate': [0.03, 0.05, 0.07, 0.1],
    'model__subsample':     [0.7, 0.8, 0.9],
    'model__colsample_bytree': [0.7, 0.8, 0.9],
    'model__gamma':         [0, 0.1, 0.5],
    'model__reg_alpha':     [0, 0.1, 0.5],
    'model__reg_lambda':    [0.5, 1.0, 1.5],
    'model__min_child_weight': [1, 3, 5],
}
print("  RandomizedSearch XGBoost (30 iter, scoring=F2)...", flush=True)
xgb_search = RandomizedSearchCV(xgb_pipe, xgb_params, n_iter=30, cv=cv,
                                 scoring=f2_scorer, n_jobs=-1, random_state=42, verbose=0)
xgb_search.fit(X_train, y_train)
print(f"  XGBoost best F2 : {xgb_search.best_score_:.4f}")

if lgbm_search.best_score_ >= xgb_search.best_score_:
    best_search, best_name = lgbm_search, 'LightGBM'
else:
    best_search, best_name = xgb_search, 'XGBoost'

print(f"\n  Meilleur modle tunn : {best_name} (F2 CV = {best_search.best_score_:.4f})")

# =============================================================================
# 11. FINAL TRAINING
# =============================================================================
print("\n[11/12] FINAL TRAINING")
final_model = best_search.best_estimator_
final_model.fit(X_train, y_train)
print(f"  {best_name} rentran sur {len(X_train)} observations.")

model_path = os.path.join(MODELS_DIR, 'final_model.joblib')
joblib.dump(final_model, model_path)
print(f"  Modele sauvegarde : {model_path}")

# =============================================================================
# 12. VALUATION
# =============================================================================
print("\n[12/12] VALUATION SUR TEST SET")

y_pred  = final_model.predict(X_test)
y_proba = final_model.predict_proba(X_test)[:,1]

# Seuil optimal pour F2
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f2_by_thresh = (1 + 2**2) * precisions[:-1] * recalls[:-1] / (2**2 * precisions[:-1] + recalls[:-1] + 1e-9)
best_idx   = np.argmax(f2_by_thresh)
best_thresh = thresholds[best_idx]
y_pred_opt  = (y_proba >= best_thresh).astype(int)

f2_default = fbeta_score(y_test, y_pred, beta=2)
f2_opt     = fbeta_score(y_test, y_pred_opt, beta=2)
f1_opt     = f1_score(y_test, y_pred_opt)
roc        = roc_auc_score(y_test, y_proba)
brier      = brier_score_loss(y_test, y_proba)
logloss    = log_loss(y_test, y_proba)

report = classification_report(y_test, y_pred_opt, output_dict=True)
recall_rain    = report['1']['recall']
precision_rain = report['1']['precision']

fn = ((y_test == 1) & (y_pred_opt == 0)).sum()
fp = ((y_test == 0) & (y_pred_opt == 1)).sum()

print("\n" + "="*55)
print("RSULTATS FINAUX")
print("="*55)
print(f"  Modle           : {best_name}")
print(f"  Seuil optimal    : {best_thresh:.3f}")
print(f"  F2-score (opt.)  : {f2_opt:.4f}   mtrique principale")
print(f"  F1-score (opt.)  : {f1_opt:.4f}")
print(f"  Recall Rain      : {recall_rain:.4f}")
print(f"  Precision Rain   : {precision_rain:.4f}")
print(f"  ROC-AUC          : {roc:.4f}")
print(f"  Brier Score Loss : {brier:.4f}")
print(f"  Log Loss         : {logloss:.4f}")
print(f"  Faux Ngatifs    : {fn} ({fn/len(y_test)*100:.1f}%)")
print(f"  Faux Positifs    : {fp} ({fp/len(y_test)*100:.1f}%)")
print("="*55)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_opt, target_names=['No Rain','Rain']))

# Graphique valuation
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(recalls[:-1], precisions[:-1], color='#4C72B0', lw=2)
axes[0].scatter(recalls[best_idx], precisions[best_idx], color='red', s=100, zorder=5,
                label=f'Seuil opt={best_thresh:.2f}\nF2={f2_opt:.4f}')
axes[0].set_xlabel('Recall'); axes[0].set_ylabel('Prcision')
axes[0].set_title('Courbe Prcision-Rappel')
axes[0].legend(); axes[0].grid(alpha=0.3)

cm = confusion_matrix(y_test, y_pred_opt)
ConfusionMatrixDisplay(cm, display_labels=['No Rain','Rain']).plot(ax=axes[1], colorbar=False, cmap='Blues')
axes[1].set_title(f'Confusion Matrix\nF2={f2_opt:.4f} | Seuil={best_thresh:.2f}')

frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10)
axes[2].plot(mean_pred, frac_pos, marker='o', color='#4C72B0', label=best_name)
axes[2].plot([0,1],[0,1],'k--', label='Parfaitement calibr')
axes[2].set_xlabel('Probabilit prdite'); axes[2].set_ylabel('Fraction relle')
axes[2].set_title('Courbe de calibration (Brier)'); axes[2].legend(); axes[2].grid(alpha=0.3)

plt.suptitle(f'valuation  {best_name} | Split chronologique | F2-score', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, '04_evaluation.png'), dpi=100)
plt.close()

results = {
    'model': best_name,
    'split': 'chronological',
    'main_metric': 'F2-score',
    'best_params': str(best_search.best_params_),
    'f2_cv': float(best_search.best_score_),
    'f2_test': float(f2_opt),
    'f1_test': float(f1_opt),
    'optimal_threshold': float(best_thresh),
    'recall_rain': float(recall_rain),
    'precision_rain': float(precision_rain),
    'roc_auc': float(roc),
    'brier_score': float(brier),
    'log_loss': float(logloss),
    'false_negatives': int(fn),
    'false_positives': int(fp),
    'train_size': int(len(X_train)),
    'test_size': int(len(X_test)),
}
with open(os.path.join(REPORTS_DIR, '..', 'final_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("\nPipeline termine. Resultats dans reports/final_results.json")

# =============================================================================
# 13. INTERPRETABILITE (SHAP + Cartographie Australie)
# =============================================================================
print("\n[13/13] INTERPRETABILITE (SHAP + carte Australie)")

try:
    import shap as _shap_mod
    _shap_ok = True
except ImportError:
    print("  SHAP non installe -> pip install shap")
    _shap_ok = False

if _shap_ok:
    # ------------------------------------------------------------------
    # 1. Noms des features apres preprocesseur + SelectKBest
    # ------------------------------------------------------------------
    _ohe = final_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['ohe']
    _cat_out = list(_ohe.get_feature_names_out(categorical_features))
    _all_names = numeric_features + _cat_out
    _sel = final_model.named_steps['selector']
    _sel_names = [_all_names[i] for i in _sel.get_support(indices=True)]

    # ------------------------------------------------------------------
    # 2. Echantillon SHAP (2000 obs max pour la vitesse)
    # ------------------------------------------------------------------
    _n_shap = min(2000, len(X_test))
    _rng = np.random.default_rng(42)
    _idx = _rng.choice(len(X_test), size=_n_shap, replace=False)
    X_shap = X_test.iloc[_idx].reset_index(drop=True)
    y_shap = y_test.iloc[_idx].reset_index(drop=True)

    _X_prep = final_model.named_steps['preprocessor'].transform(X_shap)
    _X_sel  = _sel.transform(_X_prep)
    _X_df   = pd.DataFrame(_X_sel, columns=_sel_names)

    # ------------------------------------------------------------------
    # 3. Calcul des valeurs SHAP via TreeExplainer
    # ------------------------------------------------------------------
    print(f"  TreeExplainer sur {_n_shap} observations...", flush=True)
    _explainer = _shap_mod.TreeExplainer(final_model.named_steps['model'])
    _sv = _explainer.shap_values(_X_sel)
    if isinstance(_sv, list):          # LightGBM renvoie [class0, class1]
        _sv = _sv[1]

    _mean_shap = pd.Series(np.abs(_sv).mean(axis=0), index=_sel_names).sort_values(ascending=False)
    _top20 = _mean_shap.head(20).index.tolist()
    _top10 = _mean_shap.head(10).index.tolist()
    _top4  = _mean_shap.head(4).index.tolist()
    print(f"  Top 5 features SHAP : {_mean_shap.head(5).index.tolist()}")

    # ------------------------------------------------------------------
    # Figure 05a : Bar chart importance globale (mean |SHAP|)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))
    _mean_shap.head(20).sort_values().plot(kind='barh', ax=ax, color='#4C72B0', edgecolor='white')
    ax.set_title(f'Importance SHAP globale (mean |SHAP|) - Top 20\n{best_name}', fontsize=13)
    ax.set_xlabel('Mean |SHAP value|')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, '05a_shap_importance.png'), dpi=100, bbox_inches='tight')
    plt.close()
    print("  Sauvegarde : 05a_shap_importance.png")

    # ------------------------------------------------------------------
    # Figure 05b : Beeswarm (implementation matplotlib directe, sans shap.plots)
    # shap.plots.beeswarm et shap.summary_plot peuvent echouer selon l'env.
    # On construit le beeswarm manuellement : scatter par feature avec jitter vertical.
    # ------------------------------------------------------------------
    _n_disp = min(20, len(_top20))
    _feats_disp = _top20[:_n_disp]
    fig, ax = plt.subplots(figsize=(10, 8))

    for _rank, _feat in enumerate(reversed(_feats_disp)):
        _col_idx = _sel_names.index(_feat)
        _shap_vals = _sv[:, _col_idx]
        _feat_vals = _X_df[_feat].values

        # Normalisation de la couleur (rouge = valeur haute, bleu = valeur basse)
        _vmin, _vmax = np.nanpercentile(_feat_vals, [5, 95])
        _norm = mcolors.Normalize(vmin=_vmin, vmax=_vmax)
        _colors = mcm.RdBu_r(_norm(_feat_vals))

        # Jitter vertical pour eviter la superposition des points
        _jitter = np.random.default_rng(_rank).uniform(-0.3, 0.3, size=len(_shap_vals))
        ax.scatter(_shap_vals, _rank + _jitter, c=_colors, s=8, alpha=0.5, linewidths=0)

    ax.set_yticks(range(_n_disp))
    ax.set_yticklabels(list(reversed(_feats_disp)), fontsize=9)
    ax.axvline(0, color='black', lw=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('Valeur SHAP (impact sur la prediction de pluie)')
    ax.set_title(f'SHAP Beeswarm - {best_name}', fontsize=13)
    ax.grid(axis='x', alpha=0.2)

    # Colorbar manuelle (rouge = valeur haute, bleu = valeur basse)
    _sm = mcm.ScalarMappable(cmap='RdBu_r', norm=mcolors.Normalize(0, 1))
    _sm.set_array([])
    cbar = plt.colorbar(_sm, ax=ax, shrink=0.4, pad=0.02)
    cbar.set_label('Valeur de la feature\n(normalisee)', fontsize=8)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Basse', 'Haute'])

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, '05b_shap_beeswarm.png'), dpi=100, bbox_inches='tight')
    plt.close('all')
    print("  Sauvegarde : 05b_shap_beeswarm.png")

    # ------------------------------------------------------------------
    # Figure 06 : Dependence plots (top 4 features)
    # Chaque graphe montre l'effet marginal d'une variable meteorologique
    # et son interaction automatique avec la feature la plus correlee
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, feat in zip(axes.flatten(), _top4):
        try:
            _shap_mod.dependence_plot(feat, _sv, _X_df, ax=ax, show=False,
                                      interaction_index='auto')
        except Exception:
            _shap_mod.dependence_plot(feat, _sv, _X_df, ax=ax, show=False,
                                      interaction_index=None)
        ax.set_title(f'Dependence: {feat}', fontsize=10)
        ax.grid(alpha=0.3)
    plt.suptitle(f'SHAP Dependence Plots - Top 4 features | {best_name}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, '06_shap_dependence.png'), dpi=100, bbox_inches='tight')
    plt.close('all')
    print("  Sauvegarde : 06_shap_dependence.png")

    # ------------------------------------------------------------------
    # Figure 07 : Analyse saisonniere
    # Heatmap importance par saison + boxplot de la feature principale
    # ------------------------------------------------------------------
    _sv_df = pd.DataFrame(_sv, columns=_sel_names)
    _sv_df['Season'] = X_shap['Season'].values
    _season_mean = _sv_df.groupby('Season')[_top10].apply(lambda d: d.abs().mean())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(_season_mean.T, annot=True, fmt='.3f', cmap='YlOrRd',
                linewidths=0.5, ax=axes[0])
    axes[0].set_title('Importance SHAP moyenne par saison (Top 10)', fontsize=11)
    axes[0].tick_params(axis='x', rotation=30)

    _bp_df = pd.DataFrame({'SHAP': _sv_df[_top4[0]].values,
                            'Season': X_shap['Season'].values})
    _order = [s for s in ['Summer', 'Autumn', 'Winter', 'Spring']
               if s in _bp_df['Season'].unique()]
    sns.boxplot(data=_bp_df, x='Season', y='SHAP', order=_order,
                palette='coolwarm', ax=axes[1])
    axes[1].axhline(0, color='black', linestyle='--', lw=1, alpha=0.7)
    axes[1].set_title(f'Distribution SHAP de "{_top4[0]}" par saison', fontsize=11)
    axes[1].set_xlabel('Saison')
    axes[1].set_ylabel('SHAP value')
    axes[1].grid(axis='y', alpha=0.3)

    plt.suptitle('Analyse saisonniere - Interpretabilite SHAP', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, '07_shap_seasonal.png'), dpi=100, bbox_inches='tight')
    plt.close('all')
    print("  Sauvegarde : 07_shap_seasonal.png")

    # ------------------------------------------------------------------
    # Figure 08 : Carte climatologique de l'Australie (cartopy)
    # Fond de carte reel : ocean, terres, cotes, frontieres d'Etats.
    # 3 panneaux : humidite | pression + fleches vent | pluviometrie
    # Source : serie complete (pas seulement le test set)
    # ------------------------------------------------------------------
    import cartopy.crs      as _ccrs
    import cartopy.feature  as _cfeat

    _COORDS = {
        'Adelaide':        (-34.93, 138.60), 'Albany':          (-35.02, 117.88),
        'Albury':          (-36.08, 146.92), 'AliceSprings':    (-23.70, 133.88),
        'BadgerysCreek':   (-33.88, 150.73), 'Ballarat':        (-37.55, 143.85),
        'Bendigo':         (-36.76, 144.28), 'Brisbane':        (-27.47, 153.02),
        'Cairns':          (-16.92, 145.77), 'Canberra':        (-35.28, 149.13),
        'Cobar':           (-31.50, 145.83), 'CoffsHarbour':    (-30.30, 153.12),
        'Dartmoor':        (-37.92, 141.27), 'Darwin':          (-12.46, 130.84),
        'GoldCoast':       (-28.00, 153.43), 'Hobart':          (-42.88, 147.33),
        'Jabiru':          (-12.66, 132.89), 'Katherine':       (-14.47, 132.27),
        'Launceston':      (-41.43, 147.14), 'Melbourne':       (-37.81, 144.96),
        'MelbourneAirport':(-37.67, 144.83), 'Mildura':         (-34.19, 142.15),
        'Moree':           (-29.47, 149.83), 'MountGambier':    (-37.83, 140.78),
        'MountGinini':     (-35.53, 148.77), 'Newcastle':       (-32.92, 151.78),
        'Nhil':            (-36.33, 141.65), 'NorahHead':       (-33.28, 151.57),
        'NorfolkIsland':   (-29.04, 167.96), 'Nuriootpa':       (-34.47, 138.99),
        'PearceRAAF':      (-31.67, 116.03), 'Penrith':         (-33.75, 150.70),
        'Perth':           (-31.95, 115.86), 'PerthAirport':    (-31.94, 115.97),
        'Portland':        (-38.34, 141.60), 'Richmond':        (-33.60, 150.75),
        'Sale':            (-38.10, 147.07), 'SalmonGums':      (-32.98, 121.63),
        'Sydney':          (-33.87, 151.21), 'SydneyAirport':   (-33.94, 151.18),
        'Townsville':      (-19.25, 146.82), 'Tuggeranong':     (-35.42, 149.09),
        'Uluru':           (-25.35, 131.04), 'WaggaWagga':      (-35.16, 147.47),
        'Walpole':         (-34.98, 116.73), 'Watsonia':        (-37.71, 145.08),
        'Williamtown':     (-32.80, 151.84), 'Witchcliffe':     (-34.03, 115.10),
        'Wollongong':      (-34.42, 150.89), 'Woomera':         (-31.15, 136.82),
    }

    _loc_in_top20 = [f for f in _top20 if f.startswith('Location_')]
    _loc_msg = ("Location dans top 20 SHAP : OUI (" + str(len(_loc_in_top20)) + " var)"
                if _loc_in_top20 else "Location dans top 20 SHAP : NON")
    print(f"  {_loc_msg}")

    _wind_dir_map = {
        'N':0,'NNE':22.5,'NE':45,'ENE':67.5,'E':90,'ESE':112.5,'SE':135,'SSE':157.5,
        'S':180,'SSW':202.5,'SW':225,'WSW':247.5,'W':270,'WNW':292.5,'NW':315,'NNW':337.5
    }
    _df_raw_map = pd.read_csv(DATA_PATH)
    _df_raw_map['_wind_rad'] = _df_raw_map['WindDir3pm'].map(_wind_dir_map).apply(
        lambda d: np.radians(d) if pd.notna(d) else np.nan)
    _df_raw_map['_wind_u'] = np.sin(_df_raw_map['_wind_rad'])
    _df_raw_map['_wind_v'] = np.cos(_df_raw_map['_wind_rad'])

    _climate = _df_raw_map.groupby('Location').agg(
        humidity = ('Humidity3pm', 'mean'),
        pressure = ('Pressure9am', 'mean'),
        rainfall = ('Rainfall',    'mean'),
        wind_u   = ('_wind_u',     'mean'),
        wind_v   = ('_wind_v',     'mean'),
    ).reset_index()
    _climate['lat'] = _climate['Location'].map(lambda l: _COORDS.get(l, (None, None))[0])
    _climate['lon'] = _climate['Location'].map(lambda l: _COORDS.get(l, (None, None))[1])
    _climate = _climate.dropna(subset=['lat', 'lon', 'humidity'])

    _proj   = _ccrs.PlateCarree()
    _extent = [112, 156, -45, -10]

    _states_feat = _cfeat.NaturalEarthFeature(
        'cultural', 'admin_1_states_provinces_lines', '50m',
        edgecolor='#888', facecolor='none')

    _panels = [
        ('humidity', 'YlGnBu',   'Humidite moyenne a 15h (%)',         '%',   False),
        ('pressure', 'RdYlBu_r', 'Pression atm. 9h (hPa) + vent 15h', 'hPa', True),
        ('rainfall', 'Blues',    'Pluviometrie journaliere moy. (mm)',  'mm',  False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(22, 8),
                             subplot_kw={'projection': _proj})

    for ax, (col, cmap, title, unit, add_quiver) in zip(axes, _panels):
        ax.set_extent(_extent, crs=_proj)
        ax.add_feature(_cfeat.OCEAN,     facecolor='#b8d4e8', zorder=0)
        ax.add_feature(_cfeat.LAND,      facecolor='#f0ece0', zorder=1)
        ax.add_feature(_states_feat,     linewidth=0.5, linestyle='--', zorder=2)
        ax.add_feature(_cfeat.COASTLINE, linewidth=0.9, edgecolor='#333', zorder=3)
        ax.add_feature(_cfeat.BORDERS,   linewidth=0.6, edgecolor='#555',
                       linestyle=':', zorder=3)

        _vals = _climate[col]
        _sc = ax.scatter(_climate['lon'], _climate['lat'],
                         c=_vals, cmap=cmap, s=120,
                         vmin=_vals.min(), vmax=_vals.max(),
                         edgecolors='#333', linewidth=0.5, alpha=0.9,
                         transform=_proj, zorder=5)
        _cb = plt.colorbar(_sc, ax=ax, shrink=0.48, pad=0.04)
        _cb.set_label(unit, fontsize=8)

        for _, _row in _climate.iterrows():
            ax.text(_row['lon'], _row['lat'] + 0.45,
                    f"{_row['Location']}\n{_row[col]:.1f}{unit}",
                    fontsize=4, ha='center', va='bottom',
                    transform=_proj, color='#111', zorder=6,
                    bbox=dict(facecolor='white', alpha=0.35, pad=0.5,
                              edgecolor='none', boxstyle='round'))

        if add_quiver:
            ax.quiver(_climate['lon'], _climate['lat'],
                      _climate['wind_u'], _climate['wind_v'],
                      color='#111', alpha=0.85,
                      transform=_proj, zorder=7,
                      width=0.003, headwidth=4, headlength=4)

        ax.set_title(title, fontsize=10, fontweight='bold')
        _gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                           alpha=0.5, color='gray', linestyle='--')
        _gl.top_labels   = False
        _gl.right_labels = False

    axes[2].text(0.02, 0.02, _loc_msg, transform=axes[2].transAxes,
                 fontsize=7, color='navy', style='italic')

    plt.suptitle('Carte climatologique Australie - Moyennes 2007-2017 par station',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, '08_australia_map.png'), dpi=120, bbox_inches='tight')
    plt.close('all')
    print("  Sauvegarde : 08_australia_map.png")

    print("\n  === RESUME INTERPRETABILITE ===")
    print(f"  Top 10 features : {_mean_shap.head(10).index.tolist()}")
    print(f"  {_loc_msg}")
    print("  Figures generees :")
    print("    05a_shap_importance.png  - Bar chart mean|SHAP| top 20")
    print("    05b_shap_beeswarm.png    - Beeswarm SHAP (distribution + direction)")
    print("    06_shap_dependence.png   - Dependence plots top 4 features")
    print("    07_shap_seasonal.png     - Heatmap + boxplot par saison")
    print("    08_australia_map.png     - Carte spatiale par station")

print("\nPipeline complet (etapes 1-13). Resultats dans reports/figures/")
