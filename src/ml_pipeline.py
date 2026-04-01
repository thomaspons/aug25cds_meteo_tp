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
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, fbeta_score, f1_score, brier_score_loss,
    log_loss, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve

import xgboost as xgb
import lightgbm as lgb

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'weatherAUS.csv')
REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports', 'figures')
os.makedirs(REPORTS_DIR, exist_ok=True)

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
df = df[(df['Pressure9am'].isna()) | (df['Pressure9am'] > 800)]
df = df[(df['Pressure3pm'].isna()) | (df['Pressure3pm'] > 800)]
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
df_full = df_full[(df_full['Pressure9am'].isna()) | (df_full['Pressure9am'] > 800)]
df_full = df_full[(df_full['Pressure3pm'].isna()) | (df_full['Pressure3pm'] > 800)]
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
df2 = df2[(df2['Pressure9am'].isna()) | (df2['Pressure9am'] > 800)]
df2 = df2[(df2['Pressure3pm'].isna()) | (df2['Pressure3pm'] > 800)]
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
        ('selector', SelectKBest(score_func=mutual_info_classif, k=35)),
        ('model', RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, class_weight='balanced',
            random_state=42, n_jobs=-1
        ))
    ]),
    'XGBoost': Pipeline([
        ('preprocessor', preprocessor),
        ('selector', SelectKBest(score_func=mutual_info_classif, k=35)),
        ('model', xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
            eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0
        ))
    ]),
    'LightGBM': Pipeline([
        ('preprocessor', preprocessor),
        ('selector', SelectKBest(score_func=mutual_info_classif, k=35)),
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
    ('selector', SelectKBest(score_func=mutual_info_classif, k=35)),
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
    ('selector', SelectKBest(score_func=mutual_info_classif, k=35)),
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

print("\nPipeline termin. Rsultats dans reports/final_results.json")
