# Sources et references du projet

## Dataset

**weatherAUS.csv**
Source : Bureau of Meteorology, Commonwealth of Australia (Copyright 2010)
Kaggle : https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package
Issu du package R "rattle" (Williams, 2011)

---

## Records meteorologiques australiens (seuils de nettoyage)

**Vitesse du vent maximale**
Record : 408 km/h le 10 avril 1996, Barrow Island (Cyclone tropical Olivia)
Reconnu par l'Organisation Meteorologique Mondiale (2010) comme le plus fort vent non-tornadique jamais enregistre.
Source : Bureau of Meteorology + WMO

**Pression atmospherique minimale**
Cyclone Monica (avril 2006) : pression mesuree ~916 hPa a Cape Wessel
Note : les estimations satellite donnaient ~868 hPa mais la mesure directe est 916 hPa.
Seuil de nettoyage dans le pipeline : 900 hPa (conserve les observations de cyclones du dataset, ex. Yasi 2011 ~929 hPa).
Source : https://www.bom.gov.au/cyclone/history/monica.shtml

---

## Algorithmes et methodes

**SHAP (TreeExplainer)**
Lundberg, S. M., & Lee, S.-I. (2017).
A Unified Approach to Interpreting Model Predictions.
Advances in Neural Information Processing Systems (NeurIPS), 30, 4766-4777.
arXiv : https://arxiv.org/abs/1705.07874
Proceedings : https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions

**XGBoost**
Chen, T., & Guestrin, C. (2016).
XGBoost: A Scalable Tree Boosting System.
Proceedings of the 22nd ACM SIGKDD, 785-794.
arXiv : https://arxiv.org/abs/1603.02754
ACM : https://dl.acm.org/doi/10.1145/2939672.2939785

**LightGBM**
Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017).
LightGBM: A Highly Efficient Gradient Boosting Decision Tree.
Advances in Neural Information Processing Systems (NIPS), 30, 3149-3157.
Proceedings : https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree

---

## Metriques

**Brier Score**
Brier, G. W. (1950).
Verification of forecasts expressed in terms of probability.
Monthly Weather Review, 78(1), 1-3.
https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2
Note : publie dans une revue de meteorologie — la metrique est nee pour evaluer les previsions de pluie, exactement notre cas d'usage.

**F-beta score**
Van Rijsbergen, C. J. (1979). Information Retrieval (2nd ed.). Butterworth.
Le F2-score (beta=2) penalise deux fois plus les faux negatifs que les faux positifs.
Review : https://dl.acm.org/doi/10.1145/3606367

**Mutual Information (feature selection)**
Implementation sklearn : sklearn.feature_selection.mutual_info_classif
Estimation non-parametrique par k-plus-proches-voisins.
Doc : https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html
