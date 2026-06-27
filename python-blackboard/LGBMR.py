# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 22:21:44 2025

@author: 霍湘月
"""
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "20"

import pandas as pd
import optuna
from sklearn.model_selection import cross_val_score, KFold
from lightgbm import LGBMRegressor


optuna.logging.set_verbosity(optuna.logging.WARNING)

df = pd.read_csv("mcovs.csv")
#df = pd.read_csv("mcovs_ML_2.csv")
#df = pd.get_dummies(df, columns=["set_typ", "region"], drop_first=True)

X = df.drop(columns=["resp"])
y = df["resp"]
               

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 320, 370),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.066, 0.0666),
        "num_leaves": trial.suggest_int("num_leaves", 4, 64),
        "min_child_samples": trial.suggest_int("min_child_samples", 25, 30),
        "subsample": trial.suggest_float("subsample", 0.51, 0.66),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 0.88),
        "random_state": 42,
        "verbosity": -1 #close logging
    }

    model = LGBMRegressor(**params)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

    return scores.mean()


study = optuna.create_study(
    direction="maximize",
    study_name="LightGBM_4_x",
    storage="sqlite:///ML_x1.db",  
    load_if_exists=True
)

study.optimize(objective, n_trials=1000)

print("Best Parameters:")
print(study.best_params)
print("Best R² Score:", round(study.best_value, 4))


best_params = study.best_params
best_params.update(objective="regression", n_jobs=20)
best_model = LGBMRegressor(**best_params).fit(X, y)


from sklearn.inspection import permutation_importance
perm = permutation_importance(best_model, X, y, scoring="r2", n_repeats=10,
                              random_state=42)
perm_df = (pd.DataFrame({"feature": X.columns,
                         "perm_imp": perm.importances_mean})
             .sort_values("perm_imp", ascending=False))
print("Top permutation importance:\n", perm_df.head(30))


import matplotlib.pyplot as plt

# 选出前 8 个重要的变量（已经是按 importance 降序排列的）
top8 = perm_df.head(8).sort_values("perm_imp", ascending=True)  # 反转便于画图时从小到大

# 画散点图
plt.figure(figsize=(8, 5))
plt.scatter(top8["perm_imp"], top8["feature"], s=80, color="blue")
plt.xlabel("Permutation Importance (mean R² drop)")
plt.ylabel("Feature")
plt.title("Top 8 Feature Importances")
plt.grid(True)
plt.tight_layout()
plt.show()



