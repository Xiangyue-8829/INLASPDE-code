# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 23:50:26 2025

@author: 霍湘月
"""


# pip install pyjanitor
# pip uninstall ConfigParser
#pip install optuna
#pip install optuna-dashboard  # 推荐一起装，交互性更强

import pandas as pd
import janitor   
#import configparser  
import optuna

import seaborn as sns# corr
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

df = pd.read_csv("mcovs.csv")
#df = pd.read_csv("mcovs_ML_2.csv")
#df = pd.get_dummies(df, columns=["set_typ", "region"], drop_first=True)

X = df.drop(columns=["resp"])
y = df["resp"]
               
optuna.logging.set_verbosity(optuna.logging.WARNING)  # close Optuna 日志输出
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300,2500),
        "max_depth": trial.suggest_int("max_depth", 5, 40),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_float("max_features", 0.1, 1),
        "n_jobs": 20
    }
    model = RandomForestRegressor(**params, random_state=42)
    cv = KFold(n_splits=2, shuffle=True, random_state=42)
    score = cross_val_score(model,X, y, scoring="r2", cv=cv)
    return score.mean()

#study = optuna.create_study(direction="maximize")
#study.optimize(objective, n_trials=500)

study = optuna.create_study(
    direction="maximize", 
    study_name="RF_1_x", 
    storage="sqlite:///ML_x.db",
    load_if_exists=True  # keep the old recording
)
study.optimize(objective, n_trials=2000)

print("Best parameters:", study.best_params)
print("Best R²:", study.best_value)


#optuna-dashboard sqlite:///RF_BO.db # --port 8080
#http://127.0.0.1:8080















