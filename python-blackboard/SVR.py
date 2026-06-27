# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 23:50:44 2025

@author: 霍湘月
"""
import pandas as pd
import optuna
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("mcovs.csv")
#df = pd.read_csv("mcovs_ML_2.csv")
#df = pd.get_dummies(df, columns=["set_typ", "region"], drop_first=True)

X = df.drop(columns=["resp"])
y = df["resp"]
               
###pca     
#X_pca= X.remove_columns(["x8", "x15", "x10"])   

###corr
#X_corr= X.remove_columns(["x", "x", "x"])   # <0.2:  "x6"  "x7"  "x8"  "x18" "x19" "x20" "x21" "x23" "x25" "x27" "x29" "x41"


optuna.logging.set_verbosity(optuna.logging.WARNING)  # close Optuna 日志输出
def objective(trial):
    params = {
        "C": trial.suggest_loguniform("C", 0.6, 15),
        #log(C)~Uniform(log(0.6),log(15))

        "gamma": trial.suggest_loguniform("gamma", -2, 0.05),
        "epsilon": trial.suggest_float("epsilon", 0.01, 0.2)
    }
    model = make_pipeline(StandardScaler(), SVR(**params))
    cv = KFold(n_splits=2, shuffle=True, random_state=42)
    score = cross_val_score(model,X, y, scoring="r2", cv=cv)
    return score.mean()

study = optuna.create_study(
    direction="maximize", 
    study_name="svr1_r", 
    storage="sqlite:///ML_x1.db",
    load_if_exists=True  # keep the old recording
)
study.optimize(objective, n_trials=600)

print("Best parameters:", study.best_params)
print("Best R²:", study.best_value) #0.45
#optuna-dashboard sqlite:///SVR_BO.db --port 8081
#http://127.0.0.1:8081
