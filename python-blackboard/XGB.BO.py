# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 13:31:41 2025

@author: 霍湘月
"""

import pandas as pd
import optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.inspection import permutation_importance

# Load data
df = pd.read_csv("mcovs.csv")
X = df.drop(columns=["resp"])
y = df["resp"]

# Set Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Objective function
def objective(trial):
    params = {
        "objective": "reg:squarederror",
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.01, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 300, 3000),
        "subsample": trial.suggest_float("subsample", 0.3, 0.99),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "gamma": trial.suggest_float("gamma", 1e-5, 0.5),
        "n_jobs": 20
    }
    model = xgb.XGBRegressor(**params)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X, y, scoring="r2", cv=cv, n_jobs=-1)
    return score.mean()

# Create study with TPE (Bayesian optimization)
sampler = optuna.samplers.TPESampler(n_startup_trials=20, seed=42)
study = optuna.create_study(
    direction="maximize",
    study_name="XGB_Bayesian_HPO",
    storage="sqlite:///ML_x.db",
    load_if_exists=True,
    sampler=sampler
)
study.optimize(objective, n_trials=100)

# Output best result
print("Best parameters:", study.best_params)
print("Best R²:", study.best_value)

# Train best model
best_params = study.best_params
best_params.update(objective="reg:squarederror", n_jobs=20)
best_model = xgb.XGBRegressor(**best_params).fit(X, y)

# Feature importance (permutation)
perm = permutation_importance(best_model, X, y, scoring="r2", n_repeats=10,
                              random_state=42, n_jobs=20)
perm_df = pd.DataFrame({
    "feature": X.columns,
    "perm_imp": perm.importances_mean
}).sort_values("perm_imp", ascending=False)
print("Top permutation importance:\n", perm_df.head(10))
