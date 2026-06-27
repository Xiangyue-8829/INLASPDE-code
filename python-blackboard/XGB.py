# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 22:46:31 2025

@author: 霍湘月
"""

#pip install optuna
#pip install xgboost

import pandas as pd
import optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold


df = pd.read_csv("mcovs.csv")
#df = pd.read_csv("mcovs_ML_2.csv")
#df = pd.get_dummies(df, columns=["set_typ", "region"], drop_first=True)

X = df.drop(columns=["resp"])
y = df["resp"]
               

optuna.logging.set_verbosity(optuna.logging.WARNING)  # close Optuna 日志输出
def objective(trial):
  params = {
    "objective": "reg:squarederror",
    "max_depth": trial.suggest_int("max_depth", 5,20), 
    "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.01),
    "n_estimators": trial.suggest_int("n_estimators", 300,3000), 
    "subsample": trial.suggest_uniform("subsample", 0.3, 0.99),  
    "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.1, 1), 
    "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
    "gamma": trial.suggest_uniform("gamma", 0.00001, 0.5), 
    "n_jobs":20
    }
  model = xgb.XGBRegressor(**params)
  cv = KFold(n_splits=2, shuffle=True, random_state=42)
  score = cross_val_score(model, X, y, scoring="r2", cv=cv)
  return score.mean()


study = optuna.create_study(
    direction="maximize", 
    study_name="XGB_1_x", 
    storage="sqlite:///ML_x.db",
    load_if_exists=True  # keep the old recording
)
study.optimize(objective, n_trials=1000)

print("Best parameters:", study.best_params)
print("Best R²:", study.best_value)#0.4805- 1648 trial

#optuna-dashboard sqlite:///XGB_reg_typ.db --port 8082
#http://127.0.0.1:8082

#optuna-dashboard sqlite:///ML_x.db --port 8086
#http://127.0.0.1:8086


# Train
best_params = study.best_params
best_params.update(objective="reg:squarederror", n_jobs=20)
best_model = xgb.XGBRegressor(**best_params).fit(X, y)


from sklearn.inspection import permutation_importance
perm = permutation_importance(best_model, X, y, scoring="r2", n_repeats=10,
                              random_state=42, n_jobs=20)
perm_df = (pd.DataFrame({"feature": X.columns,
                         "perm_imp": perm.importances_mean})
             .sort_values("perm_imp", ascending=False))
print("Top permutation importance:\n", perm_df.head(10))

# shap
# import shap
# shap.initjs()
# explainer = shap.TreeExplainer(best_model)
# shap_vals = explainer.shap_values(X)
# shap.summary_plot(shap_vals, X, plot_type="bar")













