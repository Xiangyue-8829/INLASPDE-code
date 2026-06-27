# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 19:35:34 2025

@author: 霍湘月
"""

#optuna-dashboard sqlite:///C:/Users/霍湘月/Desktop/p4-INLASPDE/Python_code/ML_x1.db --port 8080
#http://127.0.0.1:8080
import optuna
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import pandas as pd

study = optuna.load_study(
    study_name="svr_2_x",
    storage="sqlite:///C:/Users/霍湘月/Desktop/p4-INLASPDE/Python_code/ML_x1.db"
)

best_params = study.best_params.copy()

allowed_keys = ['C', 'epsilon', 'gamma', 'kernel', 'degree']
best_params = {k: v for k, v in best_params.items() if k in allowed_keys}

# Construct model
svr_model = make_pipeline(
    StandardScaler(),
    SVR(**best_params)
)

# Train
df = pd.read_csv("mcovs.csv")
#df = pd.read_csv("mcovs_ML_2.csv")
#df = pd.get_dummies(df, columns=["set_typ", "region"], drop_first=True)

X = df.drop(columns=["resp"])
y = df["resp"]
svr_model.fit(X, y)

# permutation importance
perm = permutation_importance(svr_model, X, y, scoring="r2", n_repeats=10,
                              random_state=42, n_jobs=20)

perm_df = (pd.DataFrame({
    "feature": X.columns,
    "perm_imp": perm.importances_mean
}).sort_values("perm_imp", ascending=False))

print("Top permutation importance-svr:\n", perm_df.head(30))


from sklearn.ensemble import RandomForestRegressor
study = optuna.load_study(
    study_name="RF_2_x",
    storage="sqlite:///C:/Users/霍湘月/Desktop/p4-INLASPDE/Python_code/ML_x1.db"
)

best_params = study.best_params.copy()

allowed_keys = [
    "n_estimators",         # number of trees
    "max_depth",          
    "min_samples_split",   
    "min_samples_leaf",     
    "max_features",        
    "max_leaf_nodes",      
    "bootstrap",           
    "n_jobs",             
    "random_state"         
]

best_params = {k: v for k, v in best_params.items() if k in allowed_keys}

# Construct model
RF_model = make_pipeline(
    StandardScaler(),
    RandomForestRegressor(**best_params)
)

# Train
df = pd.read_csv("mcovs.csv")
#df = pd.read_csv("mcovs_ML_2.csv")
#df = pd.get_dummies(df, columns=["set_typ", "region"], drop_first=True)

X = df.drop(columns=["resp"])
y = df["resp"]
RF_model.fit(X, y)

# permutation importance
perm = permutation_importance(RF_model, X, y, scoring="r2", n_repeats=10,
                              random_state=42, n_jobs=20)

perm_df = (pd.DataFrame({
    "feature": X.columns,
    "perm_imp": perm.importances_mean
}).sort_values("perm_imp", ascending=False))

print("Top permutation importance-RF:\n", perm_df.head(30))