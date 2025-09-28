from data import *

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score
from matplotlib import pyplot as plt
from numpy import mean, std
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import numpy as np

# ------------------------------
# Large hyperparameter ranges
# ------------------------------

# Logistic Regression
param_dist_lr = {
    'C': np.linspace(0.01, 5, 30),        # autour de 1.6237
    'penalty': ['l2'],
    'solver': ['lbfgs', 'saga'],
    'class_weight': [None, 'balanced'],
    'max_iter': [900000]  # on fixe, pas besoin d'optimiser
}

# LinearSVC
param_dist_svc = {
    'C': np.linspace(0.01, 2, 30),        # autour de 0.23357
    'penalty': ['l2'],
    'loss': ['hinge', 'squared_hinge'],
    'dual': [True, False],
    'class_weight': [None, 'balanced'],
    'max_iter': [900000]  # fixé aussi
}

# RandomForest
param_dist_rf = {
    'n_estimators': [500, 800, 1000, 1200],
    'max_depth': [None, 10, 20, 50],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini'],
    'max_features': ['log2', 'sqrt', None],
    'class_weight': ['balanced', 'balanced_subsample', None],
    'bootstrap': [True, False]
}

# ExtraTrees
param_dist_et = {
    'n_estimators': [500, 800, 1000, 1200],
    'max_depth': [None, 10, 20, 50],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', None],
    'bootstrap': [False, True]
}

# ------------------------------
# Function for RandomizedSearchCV
# ------------------------------
def optimize_model(model, param_dist, X_train, y_train, n_iter=500):
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='f1_weighted',
        cv=5,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    print(f"Best params for {model.__class__.__name__}: {search.best_params_}")
    print(f"Best F1-score (CV) : {search.best_score_:.4f}\n")
    return search.best_estimator_

# ------------------------------
# Optimize each model
# ------------------------------
best_lr = optimize_model(LogisticRegression(max_iter=90000, random_state=42), param_dist_lr, X_train, y_train)
best_svc = optimize_model(LinearSVC(max_iter=90000, random_state=42), param_dist_svc, X_train, y_train)
best_rf = optimize_model(RandomForestClassifier(random_state=42, n_jobs=-1), param_dist_rf, X_train, y_train)
best_et = optimize_model(ExtraTreesClassifier(random_state=42, n_jobs=-1), param_dist_et, X_train, y_train)

# ------------------------------
# Optimized models ready
# ------------------------------
optimized_models = [
    ('lr', best_lr),
    ('linsvc', best_svc),
    ('rf', best_rf),
    ('et', best_et)
]

print("\n==============================")
print(" ✅ Hyperparamètres finaux des modèles optimisés")
print("==============================\n")

for name, model in optimized_models:
    print(f"▶ {name.upper()} :")
    for param, value in model.get_params().items():
        print(f"   - {param}: {value}")
    print("\n------------------------------\n")
