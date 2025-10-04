from data import * 

import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from scipy.stats import randint, uniform, loguniform

# Pour le XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# Paramètres de la recherche

N_ITER = 200          # nombre d'itérations RandomizedSearchCV
CV = 5                # nombre de folds CV
SCORING = make_scorer(f1_score, average='macro', zero_division=0)
RANDOM_STATE = 42
N_JOBS = -1
VERBOSE = 2


# Distributions larges par modèle

param_dist = {
    'lr_l2': {
        'C': loguniform(1e-6, 3),
        'penalty': ['l2'],
        'solver': ['lbfgs', 'saga'],
        'class_weight': [None, 'balanced']
    },
    'lr_en': {
        'C': loguniform(1e-6, 3),
        'penalty': ['elasticnet'],
        'l1_ratio': uniform(0.0, 1.0),
        'solver': ['saga'],
        'class_weight': [None, 'balanced']
    },
    'svm_linear': {
        'C': loguniform(1e-6, 3),
        'kernel': ['linear'],
        'class_weight': [None, 'balanced']
    },
    'svm_rbf': {
        'C': loguniform(1e-6, 3),
        'gamma': ['scale', 'auto', loguniform(1e-6, 1)],
        'kernel': ['rbf'],
        'class_weight': [None, 'balanced']
    },
    'sgd': {
        'alpha': loguniform(1e-8, 1e-1),
        'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
        'penalty': ['l2', 'elasticnet', 'l1', 'none'],
        'l1_ratio': uniform(0.0, 1.0)
    },
    'lda': {
        'solver': ['svd', 'lsqr', 'eigen'],
        'shrinkage': [None, 'auto', uniform(0.0, 1.0)]
    },
    'qda': {
        'reg_param': uniform(0.0, 0.9)
    },
    'knn': {
        'n_neighbors': randint(1, 50),
        'weights': ['uniform', 'distance'],
        'p': [1, 2],
        'metric': ['minkowski', 'euclidean', 'manhattan']
    },
    'rf': {
        'n_estimators': randint(150, 1200),
        'max_depth': [None] + list(randint(3, 100).rvs(20)),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'max_features': ['sqrt', 'log2', None],
        'class_weight': [None, 'balanced', 'balanced_subsample'],
        'bootstrap': [True, False]
    },
    'et': {
        'n_estimators': randint(150, 1000),
        'max_depth': [None] + list(randint(3, 100).rvs(20)),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'max_features': ['sqrt', 'log2', None],
        'class_weight': [None, 'balanced']
    },
    'gb': {
        'n_estimators': randint(150, 1000),
        'learning_rate': loguniform(1e-4, 1.0),
        'max_depth': randint(2, 20),
        'subsample': uniform(0.3, 0.7)
    }
}

if HAS_XGB:
    param_dist['xgb'] = {
        'n_estimators': randint(50, 1000),
        'max_depth': randint(2, 16),
        'learning_rate': loguniform(1e-4, 1.0),
        'subsample': uniform(0.3, 1.0),
        'colsample_bytree': uniform(0.3, 1.0),
        'gamma': loguniform(1e-8, 10)
    }


# Estimateurs initiaux

estimators_init = {
    'lr_l2': LogisticRegression(random_state=RANDOM_STATE, max_iter=10000),
    'lr_en': LogisticRegression(random_state=RANDOM_STATE, max_iter=10000, solver='saga'),
    'svm_linear': SVC(random_state=RANDOM_STATE, probability=True),
    'svm_rbf': SVC(random_state=RANDOM_STATE, probability=True),
    'sgd': SGDClassifier(random_state=RANDOM_STATE, max_iter=10000),
    'lda': LinearDiscriminantAnalysis(),
    'qda': QuadraticDiscriminantAnalysis(),
    'knn': KNeighborsClassifier(),
    'rf': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),  # n_jobs -1 sometimes collides with outer parallelism
    'et': ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=1),
    'gb': GradientBoostingClassifier(random_state=RANDOM_STATE)
}
if HAS_XGB:
    estimators_init['xgb'] = XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss')


# Fonction utilitaire

def run_random_search(name, estimator, param_distribution):
    if param_distribution is None:
        print(f"→ Aucun espace de recherche pour {name}, on garde l'estimateur par défaut.")
        return estimator, None

    rs = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distribution,
        n_iter=N_ITER,
        cv=CV,
        scoring=SCORING,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        verbose=VERBOSE,
        return_train_score=False,
        error_score=np.nan
    )
    t0 = time.time()
    try:
        rs.fit(X_train, y_train)
        t1 = time.time()
        print(f"\n✅ {name} — best {SCORING} = {rs.best_score_:.5f} (temps {t1 - t0:.1f}s)")
        print("   best params:", rs.best_params_)
        return rs.best_estimator_, rs
    except Exception as exc:
        print(f"\n Erreur lors de RandomizedSearchCV pour {name} : {exc}")
        print("   -> On retourne l'estimateur initial non optimisé.")
        return estimator, None


# Boucle d'optimisation

search_results = {}
best_estimators = {}

for name, est in estimators_init.items():
    print(f"\n--- Lancement recherche pour: {name} ---")
    pdist = param_dist.get(name, None)
    best_est, rs_obj = run_random_search(name, est, pdist)
    best_estimators[name] = best_est
    search_results[name] = rs_obj


# Résumé final

print("\n\n==================== RÉSUMÉ DES MEILLEURS RÉSULTATS ====================")
summary = []
for name, rs in search_results.items():
    if rs is None:
        print(f" - {name}: pas d'objet RandomizedSearchCV (aucune recherche ou erreur).")
        continue
    best_score = rs.best_score_ if hasattr(rs, 'best_score_') else np.nan
    best_params = rs.best_params_ if hasattr(rs, 'best_params_') else {}
    print(f"\n{name}:\n  best_{SCORING} = {best_score:.5f}\n  best_params = {best_params}")
    summary.append((name, best_score, best_params))

final_estimators_for_stacking = [(name, est) for name, est in best_estimators.items()]

print("\n✅ Script terminé — final_estimators_for_stacking prêt (liste d'(nom, estimator)).")
