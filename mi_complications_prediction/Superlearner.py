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

estimators = [
    # Modèle linéaire classique
    ('lr', LogisticRegression(
        solver='lbfgs',       # optimise via quasi-Newton
        max_iter=90000,        # nombre max d'itérations
        penalty='l2',         # régularisation L2 par défaut
        C=2.246896,                # inverse du paramètre de régularisation (1 = default)
        class_weight=None,    # pas de pondération, peut mettre 'balanced' si dataset déséquilibré
        random_state=42
    )),
    # Random Forest
    ('rf', RandomForestClassifier(
        n_estimators=800,
        min_samples_split=5,
        min_samples_leaf=1,
        criterion='gini',
        max_features='sqrt',
        max_depth=50,
        class_weight='balanced_subsample',
        bootstrap=False,
        random_state=42,
        n_jobs=-1
    )),

    # Extra Trees
    ('et', ExtraTreesClassifier(
        n_estimators=500,
        max_depth=50,
        min_samples_split=10,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
]

# ------------------------------
# Méta-modèle
# ------------------------------
meta_model = LogisticRegression(solver='lbfgs', max_iter=10000)

# ------------------------------
# StackingClassifier
# ------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_model,
    cv=cv,
    n_jobs=-1,
    passthrough=False
)

# F1 scorer pondéré
f1_scorer = make_scorer(f1_score, average='weighted', zero_division=0)

# ------------------------------
# Entraînement du SuperLearner
# ------------------------------
stack_model.fit(X_train, y_train)

# ------------------------------
# Poids réel des modèles dans le méta-modèle
# ------------------------------
# Avec passthrough=True, le méta-modèle reçoit aussi toutes les features
# On ne prend que les coefficients correspondant aux modèles de base
coefs = stack_model.final_estimator_.coef_[0][:len(estimators)]
base_models = [name for name, _ in estimators]

df_coefs = pd.DataFrame({'Model': base_models, 'Weight': coefs})

plt.figure(figsize=(8,5))
sns.barplot(x='Weight', y='Model', data=df_coefs, palette='viridis')
plt.title("Poids des modèles dans le SuperLearner")
plt.xlabel("Coefficient du méta-modèle")
plt.ylabel("Modèle de base")
plt.show()

# ------------------------------
# Évaluation des modèles individuels + stacking (CV)
# ------------------------------
models = estimators + [('stacking', stack_model)]
results, names = [], []

for name, model in models:
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=f1_scorer, n_jobs=-1)
    results.append(cv_scores)
    names.append(name)
    print(f'>{name} F1-score moyen (CV) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

# Boxplot comparatif
plt.figure(figsize=(10,6))
plt.boxplot(results, tick_labels=names, showmeans=True)
plt.title("Comparaison des modèles (F1-score pondéré)")
plt.ylabel("F1-score")
plt.grid(axis='y')
plt.show()

# ------------------------------
# Évaluation finale sur X_test
# ------------------------------
stack_preds = stack_model.predict(X_test)
f1_final = f1_score(y_test, stack_preds, average='weighted', zero_division=0)
print(f"F1-score final sur X_test : {f1_final:.4f}")



# # Ajouter les prédictions dans X_new
# X_new["Prediction"] = preds_new

# # Exporter dans un fichier Excel
# X_new.to_excel("predictions.xlsx", index=False)

# print("Les prédictions ont été sauvegardées dans 'predictions.xlsx'")
