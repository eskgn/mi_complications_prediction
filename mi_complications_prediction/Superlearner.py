from data import *

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import mean, std
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score, confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier

# Définition des modèles de base

estimators = [
    ('lr', LogisticRegression(
        solver='lbfgs',
        max_iter=90000,
        penalty='l2',
        C=2.246896,
        class_weight=None,
        random_state=42
    )),
    ('lr_en', LogisticRegression(
    solver='saga',
    penalty='elasticnet',
    l1_ratio=0.66666,
    max_iter=10000,
    random_state=42
    )),
    ('svm', SVC(
    kernel='linear',  # 
    C=0.182,
    probability=True,
    class_weight='balanced',
    random_state=42
    )),
    ('sgd', SGDClassifier(
    loss='hinge',
    penalty='l2',
    alpha=0.00464,
    max_iter=1000,
    tol=1e-3,
    random_state=42
    )),
    ('lda', LinearDiscriminantAnalysis(
        solver='lsqr',
        shrinkage=0.1,
        n_components=None,
        priors=None,
        store_covariance=False,
        tol=0.0001
    )),
    ('rf', RandomForestClassifier(
        n_estimators=240,
        min_samples_split=2,
        min_samples_leaf=2,
        criterion='gini',
        max_features='sqrt',
        max_depth=30,
        class_weight='balanced_subsample',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )),
    ('et', ExtraTreesClassifier(
        n_estimators=500,
        max_depth=30,
        min_samples_split=10,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
]

# Méta-modèle
meta_model = LogisticRegression(solver='lbfgs', max_iter=10000)


# StackingClassifier (SuperLearner)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_model,
    cv=cv,
    n_jobs=-1,
    passthrough=False
)

f1_scorer = make_scorer(f1_score, average='weighted', zero_division=0)

# Entraînement du SuperLearner
stack_model.fit(X_train, y_train)


# Poids des modèles dans le SuperLearner

coefs = stack_model.final_estimator_.coef_[0][:len(estimators)]
base_models = [name for name, _ in estimators]

df_coefs = pd.DataFrame({'Model': base_models, 'Weight': coefs})

plt.figure(figsize=(7,4))
sns.barplot(x='Weight', y='Model', data=df_coefs, palette='viridis')
plt.title("Poids des modèles dans le SuperLearner")
plt.xlabel("Coefficient du méta-modèle")
plt.ylabel("Modèle de base")
for i, val in enumerate(df_coefs["Weight"]):
    plt.text(val, i, f"{val:.3f}", va='center', ha='left', fontsize=9)
plt.show()


# Évaluation des modèles individuels + stacking (CV)

models = estimators + [('stacking', stack_model)]
results, names = [], []

for name, model in models:
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=f1_scorer, n_jobs=-1)
    results.append(cv_scores)
    names.append(name)
    print(f'>{name} F1-score moyen (CV) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

mean_scores = [mean(r) for r in results]
std_scores = [std(r) for r in results]

plt.figure(figsize=(8,5))
sns.barplot(x=names, y=mean_scores, palette="crest", ci=None)
plt.errorbar(x=range(len(names)), y=mean_scores, yerr=std_scores, fmt='none', c='black', capsize=5)
plt.title("Comparaison des modèles (F1-score pondéré - CV)")
plt.ylabel("F1-score")
plt.xticks(rotation=30)
for i, val in enumerate(mean_scores):
    plt.text(i, val + 0.02, f"{val:.3f}", ha='center', fontsize=10)
plt.ylim(0,1)
plt.show()


# Évaluation finale sur X_test

stack_preds = stack_model.predict(X_test)
f1_final = f1_score(y_test, stack_preds, average='weighted', zero_division=0)
print(f"F1-score final sur X_test : {f1_final:.4f}")

# Matrice de confusion
cm = confusion_matrix(y_test, stack_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False)
plt.title("Matrice de confusion - SuperLearner")
plt.xlabel("Prédictions")
plt.ylabel("Valeurs réelles")
plt.show()

# Graphique du F1-score final
plt.figure(figsize=(5,4))
sns.barplot(x=["SuperLearner"], y=[f1_final], palette="crest")
plt.title("F1-score final sur le jeu de test")
plt.ylabel("F1-score pondéré")
plt.ylim(0,1)
plt.text(0, f1_final + 0.02, f"{f1_final:.3f}", ha='center', fontsize=12)
plt.show()
