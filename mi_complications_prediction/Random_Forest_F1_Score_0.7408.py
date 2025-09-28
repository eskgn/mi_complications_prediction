from data import *

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Définition du modèle
rf_model = RandomForestClassifier(
    n_estimators=240,
    min_samples_split=2,
    criterion='gini',
    min_samples_leaf=2,
    class_weight='balanced_subsample',
    max_features='sqrt',
    max_depth=9,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

# Définir un F1 scorer
f1_scorer = make_scorer(f1_score, average='weighted', zero_division=0)

# Validation croisée stratifiée
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Évaluation par cross-validation
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring=f1_scorer, n_jobs=-1)

print(f"F1-score moyen (CV) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Ensuite on peut entraîner sur tout le train et évaluer sur ton X_test
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
f1_test = f1_score(y_test, rf_preds, average='weighted', zero_division=0)
print(f"F1-score sur X_test : {f1_test:.4f}")

# Calcul et affichage de la matrice de confusion
cm = confusion_matrix(y_test, rf_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')  # tu peux changer la couleur
plt.title("Matrice de confusion")
plt.show() 


# # Ajouter les prédictions dans X_new
# X_new["Prediction"] = preds_new

# # Exporter dans un fichier Excel
# X_new.to_excel("predictions.xlsx", index=False)

# print("Les prédictions ont été sauvegardées dans 'predictions.xlsx'")

# Afficher l'importance des caractéristiques
# importances = rf_model.feature_importances_
# feature_names = X_train.columns
# feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
# plt.figure(figsize=(10, 6))
# sns.barplot(x=feature_importances.values, y=feature_importances.index)
# plt.title("Importance des caractéristiques - Random Forest")
# plt.xlabel("Importance")
# plt.ylabel("Caractéristiques")
# plt.tight_layout()
# plt.show()