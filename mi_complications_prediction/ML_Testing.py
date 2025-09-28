from data import *

from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

models = {
    "LogisticRegression": LogisticRegression(C=1.0, max_iter=5000),
    "SVM_linear": CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=5000)),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss", random_state=42
    )
}

results = {}
plt.figure(figsize=(8, 6))

for name, model in models.items():
    print(f"Entraînement de {name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # Pour les modèles sans predict_proba (ex: LinearSVC non calibré)
        y_proba = model.decision_function(X_test)
    
    f1 = f1_score(y_test, y_pred)
    results[name] = f1
    
    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

# Résultats F1
print("\nRésultats F1-score :")
for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: {score:.5f}")

# Finalisation courbe ROC
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("Taux de faux positifs (FPR)")
plt.ylabel("Taux de vrais positifs (TPR)")
plt.title("Courbes ROC comparées")
plt.legend()
plt.show()














# # Ajouter les prédictions dans X_new
# X_new["Prediction"] = preds_new

# # Exporter dans un fichier Excel
# X_new.to_excel(r"C:\Users\enisk\Documents\DOCUMENTS\PROGRAMMING\Python\predictions.xlsx", index=False)

# print("Les prédictions ont été sauvegardées dans 'predictions.xlsx'")

