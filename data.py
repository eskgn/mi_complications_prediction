import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Afficher toutes les colonnes
pd.set_option("display.max_columns", None)

# (Optionnel) afficher toutes les lignes
pd.set_option("display.max_rows", None)

# Lire le fichier CSV
data = pd.read_csv(r'MI.csv', sep=',')
variable = pd.read_csv(r'CahierVariables.csv', sep=';')
# df_new = pd.read_csv(r'', sep=',')

# X_new = df_new.copy()

# Afficher les premières lignes
print(data.shape)
print(variable.shape)

new_columns = variable.iloc[:, 0].astype(str).tolist()

# Créer new_df comme copie des données de df
df = data.copy()

# Renommer les colonnes en utilisant les valeurs de la première colonne de `variable`
df.columns = new_columns
print(variable.columns)

print(variable["Type"])

# print(df.head(10))
print(df.shape)
df = df.apply(pd.to_numeric, errors='coerce')

# Colonnes quantitatives : Integer + Continuous
quantitative_cols = variable.loc[variable['Type'].isin(['Integer', 'Continuous']), 'Variable Name'].tolist()
# Colonnes catégorielles : Categorical + Binary
categorical_cols = variable.loc[variable['Type'].isin(['Categorical', 'Binary']), 'Variable Name'].tolist()

print("Colonnes quantitatives :", quantitative_cols)
print("Colonnes catégorielles :", categorical_cols)


df.drop(columns=["ID"], inplace=True)

print(df.dtypes)
df = df.replace("?", np.nan)
summary = pd.DataFrame({
    "dtype": df.dtypes,
    "% manquant": df.isna().mean().mul(100).round(2)
})
# print(summary)

summary_sorted = summary.sort_values("% manquant", ascending=False).head(30)

plt.figure(figsize=(12,6))
sns.barplot(x=summary_sorted.index, y=summary_sorted["% manquant"])
plt.xticks(rotation=90)
plt.title("Top 30 colonnes avec le plus de valeurs manquantes")
plt.ylabel("% manquant")
plt.show()

# Visualiser les valeurs manquantes #1
plt.figure(figsize=(30, 10))
sns.heatmap(df.isna(), cbar=False)
plt.xticks(rotation=90)  # rotation pour voir les noms
plt.show()


# Supprimer les colonnes avec plus de 30% de valeurs manquantes
threshold = 30  # pourcentage
cols_to_drop = summary[summary["% manquant"] > threshold].index
df = df.drop(columns=cols_to_drop)

# X_new = X_new.drop(columns=cols_to_drop, errors='ignore')

# print(df.dtypes)
df = df.replace("?", np.nan)
summary = pd.DataFrame({
    "dtype": df.dtypes,
    "% manquant": df.isna().mean().mul(100).round(2)
})
print(summary)


# Visualiser les valeurs manquantes #2
plt.figure(figsize=(30, 10))
sns.heatmap(df.isna(), cbar=False)
plt.xticks(rotation=90)  # rotation pour voir les noms
plt.show()

summary_sorted = summary.sort_values("% manquant", ascending=False).head(30)

plt.figure(figsize=(12,6))
sns.barplot(x=summary_sorted.index, y=summary_sorted["% manquant"])
plt.xticks(rotation=90)
plt.title("Top 30 colonnes avec le plus de valeurs manquantes")
plt.ylabel("% manquant")
plt.show()


target_cols = variable.loc[variable['Role'] == 'Target', 'Variable Name'].tolist()
print("Colonnes cibles :", target_cols)


# Visualisation de la distribution des classes pour chaque target

# Nombre de targets
n_targets = len(target_cols)

# Taille de la figure
plt.figure(figsize=(5*n_targets, 4))

# Boucle sur chaque target
for i, col in enumerate(target_cols):
    plt.subplot(1, n_targets, i+1)
    sns.countplot(x=df[col])
    plt.title(f"{col}")
    plt.xlabel("Classe")
    plt.ylabel("Effectif")
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# Création d'une target binaire "complication" : 1 si au moins une complication, 0 sinon
target = df[target_cols]
target["LET_IS"] = (df["LET_IS"] != 0).astype(int) # transformer LET_IS en binaire
# Si au moins une des complications est présente, on met 1, sinon 0
target = (target.sum(axis=1) > 0).astype(int)

# Vérification de la distribution
print("Répartition de la target (0 = pas de complication, 1 = complication) :")
print(target.value_counts(normalize=True))

plt.figure(figsize=(4,4))
target.value_counts().plot(kind="bar")
plt.title("Répartition de la variable cible binaire")
plt.xlabel("Classe")
plt.ylabel("Effectif")
plt.xticks(rotation=0)
plt.show()


#On remarque que le jeu de donnée est déséquilibré.

#Charger les données 
X = df.drop(columns=target_cols)
y = target


# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.impute import SimpleImputer

# Ne garder que les colonnes existantes dans X_train
quantitative_cols_existing = [col for col in quantitative_cols if col in X_train.columns]
categorical_cols_existing = [col for col in categorical_cols if col in X_train.columns]

# Imputation pour les colonnes quantitatives (médiane)
quant_imputer = SimpleImputer(strategy='median')
X_train[quantitative_cols_existing] = quant_imputer.fit_transform(X_train[quantitative_cols_existing])
X_test[quantitative_cols_existing]  = quant_imputer.transform(X_test[quantitative_cols_existing])

# Imputation pour les colonnes catégorielles (plus fréquent)
cat_imputer = SimpleImputer(strategy='most_frequent')
X_train[categorical_cols_existing] = cat_imputer.fit_transform(X_train[categorical_cols_existing])
X_test[categorical_cols_existing]  = cat_imputer.transform(X_test[categorical_cols_existing])

# # Pour X_new:
# X_new[quantitative_cols] = quant_imputer.transform(X_new[quantitative_cols])
# X_new[categorical_cols]  = cat_imputer.transform(X_new[categorical_cols])

# Vérification
print("Valeurs manquantes dans X_train :", X_train.isna().sum().sum())
print("Valeurs manquantes dans X_test :", X_test.isna().sum().sum())

from sklearn.preprocessing import StandardScaler

# Standardisation des colonnes quantitatives
scaler = StandardScaler()
X_train[quantitative_cols_existing] = scaler.fit_transform(X_train[quantitative_cols_existing])
X_test[quantitative_cols_existing]  = scaler.transform(X_test[quantitative_cols_existing])
