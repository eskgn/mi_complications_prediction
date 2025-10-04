from data import *
import torch
import torch.nn as nn # Pour les modules de réseau de neurones
import torch.optim as optim # Pour les optimizers comme Adam
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm # Pour la barre de progression

seed = 42  # graine pour la reproductibilité
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # pour multi-GPU
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# 0. Gestion automatique du device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("⚡ Device utilisé :", device)

# Conversion des données sur CPU (on les enverra les batchs sur GPU si dispo)
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
y_test_t  = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)


# 1. Définition du MLP

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.layers(x)


# 2. DataLoader

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)


# 3. Modèle, critère et optimizer

model = MLP(X_train.shape[1]).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=6e-5, weight_decay=6e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.8, patience=4
)

# 4. Entraînement avec suivi du loss et F1

epochs = 18
train_losses, test_losses = [], []
train_f1s, test_f1s = [], []

for epoch in range(epochs):
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    running_loss = 0
    for xb, yb in loop:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
        loop.set_postfix(loss=loss.item())
    
    # Loss moyen sur l'epoch
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    
    # Évaluation sur test
    model.eval()
    with torch.no_grad():
        logits_test = model(X_test_t.to(device))
        test_loss = criterion(logits_test, y_test_t.to(device)).item()
        test_losses.append(test_loss)
        preds_train = (torch.sigmoid(model(X_train_t.to(device))) > 0.5).int().cpu().numpy()
        preds_test = (torch.sigmoid(logits_test) > 0.5).int().cpu().numpy()
        train_f1 = f1_score(y_train, preds_train, average='macro', zero_division=0)
        test_f1 = f1_score(y_test, preds_test, average='macro', zero_division=0)
        train_f1s.append(train_f1)
        test_f1s.append(test_f1)

    scheduler.step(test_f1)


# Visualisation Loss et F1

plt.figure(figsize=(12,5))

# Courbe des losses
plt.subplot(1,2,1)
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), test_losses, label='Test Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Évolution du Loss")
plt.legend()

# Courbe du F1-score
plt.subplot(1,2,2)
plt.plot(range(1, epochs+1), train_f1s, label='Train F1')
plt.plot(range(1, epochs+1), test_f1s, label='Test F1')
plt.xlabel("Epoch")
plt.ylabel("F1-score")
plt.title("Évolution du F1-score")
plt.legend()

plt.tight_layout()
plt.show()

# Matrice de confusion 

cm = confusion_matrix(y_test, preds_test)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Prédiction")
plt.ylabel("Vérité")
plt.title("Matrice de confusion")
plt.show()

X_new_t = torch.tensor(X_new, dtype=torch.float32).to(device)

# Mode évaluation
model.eval()
with torch.no_grad():
    # logits
    logits_new = model(X_new_t)
    # appliquer sigmoid puis threshold à 0.5 pour obtenir 0/1
    pred_new = (torch.sigmoid(logits_new) > 0.5).int().cpu().numpy()

pred_new = pred_new.flatten()

df_export = pd.DataFrame({
    "ID": ID_new,   # garde seulement l'identifiant
    "pred": pred_new    # ajoute les prédictions
})

# Export en csv
df_export.to_csv("predictions.csv", index=False)

print("Les prédictions ont été sauvegardées dans 'predictions.csv'")