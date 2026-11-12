
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. CHARGEMENT DU DATASET
# ==========================================

dossier = Path(__file__).resolve().parent

fichier = (
    dossier
    / "CWRU_48k_load_1_CNN_data.npz"
    / "CWRU_48k_load_1_CNN_data.npz"
)

with np.load(fichier) as dataset:
    X = dataset["data"]
    y = dataset["labels"]

# Transformer 32x32 en 1024 caractéristiques
X = X.reshape(X.shape[0], -1)

print("=== DATASET CWRU ===")
print("Dimensions X :", X.shape)
print("Dimensions y :", y.shape)

# ==========================================
# 2. DIVISION TRAIN / TEST
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n=== DIVISION ===")
print("Train :", X_train.shape)
print("Test :", X_test.shape)

# ==========================================
# 3. NORMALISATION
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 4. ENTRAÎNEMENT SVM
# ==========================================

print("\n=== ENTRAÎNEMENT SVM ===")

svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    random_state=42
)

svm.fit(X_train_scaled, y_train)

print("Entraînement terminé !")

# ==========================================
# 5. PRÉDICTION
# ==========================================

y_pred = svm.predict(X_test_scaled)

# ==========================================
# 6. ÉVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n=== RÉSULTATS SVM ===")
print(f"Accuracy : {accuracy * 100:.2f}%")

print("\n=== RAPPORT DE CLASSIFICATION ===")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

print("\n=== MATRICE DE CONFUSION ===")
print(confusion_matrix(y_test, y_pred))
