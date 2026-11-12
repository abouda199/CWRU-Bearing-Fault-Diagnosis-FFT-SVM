
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =====================================
# 1. CHARGEMENT DU DATASET
# =====================================

dossier = Path(__file__).resolve().parent

fichier = (
    dossier
    / "CWRU_48k_load_1_CNN_data.npz"
    / "CWRU_48k_load_1_CNN_data.npz"
)

with np.load(fichier) as dataset:
    X = dataset["data"]
    y = dataset["labels"]

print("=== DATASET ORIGINAL ===")
print("X :", X.shape)
print("y :", y.shape)

# =====================================
# 2. TRANSFORMATION EN 2D
# =====================================

X = X.reshape(X.shape[0], -1)

print("\n=== APRÈS TRANSFORMATION ===")
print("X :", X.shape)

# =====================================
# 3. DIVISION TRAIN / TEST
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n=== DIVISION DES DONNÉES ===")
print("X_train :", X_train.shape)
print("X_test :", X_test.shape)
print("y_train :", y_train.shape)
print("y_test :", y_test.shape)

# =====================================
# 4. NORMALISATION
# =====================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n=== NORMALISATION ===")
print("Normalisation terminée !")

# =====================================
# 5. VÉRIFICATION DES CLASSES
# =====================================

print("\n=== CLASSES TRAIN ===")
classes, counts = np.unique(y_train, return_counts=True)

for classe, count in zip(classes, counts):
    print(classe, ":", count)

print("\n=== CLASSES TEST ===")
classes, counts = np.unique(y_test, return_counts=True)

for classe, count in zip(classes, counts):
    print(classe, ":", count)
