
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

# =====================================
# 1. CHARGER LE DATASET
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

X = X.reshape(X.shape[0], -1)

print("Dataset :", X.shape)

# =====================================
# 2. EXTRACTION FFT
# =====================================

# Supprimer la composante continue
X_centered = X - X.mean(axis=1, keepdims=True)

# Fenêtre de Hann pour limiter la fuite spectrale
window = np.hanning(X.shape[1])

# Calculer la FFT réelle
X_fft = np.abs(
    np.fft.rfft(X_centered * window, axis=1)
)

# Compression logarithmique
X_fft = np.log1p(X_fft)

print("Dimensions FFT :", X_fft.shape)

# =====================================
# 3. DIVISION TRAIN / TEST
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X_fft,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =====================================
# 4. PIPELINE SVM
# =====================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf"))
])

# =====================================
# 5. OPTIMISATION
# =====================================

parametres = {
    "svm__C": [10, 100, 1000],
    "svm__gamma": [
        0.0001, 0.001, 0.01
    ]
}

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

grid = GridSearchCV(
    pipeline,
    parametres,
    cv=cv,
    scoring="accuracy",
    n_jobs=2,
    verbose=2
)

print("\nOptimisation FFT + SVM...")

grid.fit(X_train, y_train)

# =====================================
# 6. MEILLEURS PARAMÈTRES
# =====================================

print("\n=== MEILLEURS PARAMÈTRES ===")
print(grid.best_params_)

print("\nMeilleur score CV :")
print(f"{grid.best_score_ * 100:.2f}%")

# =====================================
# 7. PRÉDICTIONS
# =====================================

y_pred = grid.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n=== RÉSULTATS FFT + SVM ===")
print(f"Accuracy test : {accuracy * 100:.2f}%")

print("\n=== RAPPORT ===")
print(classification_report(y_test, y_pred))

# =====================================
# 8. MATRICE DE CONFUSION
# =====================================

fig, ax = plt.subplots(figsize=(12, 10))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    cmap="Blues",
    xticks_rotation=45,
    ax=ax
)

plt.title("Matrice de confusion - FFT + SVM")
plt.tight_layout()

plt.savefig(
    dossier / "matrice_FFT_SVM.png",
    dpi=300
)

plt.show()
