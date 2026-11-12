
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
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

X = X.reshape(X.shape[0], -1)

print("Dataset :", X.shape)

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

# ==========================================
# 3. PIPELINE PCA + SVM
# ==========================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(
        svd_solver="randomized",
        random_state=42
    )),
    ("svm", SVC(kernel="rbf"))
])

# ==========================================
# 4. PARAMÈTRES À TESTER
# ==========================================

parametres = {
    "pca__n_components": [50, 100, 200],
    "svm__C": [10, 100, 1000],
    "svm__gamma": [0.0001, 0.001, 0.01]
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

# ==========================================
# 5. ENTRAÎNEMENT
# ==========================================

print("\nOptimisation PCA + SVM en cours...")

grid.fit(X_train, y_train)

print("\n=== MEILLEURS PARAMÈTRES ===")
print(grid.best_params_)

print("\nMeilleur score CV :")
print(f"{grid.best_score_ * 100:.2f}%")

# ==========================================
# 6. ÉVALUATION
# ==========================================

y_pred = grid.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n=== RÉSULTATS PCA + SVM ===")
print(f"Accuracy test : {accuracy * 100:.2f}%")

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# ==========================================
# 7. MATRICE DE CONFUSION
# ==========================================

fig, ax = plt.subplots(figsize=(12, 10))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    cmap="Blues",
    xticks_rotation=45,
    ax=ax
)

plt.title("Matrice de confusion - PCA + SVM")

plt.tight_layout()
plt.savefig(
    dossier / "matrice_PCA_SVM.png",
    dpi=300
)
plt.show()
