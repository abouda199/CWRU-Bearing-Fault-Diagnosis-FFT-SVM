
import numpy as np
import matplotlib.pyplot as plt
import joblib

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

# ============================================
# 1. CHARGEMENT DU DATASET CWRU
# ============================================

dossier = Path(__file__).resolve().parent

fichier = (
    dossier
    / "CWRU_48k_load_1_CNN_data.npz"
    / "CWRU_48k_load_1_CNN_data.npz"
)

with np.load(fichier) as dataset:
    X = dataset["data"]
    y = dataset["labels"]

print("\n====================================")
print("DATASET CWRU")
print("====================================")

print("Dimensions originales :", X.shape)
print("Nombre de classes :", len(np.unique(y)))

# Transformer 32x32 en 1024 valeurs
X = X.reshape(X.shape[0], -1)

print("Dimensions apres reshape :", X.shape)

# ============================================
# 2. EXTRACTION FFT
# ============================================

print("\nExtraction FFT en cours...")

# Centrage de chaque signal
X_centered = X - X.mean(
    axis=1,
    keepdims=True
)

# Fenetre de Hann
window = np.hanning(X.shape[1])

# Transformee de Fourier
X_fft = np.abs(
    np.fft.rfft(
        X_centered * window,
        axis=1
    )
)

# Compression logarithmique
X_fft = np.log1p(X_fft)

print("Dimensions FFT :", X_fft.shape)

# ============================================
# 3. DIVISION TRAIN / TEST
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X_fft,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n====================================")
print("DIVISION DES DONNEES")
print("====================================")

print("Train :", X_train.shape)
print("Test  :", X_test.shape)

# ============================================
# 4. CONSTRUCTION DU PIPELINE
# ============================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf"))
])

# ============================================
# 5. OPTIMISATION DES HYPERPARAMETRES
# ============================================

parametres = {
    "svm__C": [10, 100, 1000],
    "svm__gamma": [0.0001, 0.001, 0.01]
}

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=parametres,
    cv=cv,
    scoring="accuracy",
    n_jobs=2,
    verbose=2,
    return_train_score=True,
    refit=True
)

# ============================================
# 6. ENTRAINEMENT DU MODELE
# ============================================

print("\n====================================")
print("ENTRAINEMENT FFT + SVM")
print("====================================")

grid.fit(X_train, y_train)

# ============================================
# 7. RECUPERER LE MEILLEUR MODELE
# ============================================

best_model = grid.best_estimator_

print("\n====================================")
print("MEILLEURS PARAMETRES")
print("====================================")

print(grid.best_params_)

print(
    "Meilleur score CV :",
    f"{grid.best_score_ * 100:.2f}%"
)

# ============================================
# 8. SAUVEGARDER LE MODELE .PKL
# ============================================

chemin_modele = (
    dossier / "modele_FFT_SVM_CWRU.pkl"
)

joblib.dump(best_model, chemin_modele)

print("\n====================================")
print("SAUVEGARDE DU MODELE")
print("====================================")

print("Modele sauvegarde avec succes !")
print("Fichier :", chemin_modele)

# ============================================
# 9. PREDICTIONS TRAIN ET TEST
# ============================================

y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

train_accuracy = accuracy_score(
    y_train,
    y_train_pred
)

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

cv_accuracy = grid.best_score_

# ============================================
# 10. ANALYSE OVERFITTING
# ============================================

ecart = (train_accuracy - cv_accuracy) * 100

print("\n====================================")
print("ANALYSE OVERFITTING")
print("====================================")

print(
    f"Train Accuracy : {train_accuracy * 100:.2f}%"
)

print(
    f"CV Accuracy    : {cv_accuracy * 100:.2f}%"
)

print(
    f"Test Accuracy  : {test_accuracy * 100:.2f}%"
)

print(
    f"Ecart Train - CV : {ecart:.2f} points"
)

# ============================================
# 11. SCORES DES PLIS CV
# ============================================

idx = grid.best_index_

cv_train_scores = [
    grid.cv_results_[
        f"split{i}_train_score"
    ][idx]
    for i in range(cv.get_n_splits())
]

cv_val_scores = [
    grid.cv_results_[
        f"split{i}_test_score"
    ][idx]
    for i in range(cv.get_n_splits())
]

print("\nScores moyens sur les plis CV :")

print(
    f"Train CV : "
    f"{np.mean(cv_train_scores)*100:.2f}% "
    f"+/- {np.std(cv_train_scores)*100:.2f}"
)

print(
    f"Validation CV : "
    f"{np.mean(cv_val_scores)*100:.2f}% "
    f"+/- {np.std(cv_val_scores)*100:.2f}"
)

# ============================================
# 12. RAPPORT DE CLASSIFICATION
# ============================================

print("\n====================================")
print("RAPPORT DE CLASSIFICATION - TEST")
print("====================================")

print(
    classification_report(
        y_test,
        y_test_pred
    )
)

# ============================================
# 13. MATRICE DE CONFUSION
# ============================================

fig, ax = plt.subplots(figsize=(12, 10))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_test_pred,
    cmap="Blues",
    xticks_rotation=45,
    ax=ax
)

plt.title(
    "Matrice de confusion - FFT + SVM"
)

plt.tight_layout()

plt.savefig(
    dossier / "matrice_FFT_SVM_final.png",
    dpi=300
)

plt.close(fig)

# ============================================
# 14. GRAPHIQUE TRAIN / CV / TEST
# ============================================

scores = [
    train_accuracy * 100,
    cv_accuracy * 100,
    test_accuracy * 100
]

noms = [
    "Train",
    "Validation CV",
    "Test"
]

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(noms, scores)

ax.set_ylim(0, 105)
ax.set_ylabel("Accuracy (%)")
ax.set_title(
    "Comparaison Train / Validation / Test"
)

for bar, score in zip(bars, scores):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        score + 1,
        f"{score:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    dossier / "comparaison_train_cv_test_final.png",
    dpi=300
)

plt.close(fig)

# ============================================
# 15. VERIFIER LE FICHIER SAUVEGARDE
# ============================================

print("\n====================================")
print("VERIFICATION DU MODELE .PKL")
print("====================================")

modele_charge = joblib.load(chemin_modele)

predictions_chargees = modele_charge.predict(X_test)

accuracy_chargee = accuracy_score(
    y_test,
    predictions_chargees
)

print("Modele recharge avec succes !")

print(
    f"Accuracy apres rechargement : "
    f"{accuracy_chargee * 100:.2f}%"
)

print("\n====================================")
print("PROGRAMME TERMINE")
print("====================================")

print("Fichiers generes :")
print("1. modele_FFT_SVM_CWRU.pkl")
print("2. matrice_FFT_SVM_final.png")
print("3. comparaison_train_cv_test_final.png")
