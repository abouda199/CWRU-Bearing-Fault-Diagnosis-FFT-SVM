
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================
# 1. CHEMINS DES FICHIERS
# ============================================

dossier = Path(__file__).resolve().parent

fichier_dataset = (
    dossier
    / "CWRU_48k_load_1_CNN_data.npz"
    / "CWRU_48k_load_1_CNN_data.npz"
)

fichier_modele = dossier / "modele_FFT_SVM_CWRU.pkl"

# ============================================
# 2. CHARGEMENT DU MODELE ENTRAINE
# ============================================

print("Chargement du modele...")

modele = joblib.load(fichier_modele)

print("Modele .pkl charge avec succes !")

# ============================================
# 3. CHARGEMENT DU DATASET
# ============================================

with np.load(fichier_dataset) as dataset:
    X = dataset["data"]
    y = dataset["labels"]

print("\nDimensions originales :", X.shape)

X = X.reshape(X.shape[0], -1)

# ============================================
# 4. TRANSFORMATION FFT
# ============================================

X_centered = X - X.mean(
    axis=1,
    keepdims=True
)

window = np.hanning(X.shape[1])

X_fft = np.abs(
    np.fft.rfft(
        X_centered * window,
        axis=1
    )
)

X_fft = np.log1p(X_fft)

print("Dimensions FFT :", X_fft.shape)

# ============================================
# 5. RECONSTRUIRE LE TEST ORIGINAL
# ============================================

_, X_test, _, y_test = train_test_split(
    X_fft,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Nombre de donnees test :", len(y_test))

# ============================================
# 6. PREDICTIONS SANS ENTRAINEMENT
# ============================================

print("\nPrediction en cours...")

y_pred = modele.predict(X_test)

# ============================================
# 7. EVALUATION
# ============================================

accuracy = accuracy_score(y_test, y_pred)

print("\n====================================")
print("RESULTATS DU MODELE .PKL")
print("====================================")

print(f"Accuracy : {accuracy * 100:.2f}%")

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred))

# ============================================
# 8. EXEMPLES DE PREDICTIONS
# ============================================

print("\n====================================")
print("10 EXEMPLES DE PREDICTIONS")
print("====================================")

for i in range(10):
    print(
        f"Echantillon {i + 1} | "
        f"Reel : {y_test[i]} | "
        f"Predit : {y_pred[i]}"
    )

print("\nTest termine sans reentrainement !")
