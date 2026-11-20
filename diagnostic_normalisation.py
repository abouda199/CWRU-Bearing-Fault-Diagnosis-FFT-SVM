
from pathlib import Path
import numpy as np
from scipy.io import loadmat

dossier = Path(__file__).resolve().parent

fichier_npz = (
    dossier
    / "CWRU_48k_load_1_CNN_data.npz"
    / "CWRU_48k_load_1_CNN_data.npz"
)

fichier_mat = dossier / "raw" / "B007_1_123.mat"

with np.load(fichier_npz) as ds:
    X = ds["data"]
    y = ds["labels"]

segments = X[y == "Ball_007"].reshape(-1, 1024)

signal = loadmat(fichier_mat)["X123_DE_time"].ravel()

n = len(signal) // 1024
fenetres = signal[:n * 1024].reshape(n, 1024)

def normaliser(a):
    a = a - a.mean(axis=1, keepdims=True)
    norme = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.maximum(norme, 1e-12)

a = normaliser(segments)
b = normaliser(fenetres)

# Trouver les fenetres les plus correlees
correlations = a @ b.T

indices = np.argmax(np.abs(correlations), axis=1)
meilleures = correlations[
    np.arange(len(segments)), indices
]

print("\nDIAGNOSTIC NORMALISATION BALL_007")
print("=" * 55)

print("Correlation absolue maximale :",
      np.max(np.abs(meilleures)))

print("Correlation absolue moyenne :",
      np.mean(np.abs(meilleures)))

print("Correspondances correlation > 0.9999 :",
      np.sum(np.abs(meilleures) > 0.9999))

print("\nExemple du meilleur couple :")

i = np.argmax(np.abs(meilleures))
j = indices[i]

x = segments[i]
s = fenetres[j]

# Ajustement affine : x ~= a*s + b
coefficient, decalage = np.polyfit(s, x, 1)

erreur = np.max(
    np.abs(x - (coefficient * s + decalage))
)

print("Indice segment NPZ :", i)
print("Indice fenetre MAT :", j)
print("Coefficient echelle :", coefficient)
print("Decalage :", decalage)
print("Erreur maximale apres ajustement :", erreur)

print("\nDiagnostic termine !")
