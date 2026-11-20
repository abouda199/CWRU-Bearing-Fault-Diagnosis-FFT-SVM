
from pathlib import Path
import numpy as np
from scipy.io import loadmat

dossier = Path(__file__).resolve().parent

fichier_mat = dossier / "raw" / "IR014_1_175.mat"

fichier_npz = (
    dossier
    / "CWRU_48k_load_1_CNN_data.npz"
    / "CWRU_48k_load_1_CNN_data.npz"
)

# Charger le dataset NPZ
with np.load(fichier_npz) as ds:
    X = ds["data"]
    y = ds["labels"]

X = X.reshape(len(X), -1)
segments = X[y == "IR_014"]

# Charger les deux signaux
mat = loadmat(fichier_mat)

signaux = [
    "X175_DE_time",
    "X217_DE_time"
]

def normaliser(a):
    a = a - a.mean(axis=1, keepdims=True)
    norme = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.maximum(norme, 1e-12)

print("\nVERIFICATION IR_014")
print("=" * 50)

for nom in signaux:

    signal = mat[nom].ravel()

    n = min(100, len(signal) // 1024)

    fenetres = signal[:n * 1024].reshape(n, 1024)

    a = normaliser(segments[:20])
    b = normaliser(fenetres)

    correlation = np.abs(a @ b.T).max()

    print(f"{nom} : correlation maximale = {correlation:.6f}")

print("\nVerification terminee !")
