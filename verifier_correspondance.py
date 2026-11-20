
from pathlib import Path
import numpy as np
from scipy.io import loadmat

dossier = Path(__file__).resolve().parent

fichier_npz = (
    dossier
    / "CWRU_48k_load_1_CNN_data.npz"
    / "CWRU_48k_load_1_CNN_data.npz"
)

dossier_raw = dossier / "raw"

classes = {
    "Ball_007": ("B007_1_123.mat", "X123_DE_time"),
    "Ball_014": ("B014_1_190.mat", "X190_DE_time"),
    "Ball_021": ("B021_1_227.mat", "X227_DE_time"),
    "IR_007": ("IR007_1_110.mat", "X110_DE_time"),
    "IR_014": ("IR014_1_175.mat", "X175_DE_time"),
    "IR_021": ("IR021_1_214.mat", "X214_DE_time"),
    "Normal": ("Time_Normal_1_098.mat", "X098_DE_time"),
    "OR_007": ("OR007_6_1_136.mat", "X136_DE_time"),
    "OR_014": ("OR014_6_1_202.mat", "X202_DE_time"),
    "OR_021": ("OR021_6_1_239.mat", "X239_DE_time"),
}

with np.load(fichier_npz) as ds:
    X = ds["data"]
    y = ds["labels"]

X = X.reshape(len(X), -1)

print("\nVERIFICATION DE CORRESPONDANCE NPZ / MAT")
print("=" * 70)

for classe, (nom_fichier, cle) in classes.items():

    signal = loadmat(dossier_raw / nom_fichier)[cle].ravel()
    segments = X[y == classe]

    # Chercher si les segments NPZ ressemblent
    # a des fenetres temporelles du fichier MAT.
    # Test indicatif : correlation avec les
    # premieres fenetres non chevauchantes.
    n = min(100, len(signal) // 1024)
    fenetres = signal[:n * 1024].reshape(n, 1024)

    # Normalisation pour comparer les formes
    def normaliser(a):
        a = a - a.mean(axis=1, keepdims=True)
        norme = np.linalg.norm(a, axis=1, keepdims=True)
        return a / np.maximum(norme, 1e-12)

    a = normaliser(segments[:20])
    b = normaliser(fenetres)

    correlations = np.abs(a @ b.T)
    meilleur = correlations.max()

    print(f"{classe:<12} | correlation maximale : {meilleur:.4f}")

print("\nVerification terminee !")
