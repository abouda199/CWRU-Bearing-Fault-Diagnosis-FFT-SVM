
from pathlib import Path
import numpy as np
from scipy.io import loadmat

dossier = Path(__file__).resolve().parent

npz_path = (
    dossier
    / "CWRU_48k_load_1_CNN_data.npz"
    / "CWRU_48k_load_1_CNN_data.npz"
)

mat_path = dossier / "raw" / "B007_1_123.mat"

with np.load(npz_path) as ds:
    X = ds["data"]
    y = ds["labels"]

segments = X[y == "Ball_007"].reshape(-1, 1024)

signal = loadmat(mat_path)["X123_DE_time"].ravel()

n = len(signal) // 1024

fenetres = signal[:n * 1024].reshape(n, 1024)

print("\nVERIFICATION DES FENETRES BALL_007")
print("=" * 55)

print("Segments NPZ :", len(segments))
print("Fenetres MAT :", len(fenetres))

# Comparaison de chaque segment NPZ avec les
# fenetres MAT non chevauchantes
correspondances = 0

for segment in segments:

    differences = np.max(
        np.abs(fenetres - segment),
        axis=1
    )

    if np.min(differences) < 1e-8:
        correspondances += 1

print("Correspondances exactes :", correspondances)
print(
    "Pourcentage :",
    round(correspondances / len(segments) * 100, 2),
    "%"
)

print("\nVerification terminee !")
