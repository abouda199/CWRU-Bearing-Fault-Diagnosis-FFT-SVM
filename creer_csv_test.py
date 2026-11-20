
import numpy as np
import pandas as pd
from pathlib import Path

dossier = Path(__file__).resolve().parent

npz_path = (
    dossier
    / "CWRU_48k_load_1_CNN_data.npz"
    / "CWRU_48k_load_1_CNN_data.npz"
)

with np.load(npz_path) as dataset:
    X = dataset["data"]
    y = dataset["labels"]

# Selectionner un vrai signal Ball_007
indice = np.where(y == "Ball_007")[0][0]
signal = X[indice].reshape(-1)

# Exporter au format CSV
sortie = dossier / "test_Ball_007.csv"
pd.DataFrame({"vibration": signal}).to_csv(
    sortie, index=False
)

print("Fichier cree :", sortie)
print("Nombre de points :", len(signal))
print("Classe reelle : Ball_007")
