
from pathlib import Path
from scipy.io import loadmat
import numpy as np

dossier = Path(__file__).resolve().parent / "raw"

fichiers = sorted(dossier.glob("*.mat"))

print("\nINSPECTION DES FICHIERS CWRU")
print("=" * 65)

for fichier in fichiers:
    print(f"\nFichier : {fichier.name}")

    donnees = loadmat(fichier)

    for cle, valeur in donnees.items():
        if cle.startswith("__"):
            continue

        if isinstance(valeur, np.ndarray):
            print(
                f"  {cle} | "
                f"Shape : {valeur.shape} | "
                f"Type : {valeur.dtype}"
            )

print("\nInspection terminee !")
