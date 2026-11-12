
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Chemin du fichier Excel
dossier = Path(__file__).resolve().parent
fichier = dossier / "CWRU_dataset.xlsx"

# Charger uniquement la colonne label
df = pd.read_excel(fichier, usecols=["label"])

# Compter les échantillons
classes = df["label"].value_counts().sort_index()

print("=== RÉPARTITION DES CLASSES ===")
print(classes)

# Créer le graphique
plt.figure(figsize=(12, 6))

classes.plot(kind="bar", color="steelblue")

plt.title("Répartition des classes - Dataset CWRU")
plt.xlabel("Types de défauts")
plt.ylabel("Nombre d'échantillons")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
