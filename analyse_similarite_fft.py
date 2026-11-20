
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

dossier = Path(__file__).resolve().parent

fichier = (
    dossier
    / "CWRU_48k_load_1_CNN_data.npz"
    / "CWRU_48k_load_1_CNN_data.npz"
)

with np.load(fichier) as ds:
    X = ds["data"]
    y = ds["labels"]

# Transformation FFT identique au modele
X = X.reshape(len(X), -1)
X = X - X.mean(axis=1, keepdims=True)

X_fft = np.log1p(
    np.abs(
        np.fft.rfft(
            X * np.hanning(X.shape[1]),
            axis=1
        )
    )
)

# Meme separation 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X_fft,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Standardisation sans fuite du test
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Recherche du plus proche voisin d'entrainement
voisins = NearestNeighbors(
    n_neighbors=1,
    metric="euclidean"
)

voisins.fit(X_train)

distances, indices = voisins.kneighbors(X_test)

distances = distances.ravel()
indices = indices.ravel()

memes_classes = y_test == y_train[indices]

print("\nANALYSE DE SIMILARITE FFT")
print("=" * 45)

print(f"Distance minimale : {distances.min():.6f}")
print(f"Distance moyenne  : {distances.mean():.6f}")
print(f"Distance mediane  : {np.median(distances):.6f}")
print(f"Distance maximale : {distances.max():.6f}")

print(
    f"\nVoisins de meme classe : "
    f"{memes_classes.mean()*100:.2f}%"
)

print("\n10 premiers exemples :")

for i in range(10):
    print(
        f"Test {i+1} | "
        f"Distance : {distances[i]:.4f} | "
        f"Classe test : {y_test[i]} | "
        f"Voisin : {y_train[indices[i]]}"
    )

print("\nAnalyse terminee !")
