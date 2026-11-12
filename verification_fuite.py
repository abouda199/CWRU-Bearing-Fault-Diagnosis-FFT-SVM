
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# =====================================
# 1. CHARGEMENT DU DATASET
# =====================================

dossier = Path(__file__).resolve().parent

fichier = (
    dossier
    / "CWRU_48k_load_1_CNN_data.npz"
    / "CWRU_48k_load_1_CNN_data.npz"
)

with np.load(fichier) as dataset:
    print("Variables disponibles :", dataset.files)

    X = dataset["data"]
    y = dataset["labels"]

X = X.reshape(X.shape[0], -1)

print("\nDimensions :", X.shape)

# =====================================
# 2. DIVISION TRAIN / TEST
# =====================================

indices = np.arange(len(y))

idx_train, idx_test = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train = X[idx_train]
X_test = X[idx_test]

y_train = y[idx_train]
y_test = y[idx_test]

print("Train :", len(X_train))
print("Test :", len(X_test))

# =====================================
# 3. RECHERCHE DES DOUBLONS EXACTS
# =====================================

print("\n=== DOUBLONS EXACTS ===")

train_signatures = {
    row.tobytes()
    for row in np.ascontiguousarray(X_train)
}

nb_doublons = sum(
    row.tobytes() in train_signatures
    for row in np.ascontiguousarray(X_test)
)

print("Doublons Train/Test :", nb_doublons)

print(
    "Pourcentage :",
    round(100 * nb_doublons / len(X_test), 2),
    "%"
)

# =====================================
# 4. RECHERCHE DE SIGNAUX SIMILAIRES
# =====================================

print("\n=== SIMILARITÉ TRAIN / TEST ===")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

voisins = NearestNeighbors(
    n_neighbors=1,
    metric="euclidean",
    n_jobs=-1
)

voisins.fit(X_train_scaled)

distances, positions = voisins.kneighbors(
    X_test_scaled
)

distances = distances.ravel()
positions = positions.ravel()

print("Distance minimale :", distances.min())
print("Distance moyenne  :", distances.mean())
print("Distance médiane  :", np.median(distances))
print("Distance maximale :", distances.max())

# =====================================
# 5. COMPARAISON DES ÉTIQUETTES
# =====================================

labels_proches = y_train[positions]

meme_classe = np.mean(
    labels_proches == y_test
)

print("\n=== PLUS PROCHE VOISIN ===")

print(
    "Même classe :",
    f"{meme_classe * 100:.2f}%"
)

# =====================================
# 6. EXEMPLES DES SIGNAUX PROCHES
# =====================================

print("\n=== 10 PAIRES LES PLUS PROCHES ===")

ordre = np.argsort(distances)[:10]

for i in ordre:
    print(
        f"Test {idx_test[i]} | "
        f"Train {idx_train[positions[i]]} | "
        f"Distance {distances[i]:.6f} | "
        f"Test={y_test[i]} | "
        f"Train={labels_proches[i]}"
    )

print("\nVérification terminée.")
