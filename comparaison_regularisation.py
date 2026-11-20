
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Charger le dataset
dossier = Path(__file__).resolve().parent

fichier = (
    dossier
    / "CWRU_48k_load_1_CNN_data.npz"
    / "CWRU_48k_load_1_CNN_data.npz"
)

with np.load(fichier) as ds:
    X = ds["data"]
    y = ds["labels"]

# 2. Transformation FFT
X = X.reshape(len(X), -1)
X = X - X.mean(axis=1, keepdims=True)

fenetre = np.hanning(X.shape[1])
X_fft = np.log1p(
    np.abs(np.fft.rfft(X * fenetre, axis=1))
)

# 3. Separation des donnees
X_train, X_test, y_train, y_test = train_test_split(
    X_fft,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Comparaison des regularisations
configurations = [
    (0.1, 0.0001),
    (1, 0.0001),
    (10, 0.0001),
    (1, 0.001),
    (10, 0.001),
    (100, 0.001),
]

print("\nCOMPARAISON DES MODELES FFT + SVM")
print("-" * 65)
print(f"{'C':<8} {'gamma':<12} {'Train':<12} {'Test':<12} {'Ecart'}")
print("-" * 65)

for C, gamma in configurations:

    modele = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=C,
            gamma=gamma
        ))
    ])

    modele.fit(X_train, y_train)

    train_acc = accuracy_score(
        y_train,
        modele.predict(X_train)
    )

    test_acc = accuracy_score(
        y_test,
        modele.predict(X_test)
    )

    ecart = train_acc - test_acc

    print(
        f"{C:<8} "
        f"{gamma:<12} "
        f"{train_acc * 100:<12.2f} "
        f"{test_acc * 100:<12.2f} "
        f"{ecart * 100:.2f}"
    )

print("\nComparaison terminee !")
