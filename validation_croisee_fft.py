
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Charger le dataset
dossier = Path(__file__).resolve().parent

fichier = (
    dossier
    / "CWRU_48k_load_1_CNN_data.npz"
    / "CWRU_48k_load_1_CNN_data.npz"
)

with np.load(fichier) as ds:
    X = ds["data"]
    y = ds["labels"]

# Transformation FFT
X = X.reshape(len(X), -1)
X = X - X.mean(axis=1, keepdims=True)

fenetre = np.hanning(X.shape[1])

X_fft = np.log1p(
    np.abs(np.fft.rfft(X * fenetre, axis=1))
)

# Validation croisee a 5 plis
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

configurations = [
    (0.1, 0.0001),
    (0.3, 0.0001),
    (0.5, 0.0001),
    (0.7, 0.0001),
    (1, 0.0001)
]

print("\nVALIDATION CROISEE FFT + SVM")
print("=" * 65)

for C, gamma in configurations:

    modele = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=C,
            gamma=gamma
        ))
    ])

    scores = cross_validate(
        modele,
        X_fft,
        y,
        cv=cv,
        scoring="accuracy",
        return_train_score=True,
        n_jobs=-1
    )

    train = scores["train_score"]
    test = scores["test_score"]

    print(f"\nC = {C} | gamma = {gamma}")
    print(f"Train moyen : {train.mean()*100:.2f}%")
    print(f"CV moyenne  : {test.mean()*100:.2f}%")
    print(f"CV ecart-type : {test.std()*100:.2f} points")
    print("Scores des 5 plis :", np.round(test*100, 2))

print("\nValidation terminee !")
