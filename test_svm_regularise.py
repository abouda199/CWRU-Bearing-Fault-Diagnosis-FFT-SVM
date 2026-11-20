
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Chargement
dossier = Path(__file__).resolve().parent
fichier = (
    dossier
    / "CWRU_48k_load_1_CNN_data.npz"
    / "CWRU_48k_load_1_CNN_data.npz"
)

with np.load(fichier) as ds:
    X = ds["data"]
    y = ds["labels"]

# FFT
X = X.reshape(len(X), -1)
X = X - X.mean(axis=1, keepdims=True)
X_fft = np.log1p(
    np.abs(np.fft.rfft(X * np.hanning(X.shape[1]), axis=1))
)

# Separation 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X_fft, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Nouveau modele regularise
modele = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1, gamma=0.0001))
])

modele.fit(X_train, y_train)

# Evaluation
train_acc = accuracy_score(y_train, modele.predict(X_train))
test_pred = modele.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)

print("\nRESULTATS SVM REGULARISE")
print("=" * 40)
print(f"Train accuracy : {train_acc*100:.2f}%")
print(f"Test accuracy  : {test_acc*100:.2f}%")
print(f"Ecart          : {(train_acc-test_acc)*100:.2f} points")
print("\n", classification_report(y_test, test_pred))
