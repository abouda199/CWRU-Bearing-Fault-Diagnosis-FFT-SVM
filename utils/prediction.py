"""Trusted local model loading and reproducible evaluation; never fits a model."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils.preprocessing import fft_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "modele_FFT_SVM_CWRU.pkl"
DATASET_PATH = ROOT / "CWRU_48k_load_1_CNN_data.npz" / "CWRU_48k_load_1_CNN_data.npz"
CLASSES = ("Ball_007", "Ball_014", "Ball_021", "IR_007", "IR_014", "IR_021",
           "Normal", "OR_007", "OR_014", "OR_021")
LIMITATION = ("Le test aléatoire par segments peut placer des fenêtres d'un même "
              "enregistrement dans le train et le test. Ce score ne valide pas la "
              "généralisation à des enregistrements indépendants.")
TEMPORAL_WARNING = ("L'ordre temporel des 1 024 valeurs du NPZ n'est pas confirmé. "
                    "Le reshape en ordre C reproduit le code existant, mais la compatibilité "
                    "avec un signal externe chronologique reste une hypothèse à valider. "
                    "Les unités, le capteur et la fréquence doivent aussi correspondre à l'entraînement.")


def load_model():
    """Only load the project's trusted artifact. No uploaded pickle is accepted."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Modèle absent : {MODEL_PATH.name}. Restaurer votre artefact local fiable.")
    model = joblib.load(MODEL_PATH)
    if (not isinstance(model, Pipeline) or len(model.steps) != 2
            or not isinstance(model.steps[0][1], StandardScaler)
            or not isinstance(model.steps[1][1], SVC)
            or model.n_features_in_ != 513
            or set(model.classes_) != set(CLASSES)):
        raise ValueError("Artefact incompatible : StandardScaler + SVC, 513 features et 10 classes attendus.")
    return model


def load_dataset():
    with np.load(DATASET_PATH, allow_pickle=False) as dataset:
        x, y = dataset["data"], dataset["labels"]
    if x.shape != (4600, 32, 32) or y.shape != (4600,):
        raise ValueError("Dimensions du dataset incompatibles avec le protocole documenté.")
    names, counts = np.unique(y, return_counts=True)
    if set(names) != set(CLASSES) or not np.all(counts == 460) or not np.isfinite(x).all():
        raise ValueError("Classes, équilibre ou valeurs du dataset invalides.")
    return x, y


def describe_class(label):
    if label not in CLASSES:
        raise ValueError(f"Classe inconnue : {label}")
    if label == "Normal":
        return {"Classe": label, "Catégorie": "Normal", "Diamètre nominal (in)": None}
    kind, size = label.split("_")
    return {"Classe": label,
            "Catégorie": {"Ball": "Ball", "IR": "Inner Race", "OR": "Outer Race"}[kind],
            "Diamètre nominal (in)": int(size) / 1000}


def predict_windows(model, windows):
    labels = model.predict(fft_features(windows))
    return pd.DataFrame([{"Fenêtre": i + 1, "Début (point)": i * 1024,
                          "Fin exclusive (point)": (i + 1) * 1024,
                          **describe_class(str(label))} for i, label in enumerate(labels)])


def evaluate_model(model):
    """Reconstruct tester_modele.py's original stratified 20% test, seed 42."""
    x, y = load_dataset()
    features = fft_features(x)
    _, test, _, truth = train_test_split(features, y, test_size=0.20,
                                        random_state=42, stratify=y)
    predicted = model.predict(test)
    report = classification_report(truth, predicted, labels=CLASSES,
                                   output_dict=True, zero_division=0)
    # Accuracy is a scalar, shown separately; do not broadcast it into support.
    report.pop("accuracy", None)
    return {"accuracy": accuracy_score(truth, predicted), "n_test": len(truth),
            "matrix": confusion_matrix(truth, predicted, labels=CLASSES),
            "report": pd.DataFrame(report).T}
