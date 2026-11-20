"""Vérifications fonctionnelles CWRU sur dix segments connus, sans entraînement.

Exécution : python tester_10_classes.py
Les CSV exportés utilisent une colonne « amplitude » avec un en-tête.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from utils.prediction import CLASSES, load_dataset, load_model
from utils.preprocessing import (WINDOW_SIZE, fft_features, numeric_signal,
                                 read_csv_signal, segment_signal)

OUTPUT_DIR = Path(__file__).resolve().parent / "tests_csv"


def main() -> int:
    """Retourne 0 si tout réussit, 1 pour un échec par classe, 2 en cas d'erreur globale."""
    print("BearingAI — vérification fonctionnelle des 10 classes")
    print("Données connues : ces tests ne constituent pas une validation indépendante.")
    print("Sélection déterministe : premier segment de chaque classe, indices NPZ à partir de 0.")
    print("Reshape en ordre C identique à l'application ; ordre temporel du NPZ non confirmé.\n")
    try:
        data, labels = load_dataset()
        model = load_model()  # Artefact local fiable uniquement ; aucun fit().
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"ERREUR de chargement ou de préparation : {exc}", file=sys.stderr)
        return 2

    rows = []
    for true_class in CLASSES:
        row = {"Classe réelle": true_class, "Indice NPZ": None,
               "Classe prédite": "—", "Points": None, "Features FFT": None,
               "CSV": f"test_{true_class}.csv", "Résultat": "ERREUR", "Détail": ""}
        try:
            indices = np.flatnonzero(labels == true_class)
            if not indices.size:
                raise ValueError(f"Classe absente du NPZ : {true_class}")
            index = int(indices[0])
            row["Indice NPZ"] = index
            signal = data[index].reshape(-1, order="C")
            row["Points"] = int(signal.size)
            if signal.shape != (WINDOW_SIZE,):
                raise ValueError(f"Dimension invalide : {signal.shape}, attendu (1024,).")
            windows, remainder = segment_signal(signal)
            if windows.shape != (1, WINDOW_SIZE) or remainder:
                raise ValueError("Segmentation incompatible : une fenêtre de 1024 points attendue.")

            # Import direct de la fonction utilisée par Streamlit, sans copie de formule.
            features = fft_features(windows)
            row["Features FFT"] = int(features.shape[1])
            if features.shape != (1, model.n_features_in_) or not np.isfinite(features).all():
                raise ValueError(f"Prétraitement incompatible : FFT {features.shape}, "
                                 f"modèle attendu (1, {model.n_features_in_}).")
            predicted = str(model.predict(features)[0])
            row["Classe prédite"] = predicted

            csv_path = OUTPUT_DIR / row["CSV"]
            np.savetxt(csv_path, signal, delimiter=",", header="amplitude",
                       comments="", fmt="%.17g", encoding="utf-8")

            # Vérifier aussi le parcours CSV réel de l'application, y compris l'en-tête.
            frame = read_csv_signal(csv_path.read_bytes(), header=True, separator=",")
            restored = numeric_signal(frame, "amplitude")
            if restored.shape != signal.shape or not np.allclose(
                    restored, signal, rtol=1e-12, atol=1e-12):
                raise ValueError("Le CSV relu ne restitue pas les 1024 valeurs originales.")
            restored_windows, _ = segment_signal(restored)
            restored_features = fft_features(restored_windows)
            if not np.allclose(restored_features, features, rtol=1e-12, atol=1e-12):
                raise ValueError("Écart de prétraitement FFT après relecture du CSV.")
            if str(model.predict(restored_features)[0]) != predicted:
                raise ValueError("La prédiction du CSV diffère de celle du segment NPZ.")
            row["Résultat"] = "OK" if predicted == true_class else "ÉCHEC"
            row["Détail"] = "CSV relu et FFT vérifiés" if predicted == true_class else "Classe différente"
        except Exception as exc:
            row["Détail"] = f"{type(exc).__name__} : {exc}"
        rows.append(row)

    report = pd.DataFrame(rows)
    print(report.to_string(index=False))
    passed = int((report["Résultat"] == "OK").sum())
    print(f"\nVérifications réussies : {passed}/{len(CLASSES)} (données connues uniquement).")
    print(f"Dossier des segments CSV : {OUTPUT_DIR}")
    print("Streamlit : sélectionner CSV avec en-tête, séparateur virgule, colonne amplitude.")
    print("Aucune fréquence d'échantillonnage n'est supposée.")
    return 0 if passed == len(CLASSES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
