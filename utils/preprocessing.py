"""Signal validation and the FFT transform used by svm_fft_overfitting.py."""
from io import BytesIO

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

WINDOW_SIZE = 1024


def read_csv_signal(content: bytes, header: bool, separator: str) -> pd.DataFrame:
    """Read an explicitly configured CSV without discarding its first sample."""
    if not content:
        raise ValueError("Le fichier CSV est vide.")
    try:
        frame = pd.read_csv(BytesIO(content), sep=separator,
                            header=0 if header else None, encoding="utf-8-sig",
                            skip_blank_lines=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
        raise ValueError("CSV invalide : vérifier UTF-8, l'en-tête et le séparateur.") from exc
    if frame.empty:
        raise ValueError("Le CSV ne contient aucune observation.")
    frame.columns = [str(c) for c in frame.columns]
    return frame


def numeric_signal(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Reject missing/non-numeric values; never delete or interpolate samples."""
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        invalid = int((~np.isfinite(values)).sum())
        raise ValueError(f"{invalid} valeur(s) manquante(s), non numérique(s) ou infinie(s). "
                         "Corriger le CSV : aucun point ne sera supprimé ou interpolé.")
    if values.size < 2:
        raise ValueError("Le signal doit contenir au moins deux valeurs.")
    return values


def segment_signal(signal: np.ndarray, discard_tail: bool = False):
    """Return complete, non-overlapping windows and the explicitly discarded tail."""
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or not np.isfinite(signal).all():
        raise ValueError("Un signal unidimensionnel fini est requis.")
    count, remainder = divmod(signal.size, WINDOW_SIZE)
    if not count:
        raise ValueError("La prédiction exige au moins 1 024 points. Aucun padding n'est appliqué.")
    if remainder and not discard_tail:
        raise ValueError(f"Il reste {remainder} points : autoriser explicitement leur exclusion.")
    return signal[:count * WINDOW_SIZE].reshape(count, WINDOW_SIZE), remainder


def fft_features(samples: np.ndarray) -> np.ndarray:
    """Exactly reproduce training, retaining input precision and C-order flattening."""
    samples = np.asarray(samples)
    if samples.ndim not in (2, 3) or not len(samples):
        raise ValueError("Entrée attendue : (n, 1024) ou (n, 32, 32).")
    x = samples.reshape(len(samples), -1)
    if x.shape[1] != WINDOW_SIZE or not np.isfinite(x).all():
        raise ValueError("Chaque fenêtre doit contenir exactement 1 024 valeurs finies.")
    centered = x - x.mean(axis=1, keepdims=True)
    return np.log1p(np.abs(np.fft.rfft(centered * np.hanning(x.shape[1]), axis=1)))


def signal_statistics(signal):
    return {"Moyenne": float(np.mean(signal)), "Écart-type": float(np.std(signal)),
            "RMS": float(np.sqrt(np.mean(np.square(signal)))),
            "Maximum": float(np.max(signal)), "Minimum": float(np.min(signal)),
            "Kurtosis (Pearson)": float(kurtosis(signal, fisher=False))
            if np.ptp(signal) > 0 else float("nan")}
