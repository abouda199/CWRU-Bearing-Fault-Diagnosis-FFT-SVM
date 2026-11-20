"""BearingAI — local Streamlit application. Run: python -m streamlit run app.py."""
import hashlib

import numpy as np
import pandas as pd
import streamlit as st

from utils.prediction import (CLASSES, LIMITATION, TEMPORAL_WARNING, evaluate_model,
                              load_dataset, load_model, predict_windows)
from utils.preprocessing import (numeric_signal, read_csv_signal, segment_signal,
                                 signal_statistics)
from utils.visualization import (confusion_plot, distribution_plot, spectrum_plot,
                                  temporal_plot)

st.set_page_config(page_title="BearingAI | Intelligent Bearing Fault Diagnosis",
                   page_icon="⚙", layout="wide")
st.markdown("""<style>
.stApp {background: radial-gradient(ellipse at top right, #12304a 0%, #081221 55%);}
.block-container {max-width:1400px; padding-top:2rem; padding-bottom:3rem;}
h1,h2,h3 {letter-spacing:-.035em;}
[data-testid="stMetric"] {background:#102238; border:1px solid #20415b;
    border-radius:14px; padding:18px; min-height:110px;}
[data-testid="stSidebar"] {border-right:1px solid #20415b;}
.eyebrow {color:#22d3ee; letter-spacing:.18em; font-size:.75rem; font-weight:700;}
.hero {border:1px solid #20415b; border-radius:18px; padding:26px;
    background:linear-gradient(115deg,#102b43,#102033); margin-bottom:24px;}
.hero h1 {font-size:2.7rem; margin:8px 0;}
.hero p {color:#afc5da; max-width:820px; line-height:1.7;}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def cached_model():
    return load_model()


@st.cache_data
def cached_dataset():
    return load_dataset()


@st.cache_data
def cached_evaluation():
    return evaluate_model(cached_model())


def get_model():
    try:
        return cached_model()
    except Exception as exc:
        st.error(f"Impossible de charger le modèle local : {exc}")
        st.info("Restaurer l'artefact fiable du projet et les versions de requirements.txt. "
                "Aucun entraînement automatique n'est lancé.")
        st.stop()


def signal_source():
    """Share input controls between analysis and prediction pages."""
    source = st.radio("Source du signal", ["Importer un CSV", "Exemple NPZ local"], horizontal=True,
                      key="source")
    if source == "Exemple NPZ local":
        st.caption("Donnée réelle du projet, à usage de démonstration ; ce n'est pas un test indépendant.")
        try:
            x, y = cached_dataset()
        except Exception as exc:
            st.error(f"Dataset local indisponible : {exc}")
            return None, None
        label = st.selectbox("Classe réelle", CLASSES, key="example_class")
        indices = np.flatnonzero(y == label)
        selected = st.number_input("Segment de la classe", 1, len(indices), 1, key="example_index")
        signal = x[indices[int(selected) - 1]].reshape(-1)
        st.caption(f"Classe réelle : {label} · 1 024 points · reshape C identique au code de recherche.")
    else:
        st.caption("CSV UTF-8, une ligne par point. Choisir une seule colonne d'amplitude ; "
                   "les colonnes de temps ne sont pas utilisées pour inférer la fréquence.")
        upload = st.file_uploader("Signal vibratoire (.csv)", type=["csv"], key="csv")
        left, right = st.columns(2)
        header = left.checkbox("Le CSV contient un en-tête", value=True, key="csv_header")
        separator = right.selectbox("Séparateur", [",", ";", "Tabulation"], key="separator")
        if upload is None:
            st.info("Importer un signal ou sélectionner un exemple du dataset local.")
            return None, None
        try:
            frame = read_csv_signal(upload.getvalue(), header,
                                    "\t" if separator == "Tabulation" else separator)
            column = st.selectbox("Colonne d'amplitude", frame.columns, key="amplitude_column")
            signal = numeric_signal(frame, column)
        except ValueError as exc:
            st.error(str(exc))
            return None, None
        with st.expander("Aperçu du CSV"):
            st.dataframe(frame.head(10), hide_index=True, width="stretch")
    known = st.checkbox("Je connais la fréquence d'échantillonnage", key="known_rate")
    rate = st.number_input("Fréquence d'échantillonnage (Hz)", min_value=0.001,
                           value=1.0, format="%.3f", key="rate") if known else None
    if not known:
        st.caption("Fréquence inconnue : axes en points et cycles/point. Aucune fréquence n'est supposée.")
    return signal, rate


def dashboard():
    st.markdown('''<div class="hero"><div class="eyebrow">VIBRATION INTELLIGENCE / CWRU</div>
    <h1>Un signal. Un diagnostic.</h1><p>BearingAI — Intelligent Bearing Fault Diagnosis.
    Explorez les vibrations de roulements, observez leur signature fréquentielle et
    appliquez votre modèle FFT + SVM dans une interface de recherche interactive.</p></div>''',
                unsafe_allow_html=True)
    cols = st.columns(4)
    for column, label, value in zip(cols, ["Classes de défauts", "Segments du dataset", "Modèle", "Test aléatoire¹"],
                                    ["10", "4 600", "FFT + SVM", "100 %"]):
        column.metric(label, value)
    st.caption("¹ Résultat historique documenté dans README.md ; recalcul disponible dans Model Performance.")
    st.warning(LIMITATION)
    st.subheader("Du signal à la classe")
    st.markdown("**1 024 points** → centrage → fenêtre de Hann → FFT réelle → log1p → "
                "StandardScaler → SVM RBF → **classe prédite**")
    st.divider()
    st.subheader("Comprendre les familles de défauts")
    cards = st.columns(4)
    for card, title, description in zip(cards, ["Normal", "Ball", "Inner Race", "Outer Race"],
            ["Roulement sans défaut introduit dans le dataset.",
             "Défaut sur un élément roulant (bille).", "Défaut sur la bague intérieure (IR).",
             "Défaut sur la bague extérieure (OR)."]):
        with card.container(border=True):
            st.markdown(f"### {title}")
            st.write(description)
    st.caption("Codes 007, 014 et 021 : diamètres nominaux de 0,007, 0,014 et 0,021 pouce. "
               "Le modèle attribue une classe ; il ne mesure pas physiquement le défaut.")
    st.info("Utilisez Signal Analysis pour explorer un signal, puis AI Prediction pour classifier ses fenêtres.")


def analysis():
    st.title("Signal Analysis")
    st.write("Inspectez le signal brut et sa représentation fréquentielle.")
    signal, rate = signal_source()
    if signal is None:
        return
    st.subheader(f"{len(signal):,} points analysés".replace(",", " "))
    stats = signal_statistics(signal)
    for row in (list(stats.items())[:3], list(stats.items())[3:]):
        for col, (name, value) in zip(st.columns(3), row):
            col.metric(name, f"{value:.5g}" if np.isfinite(value) else "Indéfinie")
    st.caption("Écart-type population (ddof=0). Kurtosis de Pearson : 3 pour une loi normale ; "
               "indéfinie pour un signal constant. Amplitudes dans les unités du CSV.")
    st.plotly_chart(temporal_plot(signal, rate), width="stretch")
    if len(signal) > 20000:
        st.caption("Affichage temporel sous-échantillonné à 20 000 points maximum. "
                   "Les calculs utilisent tous les points.")
    st.plotly_chart(spectrum_plot(signal, rate), width="stretch")
    st.caption("Le spectre d'analyse utilise le signal complet. L'inférence utilise toujours des fenêtres de 1 024 points.")


def prediction():
    st.title("AI Prediction")
    st.write("Diagnostic par fenêtres non chevauchantes de 1 024 points.")
    st.warning(TEMPORAL_WARNING)
    signal, _ = signal_source()
    if signal is None:
        return
    acknowledged = st.checkbox("J'ai pris connaissance de l'hypothèse d'ordre temporel et des limites "
                               "de compatibilité avec l'entraînement.")
    remainder = len(signal) % 1024
    discard = False
    if remainder:
        discard = st.checkbox(f"Exclure explicitement les {remainder} derniers points incomplets")
    try:
        windows, dropped = segment_signal(signal, discard)
    except ValueError as exc:
        st.info(str(exc))
        return
    st.caption(f"{len(windows)} fenêtre(s) complète(s) · {dropped} point(s) exclu(s) · aucun rééchantillonnage.")
    signature = hashlib.sha256(windows.tobytes()).hexdigest()
    if st.button("Analyser les fenêtres", type="primary", disabled=not acknowledged):
        model = get_model()
        try:
            with st.spinner("Extraction FFT et prédiction SVM…"):
                results = predict_windows(model, windows)
                st.session_state["prediction_result"] = (signature, results)
        except Exception as exc:
            st.error(f"La prédiction a échoué : {exc}")
            return
    saved = st.session_state.get("prediction_result")
    if acknowledged and saved is not None and saved[0] == signature:
        results = saved[1]
        counts = results["Classe"].value_counts()
        if len(results) == 1:
            row = results.iloc[0]
            cols = st.columns(3)
            cols[0].metric("Classe prédite", row["Classe"])
            cols[1].metric("Catégorie", row["Catégorie"])
            size = row["Diamètre nominal (in)"]
            cols[2].metric("Diamètre nominal", "Sans objet" if pd.isna(size) else f"{size:.3f} in")
        else:
            leaders = counts[counts == counts.max()].index.tolist()
            st.success("Classe(s) la/les plus fréquente(s) : " + ", ".join(leaders))
        st.caption("Les effectifs décrivent les classes prédites, pas une probabilité de confiance. "
                   "Aucune calibration probabiliste ni détection hors distribution n'est fournie.")
        st.plotly_chart(distribution_plot(counts), width="stretch")
        st.dataframe(results, hide_index=True, width="stretch")
        st.download_button("Télécharger les prédictions CSV", results.to_csv(index=False).encode("utf-8"),
                            "bearingai_predictions.csv", "text/csv")


def performance():
    st.title("Model Performance")
    st.warning(LIMITATION)
    st.subheader("Comparaison historique des méthodes")
    historic = pd.DataFrame({"Méthode": ["SVM initial", "SVM optimisé", "PCA + SVM", "FFT + SVM"],
                             "Accuracy test (%)": [65.43, 89.02, 90.11, 100.00]})
    st.dataframe(historic, hide_index=True, width="stretch")
    st.caption("Source : tableau préexistant du README.md. Valeurs historiques non recalculées ici ; "
               "aucun autre modèle n'est entraîné pour cette comparaison.")
    st.subheader("Évaluation du modèle sauvegardé")
    st.write("Même protocole que tester_modele.py : reshape C, FFT puis split stratifié "
             "80/20, random_state=42. Seul predict() est appelé ; aucun réentraînement.")
    try:
        with st.spinner("Reconstruction du test et calcul des résultats…"):
            results = cached_evaluation()
    except Exception as exc:
        st.error(f"Évaluation indisponible : {exc}")
        st.info("Le modèle fiable et le dataset NPZ local sont requis. Aucun résultat de remplacement n'est généré.")
        return
    cols = st.columns(2)
    cols[0].metric("Accuracy recalculée", f"{results['accuracy']:.2%}")
    cols[1].metric("Segments du test reconstruit", results["n_test"])
    st.plotly_chart(confusion_plot(results["matrix"], CLASSES), width="stretch")
    st.subheader("Classification report · test reconstruit")
    st.dataframe(results["report"], width="stretch")
    st.download_button("Télécharger le rapport CSV", results["report"].to_csv().encode("utf-8"),
                        "bearingai_classification_report.csv", "text/csv")
    st.caption("Les résultats de validation croisée ne sont pas présents sous forme structurée. "
               "Ils ne sont ni reconstitués artificiellement ni recalculés par entraînement.")


def about():
    st.title("About Project")
    st.write("BearingAI présente le projet CWRU Bearing Fault Diagnosis Using FFT and SVM.")
    st.subheader("Dataset et périmètre")
    st.write("Les données du projet proviennent du Bearing Data Center de Case Western Reserve University. "
             "Le NPZ local contient 4 600 segments de forme 32 × 32, répartis en dix classes équilibrées. "
             "La forme et le nom du fichier ne suffisent pas à prouver l'ordre temporel ou "
             "la fréquence d'un signal importé.")
    st.subheader("FFT et SVM")
    st.write("Le centrage enlève la moyenne de chaque fenêtre. La fenêtre de Hann limite la fuite "
             "spectrale. La FFT réelle produit 513 coefficients ; log1p comprime leurs amplitudes. "
             "Le StandardScaler sauvegardé normalise les caractéristiques avec les paramètres appris. "
             "Le SVM RBF applique ensuite ses frontières de décision aux dix classes.")
    st.code("1 024 points → centrage → Hann → rFFT → abs → log1p\n"
            "→ StandardScaler sauvegardé → SVC RBF sauvegardé → classe", language="text")
    st.subheader("Limites de généralisation")
    st.warning(LIMITATION)
    st.warning(TEMPORAL_WARNING)
    st.write("Une validation par enregistrement, puis sur d'autres charges, vitesses et capteurs, "
             "reste nécessaire. Ce démonstrateur de recherche n'est pas une validation industrielle. "
             "Il ne rééchantillonne pas les signaux, ne déduit pas les unités et ne détecte pas les classes inconnues.")
    st.subheader("Technologies")
    st.write("Python · Streamlit · Plotly · NumPy · Pandas · SciPy · scikit-learn · Joblib")
    st.caption("Seul l'artefact Joblib local fiable est chargé. Aucun upload de modèle, téléchargement "
               "automatique de données ou entraînement n'est effectué.")


with st.sidebar:
    st.markdown("## ⚙ BearingAI")
    st.caption("INTELLIGENT BEARING DIAGNOSTICS")
    st.divider()
    page = st.radio("Navigation", ["Dashboard", "Signal Analysis", "AI Prediction",
                                   "Model Performance", "About Project"], label_visibility="collapsed")
    st.divider()
    st.caption("FFT + SVM / 10 classes / 1 024 points")
    st.caption("Recherche appliquée · inférence locale")

{"Dashboard": dashboard, "Signal Analysis": analysis, "AI Prediction": prediction,
 "Model Performance": performance, "About Project": about}[page]()
