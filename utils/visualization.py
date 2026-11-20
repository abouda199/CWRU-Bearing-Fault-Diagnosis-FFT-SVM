"""Plotly views derived exclusively from the selected signal or computed results."""
import numpy as np
import plotly.graph_objects as go

CYAN = "#22d3ee"


def style(fig, title, x_title, y_title):
    fig.update_layout(title=title, template="plotly_dark", height=360,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1c30",
                      font=dict(family="Segoe UI, sans-serif", color="#cfdeee"),
                      margin=dict(l=20, r=20, t=60, b=30),
                      xaxis_title=x_title, yaxis_title=y_title)
    return fig


def temporal_plot(signal, sample_rate=None):
    x = np.arange(len(signal))
    if sample_rate:
        x = x / sample_rate
    # Bound rendering cost, while all statistics and inference use the full signal.
    step = max(1, int(np.ceil(len(signal) / 20000)))
    fig = go.Figure(go.Scattergl(x=x[::step], y=signal[::step], mode="lines",
                                 line=dict(color=CYAN, width=1), name="Vibration"))
    return style(fig, "Signal temporel", "Temps (s)" if sample_rate else "Indice du point",
                 "Amplitude (unité du fichier)")


def spectrum_plot(signal, sample_rate=None):
    centered = signal - np.mean(signal)
    spectrum = np.log1p(np.abs(np.fft.rfft(centered * np.hanning(len(signal)))))
    frequencies = np.fft.rfftfreq(len(signal), d=1 / sample_rate if sample_rate else 1)
    return style(go.Figure(go.Scattergl(x=frequencies, y=spectrum, mode="lines",
                                        line=dict(color=CYAN, width=1))),
                 "Spectre FFT · centrage + Hann + log1p",
                 "Fréquence (Hz)" if sample_rate else "Fréquence normalisée (cycles/point)",
                 "log(1 + |FFT|), non calibré")


def distribution_plot(counts):
    return style(go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=CYAN)),
                 "Distribution des classes prédites", "Classe", "Nombre de fenêtres")


def confusion_plot(matrix, classes):
    return style(go.Figure(go.Heatmap(z=matrix, x=list(classes), y=list(classes),
                                      colorscale="Blues", text=matrix, texttemplate="%{text}",
                                      hovertemplate="Réel : %{y}<br>Prédit : %{x}<br>Nombre : %{z}<extra></extra>")),
                 "Matrice de confusion · test reconstruit", "Classe prédite", "Classe réelle")
