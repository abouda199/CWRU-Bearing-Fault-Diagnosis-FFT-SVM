
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Classes dans l'ordre du rapport SVM
classes = [
    "Ball_007", "Ball_014", "Ball_021",
    "IR_007", "IR_014", "IR_021",
    "Normal", "OR_007", "OR_014", "OR_021"
]

# Matrice obtenue avec ton SVM
cm = np.array([
    [31, 31, 2, 0, 0, 0, 4, 0, 24, 0],
    [7, 11, 7, 6, 0, 0, 48, 0, 12, 1],
    [16, 12, 45, 2, 0, 1, 0, 0, 14, 2],
    [0, 0, 0, 92, 0, 0, 0, 0, 0, 0],
    [11, 1, 0, 0, 79, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 91, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 92, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 92, 0, 0],
    [31, 23, 0, 0, 0, 0, 12, 0, 26, 0],
    [0, 2, 5, 16, 2, 0, 0, 24, 0, 43]
])

# Afficher la matrice
fig, ax = plt.subplots(figsize=(12, 10))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    xticks_rotation=45
)

plt.title("Matrice de confusion - SVM - CWRU")
plt.xlabel("Classes prédites")
plt.ylabel("Classes réelles")

plt.tight_layout()
plt.savefig("matrice_confusion_SVM.png", dpi=300)
plt.show()
