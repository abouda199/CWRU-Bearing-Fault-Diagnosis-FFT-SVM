
# CWRU Bearing Fault Diagnosis Using FFT and SVM

Machine learning-based bearing fault diagnosis using
vibration signals from the Case Western Reserve
University (CWRU) dataset.

## Project Overview

This project develops a multiclass bearing fault
classification system combining:

- Fast Fourier Transform (FFT)
- Support Vector Machine (SVM)
- Hyperparameter optimization using GridSearchCV
- Stratified cross-validation
- Performance evaluation and confusion matrices

## Dataset

Dataset: CWRU Bearing Dataset

- Total samples: 4,600
- Number of classes: 10
- Samples per class: 460
- Input dimensions: 32 x 32
- Flattened dimensions: 1,024

### Fault Classes

- Normal
- Ball_007, Ball_014, Ball_021
- IR_007, IR_014, IR_021
- OR_007, OR_014, OR_021

## Methodology

1. Load vibration data
2. Flatten input samples
3. Extract FFT-based features
4. Apply feature standardization
5. Train an RBF-kernel SVM
6. Optimize C and gamma using GridSearchCV
7. Evaluate the model
8. Save the trained model using Joblib

## Experimental Results

| Model | Test Accuracy |
|-------|---------------|
| Initial SVM | 65.43% |
| Optimized SVM | 89.02% |
| PCA + SVM | 90.11% |
| FFT + SVM | 100.00% |

The FFT + SVM model achieved 100% accuracy on
the random stratified test split.

**Validation limitation:** The dataset contains
segmented samples without original recording IDs.
Segments from the same acquisition may appear
in both training and testing subsets.

Therefore, the reported accuracy does not establish
generalization to independent recordings.
Recording-level validation remains future work.

## BearingAI — Streamlit application

Five views: **Dashboard**, **Signal Analysis**, **AI Prediction**,
**Model Performance**, **About Project**. Dark navy/cyan styling, interactive
Plotly charts, local inference and CSV exports. No training or automatic data download.

### Windows / VS Code setup

Python **3.11** is the tested version. In PowerShell:

```powershell
cd C:\Users\MSI-PRO\Desktop\amin
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

If script activation is blocked, run directly:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the URL printed by Streamlit, normally `http://localhost:8501`.
Select `.venv\Scripts\python.exe` as the VS Code interpreter. If dependencies
already exist in your active environment, run `python -m streamlit run app.py`.

### Application layout and artifacts

```text
amin/
├── app.py
├── modele_FFT_SVM_CWRU.pkl
├── CWRU_48k_load_1_CNN_data.npz/
│   └── CWRU_48k_load_1_CNN_data.npz
├── utils/
│   ├── preprocessing.py
│   ├── prediction.py
│   └── visualization.py
├── tests/
├── .streamlit/config.toml
└── requirements.txt
```

The NPZ is optional for external CSV inference, but required for examples and
performance evaluation. Paths are anchored with pathlib to the project folder.
The existing `.gitignore` excludes datasets and the model: a fresh GitHub clone
requires manually restoring the owner's original artifacts at the paths above.
Only load trusted Joblib files; deserialization can execute code. Model uploads
are not accepted. Restart Streamlit after replacing artifacts to invalidate caches.
scikit-learn 1.7.2 is pinned to the verified local runtime; migration of serialized
models across versions requires separate validation.

### Signal input

Upload UTF-8 CSV (20 MB maximum), one point per row. Choose the header setting,
comma/semicolon/tab delimiter and amplitude column. Decimal separator is a point.
Extra columns, including timestamps, are allowed. Only the selected amplitude
column is analyzed. Missing, non-numeric or infinite values are rejected; no
samples are silently deleted or interpolated. Blank rows are retained for validation.

Enter a sampling rate only when known. Otherwise, axes use sample indices and
cycles/sample. No frequency or unit is inferred from filenames or timestamps.
Statistics use all points: mean, population standard deviation, RMS, min, max and
Pearson kurtosis (undefined for constant signals). The spectrum is a centered,
Hann-windowed log FFT magnitude, not a calibrated PSD. Temporal plots above
20,000 points are subsampled for display only.

### Inference protocol

Complete, non-overlapping windows of **1,024 points** are required. Short inputs
are rejected, without padding. Exclusion of incomplete trailing points requires
explicit consent. The exact original transform is:

```python
X = X.reshape(len(X), -1)
X_centered = X - X.mean(axis=1, keepdims=True)
window = np.hanning(X.shape[1])
X_fft = np.log1p(np.abs(np.fft.rfft(X_centered * window, axis=1)))
```

The 513 features enter the saved StandardScaler and SVC via `predict()`.
No new normalization or resampling is added. Original scripts execute training
at module level, so their exact transformation and evaluation protocol are reused
in import-safe utilities instead of importing those scripts.

**Temporal-order assumption:** the repository does not establish that C-order
flattening of the 32 × 32 NPZ samples recovers chronological measurements. Existing
MAT comparison scripts are investigative checks, not stored confirmation. The app
displays this assumption and requires acknowledgement before prediction. Verify
ordering, sampling rate, amplitude units, sensor and operating conditions against
training before interpreting external predictions.

Fault sizes are nominal class diameters: 007 = 0.007 in, 014 = 0.014 in,
021 = 0.021 in, not continuous physical measurements. Long signals show all window
predictions, class counts and CSV export. Counts are not confidence probabilities.
The saved SVC has `probability=False`; neither artificial probabilities nor decision
scores presented as confidence are displayed. No unknown-class detection is provided.

### Performance provenance

The comparison table is attributed to the pre-existing historical results above;
other methods are not retrained. The confusion matrix and classification report
are recalculated from the saved model with `tester_modele.py`'s protocol:
original NPZ row order, exact FFT, stratified 80/20 split, `random_state=42`.
This reconstructs **920 test segments** and locally reproduces **100% accuracy**.
No fitting or artifact modification occurs. No cross-validation scores are invented
or recalculated by training. This reconstruction cannot independently establish
the artifact's training history or generalization to new recordings. Recording IDs
are absent from the NPZ; the random-segment limitation is visible in the app.

### Verification

```powershell
python -m unittest discover -s tests -v
```

Tests cover exact FFT equivalence, invalid CSV data, segmentation and explicit tail
handling, trusted model loading, real predictions, reconstructed holdout and all five
Streamlit pages using AppTest. No fitting occurs. Scope discovery to `tests/`:
some original root-level research scripts named `test_*.py` can train when executed.
AppTest verifies execution and interactions; browser rendering is not visually tested.
See [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) for the initial audit.

## Technologies

- Python
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- FFT
- SVM

## Main Scripts

- `preparation_svm.py`: Data preparation
- `entrainement_svm.py`: Initial SVM training
- `optimisation_svm.py`: Hyperparameter optimization
- `svm_pca.py`: PCA + SVM
- `svm_fft.py`: FFT + SVM
- `svm_fft_overfitting.py`: Training and evaluation
- `tester_modele.py`: Load and test the saved model
- `verification_fuite.py`: Data similarity checks

## Future Improvements

- Recording-independent evaluation
- Validation across different operating conditions
- Testing on new vibration recordings
- Deployment through a prediction interface

## Author

Abdel Bari BRAHIM

Applied Computer Science Engineering Student
ENISO
