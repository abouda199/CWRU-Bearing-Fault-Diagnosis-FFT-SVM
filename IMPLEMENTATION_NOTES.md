# BearingAI repository audit

## Inspected before modification

- `preparation_svm.py`: NPZ flattening and stratified 80/20 split, seed 42.
- `svm_fft.py`, `svm_fft_overfitting.py`: centered, Hann-windowed real FFT magnitude
  and log1p; StandardScaler + RBF SVC. The latter writes the model artifact.
- `tester_modele.py`: original saved-model holdout reconstruction, without fitting.
- `validation_croisee_fft.py`: performs fitting; deliberately not executed.
- `verifier_correspondance.py`, `verifier_fenetres.py`: investigative MAT/NPZ checks;
  chronological ordering is not confirmed by stored results.
- Existing README: historical comparison metrics and segment-split limitation.

## Verified compatibility

- Python 3.11.3; required application packages already installed. Exact tested
  runtime versions are pinned in requirements.txt.
- NPZ: `data` (4600, 32, 32), float64, finite; `labels` (4600,), ten string
  classes with 460 segments each. Loaded with `allow_pickle=False`.
- Model: Pipeline(StandardScaler, SVC(C=10, gamma=0.0001)), 513 features,
  ten expected string classes, `probability=False`. Loaded without version warnings.
- Dataset location has a directory and a file with the same `.npz` name.
- Dataset and model are ignored by Git; README documents manual restoration.

## Decisions and limits

Original scripts have top-level execution, including training and model writes.
Import-safe utilities therefore reuse the exact FFT formula and holdout protocol
without importing those scripts. No model, dataset, MAT file, figure or original
Python script is replaced or deleted. No training is performed.

The app checks dimensions and class compatibility, caches the trusted local model,
and never accepts uploaded pickles. NPZ examples are identified as demonstration
data rather than independent validation. All plots derive from real selected data
or computed predictions. Historical comparison metrics retain their attribution.

The NPZ filename does not prove temporal ordering or sampling rate. External-signal
compatibility is an explicit unresolved assumption. The sampling-rate input affects
axes only; no resampling or silent repair is performed.

Unit tests verify exact numerical preprocessing and real-model inference. Streamlit
AppTest checks all pages and the prediction workflow, including result persistence.
It does not visually assess browser layout. The saved model reproduces 100% on
the documented 920-segment test; this does not validate independent recordings.
