
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
