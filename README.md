# GED_PAC_WM_PREDICTIVE_MODEL
This is a demo of the predictive model of the working memory performance from the generalized eigen decomposition phase amplitude coupling values extracted from the  EEG data

# PAC-Based Working Memory Prediction

This project demonstrates the use of Phase-Amplitude Coupling (PAC) features to predict working memory performance using deep learning. It serves as a proof of concept for using neural oscillation characteristics to predict cognitive performance.

## Overview

The project consists of two main parts:
1. PAC Feature Generation and Analysis
2. Working Memory Score Prediction using Neural Networks

### PAC Features Generation

We simulate EEG data with specific phase-amplitude coupling characteristics:
- Theta oscillations (6 Hz)
- Gamma bursts (40 Hz)
- Realistic noise components

[Placeholder: Image of PAC time series and frequency analysis]

Key features extracted:
- PAC Value: ~2.39
- Gamma Frequency: ~43.9 Hz
- Modulation Width: ~28.7 Hz
- Total Modulation: ~83.2

[Placeholder: Image of theta-gamma coupling visualization]

### Deep Learning Model

Architecture:
```python
WMPredictor(
  (layers): Sequential(
    (0): Linear(in_features=4, out_features=6, bias=True)
    (1): BatchNorm1d(6)
    (2): ReLU()
    (3): Dropout(p=0.2)
    (4): Linear(in_features=6, out_features=1, bias=True)
  )
)
```

Training Features:
- Batch size: 3
- AdamW optimizer
- Weight decay: 0.01
- MSE loss function

[Placeholder: Training loss curve]

## Results

Model Performance (2-fold CV):
```
=== Average CV Results ===
MSE: mean ± std
MAE: mean ± std
R²: mean ± std
Correlation: mean ± std
```

[Placeholder: Predictions vs Actuals scatter plot]

## Dependencies
- PyTorch
- NumPy
- Pandas
- scikit-learn
- MNE-Python
- Matplotlib

## Usage

1. Generate PAC features:
```python
# Code snippet for PAC generation
```

2. Train the model:
```python
# Code snippet for model training
```

## Future Work
- Increase dataset size
- Explore additional PAC features
- Test on real EEG data
- Implement more complex architectures

## References
[Add relevant papers and resources]

Would you like me to expand any section or add more technical details?
