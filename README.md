# GED_PAC_WM_PREDICTIVE_MODEL
This is a demo of the predictive model of the working memory performance from the generalized eigen decomposition phase amplitude coupling values extracted from the  EEG data

# PAC-Based Working Memory Prediction

This project demonstrates the use of Phase-Amplitude Coupling (PAC) features to predict working memory performance using deep learning. It serves as a proof of concept for using neural oscillation characteristics to predict cognitive performance.

## Overview

The project consists of 3 main parts:
1. Data generation
2. Multivariate generalized eigendecomposition phase-amplitude coupling
3. Working Memory Score Prediction using Neural Networks

### PAC Features Generation

# Phase-Amplitude Coupling Analysis for Working Memory Prediction

## Analysis Pipeline

### 1. Theta Component Extraction via GED
First, we extract the theta component using Generalized Eigendecomposition (GED):

![Components Analysis](https://github.com/zavesone/GED_PAC_WM_PREDICTIVE_MODEL/blob/main/component_analysis.png)
- Left: Power spectra showing clear theta peak in both component and filtered signals
- Right: Channel spectrum demonstrating raw EEG frequency content
- Bottom: GED eigenvalues showing component separation

### 2. Theta Component Characteristics
The extracted theta component shows clear oscillatory behavior:

![Theta Time Series](https://github.com/zavesone/GED_PAC_WM_PREDICTIVE_MODEL/blob/main/theta_component_time_series.png)
- Clean theta oscillations at ~6 Hz
- Stable amplitude modulation
- Clear cyclic pattern

### 3. Trough Detection and Analysis
We detect troughs in the theta signal for phase-based analysis:

![Trough Analysis](https://github.com/zavesone/GED_PAC_WM_PREDICTIVE_MODEL/blob/main/theta_components_spectral.png)
- Top: Theta component with detected troughs (magenta dots)
- Bottom: Data visualization around a specific trough showing spatial distribution of activity across channels

### 4. Frequency-Specific Modulation
Analysis of cross-frequency coupling strength:

![PAC Analysis](https://github.com/zavesone/GED_PAC_WM_PREDICTIVE_MODEL/blob/main/frequency_specific_modulation.png)
- Clear peak in gamma range (40-50 Hz)
- Shows strongest coupling between theta phase and gamma amplitude
- Demonstrates specific frequency bands involved in PAC

### 5. Working Memory Prediction
The extracted PAC features were used to predict working memory performance:
- 2-fold cross-validation
- Mean correlation: 0.387 ± 0.291
- MSE: 22.973 ± 15.734

## Technical Details
- Sampling rate: 1024 Hz
- Theta band: 4-8 Hz
- Gamma band: 30-90 Hz
- Regularized covariance matrices
- AdamW optimizer for neural network

## Key Findings
1. Successful extraction of theta component using GED
2. Clear theta-gamma coupling in expected frequency ranges
3. Proof of concept for PAC-based working memory prediction

Would you like me to elaborate on any part of this documentation?

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


## References
[Add relevant papers and resources]

Would you like me to expand any section or add more technical details?
