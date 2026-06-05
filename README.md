# Power-Quality-Classifier
# Power Quality Disturbance Classification using Attention-Based Multiple Instance Learning

## Overview

This project presents a deep learning framework for automatic classification of Power Quality Disturbances (PQDs) in electrical power systems.

The proposed system combines signal processing techniques with an Attention-Based Multiple Instance Learning (MIL) architecture to accurately classify disturbances while also identifying the most relevant signal region responsible for the prediction.

The model processes multiple representations of a power signal, including frequency-domain, harmonic, and envelope information, enabling robust disturbance recognition.

---

## Features

* Classification of 17 Power Quality Disturbance classes
* Multi-channel signal representation
* FFT-based spectral analysis
* Harmonic energy extraction
* Hilbert transform envelope extraction
* Data augmentation for improved generalization
* Attention-Based Multiple Instance Learning (MIL)
* Disturbance localization using attention weights
* Early stopping and learning-rate scheduling
* GPU acceleration with PyTorch

---

## Disturbance Classes

The model supports classification of:

1. Pure Sinusoidal
2. Sag
3. Swell
4. Interruption
5. Transient
6. Oscillatory Transient
7. Harmonics
8. Harmonics with Sag
9. Harmonics with Swell
10. Flicker
11. Flicker with Sag
12. Flicker with Swell
13. Sag with Oscillatory Transient
14. Swell with Oscillatory Transient
15. Sag with Harmonics
16. Swell with Harmonics
17. Notch

---

## Project Structure

```text
Power-Quality-Classifier/
│
├── Dataset/                  # Disturbance datasets (CSV format)
├── ltspice_schematics/       # LTspice circuits used for waveform generation
├── models/                   # Saved trained models
│
├── main.py                   # Training, evaluation and localization pipeline
├── Demo.py                   # Demonstration/inference script
├── Presentation1.pdf         # Project presentation
└── README.md
```

---

## Methodology

### 1. Signal Acquisition

Power quality disturbance waveforms are generated using LTspice simulations and stored as CSV datasets.

### 2. Feature Extraction

Each signal is transformed into four complementary channels:

#### Channel 1: Normalized Time-Domain Signal

Captures waveform shape and amplitude variations.

#### Channel 2: FFT Magnitude Spectrum

Provides frequency-domain information useful for detecting harmonics and spectral distortions.

#### Channel 3: Harmonic Energy Profile

Extracts energy corresponding to harmonic frequencies, allowing direct analysis of harmonic contamination.

#### Channel 4: Hilbert Envelope

Uses the Hilbert Transform to obtain the instantaneous amplitude envelope, highlighting transient and modulation effects.

---

## Data Augmentation

To improve model robustness, the following augmentations are applied:

* Additive Gaussian noise
* Amplitude scaling
* Temporal shifting

This increases dataset diversity and improves generalization.

---

## Multiple Instance Learning (MIL)

Instead of processing the entire signal at once, each signal is divided into overlapping windows.

```text
Signal
│
├── Window 1
├── Window 2
├── Window 3
└── ...
```

Each window becomes an instance within a bag.

This enables the model to focus on localized disturbances while maintaining global classification performance.

---

## Model Architecture

### CNN Instance Encoder

Each window is processed using:

* Conv1D + BatchNorm + ReLU
* Conv1D + BatchNorm + ReLU
* Conv1D + BatchNorm + ReLU
* Adaptive Average Pooling
* Fully Connected Layer

The encoder learns compact feature representations for every window.

### Attention Mechanism

Attention scores are assigned to all windows:

```text
Window Features
       │
       ▼
Attention Network
       │
       ▼
Importance Weights
```

The model automatically identifies the most informative regions of the signal.

### Classifier

The attention-weighted representation is passed through fully connected layers for final disturbance classification.

---

## Disturbance Localization

A unique feature of the system is attention-based localization.

After classification, attention weights are analyzed to determine:

* Which signal window contributed most to the prediction
* Approximate location of the disturbance
* Relative importance of different regions

This improves model interpretability.

---

## Training Strategy

The training pipeline includes:

* Cross Entropy Loss with Label Smoothing
* AdamW Optimizer
* Learning Rate Scheduler
* Gradient Clipping
* Early Stopping
* Stratified Train/Validation/Test Splits

---

## Requirements

Install dependencies:

```bash
pip install numpy pandas scipy scikit-learn torch torchvision
```

---

## Running the Project

### Train the Model

```bash
python main.py
```

### Run Demo

```bash
python Demo.py
```

---

## Outputs

The framework provides:

* Classification Accuracy
* Validation Accuracy
* Test Accuracy
* Confusion Analysis
* Per-Class Performance
* Attention-Based Disturbance Localization

---

## Applications

* Smart Grid Monitoring
* Industrial Power Systems
* Power Quality Assessment
* Fault Detection
* Energy Management Systems
* Predictive Maintenance

---

## Future Scope

* Real-time deployment on edge devices
* Online disturbance monitoring
* Transformer-based architectures
* Explainable AI visualizations
* FPGA and embedded implementations
* Integration with smart grid infrastructure

---

## Team Members

Siddha Nema 240002070

Savita Meena 240002068

---

## Supervisor

Prof. Vivek Kanhangad

Department of Electrical Engineering

Indian Institute of Technology Indore

---

## License

This project is intended for academic and educational purposes.

