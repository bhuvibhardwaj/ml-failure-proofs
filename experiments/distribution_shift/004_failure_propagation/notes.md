# 004 — Domino Effect: Locality Metric Under Distribution Shift

## Core Hypothesis

Machine learning failures under distribution shift do not occur instantaneously at the output layer.

Instead, spurious signals propagate progressively through internal representations, amplifying across layers and eventually resulting in prediction collapse.

This experiment introduces a locality metric:

D(k)

which measures representation divergence between clean and shifted inputs at layer k using GradCAM attribution drift.

---

# Research Question

Can internal representation drift be measured layer-by-layer during distribution shift?

Does spurious signal strength amplify deeper into the network?

---

# Experimental Setup

## Dataset
MNIST

## Model
Simple CNN:
- Conv Layer 1 → ReLU → MaxPool
- Conv Layer 2 → ReLU → MaxPool
- Fully Connected Classifier

## Distribution Shift Applied
Aggressive input corruption:
- Random Rotation (75°)
- Gaussian Blur (kernel size = 7)

The model is trained only on clean MNIST images.

---

# Locality Metric

For each layer k:

D(k) = || GradCAM_shifted(k) - GradCAM_clean(k) ||

Where:
- GradCAM_clean(k) = attribution map for clean input
- GradCAM_shifted(k) = attribution map for shifted input
- || · || = vector norm magnitude

Interpretation:
- Small D(k): stable representation
- Large D(k): strong representation drift
- Increasing D(k): spurious signal amplification

---

# Results

## Accuracy

| Condition | Accuracy |
|---|---|
| Clean | 99.01% |
| Shifted | 64.30% |

Accuracy Drop:
34.71%

---

## Locality Metric

| Layer | Average D(k) |
|---|---|
| Conv Layer 1 | 0.0913 |
| Conv Layer 2 | 0.3489 |

Observation:
D(2) > D(1)

This suggests:
- representation drift increases deeper into the network
- spurious signals amplify hierarchically
- internal instability accumulates before final prediction collapse

---

# Key Interpretation

The experiment supports the hypothesis that distribution shift behaves like a cascading internal failure process rather than a single isolated output error.

The deeper convolutional layer exhibited substantially larger attribution drift than the earlier layer, suggesting that representation corruption compounds through the network hierarchy.

This behavior is referred to here as:

"The Domino Effect"

where early perturbations propagate and intensify across internal representations.

---

# Why This Matters

Most robustness evaluation relies on:
- accuracy
- loss
- confidence

These metrics only measure final outcomes.

D(k) instead attempts to measure:
- where instability enters
- how instability propagates
- whether deeper layers amplify spurious information

This shifts robustness analysis from:
output-only evaluation

toward:
internal failure dynamics

---

# Limitations

- Single dataset (MNIST)
- Small CNN architecture
- Only two convolutional layers analyzed
- GradCAM-based locality approximation
- Limited corruption types

---

# Future Work

- CIFAR-10-C robustness evaluation
- Transformer attention drift analysis
- Layerwise instability trajectories
- Cross-architecture comparison
- Frequency-domain perturbation analysis
- Formalization of D(k) propagation dynamics
- Correlation between D(k) and calibration collapse

---

# Repository Context

This experiment is part of the broader:

ML Failure Proofs

research series investigating:
- distribution shift
- representation instability
- robustness failures
- confidence collapse
- cascading internal errors in neural networks
