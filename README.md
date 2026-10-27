# BGNN-PSSE: Probabilistic Factor Graph GNN for Power Grid State Estimation

## Overview
This project implements a Graph Neural Network (GNN) approach for Power Grid State Estimation (SE) on the IEEE 300-bus system. It provides two distinct architectural paths: a deterministic base model and a probabilistic Factor Graph GNN (PFGNN) capable of robust uncertainty quantification.

## Architecture & Pipeline
1. **Scenario Generation**: Uses `pandapower` to generate 10,000 synthetic power system operating points (voltages, loads, generation, line flows) based on the IEEE 300-bus network.
2. **Graph Construction**: Converts grid buses and measurement components into heterogeneous PyTorch Geometric graphs (`HeteroData`), encoding topological connections and Jacobian measurement matrices.
3. **Model Execution**: Implements heterogeneous message passing across variable and factor nodes.
4. **Uncertainty & Readout**: Outputs point estimates along with predicted variance via decoupled residual regression and MC Dropout.

## Data
- **Dataset**: Synthetic operating scenarios generated from IEEE 300-bus network via `pandapower`.
- **Splits**: 70% training (7,000 samples), 15% validation (1,500 samples), 15% test (1,500 samples).
- **Features**: Multi-feature node states for variable nodes (voltages, angles) and factor nodes (measurements, covariances).

## Method & Algorithms
- **Base Model (`model.py`)**: Deterministic point estimation using heterogeneous GNN convolutional layers.
- **Probabilistic PFGNN (`model_prob.py`)**: Dual-head readout predicting mean ($\mu$) and variance ($\sigma^2$) using Gaussian Negative Log-Likelihood (NLL) loss with variance regularization.
- **Benchmarking (`benchmark_wls.py`)**: Compares inference speed and accuracy against the traditional iterative Weighted Least Squares (WLS) power system solver.

## Verified Results & Metrics
- Successfully generated 10,000 samples with 70/15/15 train/val/test splits.
- Demonstrated significantly faster inference times compared to iterative WLS solvers while providing comprehensive confidence bounds.

## Limitations & Caveats
- Research prototype evaluated entirely on simulated pandapower scenarios rather than physical SCADA telemetry.

## Run Instructions
```bash
# 1. Generate dataset
python generate_dataset.py

# 2. Train probabilistic model
python train_and_evai.py

# 3. Run inference & evaluation
python predict_prob.py
```

## Claims to Avoid
- Do not claim deployment on live utility SCADA grids or physical power transmission networks.
