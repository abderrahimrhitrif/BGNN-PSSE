# Probabilistic Factor Graph GNN (PFGNN) for Power Grid State Estimation

This project implements a Graph Neural Network (GNN) approach to Power Grid State Estimation (SE) on the IEEE 300-bus system. It provides two distinct architectural paths:

## 1. Base Model (Non-Probabilistic)
The **Base Model** (`model.py`) provides a deterministic point estimate of the power system state (voltage magnitudes and angles). It utilizes a heterogeneous graph structure to model the relationships between buses and measurements but does not quantify prediction uncertainty.

## 2. PFGNN (Probabilistic Model)
The **PFGNN** (`model_prob.py`) is the core contribution of this project. It extends the GNN architecture into the probabilistic domain using a **Decoupled Residual Regression** strategy and **MC Dropout**.

- **Uncertainty Quantification:** Provides both Aleatoric uncertainty (data noise) and Epistemic uncertainty (model ignorance).
- **Dual-Head Readout:** Outputs both the predicted state mean ($\mu$) and the predicted variance ($\sigma^2$).
- **Efficiency:** Significantly faster than traditional iterative WLS solvers while providing comprehensive confidence intervals.

---

## Project Structure

### Core Implementation
- `model.py`: Deterministic GNN architecture.
- `model_prob.py`: Probabilistic PFGNN architecture.
- `train.py`: Training pipeline for the deterministic model.
- `train_and_evai.py`: Training pipeline for the probabilistic PFGNN using decoupled loss.

### Data & Processing
- `generate_dataset.py`: Synthetic data generation using `pandapower`.
- `create_graph.py`: Converts raw power system data and Jacobian matrices into PyTorch Geometric heterogeneous graphs.

### Inference & Benchmarking
- `predict.py`: Inference for the deterministic model.
- `predict_prob.py`: Probabilistic inference including uncertainty decomposition and confidence interval plotting.
- `benchmark_wls.py`: Performance benchmark comparing traditional solvers (Newton-Raphson/WLS) against the GNN.
- `plot_latency_comparison.py`: Generates visualization of the computational speedup achieved by PFGNN.

---

## Quick Start

1. **Generate Dataset:**
   ```bash
   python generate_dataset.py
   ```

2. **Process Graphs:**
   ```bash
   python create_graph.py
   ```

3. **Train Probabilistic Model:**
   ```bash
   python train_and_evai.py
   ```

4. **Run Probabilistic Inference:**
   ```bash
   python predict_prob.py
   ```
