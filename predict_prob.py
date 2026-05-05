import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

# Import our custom dataset and the BAYESIAN model
from create_graph import PowerGridDataset
from model_prob import BayesianPowerGridGNN # <--- Updated import

def predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running predictions on device: {device}")

    # ==========================================
    # 1. LOAD TEST DATASET
    # ==========================================
    print("Loading test dataset...")
    test_dataset = PowerGridDataset(root='./', dataset_type='test')
    
    # We use batch_size=1 so we can easily plot individual graphs later
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # ==========================================
    # 2. RE-INSTANTIATE THE MODEL
    # ==========================================
    sample_data = test_dataset[0]
    var_in_dim = sample_data['variable'].x.shape[1]
    factor_in_dim = sample_data['factor'].x.shape[1]

    model = BayesianPowerGridGNN(
        var_in_dim=var_in_dim,
        factor_in_dim=factor_in_dim,
        hidden_dim=32,
        readout_units=16,
        output_units=1
    ).to(device)

    # ==========================================
    # 3. LOAD TRAINED WEIGHTS
    # ==========================================
    model_path = 'best_model.pth' 
    if not os.path.exists(model_path):
        print(f"[!] Error: {model_path} not found. You must run train.py first.")
        return

    print("Loading trained weights...")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # ==========================================
    # 4. RUN INFERENCE & TIMING
    # ==========================================
    predictions_mu_whole_test_set = []
    predictions_var_whole_test_set = [] 
    true_values_whole_test_set = []
    
    # List to hold the time it takes for each individual snapshot
    snapshot_times = []

    print("Running inference...")
    total_start = time.time()
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # -- START TIMING --
            if device.type == 'cuda':
                torch.cuda.synchronize() # Wait for all previous GPU tasks to finish
            snapshot_start = time.time()
            
            out = model(batch.x_dict, batch.edge_index_dict)
            
            # -- STOP TIMING --
            if device.type == 'cuda':
                torch.cuda.synchronize() # Wait for the model to actually finish calculating
            snapshot_end = time.time()
            
            snapshot_times.append(snapshot_end - snapshot_start)
            
            # Extract mu and var, convert to numpy
            mu = out['mu'].cpu().squeeze().numpy()
            var = out['var'].cpu().squeeze().numpy()
            trues = batch['variable'].y.cpu().squeeze().numpy()
            
            predictions_mu_whole_test_set.append(mu)
            predictions_var_whole_test_set.append(var)
            true_values_whole_test_set.append(trues)

    total_end = time.time()
    
    # Print Timing Metrics
    print(f"\n--- TIMING METRICS ---")
    print(f"Total Test Set Time: {total_end - total_start:.4f} seconds")
    print(f"Per-Snapshot Average: {np.mean(snapshot_times):.6f} seconds")
    print(f"Per-Snapshot Min:     {np.min(snapshot_times):.6f} seconds")
    print(f"Per-Snapshot Max:     {np.max(snapshot_times):.6f} seconds")
    print(f"----------------------\n")

    # ==========================================
    # 5. CALCULATE & PLOT METRICS
    # ==========================================
    calculate_test_set_results(predictions_mu_whole_test_set, predictions_var_whole_test_set, true_values_whole_test_set)
    
    example_id = 0
    plot_results_for_one_test_example(predictions_mu_whole_test_set, predictions_var_whole_test_set, true_values_whole_test_set, example_id)


def calculate_test_set_results(predictions_mu_list, predictions_var_list, true_values_list):
    MSEs = []
    NLLs = []
    eps = 1e-6 # For numerical stability in NLL

    for i in range(len(predictions_mu_list)):
        mu = predictions_mu_list[i]
        var = predictions_var_list[i]
        trues = true_values_list[i]
        
        # Calculate MSE
        mse = np.mean(np.square(trues - mu))
        MSEs.append(mse)

        # Calculate Gaussian NLL in NumPy
        var_safe = var + eps
        nll = 0.5 * np.mean(np.log(var_safe) + np.square(trues - mu) / var_safe)
        NLLs.append(nll)

    # Flatten the list of arrays into one single array containing every node's variance
    all_variances = np.concatenate(predictions_var_list)

    print(f"\n--- TEST SET METRICS ---")
    print(f"Average MSE:      {np.mean(MSEs):.9f}")
    print(f"Average NLL:      {np.mean(NLLs):.9f}")
    print(f"Average Variance: {np.mean(all_variances):.9f}")
    print(f"Max Variance:     {np.max(all_variances):.9f}") # <--- ADDED
    print(f"Min Variance:     {np.min(all_variances):.9f}") # <--- ADDED
    print(f"------------------------\n")


def plot_results_for_one_test_example(predictions_mu_list, predictions_var_list, true_values_list, example_id):
    mu = predictions_mu_list[example_id]
    var = predictions_var_list[example_id]
    trues = true_values_list[example_id]

    # Convert variance to standard deviation for confidence intervals
    std = np.sqrt(var)
    upper_bound = mu + 2 * std  # ~95% confidence interval
    lower_bound = mu - 2 * std

    mse = np.mean(np.square(trues - mu))
    print(f'MSE first example ({example_id}): {mse:.9f}')

    linewidth = 1
    plt.figure(figsize=(10, 5)) # Made slightly wider to better see intervals
    
    # Plot true values and predicted mean
    plt.plot(range(len(trues)), trues, color="blue", label="True Values", linewidth=linewidth)
    plt.plot(range(len(mu)), mu, color="red", label="Predicted Mean (mu)", linewidth=linewidth)
    
    # Plot the uncertainty bounds as a shaded region
    plt.fill_between(
        range(len(mu)), 
        lower_bound, 
        upper_bound, 
        color="red", 
        alpha=0.2, 
        label="95% Confidence Interval (\u00B1 2\u03C3)"
    )
    
    plt.legend()
    plt.title(f"Test Example {example_id} - Probabilistic Predictions")
    plt.xlabel("Node Index")
    plt.ylabel("Target Value")
    
    # Ensure the directory exists before trying to save
    os.makedirs('data/test', exist_ok=True)
    save_path = os.path.join('data/test', 'singleTestSampleSetRezPerNodeID.png')
    
    plt.savefig(save_path)
    print(f"Plot saved successfully to: {save_path}")
    plt.clf()

if __name__ == "__main__":
    predict()