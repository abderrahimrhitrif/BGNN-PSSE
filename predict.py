import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

# Import our custom dataset and model
from create_graph import PowerGridDataset
from model import PowerGridGNN

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

    model = PowerGridGNN(
        var_in_dim=var_in_dim,
        factor_in_dim=factor_in_dim,
        hidden_dim=32,
        readout_units=16,
        output_units=1,
        num_iterations=3
    ).to(device)

    # ==========================================
    # 3. LOAD TRAINED WEIGHTS
    # ==========================================
    model_path = 'best_powergrid_gnn.pth'
    if not os.path.exists(model_path):
        print(f"[!] Error: {model_path} not found. You must run train.py first.")
        return

    print("Loading trained weights...")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # ==========================================
    # 4. RUN INFERENCE
    # ==========================================
    predictions_whole_test_set = []
    true_values_whole_test_set = []

    start = time.time()
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)
            
            # Extract arrays for this specific graph and move to CPU for NumPy/Matplotlib
            preds = out['variable'].cpu().squeeze().numpy()
            trues = batch['variable'].y.cpu().squeeze().numpy()
            
            predictions_whole_test_set.append(preds)
            true_values_whole_test_set.append(trues)

    end = time.time()
    print(f"Time elapsed: {end - start:.9f} seconds")

    # ==========================================
    # 5. CALCULATE & PLOT METRICS
    # ==========================================
    calculate_test_set_results(predictions_whole_test_set, true_values_whole_test_set)
    
    example_id = 0
    plot_results_for_one_test_example(predictions_whole_test_set, true_values_whole_test_set, example_id)


def calculate_test_set_results(predictions_list, true_values_list):
    MSEs = []
    # We loop through the lists we just generated instead of opening CSVs
    for i in range(len(predictions_list)):
        preds = predictions_list[i]
        trues = true_values_list[i]
        
        mse = np.mean(np.square(trues - preds))
        MSEs.append(mse)

    print(f"Average MSE over the test set: {np.mean(MSEs):.9f}")


def plot_results_for_one_test_example(predictions_list, true_values_list, example_id):
    preds = predictions_list[example_id]
    trues = true_values_list[example_id]

    mse = np.mean(np.square(trues - preds))
    print(f'MSE first example ({example_id}): {mse:.9f}')

    linewidth = 1
    plt.plot(range(len(preds)), preds, color="red", label="predictions", linewidth=linewidth)
    plt.plot(range(len(trues)), trues, color="blue", label="true values", linewidth=linewidth)
    plt.legend()
    
    # Ensure the directory exists before trying to save
    os.makedirs('data/test', exist_ok=True)
    save_path = os.path.join('data/test', 'singleTestSampleSetRezPerNodeID.png')
    
    plt.savefig(save_path)
    print(f"Plot saved successfully to: {save_path}")
    plt.clf()

if __name__ == "__main__":
    predict()