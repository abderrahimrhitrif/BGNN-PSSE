import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

# Import your dataset
from create_graph import PowerGridDataset 

# Import the model we just defined
from model import PowerGridGNN 

def train():
    # ==========================================
    # 1. HARDWARE & CONFIGURATION
    # ==========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    batch_size = 32
    epochs = 500  # You can safely leave this high now thanks to early stopping
    learning_rate = 0.001
    clipnorm = 1.0
    clipvalue = 0.5
    
    # Early Stopping Configuration
    patience = 30  # Stop if no improvement for 15 epochs
    
    # Model hyperparameters
    hidden_dim = 32
    readout_units = 16
    output_units = 1

    # ==========================================
    # 2. DATA LOADERS
    # ==========================================
    print("Loading datasets...")
    train_dataset = PowerGridDataset(root='./', dataset_type='train')
    val_dataset = PowerGridDataset(root='./', dataset_type='validation')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) 

    # ==========================================
    # 3. DYNAMIC MODEL INSTANTIATION
    # ==========================================
    sample_data = train_dataset[0]
    var_in_dim = sample_data['variable'].x.shape[1]
    factor_in_dim = sample_data['factor'].x.shape[1]

    print(f"Instantiating model with Var_In: {var_in_dim}, Factor_In: {factor_in_dim}")
    
    model = PowerGridGNN(
        var_in_dim=var_in_dim, 
        factor_in_dim=factor_in_dim, 
        hidden_dim=hidden_dim, 
        readout_units=readout_units, 
        output_units=output_units, 
        num_iterations=3
    ).to(device)
    
    # UPGRADE 1: Swapped Adam for AdamW. Decouples weight decay for much better generalization.
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,      # reduce LR by half
        patience=5,      # wait 5 epochs
        min_lr=1e-6
    )

    # ==========================================
    # 4. TRAINING LOOP & EARLY STOPPING SETUP
    # ==========================================
    best_val_mse = float('inf')
    stagnation_counter = 0
    best_model_path = 'best_powergrid_gnn.pth'

    print("Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            out = model(batch.x_dict, batch.edge_index_dict)

            # Extract targets 
            target = batch['variable'].y.view(-1, 1)
            
            # Loss: MeanSquaredError
            loss = F.mse_loss(out['variable'], target)
            
            # Backward pass
            loss.backward()

            # Gradient Clipping
            clip_grad_norm_(model.parameters(), max_norm=clipnorm)
            clip_grad_value_(model.parameters(), clip_value=clipvalue)

            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        avg_train_loss = total_loss / len(train_dataset)

        # ==========================================
        # 5. VALIDATION LOOP
        # ==========================================
        model.eval()
        val_mse = 0
        val_mae = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x_dict, batch.edge_index_dict)
                target = batch['variable'].y.view(-1, 1)

                # Metrics: MAE + MSE
                val_mse += F.mse_loss(out['variable'], target, reduction='sum').item()
                val_mae += F.l1_loss(out['variable'], target, reduction='sum').item()
                
        num_val_nodes = len(val_dataset) * sample_data['variable'].x.shape[0]
        avg_val_mse = val_mse / num_val_nodes
        avg_val_mae = val_mae / num_val_nodes

        print(f'Epoch: {epoch:03d}/{epochs} | Train MSE: {avg_train_loss:.9f} | Val MSE: {avg_val_mse:.9f} | Val MAE: {avg_val_mae:.9f}')
        scheduler.step(avg_val_mse)

        # ==========================================
        # 6. EARLY STOPPING LOGIC
        # ==========================================
        if avg_val_mse < best_val_mse:
            best_val_mse = avg_val_mse
            stagnation_counter = 0
            # Save the best weights to disk
            torch.save(model.state_dict(), best_model_path)
        else:
            stagnation_counter += 1
            print(f"   -> No improvement for {stagnation_counter} epoch(s).")
            if stagnation_counter >= patience:
                print(f"\n[!] Early stopping triggered! Validation MSE has not improved for {patience} epochs.")
                break

    # Reload the absolute best weights before finishing
    print(f"\nTraining complete. Loading best model weights (Val MSE: {best_val_mse:.6f})...")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    
    return model

if __name__ == "__main__":
    trained_model = train()