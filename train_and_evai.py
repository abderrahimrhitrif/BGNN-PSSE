import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau # <-- ADDED IMPORT

from create_graph import PowerGridDataset
from model_prob import BayesianPowerGridGNN

def gaussian_nll(mu, target, var, eps=1e-6, reduction="mean"):
    var = var + eps  # stability
    loss = 0.5 * (torch.log(var) + (target - mu) ** 2 / var)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss

def train():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 32
    epochs = 300
    lr = 1e-3
    
    # --- PATIENCE SETTINGS ---
    # The scheduler patience MUST be lower than early stopping patience.
    # This allows the model to try lowering the learning rate a few times 
    # before finally giving up and stopping entirely.
    scheduler_patience = 5 
    early_stopping_patience = 30 

    lambda_var = 1e-4   # VERY IMPORTANT (small)

    train_dataset = PowerGridDataset(root='./', dataset_type='train')
    val_dataset = PowerGridDataset(root='./', dataset_type='validation')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    sample = train_dataset[0]

    model = BayesianPowerGridGNN(
        sample['variable'].x.shape[1],
        sample['factor'].x.shape[1],
        hidden_dim=32,
        readout_units=16,
        output_units=1
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # --- ADDED SCHEDULER ---
    # mode='min': we want val_mse to go down.
    # factor=0.5: cut the learning rate in half when plateauing.
    # min_lr: don't let the learning rate drop to absolute zero.
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=scheduler_patience, 
        min_lr=1e-6
    )

    best_val = float('inf')
    patience_counter = 0

    for epoch in range(epochs):

        # ===== TRAIN =====
        model.train()
        total_loss = 0
        total_nll = 0

        for batch in train_loader:
            batch = batch.to(device)
            target = batch['variable'].y.view(-1, 1)

            optimizer.zero_grad()

            out = model(batch.x_dict, batch.edge_index_dict)
            mu, var = out['mu'], out['var']

            mse = F.mse_loss(mu, target)

            # tiny regularization to prevent var explosion
            var_reg = var.mean()
            error = (target - mu).detach() ** 2
            var_loss = F.mse_loss(var, error)

            nll = gaussian_nll(mu, target, var)
            loss = mse + lambda_var * var_loss

            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += mse.item() * batch.num_graphs
            total_nll += nll.item() * batch.num_graphs

        # ===== VALID =====
        model.eval()
        val_mse = 0
        val_nll = 0
        nodes = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                target = batch['variable'].y.view(-1, 1)

                out = model(batch.x_dict, batch.edge_index_dict)
                mu = out['mu']
                var = out['var']
                nll = gaussian_nll(mu, target, var, reduction='sum')

                val_mse += F.mse_loss(mu, target, reduction='sum').item()
                val_nll += nll.item()
                nodes += mu.size(0)

        val_mse /= nodes
        val_nll /= nodes

        # --- STEP SCHEDULER ---
        # Pass the validation metric so the scheduler knows if we plateaued
        scheduler.step(val_mse)
        
        # Get the current learning rate to print it
        current_lr = optimizer.param_groups[0]['lr']

        print(
                f"Epoch {epoch} | "
                f"LR: {current_lr:.6f} | " # <-- ADDED TO PRINT
                f"Train MSE: {total_loss/len(train_dataset):.8f} | "
                f"Train NLL: {total_nll/len(train_dataset):.8f} | "
                f"Val MSE: {val_mse:.8f} | "
                f"Val NLL: {val_nll:.8f}"
            )

        # --- EARLY STOPPING LOGIC ---
        if val_mse < best_val:
            best_val = val_mse
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    model.load_state_dict(torch.load("best_model.pth"))
    return model


if __name__ == "__main__":
    model = train()