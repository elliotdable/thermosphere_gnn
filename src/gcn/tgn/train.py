import os
import sys
import time
import math
import torch
import numpy as np
from torch.nn import MSELoss
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.loader import TemporalDataLoader

from model import SpatioTemporalTGN

# Utility: Physical RMSE evaluation
def eval_physical(loader, model, device, data, y_scaler):
    """
    Evaluate TGN model in physical units (Kelvin).
    Each event is predicted using y.
    """
    model.memory.reset_state()
    model.eval()
    se_sum = 0.0
    n = 0

    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            pred = model(
                batch.src,
                batch.dst,
                batch.t,
                data.x.to(device),
                batch.msg,
            )

            # event-level target: node target at dst index
            targ = data.y[batch.dst].to(device)

            # convert to CPU numpy
            pred_np = pred.cpu().numpy().reshape(-1, 1)
            targ_np = targ.cpu().numpy().reshape(-1, 1)

            # inverse-transform to physical units
            pred_phys = y_scaler.inverse_transform(pred_np).squeeze(1)
            targ_phys = y_scaler.inverse_transform(targ_np).squeeze(1)

            se_sum += float(np.sum((pred_phys - targ_phys) ** 2))
            n += pred_phys.size
    return (se_sum / n) ** 0.5


# Training function for TGN
def train_with_neighbour_sampling_tgn(data, model_path, hidden_dim=64, batch_size=256,
    learning_rate=1e-3, epochs=1000, patience=10, use_amp=True):
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print("Using device:", device)

    # Load scalers
    y_scaler = torch.load(os.path.join(model_path, "target_scaler.pt"))

    # Build model
    model = SpatioTemporalTGN(
        in_channels=data.x.shape[1],
        edge_attr_dim=data.msg.shape[1],
        hidden_channels=hidden_dim,
    ).to(device)

    # TGN memory must be initialized once at the start of training, num_nodes set to size of data object
    max_idx = int(torch.cat([data.src, data.dst]).max().item())
    num_nodes = max_idx + 1
    model.set_num_nodes(num_nodes)
    print("Memory initialized with", num_nodes, "nodes.")

    model.memory.reset_state()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    scaler = GradScaler(enabled=use_amp)

    # Split events for train / val / test using node masks
    # Each edge/event uses dst as the supervised node

    train_evt_mask = data.train_mask
    val_evt_mask   = data.val_mask
    test_evt_mask  = data.test_mask

    train_data = data[train_evt_mask]
    val_data   = data[val_evt_mask]
    test_data  = data[test_evt_mask]

    train_loader = TemporalDataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_loader   = TemporalDataLoader(val_data,   batch_size=batch_size, shuffle=False)
    test_loader  = TemporalDataLoader(test_data,  batch_size=batch_size, shuffle=False)

    print(f"Train events: {train_evt_mask.sum().item():,}")
    print(f"Val events:   {val_evt_mask.sum().item():,}")
    print(f"Test events:  {test_evt_mask.sum().item():,}")

    # Loss
    mse_loss = MSELoss()

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(model_path, "tgn_model_best.pth")

    train_losses = []
    val_losses   = []

    for epoch in range(1, epochs + 1):
        # Reset TGN memory *only at epoch boundaries*
        model.memory.reset_state()

        model.train()
        total_loss = 0.0
        total_mse = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                pred = model(
                    batch.src,
                    batch.dst,
                    batch.t,
                    data.x.to(device),
                    batch.msg,
                )
                y_true = data.y[batch.dst].to(device)
                loss = mse_loss(pred, y_true)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_mse  += loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_rmse = math.sqrt(total_mse / len(train_loader))
        train_losses.append(avg_loss)

        # Validation
        model.memory.reset_state()
        model.eval()
        val_loss = 0.0
        val_mse = 0.0

        with torch.inference_mode():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(
                    batch.src,
                    batch.dst,
                    batch.t,
                    data.x.to(device),
                    batch.msg,
                )

                y_true = data.y[batch.dst].to(device)
                loss = mse_loss(pred, y_true)

                val_loss += loss.item()
                val_mse  += loss.item()

        val_loss /= len(val_loader)
        val_rmse = math.sqrt(val_mse / len(val_loader))
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        val_rmse_phys = eval_physical(val_loader, model, device, data, y_scaler)

        print(
            f"Epoch {epoch:03d} "
            f"| Train Loss: {avg_loss:.6f} RMSE: {avg_rmse:.4f} "
            f"| Val Loss: {val_loss:.6f} RMSE: {val_rmse:.4f} "
            f"| Val Physical RMSE: {val_rmse_phys:.4f}",
            flush=True,
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # TEST
    print("\nLoading best model for testing...")
    model.load_state_dict(torch.load(best_model_path))
    model.memory.reset_state()
    model.eval()

    test_loss = 0.0
    test_mse  = 0.0

    with torch.inference_mode():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(
                batch.src,
                batch.dst,
                batch.t,
                data.x.to(device),
                batch.msg,
            )

            y_true = data.y[batch.dst].to(device)
            loss = mse_loss(pred, y_true)
            test_loss += loss.item()
            test_mse  += loss.item()

    test_loss /= len(test_loader)
    test_rmse = math.sqrt(test_mse / len(test_loader))
    model.memory.reset_state()
    test_rmse_phys = eval_physical(test_loader, model, device, data, y_scaler)

    print(f"\nTEST RESULTS:")
    print(f"Test Loss:          {test_loss:.6f}")
    print(f"Test RMSE:          {test_rmse:.6f}")
    print(f"Test RMSE Physical: {test_rmse_phys:.6f} K")

    return train_losses, val_losses, test_loss, test_rmse, test_rmse_phys
