import os
import sys
import time
import signal
from xml.parsers.expat import model
import torch
from torch.nn import MSELoss
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.loader import NeighborLoader
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
from contextlib import nullcontext
import math
import numpy as np

from model import SpatioTemporalGAT

# --- Signal Handler ---
def create_signal_handler(model, model_path):
    def handler(signum):
        print(f"Received signal {signum}, saving checkpoint...")
        torch.save(model.state_dict(), os.path.join(model_path, 'checkpoint.pth'))
        sys.exit(0)
    return handler

def eval_loader_physical(loader, model, device, y_scaler):
    """
    Evaluate the model on a data loader using physical units.

    Args:
        loader: DataLoader providing batches of data.
        model: The trained GNN model.
        device: The device to run the model on (CPU or GPU).
        y_scaler: Scaler used to inverse transform the target variable. 

    Returns:
        rmse_phys: Root Mean Square Error in physical units.
    """
    model.eval()
    se_sum = 0.0
    n = 0
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)

            # Forward pass
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            seed_n = batch.batch_size

            # Get predictions and targets for the seed nodes only
            pred = out[:seed_n].detach().cpu().numpy().reshape(-1, 1)
            targ = batch.y[:seed_n].detach().cpu().numpy().reshape(-1, 1)

            # inverse transform to physical units
            pred_phys = y_scaler.inverse_transform(pred).squeeze(1)
            targ_phys = y_scaler.inverse_transform(targ).squeeze(1)

            se_sum += float(np.sum((pred_phys - targ_phys) ** 2))
            n += pred_phys.size
    rmse_phys = (se_sum / n) ** 0.5
    return rmse_phys

def train_with_neighbour_sampling_gat(data, in_channels, model_path,
                                 hidden_dim=32, dropout_in=0.1, dropout_hidden=0.3, epochs=1000,
                                 batch_size=512, learning_rate=1e-3, num_workers=10,
                                 enable_profiling='false', use_amp=True, patience=10):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    data = data.cpu()  # ensure data is on CPU before loading to device in batches

    y_scaler = torch.load(os.path.join(model_path, 'target_scaler.pt'))

    model = SpatioTemporalGAT(
        in_channels=in_channels,
        edge_attr_dim=data.edge_attr.shape[1],
        hidden_channels=hidden_dim,
        dropout_in=dropout_in,
        dropout_hidden=dropout_hidden
    ).to(device)

    # Enable higher matmul precision for better performance (A100-friendly)
    torch.set_float32_matmul_precision('medium')  # or 'medium' for more speed, less precision

    # Torch 2.x compile (if stable with your stack)
    if os.environ.get("DISABLE_COMPILE", "0") != "1":
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")

    # Optimizer, scheduler, scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = GradScaler(enabled=use_amp)

    signal.signal(signal.SIGTERM, create_signal_handler(model, model_path))
    signal.signal(signal.SIGINT, create_signal_handler(model, model_path))

    def make_loader(mask, is_train=True):
        input_nodes = mask.nonzero(as_tuple=True)[0]

        return NeighborLoader(
            data,
            input_nodes=input_nodes,
            num_neighbors=[15, 10],
            batch_size=batch_size,
            shuffle=is_train,
            pin_memory=True,       
            num_workers=num_workers,
            persistent_workers=num_workers > 0,  # keeps workers warm
            prefetch_factor=2 if num_workers > 0 else None
        )

    data.num_nodes = data.x.size(0)

    train_loader = make_loader(data.train_mask, is_train=True)
    val_loader = make_loader(data.val_mask, is_train=False)

    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")

    # Where to save traces (view in TensorBoard -> Profile)
    prof_logdir = os.path.join(model_path, "profiler")

    # Choose a profiler (real one or a no-op)
    if enable_profiling == 'true':
        prof_ctx = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
            schedule=schedule(wait=1, warmup=1, active=3, repeat=1),  # keeps overhead reasonable
            on_trace_ready=tensorboard_trace_handler(prof_logdir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
    else:
        prof_ctx = nullcontext()

    start_time = time.time()
    buffer = 300
    max_duration = 24 * 3600
    save_interval = 10800
    last_save_time = start_time
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(model_path, 'model_best.pth')

    mse_loss = MSELoss()

    def train_loop():
        nonlocal last_save_time, best_val_loss, patience_counter
        for epoch in range(1, epochs + 1):
            if time.time() - start_time > max_duration - buffer:
                torch.save(model.state_dict(), os.path.join(model_path, 'model_final.pth'))
                break

            model.train()
            total_loss = 0.0
            total_mse = 0.0

            for batch in train_loader:
                batch = batch.to(device, non_blocking=True)
                assert batch.edge_index.max().item() < batch.x.size(0)
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=use_amp):
                    E = batch.edge_index.size(1)
                    N = batch.x.size(0)

                    assert batch.edge_index.dtype == torch.long
                    assert batch.edge_index.min().item() >= 0
                    assert batch.edge_index.max().item() < N, (batch.edge_index.max().item(), N)

                    if batch.edge_attr is not None:
                        assert batch.edge_attr.size(0) == E, (batch.edge_attr.size(), E)
                        assert torch.isfinite(batch.edge_attr).all()
                    assert torch.isfinite(batch.x).all()
                    out = model(batch.x, batch.edge_index, batch.edge_attr)

                    # --- only use the seed nodes ---
                    seed_n = batch.batch_size  
                    out_seed = out[:seed_n]
                    y_seed = batch.y[:seed_n]
                    mse = mse_loss(out_seed, y_seed)

                scaler.scale(mse).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                # accumulate per-batch
                total_loss += float(mse.detach())
                total_mse += float(mse.detach())

                # advance profiler one training step
                if enable_profiling == 'true':
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    prof.step()

            avg_loss = total_loss / len(train_loader)
            avg_rmse = math.sqrt(total_mse / len(train_loader))
            train_losses.append(avg_loss)

            # Validation (optionally step the profiler here too if you want)
            model.eval()
            val_loss = 0.0
            val_mse = 0.0
            with torch.inference_mode():
                for batch in val_loader:
                    batch = batch.to(device, non_blocking=True)
                    out = model(batch.x, batch.edge_index, batch.edge_attr)

                    # --- only use the seed nodes ---
                    seed_n = batch.batch_size  
                    out_seed = out[:seed_n]
                    y_seed = batch.y[:seed_n]

                    mse = mse_loss(out_seed, y_seed)
                    val_loss += mse.item()
                    val_mse += mse.item()

            val_loss /= len(val_loader)
            val_rmse = (val_mse / len(val_loader)) ** 0.5
            val_losses.append(val_loss)

            scheduler.step(val_loss)

            val_rmse_phys = eval_loader_physical(val_loader, model, device, y_scaler)

            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.6f} | RMSE: {avg_rmse:.6f} | "
                  f"Val Loss: {val_loss:.6f} | Val RMSE: {val_rmse:.6f} | "
                  f"Val RMSE (Physical): {val_rmse_phys:.6f} K",
                  file=sys.stdout, flush=True)

            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                print(f"No improvement. Patience counter: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

            if time.time() - last_save_time > save_interval:
                ckpt_path = os.path.join(model_path, f'epoch{epoch}_checkpoint.pth')
                torch.save(model.state_dict(), ckpt_path)
                last_save_time = time.time()

    # --- Run, with or without profiler ---
    with prof_ctx as prof:
        train_loop()

    if enable_profiling:
        # Safe printout even on CPU-only or if CUDA metric missing
        sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
        try:
            print(prof.key_averages().table(sort_by=sort_key, row_limit=15))
            print(f"Profiler traces written to: {prof_logdir}")
        except Exception as e:
            print(f"Profiler summary unavailable: {e}")

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")

    test_start = time.time()
    test_loader = make_loader(data.test_mask)
    model.eval()
    test_loss = 0.0
    test_mse = 0.0
    with torch.inference_mode():
        for batch in test_loader:
            batch = batch.to(device, non_blocking=True)
            out = model(batch.x, batch.edge_index, batch.edge_attr)

            # --- only use the seed nodes ---
            seed_n = batch.batch_size  
            out_seed = out[:seed_n]
            y_seed = batch.y[:seed_n]

            mse = mse_loss(out_seed, y_seed)
            test_loss += mse.item()
            test_mse += mse.item()

    test_loss /= len(test_loader)
    test_rmse = (test_mse / len(test_loader)) ** 0.5

    test_rmse_phys = eval_loader_physical(test_loader, model, device, y_scaler)
    test_end = time.time()
    print(f"Test inference time: {test_end - test_start:.2f} seconds", file=sys.stdout, flush=True)
    print(f"Final Test Loss: {test_loss:.6f} | Final Test RMSE: {test_rmse:.6f} | "
          f"Final Test RMSE (Physical): {test_rmse_phys:.6f} K", file=sys.stdout, flush=True)

    # --- Cleanup old checkpoints ---
    for fname in os.listdir(model_path):
        if fname.endswith("_checkpoint.pth") or fname == "checkpoint.pth":
            fpath = os.path.join(model_path, fname)
            try:
                os.remove(fpath)
                print(f"Deleted checkpoint file: {fpath}")
            except Exception as e:
                print(f"Warning: Could not delete {fpath}: {e}")
                continue

    return train_losses, val_losses, test_loss, test_rmse, test_rmse_phys