import argparse
import os
import time
import torch
import matplotlib.pyplot as plt
import json
import pandas as pd

from data_processing import train_data_object

def count_total_nans_and_infs(data):
    total_nans = 0
    total_infs = 0
    for t in data.values():
        if isinstance(t, torch.Tensor):
            total_nans += torch.isnan(t).sum().item()
            total_infs += torch.isinf(t).sum().item()
    return total_nans, total_infs

def _as_serializable(v):
    """Convert lists/dicts to JSON strings so commas inside don't break the CSV."""
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, ensure_ascii=False)
    return v 

def write_run_to_csv(csv_path, args, test_loss, test_rmse, test_rmse_phys):
    """
    Append a training run’s results to a CSV, automatically handling new or changed columns.
    Old data is preserved, missing columns are filled with 'NA'.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "NA")

    # Define current expected parameters
    param_names = [
        "fpi_path", "imf_path", "geomag_path",
        "start_date", "end_date", "target_col",
        "save_path", "model_name",
        "imf_feature_cols", "geomag_feature_cols",
        "edge_count", "imf_hours", "geomag_days",
        "hidden_dim", "dropout_in", "dropout_hidden",
        "epochs", "batch_size", "num_neighbors_l1", "num_neighbors_l2", "learning_rate",
        "imf_resample", "num_workers", "enable_profiling",
        "patience", "period", "net_type", "use_pca"
    ]

    header = ["slurm_job_id"] + param_names + [
        "test_loss", "test_rmse", "test_rmse_phys"
    ]

    # Build the new row as a dict
    new_row = {
        "slurm_job_id": slurm_job_id,
        **{k: _as_serializable(getattr(args, k)) for k in param_names},
        "test_loss": float(test_loss),
        "test_rmse": float(test_rmse),
        "test_rmse_phys": float(test_rmse_phys)
    }

    # ---- Merge with existing CSV if it exists ----
    if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            df = pd.read_csv(csv_path, dtype=str)
        except Exception:
            print(f"Could not parse existing {csv_path}, rewriting it fresh.")
            df = pd.DataFrame(columns=header)

        # Ensure all expected columns exist in old data
        for col in header:
            if col not in df.columns:
                df[col] = "NA"

        # Keep any old columns not present in current header (for backwards compat)
        for col in df.columns:
            if col not in header:
                header.append(col)

        # Append new row and reorder columns
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)[header]
    else:
        # Fresh file
        df = pd.DataFrame([new_row], columns=header)

    # ---- Write final CSV ----
    df.to_csv(csv_path, index=False)
    print(f"Updated training history at {csv_path}")



# --- Main Execution ---
def main():
    execution_start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument('--fpi_path', '-fpi', type=str, required=True,
                        help='Path to the FPI dataset CSV file (must include lat/lon/alt/time/temperature).')

    parser.add_argument('--imf_path', '-imf', type=str, required=True,
                        help='Path to the IMF dataset CSV file (must include datetime and solar wind parameters).')

    parser.add_argument('--geomag_path', '-gmag', type=str, required=True,
                        help='Path to the geomagnetic indices dataset CSV file (must include datetime and indices like ap, AE, etc).')

    parser.add_argument('--start_date', '-sd', type=str, required=True,
                        help='Start date (YYYY-MM-DD) to filter the FPI dataset.')

    parser.add_argument('--end_date', '-ed', type=str, required=True,
                        help='End date (YYYY-MM-DD) to filter the FPI dataset.')

    parser.add_argument('--target_col', '-tc', type=str, required=True,
                        help='Name of the target column in the FPI dataset to predict.')

    parser.add_argument('--save_path', '-save', type=str, required=True,
                        help='Directory where the trained model and outputs will be saved.')

    parser.add_argument('--model_name', '-name', type=str, required=True,
                        help='Base name for saving the trained model files.')
    
    parser.add_argument("--imf_feature_cols", '-imf_f', nargs="+", required=True,
                        help="List of IMF feature column names to use as input features.")

    parser.add_argument("--geomag_feature_cols", '-gmag_f', nargs="+", required=True,
                        help="List of geomagnetic feature column names to use as input features.")

    parser.add_argument('--edge_count', '-etr', type=int, default=100,
                        help='Number of temporal neighbors to consider for each node.')

    parser.add_argument('--imf_hours', '-ih', type=float, default=3,
                        help='Number of hours of IMF data to include as time-series input per node.')

    parser.add_argument('--geomag_days', '-gd', type=float, default=1,
                        help='Number of past days of geomagnetic data to include as time-series input per node.')

    parser.add_argument('--hidden_dim', '-hd', type=int, default=32,
                        help='Number of hidden units in the GNN model layers.')

    parser.add_argument('--dropout_in', '-din', type=float, default=0.1,
                        help='Dropout rate applied in the GNN input layers.')
    
    parser.add_argument('--dropout_hidden', '-dhid', type=float, default=0.3,
                    help='Dropout rate applied in the GNN hidden layers.')

    parser.add_argument('--epochs', '-e', type=int, default=100,
                        help='Total number of training epochs.')

    parser.add_argument('--batch_size', '-bs', type=int, default=512,
                        help='Batch size used by NeighborLoader during training.')
    
    parser.add_argument('num_neighbors_l1', '-nn1', type=int, default=15,
                        help='Number of neighbors to sample at the first GNN layer.')   
    
    parser.add_argument('num_neighbors_l2', '-nn2', type=int, default=10,
                        help='Number of neighbors to sample at the second GNN layer.')

    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3,
                        help='Initial learning rate for the Adam optimizer.')

    parser.add_argument('--imf_resample', '-ir', type=str, default='15min',
                        help='Temporal resolution to resample IMF data (e.g., "15min", "1H").')

    parser.add_argument('--num_workers', '-nw', type=int, default=10,
                        help='Number of parallel workers for data loading and edge building.')

    parser.add_argument('--enable_profiling', '-ep', default=False, 
                        help='Enable PyTorch profiler')

    parser.add_argument('--patience', '-p', type=int, default=10,
                        help='Epochs before early stopping')
    
    parser.add_argument('--period', '-per', type=str, default='all',
                        help='Geomagnetic activity period to filter the data (quiet, active, storm).')
    
    parser.add_argument('--net_type', '-nt', type=str, default='gcn',
                        help='Type of model to use: "gcn" for SpatioTemporalGCN, "gat" for SpatioTemporalGAT.')
    
    parser.add_argument('--no_pca', action='store_false', dest='use_pca',
        help='Disable PCA for feature reduction')
    parser.set_defaults(use_pca=True)
    
    args = parser.parse_args()

    print(f'Starting {args.net_type} training with the following parameters:')
    print('Arguments:', args)
    print('IMF features:', args.imf_feature_cols)
    print('Geomag features:', args.geomag_feature_cols)

    model_path = os.path.join(args.save_path, args.model_name)
    os.makedirs(model_path, exist_ok=True)
    print(f'Created model directory at {model_path}')

    print('Creating data object...')
    start_time = time.time()    
    data = train_data_object(
        fpi_path=args.fpi_path,
        imf_path=args.imf_path,
        geomag_path=args.geomag_path,
        start_date=args.start_date,
        end_date=args.end_date,
        target_col=args.target_col,
        imf_feature_cols=args.imf_feature_cols,
        geomag_feature_cols=args.geomag_feature_cols,
        model_path=model_path,
        net_type=args.net_type,
        num_workers=args.num_workers,
        edge_count=args.edge_count,
        imf_hours=args.imf_hours,
        geomag_days=args.geomag_days,
        imf_resample=args.imf_resample,
        period=args.period,
        use_pca=args.use_pca
    )
    print(f"Took: {(time.time() - start_time)/3600:.2f} hours to create data object.")

    print(f'Data object: {data}')

    nans, infs = count_total_nans_and_infs(data)
    print(f"NaNs in Data: {nans}, Infs in Data: {infs}")

    print(f"# Nodes: {data.num_nodes}")
    print(f"Feature shape: {data.x.shape}")

    if args.net_type == 'gcn':
        print(f"# Edges: {data.num_edges}")
        print(f"Edge attr shape: {data.edge_attr.shape}")
        print("\nEdge attr stats:")
        print("Mean:", data.edge_attr.mean(0))
        print("Std:", data.edge_attr.std(0))
        print("Max:", data.edge_attr.max(0))
        print("Min:", data.edge_attr.min(0))
        print("NaNs:", torch.isnan(data.edge_attr).sum())

    print('Finished creating data object. Training model...')
    start_time = time.time()

    from train import train_with_neighbour_sampling_tgn
    train_losses, val_losses, test_loss, test_rmse, test_rmse_phys = train_with_neighbour_sampling_tgn(
        data=data,
        model_path=model_path,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        dropout_in=args.dropout_in,
        dropout_hidden=args.dropout_hidden,
        num_neighbors_l1=args.num_neighbors_l1,
        num_neighbors_l2=args.num_neighbors_l2,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        patience=args.patience,
        use_amp=True
    )

    print(f"Took: {(time.time() - start_time)/3600:.2f} hours to train model.")

    params_file = os.path.join(args.save_path, args.model_name, 'params.txt')

    with open(params_file, "w") as f:
        # Write training parameters
        f.write("Training Parameters\n")
        f.write("==================\n\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

        f.write("\n\nTest Results\n")
        f.write("============\n")
        f.write(f"Test Loss: {test_loss:.6f}\n")
        f.write(f"Test RMSE: {test_rmse:.6f}\n")
        f.write(f"Test RMSE Phys: {test_rmse_phys:.6f} K\n")

    print(f"Saved parameters and results to {params_file}.")

    print('Finished training model. Plotting loss curve...')

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(args.save_path, args.model_name, 'loss_curve.png'))
    print('Finished plotting loss curve. Training complete.')

    total_execution_time = time.time() - execution_start_time
    print(f"Total execution time: {total_execution_time/3600:.2f} hours")

    # ---- Call it after training completes ----
    results_csv = os.path.join(args.save_path, 'training_history.csv')  # change name/location if you prefer
    write_run_to_csv(
        csv_path=results_csv,
        args=args,
        test_loss=test_loss,
        test_rmse=test_rmse,
        test_rmse_phys=test_rmse_phys
    )
    print(f"Wrote/updated results CSV at: {results_csv}")

if __name__ == '__main__':
    main()