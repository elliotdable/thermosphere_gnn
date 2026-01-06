# GNN Pipeline: What Happens, in Order — and **Why**

This is a step-by-step walkthrough of your pipeline **as it runs in practice**, from `main.sh` → `run_main.py` → `data_processing.py` → `train.py` → `gnn_model.py`, including the rationale behind each step.

---

## ## 1) `main.sh` — submit & configure a training job

**What it does**

Submits a SLURM job on the GPU partition (A100), sets CPU/RAM/time limits, and points logs to `logs/`.

Starts an Apptainer/Singularity container, activates the `fpi_gnn` conda env, and **executes**:
  
  python `/mnt/fpi_gnn_training/src/GNN/run_main.py` ...args...

  Passes all run-time arguments (data paths, feature columns, lookback windows, model hyperparams, etc.).

Builds a `MODEL_NAME` using `dates/period/SLURM_JOB_ID`, sets up a per-job log folder, and monitors CPU/GPU every 5 min.

After training finishes, it moves the logs into `models/{MODEL_NAME}/logs/` and saves a copy of the script for provenance.

**Why**

Reproducible, containerized runs on the cluster; easy experiment tracking; automatic resource monitoring and log collation.

## 2) `run_main.py` — orchestrate end-to-end run

**What it does**

Parse CLI args (data paths, dates, target column, IMF/geomag feature lists, edge count, lookback windows, hidden dim, dropouts, epochs, batch size, LR, resample freq, profiling flag, patience, and activity period).

Create a model folder: `{save_path}/{model_name}`.

Build the PyG Data object by calling create_data_object(...) in data_processing.py with the parsed args.

Sanity-print graph stats (nodes/edges, feature dims, edge attribute stats, NaN/Inf counts).

Train by calling train_with_neighbor_sampling(...) from `train.py`. Collects train/val curves and final test metrics.

Persist run artifacts in the model folder:

`params.txt` with all arguments + final metrics

`loss_curve.png` (train/val loss vs. epoch)

Append a row to `results_history.csv` (under save_path/) with all run params + test metrics

**Why**

Keeps the entry point minimal and reproducible; central spot to route config → data build → train → log artifacts.

## 3) `data_processing.py` — build the graph & features
3.1 Read & time-filter data

Read CSVs: FPI (observations/target), IMF (solar wind), Geomagnetic indices.

Downcast floats to float32 (memory-friendly) and ensure datetime is parsed.

Optional activity filter (period): uses ap index thresholds (quiet: ap<15; active: 15≤ap<100; storm: ap≥100).

Filter FPI rows to [start_date, end_date].

**Why**: ensure consistent dtypes, avoid leakage across regimes, and trim to the run’s time window.

3.2 Resample IMF & add node features

IMF resampled to a uniform cadence (e.g., 30min) with .mean().interpolate().

For each FPI timestamp, concatenate:

Base/static & cyclic time features:

altitude, observation_latitude, observation_longitude

Hour-of-day (sin/cos), day-of-year (sin/cos)

11-year solar cycle proxy (sin/cos)

IMF lookback window (length = imf_hours) ending 30 min before the FPI time.

Geomag lookback window (length = geomag_days in days) ending at the FPI time.

Extraction of both time-series windows is parallelized with ThreadPoolExecutor.

**Why**: give the GNN both local context (position/time) and recent drivers (IMF/geomag history) aligned to plausible physical lags.

3.3 Train/val/test split (nodes)

Random split with seed 42: 70% / 15% / 15% over nodes. Masks are stored on the Data object.

**Why**: reproducible evaluation and to support train-only fitting of normalizers/scalers (no leakage).

3.4 Feature normalization (nodes)

StandardScaler fit only on train nodes, then applied to all node features.

The scaler for targets is also fit on train and saved as target_scaler.pt. Targets are saved standardized as y.

**Why**: stabilize optimization and ensure metrics in both standardized and physical units (via inverse transform later).

3.5 Edge construction (directed, past → present)

For each node i, find temporal neighbors within ±6 hours using a BallTree over timestamps.

Keep only past or same-time neighbors (dt>0 and ≤ 21600s) to enforce causality. (6 hours)

Among those past neighbors, compute spatial distance using great-circle (haversine) + altitude differences.

Select the top-K by spatial proximity (edge_count).

Create directed edges j → i (past → present) and attach edge attributes:

[spatial_distance, dt_seconds, dlat, dlon, dalt] (all numeric).

**Why**: messages flow from earlier observations that are spatially closest and temporally relevant to the current node.

3.6 Edge-attribute normalization

Fit a StandardScaler on train-related edges (by default edges with destination node in the train mask), then transform all edges.

Save as `edge_attr_scaler.pt`.

**Why**: lets the model see edge features on a consistent scale without train→val/test leakage.

3.7 Per-sample weights

Compute weight = 1 / (target_error^2 + 1e-3) and normalize to mean 1 (stored on Data.weight).

**Why**: encode observation uncertainty—intended for a weighted loss.

Note: In the current code, the custom weighted_mse_loss returns plain MSE (weights are not applied). If you want weighting active, switch it to use the weights (the commented code shows the intended logic).

## 4) `gnn_model.py` — the architecture (edge-conditioned GNN)

**SpatioTemporalGNN**

Two NNConv layers (edge-conditioned convolutions). Each layer computes a kernel as an MLP of edge attributes, enabling continuous, physics-aware message passing.

MLPs: edge_attr_dim → 64 → (in*hidden) and edge_attr_dim → 64 → (hidden*hidden)

Nonlinearities: ReLU.

Dropout: input (dropout_in) and hidden (dropout_hidden).

Head: Linear(hidden → hidden) + Linear(hidden → 1); Xavier init.

**Why**

NNConv lets the model condition messages on distance, time lag, and direction (dlat, dlon, dalt) instead of using fixed adjacency weights.

## 5) `train.py` — scalable training with neighbor sampling

**Loaders**

NeighborLoader over masks (train/val/test), sampling [10, 5] neighbors per hop; batch size = --batch_size.

Only seed nodes (the first batch.batch_size nodes) contribute to the loss/metrics.

**Training loop**

Device selection + CuDNN benchmark; AMP via GradScaler; optional torch.compile to JIT the model.

Optimizer: AdamW (lr=--learning_rate, weight_decay=5e-4).

Scheduler: ReduceLROnPlateau on validation loss (factor 0.5, patience 5).

Gradient clipping (1.0), mixed precision, and optional Profiler (TensorBoard traces) when --enable_profiling true.

Early stopping: track best validation loss; save model_best.pth; stop when no improvement for --patience epochs. Periodic time-based checkpoints also written.

**Metrics & logging**

Compute train/val/test loss (MSE) and RMSE (standardized).

Also compute physical-units RMSE by inverse-transforming with target_scaler.pt.

Final prints and return the histories + test metrics.

**Why**

Neighbor sampling scales to large graphs; AMP + AdamW + scheduler stabilize and speed up training; seed-node loss ensures correctness with sampled subgraphs.

## 6) Outputs, folders, and artifacts

Inside `{save_path}/{model_name}/` you’ll get:

`model_best.pth` (best weights) and time-based checkpoints during training

`edge_attr_scaler.pt`, target_scaler.pt

`params.txt`, loss_curve.png

`logs/` (SLURM/stdout, CPU/GPU utilization from main.sh)

Global: `{save_path}/results_history.csv` appends a row with all run parameters and final test metrics
