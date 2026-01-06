
import sys
import torch
from torch_geometric.data import Data
from sklearn.neighbors import BallTree
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import os
from itertools import repeat
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from torch_geometric.data import TemporalData

# --------------------------
# ---- HELPER FUNCTIONS ---- 
# --------------------------

# --- Haversine Distance Function ---
def haversine_distance(lat1, lon1, lat2, lon2, R=6371):
    '''
    Calculate the great-circle distance between two points on a sphere.
    
    Args:
        lat1, lon1: Latitude and longitude of the first point (in degrees)
        lat2, lon2: Latitude and longitude of the second point (in degrees)
    
    Returns:
        distance: Distance between the two points (in kilometers)
    '''

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# --- Feature Normalization ---
def normalize_features(features: np.ndarray, fit_mask=None, scaler=None):
    '''
    Normalize the features of a dataset.
    
    Args:
        features: Array of features to normalize
        fit_mask: Boolean mask to select rows for fitting the scaler
        scaler: Pre-fitted scaler object (if None, a new one will be created)
    
    Returns:
        normalized: Normalized features
        scaler: the scaler object (fitted if fit_mask is provided)
    '''
    if scaler is None:
        scaler = StandardScaler()
        if fit_mask is not None:
            scaler.fit(features[fit_mask])
        else:
            scaler.fit(features)
    features = scaler.transform(features).astype(np.float32)
    return features, scaler

def normalize_edge_attr(edge_index: torch.Tensor,
                        edge_attr: torch.Tensor,
                        train_mask: torch.Tensor,
                        fit_on: str = "dst",
                        scaler: StandardScaler | None = None):
    """
    Fit edge attribute scaler on a subset of edges, then transform all edges.

    Args:
        edge_index: [2, E] tensor
        edge_attr: [E, F] tensor (raw, unnormalized)
        train_mask: [N] bool tensor over nodes
        fit_on: "dst" (edges ending at train nodes),
                "both" (both ends are train), or
                "either" (either end is train)
        scaler: optional pre-fitted StandardScaler

    Returns:
        edge_attr_norm: [E, F] tensor (float32)
        scaler: fitted StandardScaler
        train_edge_mask: [E] bool tensor used for fitting
    """
    src = edge_index[0]
    dst = edge_index[1]

    if fit_on == "dst":
        train_edge_mask = train_mask[dst]
    elif fit_on == "both":
        train_edge_mask = train_mask[src] & train_mask[dst]
    elif fit_on == "either":
        train_edge_mask = train_mask[src] | train_mask[dst]
    else:
        raise ValueError("fit_on must be one of {'dst','both','either'}")

    edge_attr_np = edge_attr.detach().cpu().numpy().astype(np.float32)

    if scaler is None:
        if not torch.any(train_edge_mask):
            raise RuntimeError("No train edges found to fit the edge scaler.")
        scaler = StandardScaler()
        scaler.fit(edge_attr_np[train_edge_mask.cpu().numpy()])

    edge_attr_norm = scaler.transform(edge_attr_np).astype(np.float32)
    edge_attr_norm = torch.from_numpy(edge_attr_norm)

    return edge_attr_norm, scaler, train_edge_mask

def apply_pca(features,
              fit_mask=None,
              pca=None,
              n_components=0.95,
              whiten=False,
              random_state=42):
    """
    Fit PCA on a subset of rows (train only), then transform all rows.

    Args:
        features: array after standardization
        fit_mask: boolean mask for rows used to fit PCA (train split)
        pca: optional pre-fitted PCA
        n_components: e.0.95 for 95% variance, or an int for fixed components
        whiten: set True if your downstream model benefits from unit-variance comps
        random_state: for reproducibility

    Returns:
        features_pca: transformed features (float32)
        pca: fitted PCA object
    """
    if pca is None:
        pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
        if fit_mask is None:
            pca.fit(features)
        else:
            pca.fit(features[fit_mask])
    features_pca = pca.transform(features).astype(np.float32)
    return features_pca, pca

# --- Extract Lat, Lon, Alt Directly ---
def extract_latlonalt_positions(df):
    '''
    Extract the latitude, longitude, and altitude from a dataframe.
    
    Args:
        df: DataFrame containing latitude, longitude, and altitude
    
    Returns:
        positions: Array of latitude, longitude, and altitude
    '''
    return df[['observation_latitude', 'observation_longitude', 'altitude']].values

# --- Temporal Neighbor Search ---
def build_temporal_neighbors(unix_times, radius_seconds=21600):
    '''
    Build the temporal neighbors of a dataset.
    
    Args:
        unix_times: Array of unix times
        radius: Radius of the temporal neighbors
    
    Returns:
        indices: Array of temporal neighbors
    '''
    time_seconds = unix_times.reshape(-1, 1)
    tree = BallTree(time_seconds, metric='euclidean')
    indices = tree.query_radius(time_seconds, r=radius_seconds)
    for i, neighbors in enumerate(indices):
        indices[i] = neighbors[neighbors != i]
    return indices

def build_edges_single(i, neighbors, latlonalt_positions, unix_times, edge_count):
    '''
    Build the edges of a dataset on a single worker.
    
    Args:
        i: Index of the current node
        neighbors: Array of temporal neighbors
        latlonalt_positions: Array of latitude, longitude, and altitude
        unix_times: Array of unix times
        edge_count: number of edges to connect to (top-K by spatial proximity)
    
    Returns:
        edge_index_i: Array of edge indices
        edge_attr_i: Array of edge attributes
    '''
    if len(neighbors) == 0:
            return [], []

    # --- keep ONLY past (or same-time) neighbors relative to i ---
    dt_all = unix_times[i] - unix_times[neighbors]      # >0 means neighbor is in the past
    past_mask = (dt_all > 0) & (dt_all <= 21600) # only past neighbors within 6 hours
    if not np.any(past_mask):
        return [], []

    nbrs = neighbors[past_mask]
    dt   = dt_all[past_mask]                       # seconds, > 0 and <= 21600

    # --- gather positions ---
    lat_i, lon_i, alt_i = latlonalt_positions[i]
    lat_n, lon_n, alt_n = latlonalt_positions[nbrs].T

    # --- spatial metrics ---
    hav_dists   = haversine_distance(lat_i, lon_i, lat_n, lon_n)
    alt_diffs   = np.abs(alt_n - alt_i)
    spatial_dists = np.sqrt(hav_dists**2 + alt_diffs**2)

    # Directional deltas (dest - src) consistent with edge j->i
    dlat = lat_i - lat_n
    dlon = lon_i - lon_n
    dalt = alt_i - alt_n

    # --- choose top-K by spatial proximity (your original behavior) ---
    top_k_idx = np.argsort(spatial_dists)[:edge_count]

    nbrs_sel = nbrs[top_k_idx]
    dt_sel   = dt[top_k_idx]
    sd_sel   = spatial_dists[top_k_idx]
    dlat_sel = dlat[top_k_idx]
    dlon_sel = dlon[top_k_idx]
    dalt_sel = dalt[top_k_idx]

    # --- build edges as src=j (past) -> dst=i (present) ---
    edge_index_i = [[int(j), int(i)] for j in nbrs_sel]

    # Edge attributes: [spatial_distance, dt_seconds, dlat, dlon, dalt]
    # (dt is non-negative; consider normalizing later.)
    edge_attr_i = np.stack([sd_sel, dt_sel, dlat_sel, dlon_sel, dalt_sel], axis=1).tolist()

    return edge_index_i, edge_attr_i

def parallel_build_edges(indices, latlonalt_positions, unix_times, edge_count=100, num_workers=10):
    '''
    Build the edges of a dataset in parallel.
    
    Args:
        indices: Array of temporal neighbors
        latlonalt_positions: Array of latitude, longitude, and altitude
        unix_times: Array of unix times
        edge_count: Temporal radius of the edge
        num_workers: Number of workers to use

    Returns:
        edge_index: Array of edge indices
        edge_attr: Array of edge attributes
    '''
    
    args = [
        (i, neighbors, latlonalt_positions, unix_times, edge_count)
        for i, neighbors in enumerate(indices)
    ]

    edge_index_all = []
    edge_attr_all = []

    print("Building edges in parallel...", file=sys.stdout, flush=True)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = executor.map(lambda x: build_edges_single(*x), args)
        for edge_index_i, edge_attr_i in futures:
            edge_index_all.extend(edge_index_i)
            edge_attr_all.extend(edge_attr_i)
    print("Edges built.", file=sys.stdout, flush=True)

    # Tensors - no normalization and uni-directional edges as this means past affects present
    edge_index = torch.tensor(edge_index_all, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(np.array(edge_attr_all, dtype=np.float32), dtype=torch.float32)

    return edge_index, edge_attr

# --- Extract Time Series Features ---
def get_past_timeseries(df, current_time, feature_cols, window_hours, end_offset=pd.Timedelta(minutes=60)):
    """
    Slice a past time window and return the flattened values of selected columns.
    
    Args:
        df : DataFrame with a 'datetime' column and feature columns
        current_time : timestamp (from fpi_df) to anchor the lookback window
        feature_cols : list[str] columns to extract from df
        window_hours : float | int, how many hours to look back
        end_offset : pd.Timedelta, offset to end the window before current_time (default 60)
                        calculated roughly by the time-lag, propagation of solar wind to Earth and
                        atmospheric response delay
    Returns:
        1D np.array of the selected columns over the window, flattened row-major
    """
    end_time = current_time - pd.Timedelta(end_offset)
    start_time = end_time - pd.Timedelta(hours=window_hours) - pd.Timedelta(end_offset)
    mask = (df["datetime"] >= start_time) & (df["datetime"] < end_time)
    
    # Select only the requested feature columns
    window = df.loc[mask, feature_cols]
    # Ensure column order is exactly as provided
    window = window.reindex(columns=feature_cols)
    return window.to_numpy().ravel()  # flatten

# --- Combined Feature Builder ---
def get_node_features_with_timeseries(
    fpi_df,
    imf_df,
    geomag_df,
    imf_feature_cols,           # <-- columns to pull from imf_df
    geomag_feature_cols,        # <-- columns to pull from geomag_df
    num_workers=10,
    imf_hours=3,
    geomag_days=0.5
):
    """
    Build node features composed of:
      - base features from fpi_df (plus cyclic time features)
      - flattened IMF timeseries features
      - flattened geomagnetic timeseries features
    """
    # --- Datetime hygiene (make sure they are real datetimes and timezone-naive)
    for d in (fpi_df, imf_df, geomag_df):
        d["datetime"] = pd.to_datetime(d["datetime"], errors="raise")
        if pd.api.types.is_datetime64tz_dtype(d["datetime"]):
            d["datetime"] = d["datetime"].dt.tz_localize(None)

    # --- Add cyclic time features to fpi_df
    fpi_df = fpi_df.copy()
    fpi_df.loc[:, "hour_sin"] = np.sin(2 * np.pi * fpi_df["datetime"].dt.hour / 24)
    fpi_df.loc[:, "hour_cos"] = np.cos(2 * np.pi * fpi_df["datetime"].dt.hour / 24)
    fpi_df.loc[:, "dayofyear_sin"] = np.sin(2 * np.pi * fpi_df["datetime"].dt.dayofyear / 365)
    fpi_df.loc[:, "dayofyear_cos"] = np.cos(2 * np.pi * fpi_df["datetime"].dt.dayofyear / 365)

    # 11-year solar cycle proxy
    year_float = fpi_df["datetime"].dt.year + fpi_df["datetime"].dt.dayofyear / 365.25
    fpi_df.loc[:, "solar_cycle_sin"] = np.sin(2 * np.pi * year_float / 11)
    fpi_df.loc[:, "solar_cycle_cos"] = np.cos(2 * np.pi * year_float / 11)

    base_feature_cols = ['altitude', 'observation_latitude',  'observation_longitude'] 
    time_features = ["hour_sin", "hour_cos", "dayofyear_sin", "dayofyear_cos", "solar_cycle_sin", "solar_cycle_cos"]
    all_base_cols = base_feature_cols + time_features

    n_base = len(base_feature_cols)
    n_time = len(time_features)

    # --------------------------------------------------
    # EXTRACT TIME-SERIES FEATURES
    # --------------------------------------------------

    # --- Helper to fetch both windows for a given fpi timestamp
    def get_both_timeseries(t, imf_cols, geomag_cols):
        imf_feat = get_past_timeseries(
            imf_df, t, imf_cols, window_hours=imf_hours, end_offset=pd.Timedelta(minutes=30)
        )
        geomag_feat = get_past_timeseries(
            geomag_df, t, geomag_cols, window_hours=geomag_days * 24, end_offset=pd.Timedelta(minutes=0)
        )
        return imf_feat, geomag_feat

    # --- Parallel extraction of time series features
    print("Extracting time series features in parallel...", file=sys.stdout, flush=True)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            executor.map(
                get_both_timeseries,
                fpi_df["datetime"],          # iterable arg
                repeat(imf_feature_cols),    # constant arg
                repeat(geomag_feature_cols)  # constant arg
            )
        )
    print("Time series features extracted.", file=sys.stdout, flush=True)
    imf_ts_list, geomag_ts_list = zip(*results)  # tuples of 1D arrays

    # --------------------------------------------------
    # BUILD FEATURE MATRICES
    # --------------------------------------------------

    base_features = fpi_df[all_base_cols].to_numpy(dtype=np.float32)

    imf_array    = np.vstack(imf_ts_list).astype(np.float32)
    geomag_array = np.vstack(geomag_ts_list).astype(np.float32)

    features = np.hstack([base_features, imf_array, geomag_array])

    # --------------------------------------------------
    # COMPUTE FEATURE INDICES (NOW SAFE)
    # --------------------------------------------------

    n_imf    = imf_array.shape[1]
    n_geomag = geomag_array.shape[1]
    total_features = features.shape[1]

    idx_base   = slice(0, n_base)
    idx_time   = slice(n_base, n_base + n_time)
    idx_imf    = slice(n_base + n_time, n_base + n_time + n_imf)
    idx_geomag = slice(n_base + n_time + n_imf, total_features)

    # (Optional sanity checks)
    assert features.shape[1] == total_features
    assert idx_geomag.stop == total_features

    # --------------------------------------------------
    # RETURN FEATURES + METADATA
    # --------------------------------------------------
    return features, {
        "idx_base": idx_base,
        "idx_time": idx_time,
        "idx_imf": idx_imf,
        "idx_geomag": idx_geomag,
        "n_base": n_base,
        "n_time": n_time,
        "n_imf": n_imf,
        "n_geomag": n_geomag,
    }

# ----------------------
# ---- DATA OBJECTS ---- 
# ----------------------

def train_data_object(fpi_path, imf_path, geomag_path, start_date, end_date, imf_feature_cols,
    geomag_feature_cols, model_path, net_type, num_workers=10, target_col='temperature', 
                       edge_count=100, imf_hours=3, geomag_days=1, imf_resample='30min', period='all', use_pca=True):
    '''
    Create a data object from a dataframe.
    
    Args:
        df: DataFrame containing the timeseries
        imf_df: DataFrame containing the IMF timeseries
        geomag_df: DataFrame containing the geomagnetic timeseries
        feature_cols: List of feature columns 
        target_col: Target column
        edge_count: Radius of the edge
        imf_hours: Number of hours to look back for IMF timeseries
        geomag_days: Number of days to look back for geomagnetic timeseries
        imf_resample: Resampling frequency for IMF data (e.g., '30min')
        model_path: Path to save the model
        net_type: Type of network ('gcn' or 'tgn')
        num_workers: Number of workers to use
        period: Geomagnetic activity period to filter the data ('quiet', 'active', 'storm', 'all')
        use_pca: Whether to apply PCA to the features
    
    Returns:
        data: Data object
    '''
    # -----------------------------
    # DATA READING & FILTERING
    # -----------------------------

    # Read & preprocess CSVs
    print("Reading datafiles to CSVs...", file=sys.stdout, flush=True)

    fpi_df = pd.read_csv(fpi_path)
    imf_df = pd.read_csv(imf_path)
    geomag_df = pd.read_csv(geomag_path)

    # downcast float64 → float32
    for df in [fpi_df, imf_df, geomag_df]:
        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype(np.float32)

    print("Columns downcasted to float32.", file=sys.stdout, flush=True)

    fpi_df['datetime'] = pd.to_datetime(fpi_df['datetime'], errors='raise')
    imf_df['datetime'] = pd.to_datetime(imf_df['datetime'], errors='raise')
    geomag_df['datetime'] = pd.to_datetime(geomag_df['datetime'], errors='raise')

    # Filter by geomagnetic activity
    if period != 'all':
        if period == 'quiet':
            fpi_df = fpi_df[fpi_df['ap'] < 15]
        elif period == 'active':
            fpi_df = fpi_df[(fpi_df['ap'] >= 15) & (fpi_df['ap'] < 100)]
        elif period == 'storm':
            fpi_df = fpi_df[fpi_df['ap'] >= 100]
        else:
            raise ValueError("period must be one of: quiet, active, storm, all")

        print(f"Filtered data to {period} periods.", file=sys.stdout, flush=True)
    else:
        print("Using all geomagnetic activity.", file=sys.stdout, flush=True)

    # Date filtering
    fpi_df_filtered = fpi_df[
        (fpi_df['datetime'] >= pd.to_datetime(start_date)) &
        (fpi_df['datetime'] <= pd.to_datetime(end_date))
    ]
    del fpi_df
    print(f"Filtered date range: {start_date} → {end_date}", file=sys.stdout, flush=True)
    print(len(fpi_df_filtered), "data points after filtering.", file=sys.stdout, flush=True)

    # Resample IMF + build node features
    imf_df = (
        imf_df.set_index("datetime")
        .resample(imf_resample).mean()
        .interpolate()
        .reset_index()
    )
    float_cols = imf_df.select_dtypes(include=['float64']).columns
    imf_df[float_cols] = imf_df[float_cols].astype(np.float32)

    print("Resampled IMF data.", file=sys.stdout, flush=True)

    features, feature_layout = get_node_features_with_timeseries(
        fpi_df_filtered, imf_df, geomag_df,
        imf_feature_cols, geomag_feature_cols,
        num_workers, imf_hours, geomag_days
    )
    print("Attached node features.", file=sys.stdout, flush=True)

    # Position + temporal neighbors
    latlonalt_positions = extract_latlonalt_positions(fpi_df_filtered)
    times = pd.to_datetime(fpi_df_filtered["datetime"])
    unix_times = times.values.astype("datetime64[s]").astype(np.int64)

    temporal_window_seconds = 21600
    temporal_neighbors = build_temporal_neighbors(
        unix_times, radius_seconds=temporal_window_seconds
    )

    edge_index, edge_attr = parallel_build_edges(
        temporal_neighbors, latlonalt_positions, unix_times,
        edge_count=edge_count,
        num_workers=num_workers
    )

    print("Built edges.", file=sys.stdout, flush=True)

    # ----------------------------------------
    # TRAIN/VAL/TEST SPLIT + NORMALIZATION
    # ----------------------------------------

    # Train/val/test split
    num_nodes = features.shape[0]
    torch.manual_seed(42)
    indices = torch.randperm(num_nodes)

    train_size = int(0.7 * num_nodes)
    val_size = int(0.15 * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask  = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size+val_size]] = True
    test_mask[indices[train_size+val_size:]] = True

    train_np_mask = train_mask.cpu().numpy()

    # ----------------------------------
    # FEATURE NORMALIZATION (TRAIN ONLY)
    # ----------------------------------
    layout = feature_layout

    idx_base   = layout["idx_base"]
    idx_time   = layout["idx_time"]      # unused, but kept for clarity
    idx_imf    = layout["idx_imf"]
    idx_geomag = layout["idx_geomag"]

    train_feats = features[train_np_mask]

    # Create scalers
    base_scaler   = MinMaxScaler()
    imf_scaler    = StandardScaler()
    geomag_scaler = StandardScaler()

    # Fit scalers ONLY on train nodes
    base_scaler.fit(train_feats[:, idx_base])
    imf_scaler.fit(train_feats[:, idx_imf])
    geomag_scaler.fit(train_feats[:, idx_geomag])

    # Transform full dataset
    features[:, idx_base]   = base_scaler.transform(features[:, idx_base])
    features[:, idx_imf]    = imf_scaler.transform(features[:, idx_imf])
    features[:, idx_geomag] = geomag_scaler.transform(features[:, idx_geomag])
    # time features untouched

    # Save scalers + layout
    torch.save(
        {
            "base_scaler": base_scaler,
            "imf_scaler": imf_scaler,
            "geomag_scaler": geomag_scaler,
            "feature_layout": layout,
        },
        os.path.join(model_path, "feature_scalers.pt")
    )

    print("Applied selective feature normalization.", flush=True)

    features_before_pca = features.shape[1]

    if use_pca:
        features, pca = apply_pca(
            features,
            fit_mask=train_np_mask,
            n_components=0.999,
            whiten=False,
            random_state=42
        )
        torch.save(pca, os.path.join(model_path, "pca.pt"))
        print(f"Applied PCA: {features_before_pca} features → {features.shape[1]} features", file=sys.stdout, flush=True)

    else:
        pca = None
        print("Skipped PCA step, keeping all features.", file=sys.stdout, flush=True)

    # Edge normalization
    edge_attr, edge_scaler, train_edge_mask = normalize_edge_attr(
        edge_index=edge_index,
        edge_attr=edge_attr,
        train_mask=train_mask,
        fit_on="dst"
    )
    torch.save(edge_scaler, os.path.join(model_path, "edge_attr_scaler.pt"))

    # Prepare tensors
    x = torch.tensor(features, dtype=torch.float32)

    y_np = fpi_df_filtered[target_col].values.astype(np.float32)
    y_scaler = StandardScaler()
    y_scaler.fit(y_np[train_np_mask].reshape(-1, 1))
    y = torch.from_numpy(
        y_scaler.transform(y_np.reshape(-1, 1)).astype(np.float32).squeeze()
    )
    torch.save(y_scaler, os.path.join(model_path, "target_scaler.pt"))

    #target_error = torch.tensor(
     #   fpi_df_filtered[f"{target_col}_error"].values, dtype=torch.float32)

    src = edge_index[0]
    dst = edge_index[1]

    # TGN TemporalData requires timestamps for each edge
    dst_t = (unix_times[dst.cpu().numpy()] - unix_times.min()) / 3600.0
    edge_t = torch.from_numpy(dst_t).float()

    # extra safety – scale to [0, 1]
    edge_t = edge_t - edge_t.min()
    if edge_t.max() > 0:
        edge_t = edge_t / edge_t.max()
        
    data = TemporalData(
        src=src,
        dst=dst,
        t=edge_t,
        msg=edge_attr,
        x=x,
        y=y
    )

    # Sort events chronologically
    perm = data.t.argsort()
    data.src = data.src[perm]
    data.dst = data.dst[perm]
    data.t   = data.t[perm]
    data.msg = data.msg[perm]

    train_event_mask = train_mask[data.dst]   # boolean mask, length = num_events
    val_event_mask   = val_mask[data.dst]
    test_event_mask  = test_mask[data.dst]

    data.train_mask = train_event_mask
    data.val_mask   = val_event_mask
    data.test_mask  = test_event_mask

    print("Created TGN TemporalData object.", file=sys.stdout, flush=True)
    print("t:", data.t)
    print("shape:", data.t.shape)
    print("dtype:", data.t.dtype)
    print("Nans:", torch.isnan(data.t).any())
    print("Infs:", torch.isinf(data.t).any())
    print("max:", data.t.max().item(), "min:", data.t.min().item())
    print('edge_t.min(), edge_t.max()', edge_t.min(), edge_t.max())

    for name in [
        "fpi_df_filtered", "imf_df", "geomag_df",
        "features", "latlonalt_positions", "times", "unix_times",
        "train_feats", "y_np",
        "train_np_mask", "train_mask", "val_mask", "test_mask",
        "indices", "num_nodes",
        "base_scaler", "imf_scaler", "geomag_scaler",
        "layout", "idx_base", "idx_time", "idx_imf", "idx_geomag",
        "pca", "features_before_pca",
        "temporal_neighbors", "edge_index", "edge_attr",
        "src", "dst", "edge_scaler", "train_edge_mask",
        "x", "y",
        "y_scaler", "target_error"
    ]:
        if name in locals():
            del locals()[name]

    return data

def inference_data_object( fpi_df,imf_path,geomag_path, start_date, end_date, imf_feature_cols,
    geomag_feature_cols,model_path, imf_hours, geomag_days, imf_resample, use_pca, num_workers=10, 
    target_col="temperature", net_type="gcn"
):
    '''
    Create a data object from a dataframe.
    
    Args:
        df: DataFrame containing the timeseries
        imf_df: DataFrame containing the IMF timeseries
        geomag_df: DataFrame containing the geomagnetic timeseries
        feature_cols: List of feature columns 
        target_col: Target column
        imf_hours: Number of hours to look back for IMF timeseries
        geomag_days: Number of days to look back for geomagnetic timeseries
        imf_resample: Resampling frequency for IMF data (e.g., '30min')
        model_path: Path to save the model
        use_pca: Whether to use PCA for feature reduction
        net_type: Type of network ('gcn' or 'tgn')
        num_workers: Number of workers to use
    
    Returns:
        data: Data object for inference
    '''

    print("Running inference data builder...", flush=True)

    # Load scalers
    scaler_bundle = torch.load(os.path.join(model_path, "feature_scalers.pt"))

    base_scaler   = scaler_bundle["base_scaler"]
    imf_scaler    = scaler_bundle["imf_scaler"]
    geomag_scaler = scaler_bundle["geomag_scaler"]
    layout        = scaler_bundle["feature_layout"]

    idx_base   = layout["idx_base"]
    idx_time   = layout["idx_time"]      # unused, kept for clarity
    idx_imf    = layout["idx_imf"]
    idx_geomag = layout["idx_geomag"]

    edge_attr_scaler = torch.load(os.path.join(model_path, "edge_attr_scaler.pt"))
    target_scaler = torch.load(os.path.join(model_path, "target_scaler.pt"))

    if use_pca:
        pca = torch.load(os.path.join(model_path, "pca.pt"))
        print("Loaded scalers + PCA.", flush=True)
    else:
        pca = None
        print("Loaded scalers + PCA skipped as was not used in training.", flush=True)

    # Read auxiliary time series
    imf_df = pd.read_csv(imf_path)
    geomag_df = pd.read_csv(geomag_path)

    for df in [imf_df, geomag_df]:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="raise")

    # Resample IMF
    imf_df = (
        imf_df.set_index("datetime")
              .resample(imf_resample).mean()
              .interpolate()
              .reset_index()
    )

    # Filter FPI input times
    fpi_df["datetime"] = pd.to_datetime(fpi_df["datetime"])
    fpi_df = fpi_df[
        (fpi_df["datetime"] >= pd.to_datetime(start_date)) &
        (fpi_df["datetime"] <= pd.to_datetime(end_date))
    ]

    print(f"Inference window: {start_date} → {end_date}", flush=True)

    # Build node features (before scaling)
    raw_features, _ = get_node_features_with_timeseries(
        fpi_df,
        imf_df,
        geomag_df,
        imf_feature_cols,
        geomag_feature_cols,
        num_workers,
        imf_hours,
        geomag_days
    )
    print("Computed raw inference features.", flush=True)

    # ----------------------------------
    # APPLY TRAINED FEATURE SCALERS
    # ----------------------------------

    # Base features - MinMax
    raw_features[:, idx_base] = base_scaler.transform(
        raw_features[:, idx_base]
    )
    # IMF - Standard
    raw_features[:, idx_imf] = imf_scaler.transform(
        raw_features[:, idx_imf]
    )
    # Geomag - Standard
    raw_features[:, idx_geomag] = geomag_scaler.transform(
        raw_features[:, idx_geomag]
    )

    if use_pca:
        features = pca.transform(raw_features)
    else:
        features = raw_features

    print(f"Feature dim: {raw_features.shape[1]} - PCA - {features.shape[1]}", flush=True)

    x = torch.tensor(features, dtype=torch.float32)

    # Build edges + timestamps
    latlonalt_positions = extract_latlonalt_positions(fpi_df)
    times = pd.to_datetime(fpi_df["datetime"])
    unix_times = times.values.astype("datetime64[s]").astype(np.int64)

    temporal_neighbors = build_temporal_neighbors(
        unix_times, radius_seconds=21600
    )

    edge_index, edge_attr = parallel_build_edges(
        temporal_neighbors,
        latlonalt_positions,
        unix_times,
        edge_count=100,
        num_workers=num_workers
    )

    # Scale edge attributes using stored scaler  
    edge_attr = edge_attr_scaler.transform(edge_attr)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    src = edge_index[0]
    dst = edge_index[1]

    # Prepare the target (scaled with stored scaler)
    if target_col in fpi_df.columns:
        y_raw = fpi_df[target_col].values.astype(np.float32).reshape(-1, 1)
        y_scaled = target_scaler.transform(y_raw).astype(np.float32).squeeze()
        y = torch.tensor(y_scaled, dtype=torch.float32)
    else:
        # During inference, target might not exist → use placeholder
        y = torch.zeros(x.size(0), dtype=torch.float32)

    # Construct PyG object based on net_type
    if net_type == "gcn":
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y
        )
        # inference → no masks
        print("Created GCN inference Data object.", flush=True)

    elif net_type == "gat":
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y
        )
        # inference → no masks
        print("Created GAT inference Data object.", flush=True)
    
    elif net_type == "tgn":
        # build TGN timestamp differences
        src_t = unix_times[src.cpu().numpy()]
        dst_t = unix_times[dst.cpu().numpy()]
        edge_t = torch.from_numpy(dst_t - src_t).float().clamp(min=0)

        data = TemporalData(
            src=src,
            dst=dst,
            t=edge_t,
            msg=edge_attr,
            x=x,
            y=y
        )

        # sort chronologically
        perm = data.t.argsort()
        for k in ["src", "dst", "t", "msg"]:
            setattr(data, k, getattr(data, k)[perm])

        print("Created TGN inference TemporalData object.", flush=True)
        print("Time stats → min:", data.t.min().item(), 
              "max:", data.t.max().item(),
              "NaN:", torch.isnan(data.t).any())

    else:
        raise ValueError("net_type must be 'gcn' or 'tgn'.")
    
    for name in [
        "imf_df","geomag_df","raw_features","features",
        "y_raw","y_scaled","edge_attr_scaler","target_scaler","pca",
        "latlonalt_positions","times","unix_times","temporal_neighbors",
        "edge_index","edge_attr","x","y","src","dst",
        "edge_t","perm","k",
        "scaler_bundle","base_scaler","imf_scaler","geomag_scaler","layout"
    ]:
        if name in locals():
            del locals()[name]

    return data
