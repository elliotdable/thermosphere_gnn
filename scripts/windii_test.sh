#!/bin/bash
#SBATCH --job-name=windii_test
#SBATCH -p GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "Job started on $(hostname) at $(date)"

# --- Paths ---
CONTAINER="$HOME/fpi_gnn_training/ubuntu_anaconda_cuda12.sif"
ENV_NAME="fpi_gnn"  # Name of your conda environment
CONDA_ROOT="/miniforge_apptainer"
TRAINING_DIR="$HOME/fpi_gnn_training"

# --- Parameters ---
START_DATE="1992-01-01"
END_DATE="1992-01-06"

IMF_FEATURE_COLS="bx_gse_gsm al_index"
# OPTIONS:
# ['datetime', 'bx_gse_gsm', 'by_gse', 'bz_gse',
# 'b_vector_rms_sd', 'vx_kms', 'vy_kms', 'vz_kms', 'proton_density',
# 'proton_temperature', 'al_index', 'au_index']

GEOMAG_FEATURE_COLS="ap30 f30 dst"
# OPTIONS:
# 'datetime', 'hp30', 'ap30', 'dst'
#'f30', 'f15', 'f10_7', 'f8','f3_2'

IMF_HOURS=1
GEOMAG_DAYS=0.5
EPOCHS=3
LEARNING_RATE=0.0001
EDGE_COUNT=10
TARGET_COL="temperature"
HIDDEN_DIM=32
DROPOUT_IN=0.1
DROPOUT_HIDDEN=0.25
BATCH_SIZE=128
IMF_RESAMPLE="30min"
PATIENCE=25
PROFILING="false" 
PERIOD="all" # "quiet", "active", "storm" or "all"
NET_TYPE="gcn" # "gcn" or "tgn"
USE_PCA="false" # "true" or "false"

# FYI there is an annoying quirk to the script where I have to mount it because of the container, and this
# means I have to use the full path within the container, which is different to the host path.
# So I bind the whole training directory to /mnt/fpi_gnn_training in the container.
# Then within the container I have to use /mnt/fpi_gnn_training/...

# --- Create model name based on parameters, the python script makes this directory ---
MODEL_NAME="windii_test"
MODEL_NAME="${MODEL_NAME//-/_}" # change dashes to underscores for file naming

# --- Create log directory for this job ---
LOG_DIR="$TRAINING_DIR/logs/$SLURM_JOB_ID"
mkdir -p "$LOG_DIR"
echo "Logging to $LOG_DIR"

PCA_FLAG=""
if [[ "$USE_PCA" == "false" ]]; then
  PCA_FLAG="--no_pca"
fi
echo "PCA flag set to: $PCA_FLAG"

# --- Run inside container ---
apptainer exec --nv \
  --env CUDA_LAUNCH_BLOCKING=1 \
  --bind "$TRAINING_DIR:/mnt/fpi_gnn_training" \
  --bind "$HOME/miniforge_apptainer:/miniforge_apptainer" \
  "$CONTAINER" \
  bash -c "
    source $CONDA_ROOT/etc/profile.d/conda.sh &&
    conda activate $ENV_NAME &&
    echo 'Conda environment activated'

    # Run training in background and get PID
    python /mnt/fpi_gnn_training/src/${NET_TYPE}/run_main.py \      
      --fpi_path /mnt/fpi_gnn_training/data/sampled_windii_temps_200_300km_cleaned.csv \
      --imf_path /mnt/fpi_gnn_training/data/omni_imf_data_1990_2025_cleaned.csv \
      --geomag_path /mnt/fpi_gnn_training/data/solar_geomag_dst_data.csv \
      --start_date $START_DATE \
      --end_date $END_DATE \
      --target_col $TARGET_COL \
      --save_path /mnt/fpi_gnn_training/models/ \
      --model_name $MODEL_NAME \
      --imf_feature_cols $IMF_FEATURE_COLS \
      --geomag_feature_cols $GEOMAG_FEATURE_COLS \
      --edge_count $EDGE_COUNT \
      --imf_hours $IMF_HOURS \
      --geomag_days $GEOMAG_DAYS \
      --hidden_dim $HIDDEN_DIM \
      --dropout_in $DROPOUT_IN \
      --dropout_hidden $DROPOUT_HIDDEN \
      --epochs $EPOCHS \
      --batch_size $BATCH_SIZE \
      --learning_rate $LEARNING_RATE \
      --imf_resample $IMF_RESAMPLE \
      --patience $PATIENCE \
      --enable_profiling $PROFILING \
      --period $PERIOD \
      --net_type $NET_TYPE \
        $PCA_FLAG &

    TRAIN_PID=\$!
    echo \"Training script running with PID \$TRAIN_PID\"

    # Start CPU and GPU monitoring for this PID
    while kill -0 \$TRAIN_PID 2>/dev/null; do
      TIMESTAMP=\$(date '+%Y-%m-%d %H:%M:%S')

      # CPU usage log (single line per sample)
      CPU_LINE=\$(ps -p \$TRAIN_PID -o pid,%cpu,%mem --no-headers)
      echo \"\$TIMESTAMP, \$CPU_LINE\" >> /mnt/fpi_gnn_training/logs/$SLURM_JOB_ID/cpu_mem_log.txt

      # GPU usage log
      nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits | grep \$TRAIN_PID >> /mnt/fpi_gnn_training/logs/$SLURM_JOB_ID/gpu_mem_log.csv

      sleep 300
    done

    wait \$TRAIN_PID
"

# --- Define the final log destination ---
FINAL_LOG_DIR="$TRAINING_DIR/models/$MODEL_NAME/logs"
mkdir -p "$FINAL_LOG_DIR"

# --- Move SLURM stdout/err into the job log folder ---
mv logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out "$LOG_DIR/job.out" 2>/dev/null
mv logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err "$LOG_DIR/job.err" 2>/dev/null

# --- Move SLURM stdout/err and monitoring logs ---
mv "$LOG_DIR"/* "$FINAL_LOG_DIR"/ 2>/dev/null

# --- Delete the initial log directory if it exists ---
rm -rf "$LOG_DIR"

# --- Save a copy of this script into the final log directory ---
cp "$0" "$TRAINING_DIR/models/$MODEL_NAME/script.sh"

echo "Logs moved to $FINAL_LOG_DIR"
echo "Job finished at $(date)"

