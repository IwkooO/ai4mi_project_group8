#!/bin/bash

cd "$HOME/ai4mi_project_group8"

MODEL_NAME="segthor_aug"
DATA_NAME="SEGTHOR_CLEAN_AUG"

SRC_DIR="$HOME/ai4mi_project_group8/data/SEGTHOR_CLEAN"
DEST_DIR="$HOME/ai4mi_project_group8/data/${DATA_NAME}"

# mkdir -p "$DEST_DIR"
# cp -r "$SRC_DIR"/* "$DEST_DIR"/

jobs=(
#   "jobs/install_env.job"
#   "jobs/create_clean_dataset.job"
#   "jobs/augment_data.job"
  "jobs/train_model.job"
  "jobs/plot_perf.job"
  "jobs/make_plot_pdf.job"
  "jobs/stitch_pred.job"
  "jobs/calc_metrics.job"
  "jobs/inspect_metrics.job"
)

# ================================
# Submit jobs sequentially
# ================================
prev_jid=""

for job in "${jobs[@]}"; do
    job_basename=$(basename "$job")  # e.g. train_model.job
    log_dir="./slurm_out/$MODEL_NAME"
    mkdir -p "$log_dir"

    if [[ -z "$prev_jid" ]]; then
        jid=$(sbatch \
            --export=ALL,MODEL_NAME=$MODEL_NAME,DATA_NAME=$DATA_NAME \
            --output="$log_dir/${job_basename}.out" \
            "$job" | awk '{print $4}')
    else
        jid=$(sbatch \
            --dependency=afterok:$prev_jid \
            --export=ALL,MODEL_NAME=$MODEL_NAME,DATA_NAME=$DATA_NAME \
            --output="$log_dir/${job_basename}.out" \
            "$job" | awk '{print $4}')
    fi

    echo "Submitted $job as job $jid (MODEL_NAME=$MODEL_NAME, DATA_NAME=$DATA_NAME)"
    prev_jid=$jid
done

echo "All jobs submitted."