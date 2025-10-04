#!/bin/bash

MODEL_NAME="segthor_baseline"
DATA_NAME="SEGTHOR_CLEAN"

jobs=(
#   "jobs/install_env.job"
#   "jobs/create_clean_dataset.job"
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
    if [[ -z "$prev_jid" ]]; then
        jid=$(sbatch --export=ALL,MODEL_NAME=$MODEL_NAME,DATA_NAME=$DATA_NAME "$job" | awk '{print $4}')
    else
        jid=$(sbatch --dependency=afterok:$prev_jid --export=ALL,MODEL_NAME=$MODEL_NAME,DATA_NAME=$DATA_NAME "$job" | awk '{print $4}')
    fi
    echo "Submitted $job as job $jid (MODEL_NAME=$MODEL_NAME, DATA_NAME=$DATA_NAME)"
    prev_jid=$jid
done

echo "All jobs submitted."