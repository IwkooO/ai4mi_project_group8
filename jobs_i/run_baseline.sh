#!/bin/bash

MODEL_NAME="segthor_baseline"

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
        # First job: submit normally with MODEL_NAME exported
        jid=$(sbatch --export=ALL,MODEL_NAME=$MODEL_NAME "$job" | awk '{print $4}')
    else
        # Dependent jobs: wait for previous to finish successfully
        jid=$(sbatch --dependency=afterok:$prev_jid --export=ALL,MODEL_NAME=$MODEL_NAME "$job" | awk '{print $4}')
    fi
    echo "Submitted $job as job $jid (MODEL_NAME=$MODEL_NAME)"
    prev_jid=$jid
done
