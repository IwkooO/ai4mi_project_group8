#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=seg_seed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=03:00:00
#SBATCH --output=out/full_pipeline_seed_%x_%j.log

# Usage: sbatch jobs/run_seed_pipeline.sh <SEED> [BASE_EXPERIMENT_NAME]

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "ERROR: Need at least the SEED argument" >&2
  exit 1
fi

SEED=$1
TIMESTAMP=$(date +%Y%m%d_%H%M)
BASE_EXPERIMENT_NAME=${2:-FullVIT${TIMESTAMP}}

module load 2023
module load Anaconda3/2023.07-2

cd "$HOME/ai4mi_project_group8" || { echo "Project directory not found"; exit 1; }
source $(conda info --base)/etc/profile.d/conda.sh
source activate ai4mi_env

# Optional installs (commented if already in env)
# pip install --quiet wandb python-dotenv

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_seed${SEED}"
RESULTS_DIR="$(pwd)/results/${EXPERIMENT_NAME}"
PRED_DIR="$(pwd)/data/seghtor_predictions_${EXPERIMENT_NAME}${SEED}"
GT_DIR="${PRED_DIR}/gt"
PRED_FOLDER="${PRED_DIR}/pred"

mkdir -p "$RESULTS_DIR" "$PRED_DIR" out

echo "=== Seed ${SEED} Job Started ==="
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Results:    ${RESULTS_DIR}"
echo "Pred Dir:   ${PRED_DIR}"

####################################
# Step 1: Train
####################################
python -O main.py \
  --dataset SEGTHOR_PREPROCESSED \
  --run_name "${EXPERIMENT_NAME}" \
  --mode full \
  --epochs 75 \
  --dest "${RESULTS_DIR}" \
  --gpu \
  --seed ${SEED}

python - <<'EOF'
import torch; torch.cuda.empty_cache(); torch.cuda.synchronize(); print('GPU memory cleared after training')
EOF

####################################
# Step 2: Stitch predictions
####################################
python stitch.py \
  --data_folder "${RESULTS_DIR}/best_epoch/val" \
  --dest_folder "${PRED_FOLDER}" \
  --num_classes 255 \
  --grp_regex "(Patient_\\d\\d)_\\d\\d\\d\\d" \
  --source_scan_pattern "data/segthor_train/train/{id_}/GT.nii.gz"

python - <<'EOF'
import torch; torch.cuda.empty_cache(); torch.cuda.synchronize(); print('GPU memory cleared after stitching')
EOF

####################################
# Step 3: Copy GTs for stitched patients
####################################
GT_SOURCE_DIR="$(pwd)/data/segthor_fixed/train"
mkdir -p "${GT_DIR}"
count=0
for pred in "${PRED_FOLDER}"/*.nii.gz; do
  [ -f "$pred" ] || continue
  patient=$(basename "$pred" .nii.gz)
  gt_path="${GT_SOURCE_DIR}/${patient}/GT.nii.gz"
  if [ -f "$gt_path" ]; then
    cp "$gt_path" "${GT_DIR}/${patient}.nii.gz"
    count=$((count+1))
  else
    echo "WARNING: Missing GT for $patient" >&2
  fi
done
echo "Copied $count ground truths"

####################################
# Step 4: Metrics
####################################
python compute_metrics.py \
  --ref_folder "${GT_DIR}" \
  --pred_folder "${PRED_FOLDER}" \
  --ref_extension .nii.gz \
  --pred_extension .nii.gz \
  --num_classes 5 \
  --metrics 3d_dice 3d_hd95 3d_hd 3d_assd 3d_nsd 3d_jaccard \
  --save_folder "${RESULTS_DIR}/metrics" \
  --overwrite

echo "=== Seed ${SEED} Completed Successfully ==="
