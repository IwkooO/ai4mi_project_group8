#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=full_pipeline_parallel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=03:00:00
#SBATCH --output=out/full_pipeline_parallel_3.log
#SBATCH --error=err/full_pipeline_parallel_3.err

module load 2023
module load Anaconda3/2023.07-2

cd $HOME/ai4mi_project_group8/
source $(conda info --base)/etc/profile.d/conda.sh
source activate ai4mi_env

pip install wandb python-dotenv

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

BASE_EXPERIMENT_NAME="experiment_$(date +%Y%m%d_%H%M%S)"
SEEDS=(3)

for SEED in "${SEEDS[@]}"; do
    (
        EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_seed${SEED}"
        RESULTS_DIR="$HOME/ai4mi_project_group8/results/${EXPERIMENT_NAME}"
        PRED_DIR="$HOME/ai4mi_project_group8/data/seghtor_predictions_${SEED}"
        GT_DIR="${PRED_DIR}/gt"
        PRED_FOLDER="${PRED_DIR}/pred"

        echo "=== Processing Seed: ${SEED} ==="
        echo "Results directory: ${RESULTS_DIR}"
        echo "Predictions directory: ${PRED_DIR}"

        python -O main.py \
            --dataset SEGTHOR_CLEAN \
            --mode full \
            --epochs 25 \
            --dest "${RESULTS_DIR}" \
            --gpu \
            --seed ${SEED}
        if [ $? -ne 0 ]; then
            echo "ERROR: Training failed for seed ${SEED}!"
            exit 1
        fi
        python -c "import torch; torch.cuda.empty_cache(); torch.cuda.synchronize(); print('GPU memory cleared')" 2>/dev/null || echo "GPU cleanup completed"

        python stich.py \
          --data_folder "${RESULTS_DIR}/best_epoch/val" \
          --dest_folder "${PRED_FOLDER}" \
          --num_classes 255 \
          --grp_regex "(Patient_\\d\\d)_\\d\\d\\d\\d" \
          --source_scan_pattern "data/segthor_train/train/{id_}/GT.nii.gz"
        if [ $? -ne 0 ]; then
            echo "ERROR: Stitching failed for seed ${SEED}!"
            exit 1
        fi
        python -c "import torch; torch.cuda.empty_cache(); torch.cuda.synchronize(); print('GPU memory cleared')" 2>/dev/null || echo "GPU cleanup completed"

        GT_SOURCE_DIR="$HOME/ai4mi_project_group8/data/segthor_fixed/train"
        mkdir -p "${GT_DIR}"
        pred_count=0
        for pred_file in "${PRED_FOLDER}"/*.nii.gz; do
            if [ -f "$pred_file" ]; then
                patient_name=$(basename "$pred_file" .nii.gz)
                gt_file="${GT_SOURCE_DIR}/${patient_name}/GT.nii.gz"
                if [ -f "$gt_file" ]; then
                    cp "$gt_file" "${GT_DIR}/${patient_name}.nii.gz"
                    pred_count=$((pred_count + 1))
                fi
            fi
        done
        echo "Total ground truths copied: ${pred_count}"

        python compute_metrics.py \
          --ref_folder "${GT_DIR}" \
          --pred_folder "${PRED_FOLDER}" \
          --ref_extension .nii.gz \
          --pred_extension .nii.gz \
          --num_classes 5 \
          --metrics 3d_dice 3d_hd95 3d_hd 3d_assd 3d_nsd 3d_jaccard \
          --save_folder "${RESULTS_DIR}/metrics" \
          --overwrite 
        if [ $? -ne 0 ]; then
            echo "ERROR: Metrics computation failed for seed ${SEED}!"
            exit 1
        fi
        echo "=== Seed ${SEED} Pipeline Completed Successfully! ==="
    ) &
done

wait
echo "=== All Seeds Pipeline Completed! ==="
