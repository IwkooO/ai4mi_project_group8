#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=full_pipeline_parallel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=03:00:00
#SBATCH --output=out/full_pipeline_parallel_FINAL_preprocess_transUnet_adam.log

module load 2023
module load Anaconda3/2023.07-2

cd $HOME/ai4mi_project_group8/
source $(conda info --base)/etc/profile.d/conda.sh
source activate ai4mi_env

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

BASE_EXPERIMENT_NAME="FINAL_preprocess_transUnet_adam$(date +%Y%m%d_%H%M%S)"
SEEDS=(1 2 3)

####################################
# Pre-process the data
####################################

python -O slice_segthor.py --source_dir data/segthor_fixed --dest_dir data/SEGTHOR_PREPROCESSED \
        --shape 256 256 --retain 10 -p -1

####################################

echo "Submitting separate sbatch jobs for seeds: ${SEEDS[*]}"
for SEED in "${SEEDS[@]}"; do
  echo "Submitting seed ${SEED}"
  sbatch jobs/run_seed_pipeline.sh "$SEED" "$BASE_EXPERIMENT_NAME"
done

echo "All seed jobs submitted. Monitor with: squeue -u $USER"
