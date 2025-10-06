#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=preprocess
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=03:00:00
#SBATCH --output=out/preprocess.out

module load 2023
module load Anaconda3/2023.07-2

cd $HOME/ai4mi_project_group8/
source $(conda info --base)/etc/profile.d/conda.sh
source activate ai4mi_env

python -O preprocessing.py

# Copy processed data to new structure
python -O preprocessing.py \
    --segthor_clean data/SEGTHOR_CLEAN \
    --output_dir data/SEGTHOR_PREPROCESSED_WINDOW_GAMMA \
    --preprocessed_subfolder preprocessed3D_window_gamma

echo "Preprocessing completed."