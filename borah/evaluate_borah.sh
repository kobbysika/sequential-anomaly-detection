#!/bin/bash
#SBATCH --job-name=sequential-eval
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=evaluation-%j.log

set -e
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate sequential-anomaly
cd "$HOME/sequential-anomaly-detection"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
nvidia-smi
for model in dablog deeplog
do
    python -u evaluate_batched.py --model "$model" --batch-size 256
done
