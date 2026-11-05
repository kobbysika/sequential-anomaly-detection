#!/bin/bash
#SBATCH --job-name=sequential-ad
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=training-%j.log

set -e

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate sequential-anomaly
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX"

cd "$HOME/sequential-anomaly-detection"

export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

nvidia-smi

git rev-parse HEAD
git diff
python -m pip freeze

for model in dablog deeplog
do
    python -u -m sequential_anomaly.cli.train \
        --config "configs/${model}.yaml" \
        --prepared results/runs/shared/preprocessed \
        --output "artifacts/checkpoints/${model}" \
        --epochs 32
done