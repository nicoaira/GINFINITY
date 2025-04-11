#!/bin/bash
#SBATCH --job-name=gin_hp
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --array=0-59
#SBATCH --time=04:00:00          # Hasta 4 horas por combinación
#SBATCH --mem=20G                # 20 GB de RAM
#SBATCH --partition=short
#SBATCH --cpus-per-task=32         # Cada GPU A100 requiere 32 CPUs
#SBATCH --gres=gpu:a100:1

# Activar entorno Conda
# conda init
# conda activate ginfinity_env

# Leer combinación desde el grid
GRID="src/hp/grid.csv"
combo=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" $GRID)
IFS=',' read lr hidden_dim gin_layers <<< "$combo"

# Ejecutar entrenamiento
python -m src.hp.train_combination \
    --lr $lr \
    --hidden_dim $hidden_dim \
    --gin_layers $gin_layers \
    --input_path example_data/train.csv \
    --val_path example_data/val_dataset.csv
