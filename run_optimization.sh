#!/bin/bash
#SBATCH --job-name=gin_hp
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --array=0-59

# 🕒 Tiempo máximo por job
#SBATCH --time=01:00:00    # 1 hora por combinación

# 🧠 Memoria RAM total (por job, no por CPU)
#SBATCH --mem=8G           # Ajustalo según lo que necesites

# 🧮 Núcleos de CPU
#SBATCH --cpus-per-task=2  # Suficiente para modelos ligeros

# 🎮 GPU (si usás CUDA)
#SBATCH --gres=gpu:1

# 📦 Partición del cluster
#SBATCH --partition=gpu    # O consulta qué particiones están disponibles

# Activar entorno Conda
conda activate strusi_env

# Leer combinación desde el grid
GRID="src/hp/grid.csv"
combo=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" $GRID)
IFS=',' read lr hidden_dim gin_layers <<< "$combo"

# Ejecutar entrenamiento
python src/hp/train_combination.py \
    --lr $lr \
    --hidden_dim $hidden_dim \
    --gin_layers $gin_layers \
    --input_path path/a/train.csv \
    --val_path path/a/val_dataset.csv
