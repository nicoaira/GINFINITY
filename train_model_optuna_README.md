# README for Hyperparameter Optimization Script

## Overview

This script performs **hyperparameter optimization** for training models that generate embeddings from RNA secondary structures. It uses **Optuna**, a state-of-the-art hyperparameter optimization framework, to minimize validation loss by tuning hyperparameters. The script supports two model types: **Siamese ResNet-LSTM** and **GIN (Graph Isomorphism Network)**.

---

## Features

- **Model Support**:
  - **Siamese ResNet-LSTM**: Sequence-based RNA embeddings.
  - **GIN**: Graph-based RNA embeddings with `standard` or `forgi` graph encoding.
- **Hyperparameter Optimization**: Optimizes:
  - Hidden dimension size
  - Output dimension size
  - Learning rate
  - Number of GIN layers
- **Early Stopping**: Stops training if validation loss does not improve for a specified number of epochs.
- **Logging**: Logs trial parameters, progress, and best results.

---

## Prerequisites

### Libraries

Install the following Python libraries:

- `torch`
- `torch_geometric`
- `optuna`
- `pandas`
- `scikit-learn`
- `tqdm`

Install missing dependencies with:

```bash
pip install torch optuna pandas scikit-learn tqdm
```

For `torch_geometric`, follow the [installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

---

## Usage

### Command-line Arguments

Run the script as follows:

```bash
python optimize_hyperparameters.py --input_path <path_to_input_file> --model_type <siamese|gin> [options]
```

#### **Required Arguments**
- `--input_path`: Path to the input CSV/TSV file containing RNA secondary structures with `structure_A`, `structure_P`, and `structure_N` columns.
- `--model_type`: Model type to use (`siamese` or `gin`).

#### **Optional Arguments**
- `--optimisation_id`: Identifier for the optimization run (default: `gin_optimisation`).
- `--n_trials`: Number of Optuna trials to run (default: `50`).
- `--graph_encoding`: Graph encoding type for GIN model (`standard` or `forgi`, default: `forgi`).
- `--batch_size`: Batch size for training and validation (default: `100`).
- `--num_epochs`: Number of epochs for training (default: `1`).
- `--patience`: Patience for early stopping (default: `5`).
- `--hidden_dim`: Fixed hidden dimension size. If not provided, it will be optimized.
- `--output_dim`: Fixed output embedding size. If not provided, it will be optimized.
- `--lr`: Fixed learning rate. If not provided, it will be optimized.
- `--gin_layers`: Fixed number of GIN layers. If not provided, it will be optimized.

---

## Input Data Format

The input file must be a CSV/TSV with columns `structure_A`, `structure_P`, and `structure_N` containing RNA secondary structures in dot-bracket notation.

### Example File

| ID  | structure_A       | structure_P       | structure_N       |
|-----|-------------------|-------------------|-------------------|
| 1   | ..((..))..        | ..((..))..        | ..(((...)))..     |
| 2   | ..(((...)))..     | ..(((...)))..     | ..((..))..        |

---

## Outputs

1. **Model Checkpoints**: Saved to `output/<optimisation_id>/<optimisation_id>_<timestamp>.pth`.
2. **Logs**: Logs training progress, parameters, and best trial details to `output/<optimisation_id>/run_<timestamp>.log`.
3. **Best Trial**: Reports the best hyperparameter configuration.

---

## Example Commands

### Optimize a Siamese Model
```bash
python optimize_hyperparameters.py --input_path data/rna_structures.csv --model_type siamese --n_trials 100
```

### Optimize a GIN Model with Fixed Hyperparameters
```bash
python optimize_hyperparameters.py --input_path data/rna_structures.csv --model_type gin --hidden_dim 256 --output_dim 64 --gin_layers 3 --lr 0.001
```

### Optimize GIN Model with Variable Hyperparameters
```bash
python optimize_hyperparameters.py --input_path data/rna_structures.csv --model_type gin --n_trials 50
```

---

## How It Works

1. **Input Validation**:
   - Ensures all RNA secondary structures in the dataset are valid dot-bracket strings.

2. **Define Objective Function**:
   - The script defines an Optuna objective function that:
     - Samples hyperparameters (e.g., `hidden_dim`, `output_dim`, `lr`, `gin_layers`).
     - Loads data and splits it into training and validation sets.
     - Trains the model with early stopping and returns the validation loss.

3. **Optimize Hyperparameters**:
   - Optuna runs the objective function for the specified number of trials (`--n_trials`) and identifies the best hyperparameter configuration.

4. **Save Results**:
   - Saves the model and logs the best trial details.

---

## Notes

- **GPU Support**: The script will automatically use a GPU if available. Otherwise, it defaults to CPU.
- **Custom Hyperparameter Ranges**: Modify the `trial.suggest_*` calls in the `objective` function to change the range of hyperparameters to optimize.
- **Extensibility**: You can extend the script to include new models or additional hyperparameters.
