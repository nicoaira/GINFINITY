# RNA Structure Embedding Training Script

## Overview

This script trains a Graph Isomorphism Network (GIN) model to generate embeddings from RNA secondary structures. The script includes data preprocessing, training with early stopping, and automatic metadata logging.

---

## Features

- **GIN Model**: Graph-based RNA embedding with configurable layers and encoding
- **Early Stopping**: Stops training if validation loss doesn't improve
- **Triplet Loss**: Uses triplet loss for training embeddings
- **Data Validation**: Removes invalid RNA secondary structures
- **Logging**: Tracks training progress, parameters, and execution time
- **Device Support**: Automatically uses GPU if available

---

## Prerequisites

### Libraries

Ensure the following Python libraries are installed:

- `torch`
- `torch_geometric`
- `pandas`
- `scikit-learn`
- `tqdm`

Install missing dependencies using:

```bash
pip install torch pandas scikit-learn tqdm
```

For `torch_geometric`, follow the installation guide: [PyTorch Geometric Installation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

### Files and Directory Structure

Place the following modules in the `src/` directory:

- `early_stopping.py`: Implements early stopping logic.
- `gin_rna_dataset.py`: Dataset class for GIN models.
- `model/gin_model.py`: GIN model definition.
- `triplet_loss.py`: Implements triplet loss function.
- `utils.py`: Utility functions like logging and validation.

---

## Usage

### Command-line Arguments

Run the script using the following arguments:

```bash
python train_script.py --input_path <path_to_csv> [options]
```

#### Required Arguments
- `--input_path`: Path to the CSV/TSV file containing RNA secondary structures with `structure_A`, `structure_P`, and `structure_N` columns.

#### Optional Arguments
- `--model_id`: Identifier for the model (default: `gin_model`).
- `--graph_encoding`: Encoding for GIN model (`standard` or `forgi`) (default: `standard`).
- `--hidden_dim`: Hidden dimension size for the model (default: `256`).
- `--output_dim`: Output embedding size for GIN model (default: `128`).
- `--batch_size`: Batch size for training and validation (default: `100`).
- `--num_epochs`: Number of epochs for training (default: `10`).
- `--patience`: Patience for early stopping (default: `5`).
- `--min_delta`: Minimum validation loss decrease to qualify as improvement (default: `0.001`).
- `--lr`: Learning rate for the optimizer (default: `0.001`).
- `--gin_layers`: Number of GIN layers (default: `1`).
- `--num_workers`: Number of worker threads for data loading (default: CPU count/2).
- `--save_best_weights`: Save best model weights (default: `True`).
- `--device`: Training device (`cuda` or `cpu`).

---

## Input Data Format

The input file should be a CSV/TSV with the following columns:
- `structure_A`: Dot-bracket notation for anchor structure.
- `structure_P`: Dot-bracket notation for positive structure.
- `structure_N`: Dot-bracket notation for negative structure.

---

## Outputs

1. **Model Checkpoints**: Saved to `output/<model_id>/<model_id>.pth` with optimizer state and epoch information.
2. **Training Logs**: Saved to `output/<model_id>/train.log` containing training progress and parameters.
3. **Embeddings**: Not generated during training but can be inferred post-training using the trained model.

---

## Example

To train a GIN model:

```bash
python train_script.py --input_path data/rna_structures.csv --hidden_dim 128 --gin_layers 2
```

## Training Flow

1. **Data Preprocessing**:
   - Loads and validates RNA structures using `is_valid_dot_bracket`.
   - Splits data into training and validation sets.

2. **Model Initialization**:
   - Creates the GIN model.
   - Initializes the optimizer and loss function.

3. **Training Loop**:
   - Trains the model with early stopping monitoring.
   - Early stopping triggers when validation loss improvement is less than `min_delta` for `patience` epochs.
   - Saves best model weights when validation loss improves significantly.
   - Logs detailed progress including loss values and early stopping status.

4. **Model Saving**:
   - Saves the best model state achieved during training.
   - Records training metadata and parameters.

5. **Logging**:
   - Logs training parameters and configuration.
   - Tracks early stopping parameters and decisions.
   - Records loss metrics and execution time.

---

## Notes

- Ensure the `input_path` file contains valid RNA structures in the required format.
- For large datasets, use a GPU for faster training.
- You can extend the script by adding new models or loss functions as needed.