# RNA Embedding Prediction Script

## Overview

This script generates embeddings from RNA secondary structures using a trained GIN (Graph Isomorphism Network) model. The model parameters and configuration are automatically loaded from the checkpoint metadata.

## Features

- **GIN Model**: Graph-based RNA embedding
- **Graph Encoding Options**: `standard` or `forgi` encoding
- **Input Validation**: Validates dot-bracket structures
- **Parallel Processing**: Multi-worker embedding generation
- **Logging**: Records parameters and execution details

## Prerequisites

### Libraries

Ensure the following Python libraries are installed:

- `torch`
- `pandas`
- `tqdm`

Install missing dependencies with:

```bash
pip install torch pandas tqdm
```

For GIN model support, ensure `torch_geometric` is installed following the [installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

---

## Usage

### Command-line Arguments

Run the script as follows:

```bash
python predict_embeddings.py --input <path_to_input_file> --model_path <path_to_model_checkpoint> [options]
```

#### Required Arguments

- `--input`: Path to the input CSV/TSV file containing RNA secondary structures.
- `--model_path`: Path to the pre-trained model file.

#### Optional Arguments

- `--output`: Path to save the output embeddings (default: `output/<model_id>/<model_id>_embeddings.tsv`).
- `--model_id`: Identifier for the model, used to create the default output path.
- `--structure_column_name`: Name of the column containing RNA secondary structures.
- `--structure_column_num`: Column index (0-indexed) of RNA secondary structures if no header is present.
- `--header`: Specify if the input file has a header (`True` or `False`, default: `True`).
- `--samples`: Number of random samples to process from the input file.
- `--num_workers`: Number of worker processes to use for multiprocessing (default: `4`).

---

## Input Data Format

The input file must be a CSV/TSV with one column containing RNA secondary structures in dot-bracket notation. Specify the column using `--structure_column_name` or `--structure_column_num`.

### Example File

| ID  | secondary_structure         |
|-----|-----------------------------|
| 1   | ..((..))..                 |
| 2   | ..(((...)))..              |

---

## Outputs

1. **Embeddings File**: A TSV file containing RNA structures and their corresponding embedding vectors.
   - Default path: `output/<model_id>/<model_id>_embeddings.tsv`
   - Columns:
     - Original input columns.
     - `embedding_vector`: Comma-separated embedding vector.

2. **Logs**: Progress and parameters are logged to `output/<model_id>/predict_embedding.log`.

---

## Example

### Predict Embeddings Using a GIN Model

```bash
python predict_embeddings.py --input data/rna_structures.csv --model_path saved_model/gin_model.pth
```

### Process a Random Sample of 100 Structures

```bash
python predict_embeddings.py --input data/rna_structures.csv --samples 100 --model_path saved_model/gin_model.pth
```

---

## How It Works

1. **Load Model**:
   - Loads the pre-trained model from `--model_path`.
   - The model metadata is automatically loaded from the checkpoint.

2. **Input Validation**:
   - Ensures valid RNA structures using dot-bracket notation.

3. **Generate Embeddings**:
   - **GIN Model**: Converts RNA structures to graphs (`standard` or `forgi`) and predicts embeddings.

4. **Save Results**:
   - Appends embedding vectors to the input data and saves as a TSV file.

---

## Notes

- Ensure the input file contains valid dot-bracket RNA structures.
- Use GPU for faster predictions with large datasets.
- Extendable: You can add support for new model types or custom preprocessing steps.
