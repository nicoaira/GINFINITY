# Scripts Overview

This directory contains utility scripts for processing and modeling RNA embeddings and distances. Scripts now feature improved error handling for missing columns/files and invalid inputs, clearer output messages, and a `--quiet` flag to suppress non-essential terminal output.

## Contents

- **compute_distances.py**  
  Compute pairwise squared Euclidean distances between embedding vectors.

- **generate_embeddings.py**  
  Generate RNA structure embeddings using a pretrained GIN model.

- **generate_node_embeddings.py**  
  Generate per-node (LxD) structural embeddings before pooling.

- **generate_base_embeddings.py**  
  Generate per-base (LxD) sequence embeddings using RiNALMo.

- **generate_windows.py**  
  Extract sliding windows from sequences for downstream analysis.

- **train_model.py**  
  Train a GIN model on RNA data with triplet loss and early stopping.

---

## Prerequisites

Install required packages via `requirements.txt` or:

```bash
pip install torch pandas tqdm torch-geometric optuna
```

Ensure CUDA drivers are set up if you plan to use GPU.

---

## compute_distances.py

**Purpose**  
Compute all-vs-all or one-vs-all squared Euclidean distances between embeddings stored in a TSV.

**Usage**

```bash
python compute_distances.py \
  --input <input.tsv> \
  --output <output.tsv> \
  [--embedding-col EMB_COL] \
  [--keep-cols COL1,COL2] \
  [--mode 1|2] \
  [--query ID] \
  [--batch-size N] \
  [--num-workers N] \
  [--device cpu|cuda:0] \
  [--quiet]
```

Key arguments:

- `--mode`  
  1 = all-vs-all (default), 2 = one-vs-all (requires `--query`).
- `--keep-cols`  
  Comma-separated metadata columns to carry forward (defaults to `--id-column`).

---

## generate_embeddings.py

**Purpose**  
Convert RNA secondary structures or precomputed windowed graphs into embedding vectors using a pretrained Graph Isomorphism Network.

**Usage**

Two modes are supported:

1.  **Raw TSV/CSV input:**
    ```bash
    python generate_embeddings.py \\
      --input <input.tsv_or_csv> \\
      --output <output_embeddings.tsv> \\
      --model-path <model_checkpoint.pth> \\
      --id-column <id_column_name> \\
      --structure-column-name <structure_column_name> \\
      [--keep-cols col1,col2,...] \\
      [--device cuda|cpu] \\
      [--num-workers N] \\
      [--batch-size N] \\
      [--quiet]
    ```

2.  **Windowed graphs input:**
    ```bash
    python generate_embeddings.py \\
      --graph-pt <path_to_windows_graphs.pt> \\
      --meta-tsv <path_to_windows_metadata.tsv> \\
      --output <output_embeddings.tsv> \\
      --model-path <model_checkpoint.pth> \\
      --id-column <id_column_name_in_meta_tsv> \\
      [--keep-cols col1,col2,...] \\
      [--device cuda|cpu] \\
      [--num-workers N] \\
      [--batch-size N] \\
      [--quiet]
    ```

By default, outputs an embedding TSV file. Log files are generated alongside the output.

---

## generate_base_embeddings.py

Purpose  
Generate per-base RiNALMo embeddings from an input TSV/CSV containing sequences. Produces a TSV with a JSON column `base_embeddings` (LxD per sequence).

Usage

```bash
python src/ginfinity/scripts/generate_base_embeddings.py \
  --input sequences.tsv \
  --output base_embeds.tsv \
  --id-column Name \
  --sequence-column-name sequence \
  [--model-name giga-v1] \
  [--device cpu|cuda:0] \
  [--batch-size 8]
```

Notes
- Install RiNALMo: `pip install rinalmo`
- The output can be merged (by ID) with your structural embeddings table to enable weighted alignments.

---

## Pairwise and MSA with base embeddings

- Pairwise: `align_node_embeddings.py` now supports optional base embeddings and a weight:
  - `--base-input` (optional second TSV) and/or a `--base-embeds-col` present in `--input`
  - `--seq-weight` in [0,1] to blend base vs. structural similarities
  - Outputs the combined matrix. With `--save-components`, also writes the component matrices.

- MSA: `embed_msa.py` accepts `--base-embeds-col` and `--seq-weight` to blend similarities throughout the pipeline (pairwise posteriors and profile–profile DP).

---

## generate_windows.py

**Purpose**  
Extract sliding windows from sequences, generating graph objects and metadata for each window. This script no longer outputs a `window_sequence` column.

**Usage**
```bash
python generate_windows.py \\
  --input <input.tsv_or_csv> \\
  --output-dir <path_to_windows_output_directory> \\
  --id-column <id_column_name> \\
  --structure-column-name <structure_column_name> \\
  --L <window_length> \\
  [--keep-paired-neighbors] \\
  [--mask-threshold FLOAT] \\
  [--keep-cols col1,col2,...] \\
  [--num-workers N] \\
  [--quiet]
```
**Outputs:**
- `<output-dir>/windows_graphs.pt`: PyTorch file containing graph objects for each window.
- `<output-dir>/windows_metadata.tsv`: TSV file with metadata for each window.
The default `--output-dir` is `windows_output/`.

---

## train_model.py

**Purpose**  
Train a GIN on RNA embeddings using triplet loss, with optional hyperparameter tuning via Optuna.

**Usage**
```bash
python train_model.py \\
  --train-data <train.tsv> \\
  --val-data <val.tsv> \\
  [--batch-size N] \\
  [--epochs N] \\
  [--lr FLOAT] \\
  [--early-stopping-patience N] \\
  [--output-model <out.pth>] \\
  [--quiet]
```

---

## Logging & Outputs

- Each script prints progress to stdout (and uses `tqdm` for bars).
- Models and embeddings are saved to specified output paths.
- Optuna tuning (if invoked) stores plots under `dev/optuna_plots/`.
