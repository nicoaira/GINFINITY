# Scripts Overview

This directory contains utility scripts for processing and modeling RNA embeddings and distances.

## Contents

- **compute_distances.py**  
  Compute pairwise squared Euclidean distances between embedding vectors.

- **generate_embeddings.py**  
  Generate RNA structure embeddings using a pretrained GIN model.

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
  [--device cpu|cuda:0]
```

Key arguments:

- `--mode`  
  1 = all-vs-all (default), 2 = one-vs-all (requires `--query`).
- `--keep-cols`  
  Comma-separated metadata columns to carry forward (defaults to `--id-column`).

---

## generate_embeddings.py

**Purpose**  
Convert RNA secondary structures into embedding vectors using a pretrained Graph Isomorphism Network.

**Usage**

```bash
python generate_embeddings.py \
  --input <structures.csv/tsv> \
  --model_path <checkpoint.pth> \
  [--output <out.tsv>] \
  [--model_id ID] \
  [--structure-column-name NAME] \
  [--samples N] \
  [--num_workers N]
```

By default, outputs to `output/<model_id>/<model_id>_embeddings.tsv` and a log file.

---

## generate_windows.py

**Purpose**  
Slide a fixed-length window across input sequences (e.g., RNA/DNA) and output subsequences.

**Usage**

```bash
python generate_windows.py \
  --input <fasta/fa> \
  --window-size <int> \
  --step-size <int> \
  --output <out.tsv>
```

---

## train_model.py

**Purpose**  
Train a GIN on RNA embeddings using triplet loss, with optional hyperparameter tuning via Optuna.

**Usage**

```bash
python train_model.py \
  --train-data <train.tsv> \
  --val-data <val.tsv> \
  [--batch-size N] \
  [--epochs N] \
  [--lr FLOAT] \
  [--early-stopping-patience N] \
  [--output-model <out.pth>]
```

---

## Logging & Outputs

- Each script prints progress to stdout (and uses `tqdm` for bars).
- Models and embeddings are saved to specified output paths.
- Optuna tuning (if invoked) stores plots under `dev/optuna_plots/`.
