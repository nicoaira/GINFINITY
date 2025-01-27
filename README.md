# GINFINITY 🚀 : Graph-based RNA Structure Embedding Generator

## Introduction
GINFINITY is a tool designed to generate embeddings from RNA secondary structures using a Graph Isomorphism Network (GIN). This project converts RNA secondary structures from dot-bracket notation into graph representations, which are then processed by a GIN model to obtain meaningful embeddings. These embeddings can be used for downstream tasks such as clustering, classification, or other forms of analysis.

## Repository Structure
The repository contains the following key components:

- **`src/model/gin_model.py`**: GIN model implementation
- **`src/utils.py`**: Utility functions for RNA graph processing
- **`predict_embedding.py`**: Main script for generating embeddings
- **`train_model.py`**: Script for training new GIN models
- **`train_model_optuna.py`**: Hyperparameter optimization script
- **`src/benchmark/`**: Benchmarking tools and datasets

## Installation
To run the GINFINITY project, you will need Python and the required dependencies installed.

### Step 1: Clone the Repository
```sh
git clone https://github.com/nicoaira/GINFINITY.git
cd GINFINITY
```

### Step 2: Set Up the Environment and Install Dependencies

It is recommended to use a ```conda``` environment to manage dependencies and ensure compatibility. Follow these steps to create a new ```conda``` environment with Python 3.12.7 and install the necessary dependencies.

#### 1. Create a new conda environment:

```sh
conda create --name ginfinity_env python=3.12.7
```
#### 2. Activate the conda environment:

```sh
conda activate ginfinity_env
```

#### 2. Install the dependencies
With the environment activated, install all necessary dependencies using:

```sh
pip install -r requirements.txt
```

Dependencies include:
- `torch==2.4.1`
- `torchvision==0.19.1`
- `pandas==2.2.3`
- `numpy==2.1.2`
- `scikit-learn==1.5.2`
- `argparse==1.4.0`
- `tqdm==4.66.5`

Ensure that the versions match those in the `requirements.txt` 

### Step 3: Download Pre-trained Model
The pre-trained model file (`GIN-Secondary.pth`) sis not included in this repository due to its size. Please download it using the command below:

```sh
# Download the model from Google Drive and save it in the 'saved_model' directory
mkdir -p saved_model
wget -O saved_model/GIN-Secondary.pth "link-to-be-added"
```

## Usage
GINFINITY can generate embeddings from RNA sequences stored in a CSV file.

### Input Format
The input file should be a CSV containing at least one column with the RNA secondary structure in dot-bracket notation.

### Running the Embedding Generation Script
To generate embeddings from an RNA dataset:

```sh
python predict_embedding.py --input example_data/sample_dataset.csv --output example_data/sample_dataset_with_embeddings.tsv
```

**Arguments**:
- `--input`: Path to the input CSV/TSV file containing RNA secondary structures.
- `--output`: Path to save the output TSV file with embeddings.
- `--structure_column_name`: The column name containing RNA secondary structures (default: 'secondary_structure').
- `--structure_column_num`: (Optional) Column number of RNA secondary structures (0-indexed). If both column name and number are provided, column number will be ignored.
- `--model_path`: Path to the trained model file (default: `saved_model/GIN-Secondary.pth`).
- `--device`: Device to run the model on (`cpu` or `cuda`, default: `cpu`).
- `--header`: Specify whether the input CSV has a header row (`True` or `False`, default: `True`).

### Example Command
If your CSV doesn't have a header and the secondary structure is in the 6th column:

```sh
python predict_embedding.py --input example_data/sample_dataset.csv --output example_data/sample_dataset_with_embeddings.tsv --structure_column_num 6 --header False --device cuda
```

## Running the t-SNE Embedding Tool
After generating the RNA embeddings, you can use the `compute_tsne.py` script to visualize the embeddings using t-SNE.

### Example Command
```sh
python compute_tsne.py --input example_data/sample_dataset_with_embeddings.tsv --output example_data/sample_dataset_with_tsne.tsv --embedding_column_name embedding_vector --n_components 3
```

**Arguments**:
- `--input`: Path to the input TSV file containing embeddings (e.g., `example_data/sample_dataset_with_embeddings.tsv`).
- `--output`: Path to save the output TSV file containing the t-SNE-transformed embeddings.
- `--embedding_column_name`: The name of the column containing the embedding vectors.
- `--n_components`: Number of components for t-SNE (e.g., 2 for 2D visualization, 3 for 3D visualization).

### Explanation
- The script reads the specified `embedding_column_name` from the input file and performs t-SNE transformation with the specified number of components.
- The resulting transformed embeddings are saved to the output file, which can then be used for downstream visualization and analysis.

### Interactive Visualization with Dash
We have also developed an application using Dash to visualize the t-SNE embeddings interactively. You can find the app in the following repository: [https://github.com/nicoaira/embeddings_app](https://github.com/nicoaira/embeddings_app).

## Running the Tests
You can run the tests using:

```sh
python -m unittest discover tests
```

This will run both the unit tests for the model and the integration tests for the embedding generation pipeline.

## Important Notes
- Ensure you have the correct PyTorch version installed that supports your GPU if you're using CUDA.
- If you encounter any issues with the pre-trained model, please make sure to check the Google Drive link and download it correctly.
