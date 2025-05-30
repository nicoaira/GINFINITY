# GINFINITY 🚀 : Graph-based RNA Structure Embedding Generator

## Introduction
GINFINITY is a Python package designed to generate embeddings from RNA secondary structures using a Graph Isomorphism Network (GIN). This project converts RNA secondary structures from dot-bracket notation into graph representations, which are then processed by a GIN model to obtain meaningful embeddings. These embeddings can be used for downstream tasks such as clustering, classification, or other forms of analysis.

## Repository Structure
The repository contains the following key components:

- **`src/ginfinity/`**: The main package directory.
  - **`model/gin_model.py`**: GIN model implementation.
  - **`utils.py`**: Utility functions for RNA graph processing and general helpers.
  - **`scripts/`**: Directory containing operational scripts installable as command-line tools.
    - **`generate_embeddings.py`** (command: `ginfinity-embed`)
    - **`compute_distances.py`** (command: `ginfinity-compute-distances`)
    - **`generate_windows.py`** (command: `ginfinity-generate-windows`)
    - **`train_model.py`** (command: `ginfinity-train`)
    - **`scripts_README.md`**: Detailed documentation for all scripts in this directory.
  - **`training/`**: Modules related to model training, like custom datasets and loss functions.
  - **`weights/`**: Location for pre-trained model weights (included in the package).

## Installation

GINFINITY is packaged using modern Python standards and can be installed directly from its Git repository.

### Prerequisites
- Conda (recommended for environment management)
- Python 3.10 (the environment will be set up with this version)
- `pip` and `git` installed on your system.

### Step 1: Set Up Conda Environment

1.  Create a new conda environment named `ginfinity-env` with Python 3.10:
    ```sh
    conda create -n ginfinity-env python=3.10
    ```
2.  Activate the newly created environment:
    ```sh
    conda activate ginfinity-env
    ```

### Step 2: Install GINFINITY Package

With the `ginfinity-env` activated, install the GINFINITY package from GitHub.

To install the latest version from the `main` branch:
```sh
pip install git+https://github.com/nicoaira/GINFINITY.git#egg=ginfinity
```
To install a specific version (e.g., v0.1.0):
```sh
pip install git+https://github.com/nicoaira/GINFINITY.git@v0.1.0#egg=ginfinity
```
This command will automatically handle the core dependencies listed in `pyproject.toml`.

### Step 3: (Optional) Install Dependencies for Training
If you intend to train new models, you'll need additional dependencies. You can install them using:
```sh
pip install git+https://github.com/nicoaira/GINFINITY.git#egg=ginfinity[train]
```

### Step 4: Verify Installation
Pre-trained model weights are included within the package in the `src/ginfinity/weights` directory and should be accessible after installation. 

To quickly verify that the GINFINITY command-line tools are available, you can try:
```sh
ginfinity-embed --help
```
This should display the help message for the `generate_embeddings.py` script.

(The previous instruction to download `GIN-Secondary.pth` might be outdated if weights are now packaged. Please verify and update this section if necessary based on your `MANIFEST.in` and `pyproject.toml` setup for including weights.)

## Usage
GINFINITY provides a suite of command-line tools to process RNA secondary structures, generate embeddings, train models, and perform downstream analysis.

### Input Format
The primary input is typically a CSV or TSV file containing at least one column with RNA secondary structures in dot-bracket notation.

### Running the Scripts (Command-Line Tools)
Once installed, the scripts are available as command-line tools. For example, to generate embeddings:

```sh
ginfinity-embed --input path/to/your/rna_data.csv --output path/to/output_embeddings.tsv --model-path path/to/your/model.pth --id-column your_id_col --structure-column-name rna_structure_col
```

**Arguments and detailed usage instructions for all command-line tools (`ginfinity-embed`, `ginfinity-compute-distances`, `ginfinity-generate-windows`, and `ginfinity-train`) can be found in the dedicated scripts documentation: [`src/ginfinity/scripts/scripts_README.md`](src/ginfinity/scripts/scripts_README.md).**

(Note: The path to `scripts_README.md` will be relative to the cloned repository if you're viewing it locally. If the package is installed, users might need to refer to the repository to view this specific README.)

## Related Repositories

- **GINflow Nextflow Pipeline**: For scalable and reproducible execution of GINFINITY workflows, please refer to our Nextflow pipeline available at: [https://github.com/nicoaira/GINflow](https://github.com/nicoaira/GINflow)
- **GINFINITY API**: For programmatic access to GINFINITY functionalities, an API is under development and will be available at: `[Link to GINFINITY API Repository - Placeholder]`
