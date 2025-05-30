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
- Python 3.9 or higher.
- `pip` and `git` installed on your system.
- It is recommended to use a virtual environment (e.g., `venv` or `conda`) to manage dependencies.

### Step 1: Install GINFINITY Package
To install the latest version from the `main` branch:
```sh
pip install git+https://github.com/nicoaira/GINFINITY.git#egg=ginfinity
```
To install a specific version (e.g., v0.1.0):
```sh
pip install git+https://github.com/nicoaira/GINFINITY.git@v0.1.0#egg=ginfinity
```

This command will automatically handle the core dependencies listed in `pyproject.toml`.

### Step 2: Install PyTorch Geometric (Manual Step)

**`torch-geometric` and its related packages must be installed manually after installing GINFINITY.** This is because their specific versions depend heavily on your system's PyTorch version and CUDA (if applicable).

Please follow the [official PyTorch Geometric installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) to install `torch-geometric`, `torch-scatter`, `torch-sparse`, `torch-cluster`, and `torch-spline-conv` that match your PyTorch and CUDA setup.

Example for PyTorch 2.2.0 and CUDA 12.1:
```bash
# Ensure PyTorch is installed correctly for your CUDA version first
# pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```

### Step 3: (Optional) Install Dependencies for Training
If you intend to train new models, you'll need additional dependencies. You can install them using:
```sh
pip install git+https://github.com/nicoaira/GINFINITY.git#egg=ginfinity[train]
```

### Step 4: Download Pre-trained Model Weights
Pre-trained model weights are included within the package in the `src/ginfinity/weights` directory and should be accessible after installation. If you need to use a specific or externally hosted model, ensure it's downloaded to a known location.

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

## Important Notes
- Ensure your PyTorch and PyTorch Geometric versions are compatible and correctly installed for your system (CPU or specific CUDA version).

## License

This project is licensed under the MIT License.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Related Repositories

- **GINflow Nextflow Pipeline**: For scalable and reproducible execution of GINFINITY workflows, please refer to our Nextflow pipeline available at: [https://github.com/nicoaira/GINflow](https://github.com/nicoaira/GINflow)
- **GINFINITY API**: For programmatic access to GINFINITY functionalities, an API is under development and will be available at: `[Link to GINFINITY API Repository - Placeholder]`
