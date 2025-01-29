# Hyperparameter Tuning with Optuna

## Overview

The `hyperparameter_tuning` module is designed to facilitate the optimization of hyperparameters for the GIN (Graph Isomorphism Network) model using [Optuna](https://optuna.org/). This module automates the process of searching for the best hyperparameter combinations to enhance model performance on RNA secondary structure datasets.

## Prerequisites

Before using the hyperparameter tuning module, ensure that the following dependencies are installed:

- Python 3.6 or higher
- PyTorch
- PyTorch Geometric
- Optuna
- Other Python packages as listed in `requirements.txt`

You can install the required packages using:

```bash
pip install -r requirements.txt
```

## Configuration

The hyperparameter search space is defined in the `config.json` file located in the `src/hyperparameter_tuning/` directory. You can modify this file to adjust the ranges and choices for different hyperparameters.

```json
{
    "gin_layers": [3, 4],
    "hidden_dim": [128, 256],
    "output_dim": [128, 256],
    "lr": {
        "min": 0.00001,
        "max": 0.001
    },
    "decay_rate": {
        "min": 0.9,
        "max": 0.99
    },
    "dropout": {
        "min": 0.0,
        "max": 0.1
    }
}
```

## Usage

To start hyperparameter tuning, navigate to the project root directory and execute the tuning script:

```bash
python -m src.hyperparameter_tuning.tune \
    --input_path '/path/to/train.csv' \
    --batch_size 500 \
    --num_epochs 25 \
    --patience 4 \
    --num_workers 8 \
    --device cuda \
    --save_best_weights True \
    --pooling_type global_add_pool \
    --n_trials 8 \
    --benchmark_datasets hard_rfam_benchmark_big
```

### Parameters

- `--input_path` (str, required):  
  Path to the input CSV/TSV file containing RNA secondary structures.

- `--graph_encoding` (str, choices: `standard`, `forgi`; default: `standard`):  
  Encoding method for transforming RNA structures into graph representations.

- `--batch_size` (int, default: 100):  
  Number of samples per batch during training and validation.

- `--num_epochs` (int, default: 10):  
  Maximum number of epochs for training the model.

- `--patience` (int, default: 5):  
  Number of epochs with no improvement after which training will be stopped.

- `--num_workers` (int, default: number of CPU cores // 2):  
  Number of subprocesses to use for data loading.

- `--save_best_weights` (bool, default: True):  
  Whether to save the model weights that achieve the best validation loss.

- `--device` (str, choices: `cuda`, `cpu`; default: `cuda` if available):  
  Device to use for training (`cuda` for GPU or `cpu`).

- `--min_delta` (float, default: 0.001):  
  Minimum change in validation loss to qualify as an improvement.

- `--pooling_type` (str, choices: `global_add_pool`, `set2set`; default: `global_add_pool`):  
  Type of pooling layer to use in the GIN model.

- `--storage` (str, default: SQLite database path):  
  Storage URL for the Optuna study (e.g., SQLite database).

- `--n_trials` (int, default: 100):  
  Total number of hyperparameter trials to perform.

- `--json_config` (str, default: path to `config.json`):  
  Path to the JSON configuration file defining the hyperparameter search space.

- `--study_id` (str, default: `tuning_YYMMDD_HHMM`):  
  Identifier for the Optuna study. If not provided, a timestamp-based ID is generated.

- `--output_dir` (str, default: `output/hyperparameter_tuning`):  
  Directory where the tuning results and models will be saved.

- `--benchmark_datasets` (str, default: `hard_rfam_benchmark_big`):  
  Benchmark datasets used for evaluating the model's performance.

- `--finish_now` (flag):  
  **Special Parameter**: When provided, the tuning process will immediately terminate and save the final results using the cached parameters. This is useful for resuming or finalizing a study without running additional trials.


## Usage Examples

### Starting a New Hyperparameter Tuning Study

To initiate a new hyperparameter tuning study, execute the following command:

```bash
python -m src.hyperparameter_tuning.tune \
    --input_path 'GINFINITY/data/train/train.csv' \
    --batch_size 500 \
    --num_epochs 25 \
    --patience 4 \
    --num_workers 8 \
    --device cuda \
    --save_best_weights True \
    --pooling_type global_add_pool \
    --n_trials 8 \
    --benchmark_datasets hard_rfam_benchmark_big
```

### Finishing an Ongoing Study with `--finish_now`

If you need to prematurely terminate an ongoing hyperparameter tuning study and save the final results, use the `--finish_now` flag. This is particularly useful when you want to finalize the study without running all the planned trials.

Ensure that you provide the `--study_id` corresponding to the study you wish to finish. Optionally, you can specify the `--parameters_cache` path if it's different from the default.

```bash
python -m src.hyperparameter_tuning.tune \
    --input_path 'GINFINITY/data/train/train.csv' \
    --batch_size 500 \
    --num_epochs 25 \
    --patience 4 \
    --num_workers 8 \
    --device cuda \
    --save_best_weights True \
    --pooling_type global_add_pool \
    --n_trials 8 \
    --benchmark_datasets hard_rfam_benchmark_big \
    --study_id tuning_250128_2030 \
    --finish_now \
    --parameters_cache 'GINFINITY/output/hyperparameter_tuning/tuning_250128_2030/parameters_cache.json'
```

**Explanation of `--finish_now`:**

- **Purpose:**  
  The `--finish_now` flag allows you to immediately terminate the hyperparameter tuning process and save all the current results. This is useful if you decide to stop the tuning early due to time constraints or other reasons.

- **How It Works:**  
  When `--finish_now` is provided, the script will:
  1. Load the cached parameters from the specified `--parameters_cache` file.
  2. Verify the existence of the study with the given `--study_id` in the Optuna storage.
  3. Finalize the study by saving the results, plotting the optimization history and parameter importances, and saving the best model found up to that point.

- **Requirements:**  
  - The `--parameters_cache` must point to a valid cache file containing the parameters of the study.
  - The specified `--study_id` must correspond to an existing study in the Optuna storage.

- **Example Scenario:**  
  Suppose you started a study with `--study_id tuning_250128_2030` and planned to run 100 trials. After completing 50 trials, you decide to stop and save the current best model and results. You would run the command above with the `--finish_now` flag to achieve this.

### Resuming a Tuning Study

If you have previously started a tuning study and wish to resume it, simply run the tuning command without the `--finish_now` flag and with the same `--study_id`. The script will detect the existing study and continue adding new trials until the total number of trials (`--n_trials`) is reached.

```bash
python -m src.hyperparameter_tuning.tune \
    --input_path 'GINFINITY/data/train/train.csv' \
    --batch_size 500 \
    --num_epochs 25 \
    --patience 4 \
    --num_workers 8 \
    --device cuda \
    --save_best_weights True \
    --pooling_type global_add_pool \
    --n_trials 100 \
    --benchmark_datasets hard_rfam_benchmark_big \
    --study_id tuning_250128_2030
```

### Viewing Results

After completing the tuning process (either naturally or using `--finish_now`), you can find the results and the best model in the specified `--output_dir`. The structure will be organized as follows:

```
output/
└── hyperparameter_tuning/
    └── tuning_250128_2030/
        ├── trials_logs/
        │   ├── trial_0/
        │   │   ├── trial_0_training.log
        │   │   └── trial_0_model.pth
        │   ├── trial_1/
        │   │   ├── trial_1_training.log
        │   │   └── trial_1_model.pth
        │   └── ...
        ├── command_log.txt
        ├── parameters_cache.json
        ├── tuning_250128_2030_results.csv
        ├── tuning_250128_2030_optimization_history.html
        ├── tuning_250128_2030_param_importances.html
        └── tuning_250128_2030_best_model.pth
```

- **trials_logs/**: Contains logs and model checkpoints for each trial.
- **command_log.txt**: Records the command executed and the configuration used.
- **parameters_cache.json**: Stores the parameters of the study for potential resumption.
- **results.csv**: CSV file summarizing the results of all trials.
- **optimization_history.html**: Interactive plot of the optimization history.
- **param_importances.html**: Interactive plot showing the importance of each hyperparameter.
- **best_model.pth**: The best model weights saved during the tuning process.

## Conclusion

The `hyperparameter_tuning` module streamlines the process of optimizing your GIN model's hyperparameters using Optuna. By following the configurations and examples provided, you can efficiently enhance your model's performance tailored to your specific RNA secondary structure datasets.

For further assistance or to report issues, please refer to the project's issue tracker or contact the maintainer.
