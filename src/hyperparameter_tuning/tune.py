import os
import sys
import json
import argparse
import optuna
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader as GeoDataLoader
from src.model.gin_model import GINModel
from src.triplet_loss import TripletLoss
from src.early_stopping import EarlyStopping
from src.gin_rna_dataset import GINRNADataset
from src.utils import is_valid_dot_bracket, log_information, log_setup, get_project_root
from optuna.integration import PyTorchLightningPruningCallback
import time
from datetime import datetime
from src.benchmark.benchmark import run_benchmark

# Get the project root directory
project_root = get_project_root()

def remove_invalid_structures(df):
    valid_structures = (
        df["structure_A"].apply(is_valid_dot_bracket) & 
        df["structure_P"].apply(is_valid_dot_bracket) & 
        df["structure_N"].apply(is_valid_dot_bracket)
    )
    return df[valid_structures]

def objective(trial, args):
    try:
        # Load data
        dataset_path = args.input_path
        df = pd.read_csv(dataset_path)
        df = remove_invalid_structures(df)
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

        device = args.device

        # Load hyperparameter search space from config
        config = {}
        if args.json_config:
            with open(args.json_config, 'r') as f:
                config = json.load(f)
        else:
            with open(os.path.join(project_root, 'src/hyperparameter_tuning/config.json'), 'r') as f:
                config = json.load(f)

        # Suggest hyperparameters only if they are in config and not overridden by command line
        if "gin_layers" in config and not hasattr(args, "gin_layers"):
            gin_layers = trial.suggest_categorical("gin_layers", config["gin_layers"])
        else:
            gin_layers = getattr(args, "gin_layers", 3)  # default value if not specified

        hidden_dim = []
        if "hidden_dim" in config and not hasattr(args, "hidden_dim"):
            for i in range(gin_layers):
                hidden_dim.append(trial.suggest_categorical(f"hidden_dim_{i}", config["hidden_dim"]))
        else:
            for i in range(gin_layers):
                hidden_dim_val = getattr(args, f"hidden_dim_{i}", getattr(args, "hidden_dim", 128))
                hidden_dim.append(hidden_dim_val)

        if "output_dim" in config and not hasattr(args, "output_dim"):
            output_dim = trial.suggest_categorical("output_dim", config["output_dim"])
        else:
            output_dim = getattr(args, "output_dim", 128)

        if "lr" in config and not hasattr(args, "lr"):
            lr = trial.suggest_float("lr", config["lr"]['min'], config["lr"]['max'], log=True)
        else:
            lr = getattr(args, "lr", 0.001)

        if "decay_rate" in config and not hasattr(args, "decay_rate"):
            decay_rate = trial.suggest_float("decay_rate", config["decay_rate"]['min'], config["decay_rate"]['max'])
        else:
            decay_rate = getattr(args, "decay_rate", 0.95)

        if "dropout" in config and not hasattr(args, "dropout"):
            dropout = trial.suggest_float("dropout", config["dropout"]['min'], config["dropout"]['max'])
        else:
            dropout = getattr(args, "dropout", 0.0)

        # Handle optional hyperparameters
        graph_encoding = args.graph_encoding
        if "graph_encoding" in config and args.graph_encoding == parser.get_default("graph_encoding"):
            graph_encoding = trial.suggest_categorical("graph_encoding", config["graph_encoding"])

        batch_size = args.batch_size
        if "batch_size" in config and args.batch_size == parser.get_default("batch_size"):
            batch_size = trial.suggest_int("batch_size", min(config["batch_size"]), max(config["batch_size"]))

        pooling_type = args.pooling_type
        if "pooling_type" in config and args.pooling_type == parser.get_default("pooling_type"):
            pooling_type = trial.suggest_categorical("pooling_type", config["pooling_type"])

        # Load benchmark metadata
        with open(os.path.join(project_root, 'data/benchmark_datasets/benchmark_datasets.json'), 'r') as f:
            benchmark_metadata = json.load(f)

        # Initialize GIN model with suggested hyperparameters
        model = GINModel(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            graph_encoding=graph_encoding,
            gin_layers=gin_layers,
            pooling_type=pooling_type,
            dropout=dropout
        )

        train_dataset = GINRNADataset(train_df, graph_encoding=graph_encoding)
        val_dataset = GINRNADataset(val_df, graph_encoding=graph_encoding)
        train_loader = GeoDataLoader(train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers=args.num_workers)
        val_loader = GeoDataLoader(val_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=args.num_workers)

        criterion = TripletLoss(margin=1.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Train the model with early stopping
        model.to(device)
        early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
        best_val_loss = float('inf')

        trial_log = []

        for epoch in range(args.num_epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                anchor, positive, negative = batch
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                anchor_out, positive_out, negative_out = model(anchor, positive, negative)
                loss = criterion(anchor_out, positive_out, negative_out)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Apply learning rate decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_rate

            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    anchor, positive, negative = batch
                    anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                    anchor_out, positive_out, negative_out = model(anchor, positive, negative)
                    loss = criterion(anchor_out, positive_out, negative_out)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

            # Log epoch information including learning rate
            current_lr = optimizer.param_groups[0]['lr']
            epoch_log = {
                "Epoch": f"{epoch + 1}/{args.num_epochs}",
                "Training Loss": f"{running_loss / len(train_loader):.6f}",
                "Validation Loss": f"{avg_val_loss:.6f}",
                "Learning Rate": f"{current_lr:.4e}"
            }
            trial_log.append(epoch_log)
            print(f"Epoch {epoch + 1}/{args.num_epochs}, Training Loss: {running_loss / len(train_loader):.6f}, Validation Loss: {avg_val_loss:.6f}, Learning Rate: {current_lr:.4e}")

            # Early stopping check
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                break

        # Restore best weights
        early_stopping.restore_best_weights(model)

        # Create trial log directory inside tuning_timestamp/trials_logs
        trial_log_dir = os.path.join(args.output_dir, args.study_id, "trials_logs", f"trial_{trial.number}")
        os.makedirs(trial_log_dir, exist_ok=True)

        # Save trial log as a .log file
        trial_log_path = os.path.join(trial_log_dir, f"trial_{trial.number}_training.log")
        with open(trial_log_path, 'w') as f:
            for log_entry in trial_log:
                f.write(f"{log_entry}\n")

        # Save the model checkpoint in the trial log directory
        model_checkpoint_path = os.path.join(trial_log_dir, f"trial_{trial.number}_model.pth")
        model.save_checkpoint(model_checkpoint_path)

        # Run benchmark and get average AUCs
        benchmark_results_path = os.path.join(trial_log_dir, "benchmark")
        average_aucs = run_benchmark(
            embeddings_script=os.path.join(project_root, "predict_embedding.py"),
            benchmark_datasets=[args.benchmark_datasets],  # Pass as a list
            benchmark_metadata=benchmark_metadata,
            benchmark_metadata_path=os.path.join(project_root, 'data/benchmark_datasets/benchmark_datasets.json'),
            datasets_dir=os.path.join(project_root, 'data/benchmark_datasets'),
            save_embeddings=False,
            emb_output_path=os.path.join(benchmark_results_path, "embeddings"),
            model_weights_path=model_checkpoint_path,
            structure_column_name="secondary_structure",
            structure_column_num=None,
            header=True,
            skip_barplot=False,
            skip_auc_curve=True,
            results_path=benchmark_results_path,
            save_distances=False,
            no_save=False,
            only_needed_embeddings=True,
            no_log=False,
            device=device,
            num_workers=args.num_workers,
            distance_batch_size=1000,
            quiet=args.quiet_benchmark,
            retries=args.retries  # Pass retries argument
        )

        # Return the average of the average AUCs
        return sum(average_aucs) / len(average_aucs)
    except KeyboardInterrupt:
        raise
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise
    except Exception as e:
        print(f"Trial {trial.number} failed with exception: {e}")
        return None

def save_parameters_cache(args, cache_path):
    """Save parameters to a cache file."""
    with open(cache_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

def load_parameters_cache(cache_path):
    """Load parameters from a cache file."""
    with open(cache_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for GIN model using Optuna.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input CSV/TSV file containing RNA secondary structures.')
    parser.add_argument('--graph_encoding', type=str, choices=['standard', 'forgi'], default='standard', help='Encoding to use for the transformation to graph')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training and validation.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train the model.')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping.')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of worker threads for data loading.')
    parser.add_argument('--save_best_weights', type=bool, default=True, help='Save the best model weights during early stopping.')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training.')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum validation loss decrease to qualify as improvement (default: 0.001)')
    parser.add_argument('--pooling_type', type=str, choices=['global_add_pool', 'set2set'], default='global_add_pool', help='Pooling type to use in the GIN model.')
    parser.add_argument('--storage', type=str, default=f'sqlite:///{os.path.join(project_root, "src/hyperparameter_tuning/db/optuna_studies.db")}', help='Storage URL for Optuna study.')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of trials for hyperparameter optimization.')
    parser.add_argument('--json_config', type=str, default=os.path.join(project_root, 'src/hyperparameter_tuning/config.json'), help='Path to the JSON configuration file for hyperparameter search space.')
    parser.add_argument('--study_id', type=str, default=f"tuning_{datetime.now().strftime('%y%m%d_%H%M')}", help='Study ID for this run.')
    parser.add_argument('--output_dir', type=str, default=os.path.join(project_root, 'output/hyperparameter_tuning'), help='Directory to save the results.')
    parser.add_argument('--benchmark_datasets', type=str, default='hard_rfam_benchmark_big', help='Benchmark datasets to use for evaluation.')
    parser.add_argument('--quiet_benchmark', action='store_true', help='Suppress benchmark output.')
    parser.add_argument('--finish_now', action='store_true', help='Force finish the study and output final results.')
    parser.add_argument('--parameters_cache', type=str, help='Path to the parameters cache file.')
    parser.add_argument('--retries', type=int, default=0, help='Number of retries if the output file is not saved (default: 0).')
    args = parser.parse_args()

    if args.finish_now:
        if not args.parameters_cache:
            args.parameters_cache = os.path.join(project_root, 'output/hyperparameter_tuning', args.study_id, 'parameters_cache.json')
        
        if not os.path.exists(args.parameters_cache):
            raise FileNotFoundError(f"Parameters cache file not found: {args.parameters_cache}")
        
        cached_args = load_parameters_cache(args.parameters_cache)
        args = argparse.Namespace(**cached_args)

        # Check if the study already exists
        study_exists = optuna.study.get_all_study_summaries(storage=args.storage, include_best_trial=False)
        if not any(study.study_name == args.study_id for study in study_exists):
            raise ValueError(f"Study '{args.study_id}' does not exist in storage '{args.storage}'.")

        print(f"Finishing study '{args.study_id}' and outputting final results.")
        study = optuna.load_study(study_name=args.study_id, storage=args.storage)

        # Save study results
        study.trials_dataframe().to_csv(os.path.join(args.output_dir, args.study_id, f"{args.study_id}_results.csv"))

        # Plot optimization history
        optuna.visualization.plot_optimization_history(study).write_html(os.path.join(args.output_dir, args.study_id, f"{args.study_id}_optimization_history.html"))

        # Plot parameter importance
        optuna.visualization.plot_param_importances(study).write_html(os.path.join(args.output_dir, args.study_id, f"{args.study_id}_param_importances.html"))

        # Save best model
        best_trial = study.best_trial
        best_model_path = os.path.join(args.output_dir, args.study_id, f"{args.study_id}_best_model.pth")
        best_model = GINModel(
            hidden_dim=[best_trial.params[f"hidden_dim_{i}"] for i in range(best_trial.params["gin_layers"])],
            output_dim=best_trial.params["output_dim"],
            graph_encoding=args.graph_encoding,
            gin_layers=best_trial.params["gin_layers"],
            pooling_type=args.pooling_type,
            dropout=best_trial.params["dropout"]
        )
        best_model.save_checkpoint(best_model_path)
        print(f"Best model saved to {best_model_path}")
    else:
        # Load JSON configuration if provided
        valid_json_keys = {"graph_encoding", "hidden_dim", "output_dim",
                            "batch_size", "num_epochs", "patience",
                            "lr", "gin_layers", "min_delta", "decay_rate",
                            "pooling_type", "dropout"}
        if args.json_config:
            with open(args.json_config, 'r') as f:
                config = json.load(f)
            for key, value in config.items():
                if key not in valid_json_keys:
                    raise ValueError(f"Invalid key '{key}' in JSON configuration. Valid keys are: {valid_json_keys}")
                if hasattr(args, key):
                    if getattr(args, key) != parser.get_default(key):
                        print(f"Warning: Command line argument for {key} overrides JSON configuration.")
                    else:
                        setattr(args, key, value)

        # Create output directory
        output_dir = os.path.join(args.output_dir, args.study_id)
        os.makedirs(output_dir, exist_ok=True)

        # Save command and configuration
        command_log_path = os.path.join(output_dir, "command_log.txt")
        with open(command_log_path, 'w') as f:
            f.write("Command executed:\n")
            f.write(" ".join(os.sys.argv) + "\n\n")
            f.write("Configuration:\n")
            json.dump(vars(args), f, indent=4)

        # Save parameters cache
        parameters_cache_path = os.path.join(output_dir, 'parameters_cache.json')
        save_parameters_cache(args, parameters_cache_path)

        # Check if the study already exists
        study_exists = optuna.study.get_all_study_summaries(storage=args.storage, include_best_trial=False)
        if any(study.study_name == args.study_id for study in study_exists):
            print(f"Study '{args.study_id}' already exists in storage '{args.storage}'. Resuming the study.")
        else:
            print(f"Study '{args.study_id}' does not exist in storage '{args.storage}'. Starting a new study.")

        # Create Optuna study
        study = optuna.create_study(
            study_name=args.study_id,
            storage=args.storage,
            load_if_exists=True,
            direction='maximize'
        )

        # Optimize
        study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials, catch=(Exception,))

        # Save study results
        study.trials_dataframe().to_csv(os.path.join(output_dir, f"{args.study_id}_results.csv"))

        # Plot optimization history
        optuna.visualization.plot_optimization_history(study).write_html(os.path.join(output_dir, f"{args.study_id}_optimization_history.html"))

        # Plot parameter importance
        optuna.visualization.plot_param_importances(study).write_html(os.path.join(output_dir, f"{args.study_id}_param_importances.html"))

        # Plot timeline
        optuna.visualization.plot_timeline(study).write_html(os.path.join(output_dir, f"{args.study_id}_timeline.html"))

        # Save best model
        best_trial = study.best_trial
        best_model_path = os.path.join(output_dir, f"{args.study_id}_best_model.pth")
        best_model = GINModel(
            hidden_dim=[best_trial.params[f"hidden_dim_{i}"] for i in range(best_trial.params["gin_layers"])],
            output_dim=best_trial.params["output_dim"],
            graph_encoding=args.graph_encoding,
            gin_layers=best_trial.params["gin_layers"],
            pooling_type=args.pooling_type,
            dropout=best_trial.params["dropout"]
        )
        best_model.save_checkpoint(best_model_path)
        print(f"Best model saved to {best_model_path}")

if __name__ == "__main__":
    main()
