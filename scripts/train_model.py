import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch_geometric.loader import DataLoader as GeoDataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
from tqdm import tqdm
from src.training.early_stopping import EarlyStopping
from src.training.gin_rna_dataset import GINRNADataset
from src.model.gin_model import GINModel
from src.training.triplet_loss import TripletLoss
from src.utils import is_valid_dot_bracket, log_information, log_setup, get_project_root
import time

def remove_invalid_structures(df):
    valid_structures = (
        df["anchor_structure"].apply(is_valid_dot_bracket) & 
        df["positive_structure"].apply(is_valid_dot_bracket) & 
        df["negative_structure"].apply(is_valid_dot_bracket)
    )
    return df[valid_structures]

import torch
from datetime import datetime

def save_model_to_local(model, optimizer, epoch, model_id, log_path):
    """Save model checkpoint with metadata"""
    output_path = f"output/{model_id}/{model_id}.pth"
    model.save_checkpoint(output_path, optimizer, epoch)
    
    save_log = {
        "Model saved path": output_path
    }
    log_information(log_path, save_log)


def train_model_with_early_stopping(
        model,
        model_id,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        num_epochs,
        patience,
        min_delta,
        device,
        log_path,
        save_best_weights=True,
        decay_rate=0.1
):
    """
    Train a GIN model with early stopping and return the best validation loss.
    
    Returns:
        float: Best validation loss observed during training.
    """
    model.to(device)
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    best_val_loss = float('inf')
    best_model_state_dict = None

    early_stopping_params = {
        "Early Stopping Parameters": {
            "patience": patience,
            "min_delta": min_delta
        }
    }
    log_information(log_path, early_stopping_params)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs} - Training")
        for i, batch in progress_bar:
            optimizer.zero_grad()
            anchor, positive, negative = batch
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_out, positive_out, negative_out = model(anchor, positive, negative)
            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix({"Loss": running_loss / (i + 1)})

        # Apply learning rate decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_rate

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            progress_bar_val = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch + 1}/{num_epochs} - Validation")
            for i, batch in progress_bar_val:
                anchor, positive, negative = batch
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                anchor_out, positive_out, negative_out = model(anchor, positive, negative)
                loss = criterion(anchor_out, positive_out, negative_out)
                val_loss += loss.item()
                progress_bar_val.set_postfix({"Val Loss": val_loss / (i + 1)})

        avg_val_loss = val_loss / len(val_loader)

        epoch_log = {
            "Epoch": f"{epoch + 1}/{num_epochs}",
            "Training Loss": f"{running_loss / len(train_loader)}",
            "Validation Loss": f"{avg_val_loss}",
            "Best Validation Loss": f"{early_stopping.best_loss}",
            "Early Stopping Counter": f"{early_stopping.counter}/{patience}",
            "Learning Rate": f"{optimizer.param_groups[0]['lr']}"
        }
        log_information(log_path, epoch_log)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {running_loss / len(train_loader)}, Validation Loss: {avg_val_loss}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if save_best_weights:
                best_model_state_dict = model.state_dict()

        early_stopping(avg_val_loss, model)  # Added model parameter here
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # If early stopping was triggered and we have best weights saved
        if early_stopping.early_stop and save_best_weights:
            early_stopping.restore_best_weights(model)

    # Restore the best model weights if early stopping was triggered and save_best_weights is True
    if early_stopping.early_stop and save_best_weights and best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)

    finished_reason = "Early stopping" if early_stopping.early_stop else f"{epoch+1} epochs"
    log_information(log_path, {"Training finished": finished_reason})
    print("Training complete.")

    save_model_to_local(model, optimizer, epoch, model_id, log_path)

    return best_val_loss

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Train a GIN model on RNA secondary structures.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input CSV/TSV file containing RNA secondary structures.')
    parser.add_argument('--model_id', type=str, default='gin_model', help='Model id')
    parser.add_argument('--graph_encoding', type=str, choices=['standard', 'forgi'], default='standard', help='Encoding to use for the transformation to graph')
    parser.add_argument('--hidden_dim', type=str, default='256', help='Hidden dimension size(s) for the model. Can be a single number or a comma-separated list of numbers of the same size of gin_layers(e.g. "256,126,256)" .')
    parser.add_argument('--output_dim', type=int, default=128, help='Output embedding size for the GIN model.')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training and validation.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train the model.')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--gin_layers', type=int, default=1, help='Number of gin layers.')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of worker threads for data loading.')
    parser.add_argument('--save_best_weights', type=bool, default=True, help='Save the best model weights during early stopping.')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training.')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum validation loss decrease to qualify as improvement (default: 0.001)')
    parser.add_argument('--decay_rate', type=float, default=0.01, help='Decay rate for the learning rate.')
    parser.add_argument('--pooling_type', type=str, choices=['global_add_pool', 'set2set'], default='global_add_pool', help='Pooling type to use in the GIN model.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate for the GIN model (default: 0.0).')
    parser.add_argument('--val_fraction', type=float, default=0.2, help='Fraction of data for validation (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data splitting (default: 42)')
    args = parser.parse_args()
    
    # Process hidden_dim argument
    try:
        if ',' in args.hidden_dim:
            hidden_dim = [int(x.strip()) for x in args.hidden_dim.split(',')]
        else:
            hidden_dim = int(args.hidden_dim)
    except ValueError:
        raise ValueError("hidden_dim must be an integer or comma-separated list of integers")

    if args.num_workers is None:
        args.num_workers = max(1, os.cpu_count() // 2)

    # Load data
    dataset_path = args.input_path
    df = pd.read_csv(dataset_path, comment='#')  # Add comment parameter to skip lines starting with #
    df = remove_invalid_structures(df)
    train_df, val_df = train_test_split(df, test_size=args.val_fraction, random_state=args.seed)

    device = args.device

    # Initialize GIN model with processed hidden_dim
    model = GINModel(
        hidden_dim=hidden_dim,
        output_dim=args.output_dim,
        graph_encoding=args.graph_encoding,
        gin_layers=args.gin_layers,
        pooling_type=args.pooling_type,  # Pass the new argument
        dropout=args.dropout  # Pass the new argument
    )
    train_dataset = GINRNADataset(train_df, graph_encoding=args.graph_encoding)
    val_dataset = GINRNADataset(val_df, graph_encoding=args.graph_encoding)
    train_loader = GeoDataLoader(train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=args.num_workers)
    val_loader = GeoDataLoader(val_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=args.num_workers)

    criterion = TripletLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_time = time.time()

    output_folder = f"output/{args.model_id}"
    os.makedirs(output_folder, exist_ok=True)

    log_path = f"{output_folder}/train.log"
    log_setup(log_path)

    training_params = {
        "train_data_path": dataset_path,
        "train_data_samples": df.shape[0],
        "hidden_dims": hidden_dim if isinstance(hidden_dim, list) else [hidden_dim] * args.gin_layers,
        "output_dim": args.output_dim,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "patience": args.patience,
        "lr": args.lr,
        "criterion": "TripletLoss",
        "gin_layers": args.gin_layers,
        "graph_encoding": args.graph_encoding
    }

    log_information(log_path, training_params, "Training params")
    
    # Train the model with early stopping
    train_model_with_early_stopping(
        model,
        args.model_id,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        num_epochs=args.num_epochs,
        patience=args.patience,
        min_delta=args.min_delta,  # Added parameter
        device=device,
        log_path=log_path,
        save_best_weights=args.save_best_weights,  # Pass the new argument
        decay_rate=args.decay_rate  # Pass the new argument
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60

    print(f"Finished. Total execution time: {execution_time_minutes:.6f} minutes")
    execution_time = {
        "Total execution time" : f"{execution_time_minutes:.6f} minutes"
    }
    log_information(log_path, execution_time, "Execution time")

if __name__ == "__main__":
    main()