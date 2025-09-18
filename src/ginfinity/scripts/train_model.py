import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch_geometric.loader import DataLoader as GeoDataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
from tqdm import tqdm
from ginfinity.training.early_stopping import EarlyStopping
from ginfinity.training.gin_rna_dataset import GINRNADataset, GINRNAPairDataset
from ginfinity.model.gin_model import GINModel
from ginfinity.training.triplet_loss import TripletLoss
from ginfinity.utils import is_valid_dot_bracket, log_information, log_setup
import time
from datetime import datetime
from typing import Optional

def remove_invalid_structures(df):
    valid_structures = (
        df["anchor_structure"].apply(is_valid_dot_bracket) & 
        df["positive_structure"].apply(is_valid_dot_bracket) & 
        df["negative_structure"].apply(is_valid_dot_bracket)
    )
    return df[valid_structures]

def save_model_to_local(model, optimizer, epoch, model_id, log_path):
    """Save model checkpoint with metadata"""
    output_path = f"output/{model_id}/{model_id}.pth"
    model.save_checkpoint(output_path, optimizer, epoch)

    save_log = {
        "Model saved path": output_path
    }
    log_information(log_path, save_log)


def compute_average_loss(dataloader, model, criterion, device, training_mode, desc: Optional[str] = None):
    """Return the average loss over a dataloader without gradient updates."""
    if len(dataloader) == 0:
        return float("nan")

    iterator = enumerate(dataloader)
    if desc:
        iterator = tqdm(iterator, total=len(dataloader), desc=desc)

    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i, batch in iterator:
            if training_mode == "triplet":
                anchor, positive, negative = batch
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)
                anchor_out, positive_out, negative_out = model(anchor, positive, negative)
                loss = criterion(anchor_out, positive_out, negative_out)
            else:
                anchor, positive, target = batch
                anchor = anchor.to(device)
                positive = positive.to(device)
                target = target.to(device)
                anchor_out = model.forward_once(anchor)
                positive_out = model.forward_once(positive)
                pred = 1 - F.cosine_similarity(anchor_out, positive_out)
                loss = criterion(pred, target.view(-1))
            total_loss += loss.item()
            if desc:
                iterator.set_postfix({"Loss": total_loss / (i + 1)})

    return total_loss / len(dataloader)


def plot_loss_curves(train_losses, val_losses, output_dir, log_path, saved_epoch=None):
    """Save a PNG plot with training and validation loss curves."""
    if not train_losses or not val_losses:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        log_information(
            log_path,
            {"Loss plot": f"Skipped (matplotlib unavailable: {exc})"}
        )
        return

    epochs = list(range(len(train_losses)))
    plt.figure()
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True, alpha=0.3)

    if saved_epoch is not None:
        plt.axvline(
            saved_epoch,
            linestyle="--",
            color="red",
            linewidth=1.0,
            label="Saved Weights"
        )
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(output_path)
    plt.close()

    log_information(log_path, {"Loss plot saved": output_path})


def train_model_with_early_stopping(
        model,
        model_id,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        num_epochs,
        patience,
        min_delta,  # Added parameter
        device,
        log_path,
        save_best_weights=True,
        decay_rate=0.1,  # Add decay_rate parameter
        training_mode="triplet"
):
    """
    Train a GIN model with early stopping.
    
    Args:
        # ...existing args...
        min_delta (float): Minimum change in validation loss to qualify as improvement
    """
    model.to(device)
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    best_val_loss = float('inf')
    best_model_state_dict = None
    best_model_epoch = None
    interrupted = False
    finished_reason = None
    last_epoch = -1
    train_losses = []
    val_losses = []
    output_dir = os.path.dirname(log_path)
    saved_epoch_for_plot = None

    early_stopping_params = {
        "Early Stopping Parameters": {
            "patience": patience,
            "min_delta": min_delta
        }
    }
    log_information(log_path, early_stopping_params)

    initial_train_loss = compute_average_loss(
        train_loader,
        model,
        criterion,
        device,
        training_mode,
        desc="Initial Evaluation - Training"
    )
    initial_val_loss = compute_average_loss(
        val_loader,
        model,
        criterion,
        device,
        training_mode,
        desc="Initial Evaluation - Validation"
    )

    best_val_loss = initial_val_loss
    if save_best_weights:
        best_model_state_dict = model.state_dict()
        best_model_epoch = -1
    train_losses.append(initial_train_loss)
    val_losses.append(initial_val_loss)
    early_stopping.best_loss = initial_val_loss
    early_stopping.best_state_dict = model.state_dict().copy()
    early_stopping.counter = 0

    initial_log = {
        "Epoch": f"0/{num_epochs}",
        "Training Loss": f"{initial_train_loss}",
        "Validation Loss": f"{initial_val_loss}",
        "Best Validation Loss": f"{best_val_loss}",
        "Early Stopping Counter": f"{early_stopping.counter}/{patience}",
        "Learning Rate": f"{optimizer.param_groups[0]['lr']}"
    }
    log_information(log_path, initial_log)
    print(
        f"Epoch 0/{num_epochs}, Training Loss: {initial_train_loss}, "
        f"Validation Loss: {initial_val_loss}"
    )

    try:
        for epoch in range(num_epochs):
            last_epoch = epoch
            # Training phase
            model.train()
            running_loss = 0.0
            progress_bar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}/{num_epochs} - Training"
            )
            for i, batch in progress_bar:
                optimizer.zero_grad()
                if training_mode == "triplet":
                    anchor, positive, negative = batch
                    anchor = anchor.to(device)
                    positive = positive.to(device)
                    negative = negative.to(device)
                    anchor_out, positive_out, negative_out = model(anchor, positive, negative)
                    loss = criterion(anchor_out, positive_out, negative_out)
                else:
                    anchor, positive, target = batch
                    anchor = anchor.to(device)
                    positive = positive.to(device)
                    target = target.to(device)
                    anchor_out = model.forward_once(anchor)
                    positive_out = model.forward_once(positive)
                    pred = 1 - F.cosine_similarity(anchor_out, positive_out)
                    loss = criterion(pred, target.view(-1))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                progress_bar.set_postfix({"Loss": running_loss / (i + 1)})

            # Apply learning rate decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_rate

            avg_train_loss = running_loss / len(train_loader)
            avg_val_loss = compute_average_loss(
                val_loader,
                model,
                criterion,
                device,
                training_mode,
                desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"
            )
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if save_best_weights:
                    best_model_state_dict = model.state_dict()
                    best_model_epoch = epoch

            early_stopping(avg_val_loss, model)

            epoch_log = {
                "Epoch": f"{epoch + 1}/{num_epochs}",
                "Training Loss": f"{avg_train_loss}",
                "Validation Loss": f"{avg_val_loss}",
                "Best Validation Loss": f"{best_val_loss}",
                "Early Stopping Counter": f"{early_stopping.counter}/{patience}",
                "Learning Rate": f"{optimizer.param_groups[0]['lr']}"
            }
            log_information(log_path, epoch_log)
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss}, "
                f"Validation Loss: {avg_val_loss}"
            )

            if early_stopping.early_stop:
                print("Early stopping")
                finished_reason = "Early stopping"
                break
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        interrupted = True

    if interrupted:
        log_information(log_path, {"Training finished": "Interrupted by user"})
        if save_best_weights and best_model_state_dict is not None:
            while True:
                try:
                    response = input(
                        "Do you want to save the model with the best weights? [y/n]: "
                    ).strip().lower()
                except EOFError:
                    response = "n"
                except KeyboardInterrupt:
                    print("\nSkipping save of best weights.")
                    response = "n"
                if response in ("y", "yes"):
                    model.load_state_dict(best_model_state_dict)
                    epoch_for_save = best_model_epoch if best_model_epoch is not None else last_epoch
                    epoch_for_save = max(epoch_for_save, 0)
                    save_model_to_local(model, optimizer, epoch_for_save, model_id, log_path)
                    saved_epoch_for_plot = epoch_for_save + 1
                    log_information(log_path, {"Best weights saved after interrupt": True})
                    break
                if response in ("n", "no", ""):
                    print("Best weights were not saved.")
                    log_information(log_path, {"Best weights saved after interrupt": False})
                    break
                print("Please respond with 'y' or 'n'.")
        else:
            print("No best weights available to save.")
        plot_loss_curves(train_losses, val_losses, output_dir, log_path, saved_epoch_for_plot)
        return {"interrupted": True, "finished_reason": "Interrupted by user"}

    if finished_reason is None:
        finished_reason = f"{last_epoch + 1} epochs" if last_epoch >= 0 else "0 epochs"

    epoch_for_save = max(last_epoch, 0)
    if early_stopping.early_stop and save_best_weights and best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
        if best_model_epoch is not None:
            epoch_for_save = best_model_epoch

    log_information(log_path, {"Training finished": finished_reason})
    print("Training complete.")

    epoch_for_save = max(epoch_for_save, 0)
    save_model_to_local(model, optimizer, epoch_for_save, model_id, log_path)
    saved_epoch_for_plot = epoch_for_save + 1
    plot_loss_curves(train_losses, val_losses, output_dir, log_path, saved_epoch_for_plot)
    return {"interrupted": False, "finished_reason": finished_reason}

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
    parser.add_argument('--pooling_type', type=str, choices=['global_add_pool','global_mean_pool', 'set2set'], default='global_add_pool', help='Pooling type to use in the GIN model.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate for the GIN model (default: 0.0).')
    parser.add_argument('--val_fraction', type=float, default=0.2, help='Fraction of data for validation (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data splitting (default: 42)')
    parser.add_argument('--training_mode', choices=['triplet', 'regression'], default='triplet',
                        help='Use triplet loss or regression on f_total_modifications')
    parser.add_argument('--seq_weight', type=float, default=0.0,
                        help='Weight of nucleotide sequence one-hot features relative to pairing state (0-1).')
    parser.add_argument('--norm_type', type=str,
                        choices=['none', 'batch', 'graph', 'layer', 'instance'], default='graph',
                        help='Normalization layer to apply after each graph convolution.')
    parser.add_argument('--node_embed_norm', type=str,
                        choices=['none', 'l2', 'zscore', 'zscore_l2'], default='none',
                        help='Post-hoc normalization to apply to node embeddings.')
    parser.add_argument('--normalize_nodes_before_pool', action='store_true',
                        help='Normalize node embeddings prior to graph pooling.')
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
    loop_feature_dim = 2  # loop size + relative position
    base_feature_dim = 4 if args.seq_weight > 0 else 0
    node_feature_dim = 1 + loop_feature_dim + base_feature_dim
    model = GINModel(
        hidden_dim=hidden_dim,
        output_dim=args.output_dim,
        graph_encoding=args.graph_encoding,
        gin_layers=args.gin_layers,
        pooling_type=args.pooling_type,  # Pass the new argument
        dropout=args.dropout,  # Pass the new argument
        node_feature_dim=node_feature_dim,
        edge_feature_dim=4,
        norm_type=args.norm_type,
        node_embed_norm=args.node_embed_norm,
        normalize_nodes_before_pool=args.normalize_nodes_before_pool,
    )
    if args.training_mode == "triplet":
        train_dataset = GINRNADataset(train_df, graph_encoding=args.graph_encoding, seq_weight=args.seq_weight)
        val_dataset = GINRNADataset(val_df, graph_encoding=args.graph_encoding, seq_weight=args.seq_weight)
        criterion = TripletLoss(margin=1.0)
    else:
        train_dataset = GINRNAPairDataset(train_df, graph_encoding=args.graph_encoding, seq_weight=args.seq_weight)
        val_dataset = GINRNAPairDataset(val_df, graph_encoding=args.graph_encoding, seq_weight=args.seq_weight)
        criterion = torch.nn.MSELoss()
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
        "criterion": "TripletLoss" if args.training_mode == "triplet" else "MSELoss",
        "gin_layers": args.gin_layers,
        "graph_encoding": args.graph_encoding,
        "training_mode": args.training_mode,
        "seq_weight": args.seq_weight,
        "norm_type": args.norm_type,
        "node_embed_norm": args.node_embed_norm,
        "normalize_nodes_before_pool": args.normalize_nodes_before_pool,
    }

    log_information(log_path, training_params, "Training params")
    
    # Train the model with early stopping
    training_outcome = train_model_with_early_stopping(
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
        decay_rate=args.decay_rate,  # Pass the new argument
        training_mode=args.training_mode
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60

    outcome = training_outcome or {}
    if outcome.get("interrupted"):
        print(f"Interrupted. Total elapsed time: {execution_time_minutes:.6f} minutes")
        execution_time = {
            "Total elapsed time before interrupt": f"{execution_time_minutes:.6f} minutes"
        }
    else:
        print(f"Finished. Total execution time: {execution_time_minutes:.6f} minutes")
        execution_time = {
            "Total execution time": f"{execution_time_minutes:.6f} minutes"
        }
    log_information(log_path, execution_time, "Execution time")

if __name__ == "__main__":
    main()
