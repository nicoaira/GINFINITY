import os
import json
import random
import argparse
import math
from typing import Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

try:
    import resource
except ImportError:  # pragma: no cover - non-posix systems
    resource = None
from ginfinity.training.early_stopping import EarlyStopping
from ginfinity.training.gin_rna_dataset import (
    GINRNADataset,
    GINRNAPairDataset,
    GINAlignmentDataset,
)
from ginfinity.model.gin_model import GINModel
from ginfinity.training.triplet_loss import TripletLoss
from ginfinity.training.alignment_loss import AlignmentContrastiveLoss
from ginfinity.utils import is_valid_dot_bracket, log_information, log_setup
import time
from datetime import datetime


def _ensure_open_file_limit(min_soft: int = 4096) -> None:
    """Raise RLIMIT_NOFILE soft limit if it is suspiciously low."""
    if resource is None or min_soft is None:
        return

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (ValueError, OSError):  # pragma: no cover - platform-dependent
        return

    if soft >= min_soft:
        return

    if hard == resource.RLIM_INFINITY or hard >= min_soft:
        new_soft = min_soft
    else:
        new_soft = hard

    if new_soft <= soft:
        return

    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    except (ValueError, OSError):  # pragma: no cover - insufficient perms
        pass

def remove_invalid_structures_triplet(df):
    """Filters out rows with invalid dot-bracket structures for triplet training."""
    valid_structures = (
        df["anchor_structure"].apply(is_valid_dot_bracket) & 
        df["positive_structure"].apply(is_valid_dot_bracket) & 
        df["negative_structure"].apply(is_valid_dot_bracket)
    )
    return df[valid_structures]

def remove_invalid_structures_alignment(df, structure_column: str):
    """Filters out rows with invalid dot-bracket structures for alignment training."""
    if structure_column not in df.columns:
        raise KeyError(
            f"Structure column '{structure_column}' not found in the input data. "
            "Please specify the correct column name using --structure_column."
        )
    valid_structures = df[structure_column].apply(is_valid_dot_bracket)
    return df[valid_structures]

def save_model_to_local(model, optimizer, epoch, model_id, log_path):
    """Save model checkpoint with metadata"""
    output_path = f"output/{model_id}/{model_id}.pth"
    model.save_checkpoint(output_path, optimizer, epoch)

    save_log = {
        "Model saved path": output_path
    }
    log_information(log_path, save_log)


def alignment_collate_first(batch):
    """Collate function returning the single alignment in the batch."""
    return batch[0]


def compute_alignment_batch_loss(
    batch_item,
    model,
    criterion,
    device,
    max_unaligned_per_graph: int,
    sample_unaligned: bool,
):
    structures = batch_item.get("structures", [])
    if len(structures) < 2:
        dummy = torch.zeros((), device=device, dtype=torch.float32)
        return dummy

    batch = Batch.from_data_list(structures)
    batch = batch.to(device)
    node_embeddings = model.get_node_embeddings(batch)
    ptr = batch.ptr.tolist()

    # Pre-allocate lists with estimated capacity for better performance
    estimated_capacity = sum(len(getattr(data, "_alignment_mapping", {})) + max_unaligned_per_graph 
                           for data in structures)
    global_indices = []
    labels = []
    graph_ids = []
    categories = []
    
    total_nodes = node_embeddings.shape[0]

    # Process all structures in one pass
    for graph_idx, data in enumerate(structures):
        # Get graph boundaries
        graph_start = ptr[graph_idx]
        graph_end = ptr[graph_idx + 1] if graph_idx + 1 < len(ptr) else total_nodes
        graph_size = graph_end - graph_start
        
        # Handle conserved (aligned) positions - vectorized where possible
        mapping = getattr(data, "_alignment_mapping", {}) or {}
        node_categories_tensor = getattr(data, "node_categories", None)
        
        if mapping:
            # Convert mapping to arrays for vectorized processing
            align_positions = list(mapping.keys())
            local_indices = list(mapping.values())
            
            # Filter valid indices in one go
            valid_mask = [(0 <= idx < graph_size) for idx in local_indices]
            valid_align_pos = [align_positions[i] for i, valid in enumerate(valid_mask) if valid]
            valid_local_idx = [local_indices[i] for i, valid in enumerate(valid_mask) if valid]
            
            if valid_align_pos:
                # Vectorized global index calculation
                valid_global_idx = [graph_start + idx for idx in valid_local_idx]
                
                # Batch append to lists
                global_indices.extend(valid_global_idx)
                labels.extend([int(pos) for pos in valid_align_pos])
                graph_ids.extend([graph_idx] * len(valid_align_pos))
                
                # Vectorized category extraction
                if node_categories_tensor is not None:
                    # Ensure both tensors are on the same device
                    valid_indices_tensor = torch.tensor(valid_local_idx, device=node_categories_tensor.device, dtype=torch.long)
                    batch_categories = node_categories_tensor[valid_indices_tensor].tolist()
                else:
                    batch_categories = [2] * len(valid_align_pos)  # Default to unpaired
                categories.extend(batch_categories)

        # Handle unaligned positions - optimized sampling
        if max_unaligned_per_graph > 0:
            unaligned = getattr(data, "unaligned_indices", None)
            if unaligned is not None:
                if torch.is_tensor(unaligned):
                    candidates = unaligned.tolist()
                else:
                    candidates = list(unaligned)
                
                if candidates:
                    # Vectorized filtering
                    valid_candidates = [idx for idx in candidates if 0 <= idx < graph_size]
                    
                    if valid_candidates:
                        sample_size = min(max_unaligned_per_graph, len(valid_candidates))
                        if sample_unaligned and sample_size < len(valid_candidates):
                            selected = random.sample(valid_candidates, sample_size)
                        else:
                            selected = valid_candidates[:sample_size]
                        
                        # Vectorized processing of unaligned nodes
                        base_label = -((graph_idx + 1) * 10**6)
                        selected_global_idx = [graph_start + idx for idx in selected]
                        selected_labels = [base_label - offset for offset, _ in enumerate(selected)]
                        
                        # Batch append
                        global_indices.extend(selected_global_idx)
                        labels.extend(selected_labels)
                        graph_ids.extend([graph_idx] * len(selected))
                        
                        # Vectorized category extraction for unaligned
                        if node_categories_tensor is not None:
                            # Ensure both tensors are on the same device
                            selected_indices_tensor = torch.tensor(selected, device=node_categories_tensor.device, dtype=torch.long)
                            unaligned_categories = node_categories_tensor[selected_indices_tensor].tolist()
                        else:
                            unaligned_categories = [5] * len(selected)  # Default to unaligned-unpaired
                        categories.extend(unaligned_categories)

    if not global_indices:
        return node_embeddings.sum() * 0.0

    # Final bounds check with vectorized operations
    max_index = max(global_indices) if global_indices else 0
    if max_index >= total_nodes:
        print(f"Warning: Max index {max_index} >= total nodes {total_nodes}")
        # Vectorized filtering
        valid_mask = [idx < total_nodes for idx in global_indices]
        global_indices = [idx for idx, valid in zip(global_indices, valid_mask) if valid]
        labels = [label for label, valid in zip(labels, valid_mask) if valid]
        graph_ids = [gid for gid, valid in zip(graph_ids, valid_mask) if valid]
        categories = [cat for cat, valid in zip(categories, valid_mask) if valid]
        
        if not global_indices:
            return node_embeddings.sum() * 0.0

    # Convert to tensors in one go (more efficient than repeated tensor operations)
    index_tensor = torch.tensor(global_indices, device=device, dtype=torch.long)
    labels_tensor = torch.tensor(labels, device=device, dtype=torch.long)
    graph_ids_tensor = torch.tensor(graph_ids, device=device, dtype=torch.long)
    categories_tensor = torch.tensor(categories, device=device, dtype=torch.long)
    
    # Single embedding selection operation
    embeddings = node_embeddings.index_select(0, index_tensor)

    return criterion(embeddings, labels_tensor, graph_ids_tensor, categories_tensor)


def compute_average_loss(
    dataloader,
    model,
    criterion,
    device,
    training_mode,
    desc: Optional[str] = None,
    alignment_max_unaligned: int = 0,
    max_batch_fraction: Optional[float] = None,
):
    """Return the average loss over a dataloader without gradient updates."""
    if len(dataloader) == 0:
        return float("nan")

    total_batches = len(dataloader)
    batch_limit = total_batches
    if max_batch_fraction is not None and total_batches > 0:
        if not math.isfinite(max_batch_fraction):
            batch_limit = total_batches
        else:
            scaled = math.ceil(total_batches * max_batch_fraction)
            batch_limit = min(total_batches, max(1, scaled))

    iterable = enumerate(dataloader)
    progress_bar = None
    if desc:
        progress_bar = tqdm(iterable, total=batch_limit, desc=desc)
        iterator = progress_bar
    else:
        iterator = iterable

    total_loss = 0.0
    processed_batches = 0
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
            elif training_mode == "alignment":
                loss = compute_alignment_batch_loss(
                    batch,
                    model,
                    criterion,
                    device,
                    alignment_max_unaligned,
                    sample_unaligned=False,
                )
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
            processed_batches += 1
            if progress_bar is not None:
                progress_bar.set_postfix({"Loss": total_loss / processed_batches})
            if processed_batches >= batch_limit:
                break

    if progress_bar is not None:
        progress_bar.close()

    if processed_batches == 0:
        return float("nan")

    return total_loss / processed_batches


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
        training_mode="triplet",
        alignment_max_unaligned: int = 0,
        initial_eval_fraction: float = 0.05,
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
        desc="Initial Evaluation - Training",
        alignment_max_unaligned=alignment_max_unaligned,
        max_batch_fraction=initial_eval_fraction,
    )
    initial_val_loss = compute_average_loss(
        val_loader,
        model,
        criterion,
        device,
        training_mode,
        desc="Initial Evaluation - Validation",
        alignment_max_unaligned=alignment_max_unaligned,
        max_batch_fraction=initial_eval_fraction,
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
        "Learning Rate": f"{optimizer.param_groups[0]['lr']}",
        "Initial Evaluation Fraction": f"{initial_eval_fraction}",
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
                elif training_mode == "alignment":
                    loss = compute_alignment_batch_loss(
                        batch,
                        model,
                        criterion,
                        device,
                        alignment_max_unaligned,
                        sample_unaligned=True,
                    )
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
                desc=f"Epoch {epoch + 1}/{num_epochs} - Validation",
                alignment_max_unaligned=alignment_max_unaligned,
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
    parser.add_argument(
        '--f_sample_dataset',
        type=float,
        default=1.0,
        help='Fraction of the dataset to sample before splitting (default: 1.0).'
    )
    parser.add_argument('--initial_eval_fraction', type=float, default=0.05,
                        help='Fraction of batches used during the initial pre-training evaluation (default: 0.05).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data splitting (default: 42)')
    parser.add_argument('--training_mode', choices=['triplet', 'regression', 'alignment'], default='triplet',
                        help='Select the training strategy for the model.')
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
    parser.add_argument('--alignment_map_path', type=str, default=None,
                        help='Path to a JSON file containing conserved position mappings for each alignment.')
    parser.add_argument('--alignment_margin', type=float, default=0.2,
                        help='Cosine similarity margin for negative pairs in alignment training.')
    parser.add_argument('--alignment_unaligned_per_graph', type=int, default=16,
                        help='Maximum number of unaligned nodes per structure to include as negatives.')
    parser.add_argument('--hard_negative_fraction', type=float, default=0.85,
                        help='Fraction of negatives that should be hard negatives (same category). Default: 0.85')
    parser.add_argument('--structure_column', type=str, default='structure',
                        help='Name of the column containing dot-bracket structures.')
    parser.add_argument(
        '--no-preprocessing-progress',
        dest='preprocessing_progress',
        action='store_false',
        help='Disable the preprocessing progress bar.'
    )
    parser.set_defaults(preprocessing_progress=True)
    parser.add_argument(
        '--cache-alignments',
        dest='alignment_cache_preprocessed',
        action='store_true',
        help='Cache graph tensors after the first access (higher RAM, fewer recomputations).'
    )
    parser.add_argument(
        '--no-cache-alignments',
        dest='alignment_cache_preprocessed',
        action='store_false',
        help='Disable caching of graph tensors between iterations (default).'
    )
    parser.set_defaults(alignment_cache_preprocessed=False)
    parser.add_argument(
        '--alignment-prefetch-factor',
        type=int,
        default=1,
        help='Prefetch factor for alignment DataLoader workers (default: 1).'
    )
    parser.add_argument('--gin_eps', type=float, default=0.0,
                        help='GIN epsilon parameter value. If train_eps is True, this is the initial value (default: 0.0).')
    parser.add_argument('--train_eps', action='store_true',
                        help='Make GIN epsilon parameter learnable during training (default: False for fixed epsilon).')
    args = parser.parse_args()

    if not math.isfinite(args.initial_eval_fraction) or args.initial_eval_fraction <= 0:
        raise ValueError("initial_eval_fraction must be a positive, finite value.")

    if not math.isfinite(args.f_sample_dataset) or not (0 < args.f_sample_dataset <= 1):
        raise ValueError("f_sample_dataset must be a positive, finite fraction in the interval (0, 1].")

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

    random.seed(args.seed)

    # Load data
    dataset_path = args.input_path
    df = pd.read_csv(dataset_path, comment='#', sep='\\t', engine='python')
    
    if args.training_mode == "triplet":
        df = remove_invalid_structures_triplet(df)
    elif args.training_mode == "alignment":
        df = remove_invalid_structures_alignment(df, args.structure_column)

    if df.empty:
        raise ValueError("No data available for training after preprocessing the dataset.")

    if args.f_sample_dataset < 1.0:
        sample_size = int(len(df) * args.f_sample_dataset + 0.5)
        sample_size = max(1, min(sample_size, len(df)))
        df = df.sample(n=sample_size, random_state=args.seed, replace=False).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    alignment_map = None
    if args.training_mode == "alignment":
        if "alignment_id" not in df.columns:
            raise ValueError("alignment_id column missing from input for alignment training mode.")
        if not args.alignment_map_path:
            raise ValueError("alignment_map_path must be provided when using alignment training mode.")
        with open(args.alignment_map_path, "r", encoding="utf-8") as handle:
            alignment_map = json.load(handle)
        alignment_ids = df["alignment_id"].unique()
        if len(alignment_ids) == 0:
            raise ValueError("No alignments found in the input dataset.")
        train_ids, val_ids = train_test_split(
            alignment_ids, test_size=args.val_fraction, random_state=args.seed
        )
        train_df = df[df["alignment_id"].isin(train_ids)].reset_index(drop=True)
        val_df = df[df["alignment_id"].isin(val_ids)].reset_index(drop=True)
    else:
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
        gin_eps=args.gin_eps,
        train_eps=args.train_eps,
    )
    alignment_max_unaligned = 0
    if args.training_mode == "triplet":
        train_dataset = GINRNADataset(train_df, graph_encoding=args.graph_encoding, seq_weight=args.seq_weight)
        val_dataset = GINRNADataset(val_df, graph_encoding=args.graph_encoding, seq_weight=args.seq_weight)
        criterion = TripletLoss(margin=1.0)
        train_loader = GeoDataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=args.num_workers,
        )
        val_loader = GeoDataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=args.num_workers,
        )
    elif args.training_mode == "regression":
        train_dataset = GINRNAPairDataset(train_df, graph_encoding=args.graph_encoding, seq_weight=args.seq_weight)
        val_dataset = GINRNAPairDataset(val_df, graph_encoding=args.graph_encoding, seq_weight=args.seq_weight)
        criterion = torch.nn.MSELoss()
        train_loader = GeoDataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=args.num_workers,
        )
        val_loader = GeoDataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=args.num_workers,
        )
    else:
        train_dataset = GINAlignmentDataset(
            train_df,
            alignment_map,
            graph_encoding=args.graph_encoding,
            seq_weight=args.seq_weight,
            structure_column=args.structure_column,
            show_progress=args.preprocessing_progress,
            progress_desc="Preprocessing train alignments",
            cache_preprocessed=args.alignment_cache_preprocessed,
        )
        val_dataset = GINAlignmentDataset(
            val_df,
            alignment_map,
            graph_encoding=args.graph_encoding,
            seq_weight=args.seq_weight,
            structure_column=args.structure_column,
            show_progress=args.preprocessing_progress,
            progress_desc="Preprocessing validation alignments",
            cache_preprocessed=args.alignment_cache_preprocessed,
        )
        criterion = AlignmentContrastiveLoss(
            margin=args.alignment_margin, 
            hard_negative_fraction=args.hard_negative_fraction
        )
        alignment_max_unaligned = max(0, int(args.alignment_unaligned_per_graph))
        worker_count = max(0, int(args.num_workers))
        persistent = worker_count > 0
        prefetch_factor = max(1, int(args.alignment_prefetch_factor)) if worker_count > 0 else 2
        if worker_count > 0:
            _ensure_open_file_limit()
            try:
                torch.multiprocessing.set_sharing_strategy("file_system")
            except (AttributeError, RuntimeError):
                pass
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            pin_memory=False,
            num_workers=worker_count,
            collate_fn=alignment_collate_first,
            persistent_workers=persistent,
            prefetch_factor=prefetch_factor,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=worker_count,
            collate_fn=alignment_collate_first,
            persistent_workers=persistent,
            prefetch_factor=prefetch_factor,
        )

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
        "batch_size": args.batch_size if args.training_mode != "alignment" else 1,
        "num_epochs": args.num_epochs,
        "patience": args.patience,
        "lr": args.lr,
        "criterion": (
            "TripletLoss"
            if args.training_mode == "triplet"
            else "MSELoss"
            if args.training_mode == "regression"
            else "AlignmentContrastiveLoss"
        ),
        "gin_layers": args.gin_layers,
        "graph_encoding": args.graph_encoding,
        "training_mode": args.training_mode,
        "seq_weight": args.seq_weight,
        "norm_type": args.norm_type,
        "node_embed_norm": args.node_embed_norm,
        "normalize_nodes_before_pool": args.normalize_nodes_before_pool,
        "gin_eps": args.gin_eps,
        "train_eps": args.train_eps,
        "initial_eval_fraction": args.initial_eval_fraction,
    }

    if args.training_mode == "alignment":
        training_params.update({
            "alignment_map_path": args.alignment_map_path,
            "alignment_margin": args.alignment_margin,
            "alignment_unaligned_per_graph": alignment_max_unaligned,
        })

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
        training_mode=args.training_mode,
        alignment_max_unaligned=alignment_max_unaligned,
        initial_eval_fraction=args.initial_eval_fraction,
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
