import os
import json
import random
import argparse
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

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
from ginfinity.utils import FORGI_NODE_TYPES, is_valid_dot_bracket, log_information, log_setup
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


def _resolve_diagnostic_dataset_path() -> str:
    env_override = os.environ.get("GINFINITY_DIAGNOSTIC_ALIGNMENT_PATH")
    if env_override:
        return os.path.abspath(os.path.expanduser(env_override))
    return os.path.abspath(os.path.join(os.getcwd(), "dev", "terts.csv"))


def _setup_diagnostic_alignment_context(
    log_path: str,
    output_dir: str,
) -> Optional[Dict[str, Any]]:
    dataset_path = _resolve_diagnostic_dataset_path()
    if not os.path.exists(dataset_path):
        log_information(
            log_path,
            {"status": "missing_dataset", "path": dataset_path},
            "diagnostic_alignment_setup",
        )
        print(f"[diagnostic-alignment] Dataset not found at {dataset_path}; skipping diagnostics.")
        return None

    try:
        df = pd.read_csv(dataset_path)
    except Exception as exc:  # pragma: no cover - defensive I/O guard
        log_information(
            log_path,
            {"status": "read_error", "path": dataset_path, "error": str(exc)},
            "diagnostic_alignment_setup",
        )
        print(f"[diagnostic-alignment] Failed to read {dataset_path}: {exc}")
        return None

    required_cols = {"Name", "DotBracket"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        log_information(
            log_path,
            {
                "status": "missing_columns",
                "path": dataset_path,
                "missing": ",".join(sorted(missing_cols)),
            },
            "diagnostic_alignment_setup",
        )
        print(
            f"[diagnostic-alignment] Required columns {missing_cols} not found in {dataset_path}; skipping diagnostics."
        )
        return None

    if len(df) < 2:
        log_information(
            log_path,
            {"status": "insufficient_rows", "path": dataset_path, "rows": len(df)},
            "diagnostic_alignment_setup",
        )
        print(
            f"[diagnostic-alignment] Expected at least two sequences in {dataset_path}; skipping diagnostics."
        )
        return None

    rna1 = str(df.iloc[0]["Name"])
    rna2 = str(df.iloc[1]["Name"])
    keep_cols = [col for col in ("DotBracket", "seq") if col in df.columns]

    similarity_dir = os.path.join(output_dir, "similarity_matrices")

    log_information(
        log_path,
        {
            "status": "ready",
            "dataset": dataset_path,
            "rna1": rna1,
            "rna2": rna2,
            "keep_cols": ",".join(keep_cols) if keep_cols else "",
            "output_dir": similarity_dir,
        },
        "diagnostic_alignment_setup",
    )

    return {
        "input_path": dataset_path,
        "rna1": rna1,
        "rna2": rna2,
        "id_column": "Name",
        "structure_column": "DotBracket",
        "keep_cols": keep_cols,
        "similarity_dir": similarity_dir,
    }


def _ensure_pythonpath(env: Dict[str, str]) -> Dict[str, str]:
    src_dir = str(Path(__file__).resolve().parents[2])
    existing = env.get("PYTHONPATH")
    if existing:
        paths = existing.split(os.pathsep)
        if src_dir not in paths:
            env["PYTHONPATH"] = os.pathsep.join([src_dir] + paths)
    else:
        env["PYTHONPATH"] = src_dir
    return env


def _run_alignment_diagnostics(
    model: GINModel,
    epoch_index: int,
    cfg: Dict[str, Any],
    device: str,
    log_path: str,
) -> None:
    similarity_dir = cfg["similarity_dir"]
    os.makedirs(similarity_dir, exist_ok=True)

    env = _ensure_pythonpath(os.environ.copy())
    env["PYTHONUNBUFFERED"] = "1"

    with tempfile.TemporaryDirectory(prefix="diagnostic_alignment_") as tmpdir:
        checkpoint_path = os.path.join(tmpdir, f"epoch_{epoch_index:03d}.pth")
        model.save_checkpoint(checkpoint_path)

        node_embeddings_path = os.path.join(tmpdir, "node_embeddings.tsv")
        gen_cmd = [
            sys.executable,
            "-m",
            "ginfinity.scripts.generate_node_embeddings",
            "--input",
            cfg["input_path"],
            "--output",
            node_embeddings_path,
            "--id-column",
            cfg["id_column"],
            "--structure-column-name",
            cfg["structure_column"],
            "--device",
            device,
            "--model-path",
            checkpoint_path,
            "--num-workers",
            "0",
            "--batch-size",
            "32",
            "--quiet",
        ]
        if cfg["keep_cols"]:
            gen_cmd.extend(["--keep-cols", ",".join(cfg["keep_cols"])])

        try:
            subprocess.run(gen_cmd, check=True, env=env)
        except subprocess.CalledProcessError as exc:
            log_information(
                log_path,
                {
                    "epoch": epoch_index,
                    "stage": "generate_node_embeddings",
                    "returncode": exc.returncode,
                    "cmd": " ".join(map(str, exc.cmd)) if isinstance(exc.cmd, (list, tuple)) else str(exc.cmd),
                },
                "diagnostic_alignment_error",
            )
            print(f"[diagnostic-alignment] generate_node_embeddings failed for epoch {epoch_index}: {exc}")
            return

        align_prefix = os.path.join(tmpdir, "alignment")
        align_cmd = [
            sys.executable,
            "-m",
            "ginfinity.scripts.align_node_embeddings",
            "--input",
            node_embeddings_path,
            "--id-column",
            cfg["id_column"],
            "--rna1",
            cfg["rna1"],
            "--rna2",
            cfg["rna2"],
            "--structure-column-name",
            cfg["structure_column"],
            "--output-prefix",
            align_prefix,
            "--plot-matrix",
        ]

        try:
            subprocess.run(align_cmd, check=True, env=env)
        except subprocess.CalledProcessError as exc:
            log_information(
                log_path,
                {
                    "epoch": epoch_index,
                    "stage": "align_node_embeddings",
                    "returncode": exc.returncode,
                    "cmd": " ".join(map(str, exc.cmd)) if isinstance(exc.cmd, (list, tuple)) else str(exc.cmd),
                },
                "diagnostic_alignment_error",
            )
            print(f"[diagnostic-alignment] align_node_embeddings failed for epoch {epoch_index}: {exc}")
            return

        png_source = align_prefix + ".matrix.png"
        if not os.path.exists(png_source):
            log_information(
                log_path,
                {"epoch": epoch_index, "stage": "missing_png", "expected": png_source},
                "diagnostic_alignment_error",
            )
            print(
                f"[diagnostic-alignment] Expected PNG at {png_source} for epoch {epoch_index}, but it was not created."
            )
            return

        destination = os.path.join(similarity_dir, f"epoch_{epoch_index:03d}.png")
        shutil.move(png_source, destination)

        log_information(
            log_path,
            {
                "epoch": epoch_index,
                "png": destination,
                "dataset": cfg["input_path"],
            },
            "diagnostic_alignment",
        )
        print(
            f"[diagnostic-alignment] Saved similarity matrix for epoch {epoch_index} to {destination}"
        )

def save_model_to_local(model, optimizer, epoch, model_id, log_path, output_path=None):
    """Save model checkpoint with metadata and return the storage path."""
    if output_path is None:
        output_path = f"output/{model_id}/{model_id}.pth"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save_checkpoint(output_path, optimizer, epoch)

    save_log = {
        "Model saved path": output_path
    }
    log_information(log_path, save_log)
    return output_path


def alignment_collate_batch(batch):
    """Collate function that merges multiple alignment entries into one batch."""
    if not batch:
        return {"structures": []}

    structures = []
    alignment_ids = []

    for item in batch:
        if not item:
            continue

        item_structures = item.get("structures", [])
        if item_structures:
            structures.extend(item_structures)

        alignment_id = item.get("alignment_id")
        if alignment_id is not None:
            alignment_ids.append(alignment_id)

    result = {"structures": structures}
    if alignment_ids:
        result["alignment_ids"] = alignment_ids

    return result


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
        first_param = next(model.parameters(), None)
        if first_param is not None:
            return first_param.sum() * 0.0
        return torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)

    batch = Batch.from_data_list(structures)
    batch = batch.to(device)
    node_embeddings = model.get_node_embeddings(batch)
    ptr = batch.ptr.tolist()

    # Use wide strides to ensure unique positive labels within a batch.
    label_stride = 10**6
    alignment_offsets = {}
    next_alignment_index = 0

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
        alignment_id = getattr(data, "alignment_id", None)
        alignment_key = alignment_id if alignment_id is not None else graph_idx
        if alignment_key not in alignment_offsets:
            alignment_offsets[alignment_key] = next_alignment_index
            next_alignment_index += 1
        alignment_offset = alignment_offsets[alignment_key] * label_stride

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
                labels.extend([alignment_offset + int(pos) for pos in valid_align_pos])
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

    limit_batches = batch_limit < total_batches

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
            if limit_batches and processed_batches >= batch_limit:
                break

    if progress_bar is not None:
        progress_bar.close()

    if processed_batches == 0:
        return float("nan")

    return total_loss / processed_batches


def _require_bool(value, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"'{field_name}' must be a boolean value (true/false).")


def _read_schedule(schedule_path: str) -> dict:
    with open(schedule_path, "r", encoding="utf-8") as handle:
        schedule_data = json.load(handle)

    if isinstance(schedule_data, list):
        schedule_dict = {
            "start_from_round": 1,
            "checkpoint": None,
            "rounds": schedule_data,
        }
    elif isinstance(schedule_data, dict):
        if "rounds" not in schedule_data:
            raise ValueError("Schedule JSON must contain a 'rounds' list.")
        schedule_dict = schedule_data
    else:
        raise ValueError("Schedule file must contain either a list of rounds or an object with a 'rounds' list.")

    rounds_raw = schedule_dict.get("rounds")
    if not isinstance(rounds_raw, list):
        raise ValueError("'rounds' must be a JSON array of round definitions.")

    start_from_round = schedule_dict.get("start_from_round", 1)
    if not isinstance(start_from_round, int):
        raise ValueError("'start_from_round' must be an integer.")
    if start_from_round < 1:
        raise ValueError("'start_from_round' must be >= 1.")

    checkpoint_path = schedule_dict.get("checkpoint")
    if checkpoint_path is not None:
        if not isinstance(checkpoint_path, str) or not checkpoint_path.strip():
            raise ValueError("'checkpoint' must be a non-empty string path if provided.")
        checkpoint_path = os.path.expandvars(os.path.expanduser(checkpoint_path.strip()))
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    rounds = []
    seen_rounds = set()
    for index, raw in enumerate(rounds_raw):
        if not isinstance(raw, dict):
            raise ValueError(f"Schedule entry at index {index} is not an object.")

        if "round" not in raw:
            raise ValueError(f"Schedule entry at index {index} is missing the 'round' field.")
        round_value = raw["round"]
        if not isinstance(round_value, int):
            raise ValueError(f"Round identifier must be an integer (entry index {index}).")
        if round_value < 1:
            raise ValueError(f"Round identifier must be >= 1 (entry index {index}).")
        round_number = round_value
        if round_number in seen_rounds:
            raise ValueError(f"Duplicate round number '{round_number}' detected in schedule.")
        seen_rounds.add(round_number)

        dataset_path = None
        for key in ("input", "input_path", "dataset", "input_tsv"):
            if key in raw:
                dataset_path = raw[key]
                break
        if dataset_path is None:
            raise ValueError(f"Schedule round {round_number} must include an 'input' dataset path.")
        if not isinstance(dataset_path, str) or not dataset_path.strip():
            raise ValueError(f"Schedule round {round_number} has an invalid dataset path value.")
        dataset_path = os.path.expandvars(os.path.expanduser(dataset_path.strip()))
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"Dataset for round {round_number} not found: {dataset_path}")

        alignment_map_path = None
        for key in ("alignment_map", "alignment_map_path"):
            if key in raw:
                alignment_map_path = raw[key]
                break
        if alignment_map_path is None:
            raise ValueError(f"Schedule round {round_number} must include an 'alignment_map' path.")
        if not isinstance(alignment_map_path, str) or not alignment_map_path.strip():
            raise ValueError(f"Schedule round {round_number} has an invalid alignment_map path value.")
        alignment_map_path = os.path.expandvars(os.path.expanduser(alignment_map_path.strip()))
        if not os.path.isfile(alignment_map_path):
            raise FileNotFoundError(f"Alignment map for round {round_number} not found: {alignment_map_path}")
        try:
            with open(alignment_map_path, "r", encoding="utf-8") as handle:
                json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Alignment map for round {round_number} is not valid JSON: {exc}"
            ) from exc

        if "patience" not in raw:
            raise ValueError(f"Schedule round {round_number} must define 'patience'.")
        patience_value = raw["patience"]
        if not isinstance(patience_value, int):
            raise ValueError(f"'patience' must be an integer in schedule round {round_number}.")
        if patience_value < 1:
            raise ValueError(f"Schedule round {round_number} must have patience >= 1.")

        epoch_key = "epochs" if "epochs" in raw else "num_epochs" if "num_epochs" in raw else None
        if epoch_key is None:
            raise ValueError(f"Schedule round {round_number} must define 'epochs'.")
        epoch_value = raw[epoch_key]
        if not isinstance(epoch_value, int):
            raise ValueError(f"'epochs' must be an integer in schedule round {round_number}.")
        if epoch_value < 1:
            raise ValueError(f"Schedule round {round_number} must have epochs >= 1.")

        lr_key = "learning_rate" if "learning_rate" in raw else "lr" if "lr" in raw else None
        if lr_key is None:
            raise ValueError(f"Schedule round {round_number} must define 'learning_rate'.")
        lr_value = raw[lr_key]
        if isinstance(lr_value, bool) or not isinstance(lr_value, (int, float)):
            raise ValueError(f"'learning_rate' must be a numeric value in schedule round {round_number}.")
        learning_rate = float(lr_value)
        if learning_rate <= 0:
            raise ValueError(f"Schedule round {round_number} must have learning_rate > 0.")

        if "decay_rate" not in raw:
            raise ValueError(f"Schedule round {round_number} must define 'decay_rate'.")
        decay_value = raw["decay_rate"]
        if isinstance(decay_value, bool) or not isinstance(decay_value, (int, float)):
            raise ValueError(f"'decay_rate' must be a numeric value in schedule round {round_number}.")
        decay_rate = float(decay_value)
        if decay_rate <= 0:
            raise ValueError(f"Schedule round {round_number} must have decay_rate > 0.")

        if "keep_weights" not in raw:
            raise ValueError(f"Schedule round {round_number} must define 'keep_weights'.")
        keep_weights = _require_bool(raw["keep_weights"], "keep_weights")

        rounds.append({
            "round": round_number,
            "dataset_path": dataset_path,
            "alignment_map_path": alignment_map_path,
            "patience": patience_value,
            "num_epochs": epoch_value,
            "lr": learning_rate,
            "decay_rate": decay_rate,
            "keep_weights": keep_weights,
            "raw": raw,
        })

    if not rounds:
        raise ValueError("Schedule file does not contain any training rounds.")

    rounds.sort(key=lambda item: item["round"])
    expected_round = 1
    for round_info in rounds:
        if round_info["round"] != expected_round:
            raise ValueError(
                f"Schedule rounds must be sequential starting at 1; expected round {expected_round} but found {round_info['round']}."
            )
        expected_round += 1

    if start_from_round > len(rounds):
        raise ValueError(
            f"'start_from_round' ({start_from_round}) exceeds total rounds ({len(rounds)})."
        )

    if start_from_round > 1 and checkpoint_path is None:
        raise ValueError(
            "'checkpoint' must be provided when 'start_from_round' is greater than 1."
        )

    return {
        "rounds": rounds,
        "start_from_round": start_from_round,
        "checkpoint": checkpoint_path,
    }


def _prepare_dataset(args, dataset_path: str, alignment_map_path: Optional[str]):
    expanded_dataset_path = os.path.expandvars(os.path.expanduser(dataset_path))
    if not os.path.isfile(expanded_dataset_path):
        raise FileNotFoundError(f"Dataset not found: {expanded_dataset_path}")

    df = pd.read_csv(expanded_dataset_path, comment='#', sep='\t', engine='python')

    if args.training_mode == "triplet":
        df = remove_invalid_structures_triplet(df)
    elif args.training_mode == "alignment":
        df = remove_invalid_structures_alignment(df, args.structure_column)
        df = df.groupby("alignment_id", sort=False).filter(lambda group: len(group) >= 2)
        if df.empty:
            raise ValueError("No alignments with at least two structures available after preprocessing the dataset.")

    if df.empty:
        raise ValueError("No data available for training after preprocessing the dataset.")

    if args.f_sample_dataset < 1.0:
        if args.training_mode == "alignment":
            alignment_sizes = df.groupby("alignment_id").size()
            alignment_sizes = alignment_sizes[alignment_sizes >= 2]
            if alignment_sizes.empty:
                raise ValueError("No alignments with at least two structures available for sampling.")

            alignment_ids = alignment_sizes.index.to_list()
            random.shuffle(alignment_ids)

            total_rows = int(alignment_sizes.sum())
            target_rows = int(total_rows * args.f_sample_dataset + 0.5)
            target_rows = max(2, min(target_rows, total_rows))

            selected_ids = []
            accumulated = 0
            for alignment_id in alignment_ids:
                if accumulated >= target_rows:
                    break
                selected_ids.append(alignment_id)
                accumulated += int(alignment_sizes.loc[alignment_id])

            if not selected_ids:
                selected_ids.append(alignment_ids[0])

            df = df[df["alignment_id"].isin(selected_ids)].reset_index(drop=True)
        else:
            sample_size = int(len(df) * args.f_sample_dataset + 0.5)
            sample_size = max(1, min(sample_size, len(df)))
            df = df.sample(n=sample_size, random_state=args.seed, replace=False).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("No data available for training after applying dataset sampling.")

    alignment_map = None
    if args.training_mode == "alignment":
        if "alignment_id" not in df.columns:
            raise ValueError("alignment_id column missing from input for alignment training mode.")
        if not alignment_map_path:
            raise ValueError("alignment_map_path must be provided when using alignment training mode.")
        expanded_map_path = os.path.expandvars(os.path.expanduser(alignment_map_path))
        if not os.path.isfile(expanded_map_path):
            raise FileNotFoundError(f"Alignment map not found: {expanded_map_path}")
        with open(expanded_map_path, "r", encoding="utf-8") as handle:
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

    return df, train_df, val_df, alignment_map, expanded_dataset_path


def _build_dataloaders_and_criterion(args, train_df, val_df, alignment_map):
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
        alignment_max_negatives = args.alignment_max_negatives
        if alignment_max_negatives is not None and alignment_max_negatives <= 0:
            alignment_max_negatives = None

        criterion = AlignmentContrastiveLoss(
            margin=args.alignment_margin,
            hard_negative_fraction=args.hard_negative_fraction,
            max_negatives=alignment_max_negatives,
            temperature=args.alignment_temperature,
            debug=getattr(args, "debug", False),
        )
        alignment_max_unaligned = max(0, int(args.alignment_unaligned_per_graph))
        worker_count = max(0, int(args.num_workers))
        persistent = worker_count > 0
        prefetch_factor = max(1, int(args.alignment_prefetch_factor)) if worker_count > 0 else 2
        alignment_batch_size = args.batch_size
        if worker_count > 0:
            _ensure_open_file_limit()
            try:
                torch.multiprocessing.set_sharing_strategy("file_system")
            except (AttributeError, RuntimeError):
                pass
        train_loader = DataLoader(
            train_dataset,
            batch_size=alignment_batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=worker_count,
            collate_fn=alignment_collate_batch,
            persistent_workers=persistent,
            prefetch_factor=prefetch_factor,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=alignment_batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=worker_count,
            collate_fn=alignment_collate_batch,
            persistent_workers=persistent,
            prefetch_factor=prefetch_factor,
        )

    return train_loader, val_loader, criterion, alignment_max_unaligned


def _create_model(args, hidden_dim):
    loop_feature_dim = 2  # loop size + relative position
    base_pair_dim = 1
    if args.graph_encoding == "forgi":
        seq_feature_dim = 4  # sequence channels always reserved (zeros if seq_weight=0)
        structural_bridge_dim = 1 + len(FORGI_NODE_TYPES)
        node_feature_dim = base_pair_dim + loop_feature_dim + seq_feature_dim + structural_bridge_dim
        edge_feature_dim = 7
    else:
        seq_feature_dim = 4 if args.seq_weight > 0 else 0
        node_feature_dim = base_pair_dim + loop_feature_dim + seq_feature_dim
        edge_feature_dim = 4

    model = GINModel(
        hidden_dim=hidden_dim,
        output_dim=args.output_dim,
        graph_encoding=args.graph_encoding,
        gin_layers=args.gin_layers,
        pooling_type=args.pooling_type,
        dropout=args.dropout,
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        norm_type=args.norm_type,
        node_embed_norm=args.node_embed_norm,
        normalize_nodes_before_pool=args.normalize_nodes_before_pool,
        gin_eps=args.gin_eps,
        train_eps=args.train_eps,
    )
    model.metadata["seq_weight"] = float(args.seq_weight)
    return model


def _load_checkpoint_into_model(model, checkpoint_path: str, device: str) -> None:
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise ValueError(f"Invalid checkpoint file (missing state_dict): {checkpoint_path}")
    model.load_state_dict(state_dict)
    model.to(device)


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
        checkpoint_path: Optional[str] = None,
        diagnostic_alignment: bool = False,
        debug_enabled: bool = False,
):
    """
    Train a GIN model with early stopping.
    
    Args:
        # ...existing args...
        min_delta (float): Minimum change in validation loss to qualify as improvement
    """
    model.to(device)
    if hasattr(criterion, "configure_debug"):
        should_debug = debug_enabled and training_mode == "alignment"
        log_target = log_path if should_debug else None
        criterion.configure_debug(should_debug, log_target)
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
    diagnostic_cfg = _setup_diagnostic_alignment_context(log_path, output_dir) if diagnostic_alignment else None

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

    if diagnostic_cfg is not None:
        _run_alignment_diagnostics(
            model,
            0,
            diagnostic_cfg,
            device,
            log_path,
        )

    saved_checkpoint_path = None

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
                if diagnostic_cfg is not None:
                    _run_alignment_diagnostics(
                        model,
                        epoch + 1,
                        diagnostic_cfg,
                        device,
                        log_path,
                    )

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
                    saved_checkpoint_path = save_model_to_local(
                        model,
                        optimizer,
                        epoch_for_save,
                        model_id,
                        log_path,
                        output_path=checkpoint_path,
                    )
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
        return {
            "interrupted": True,
            "finished_reason": "Interrupted by user",
            "checkpoint_path": saved_checkpoint_path,
            "saved_epoch": saved_epoch_for_plot - 1 if saved_epoch_for_plot is not None else None,
        }

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
    saved_checkpoint_path = save_model_to_local(
        model,
        optimizer,
        epoch_for_save,
        model_id,
        log_path,
        output_path=checkpoint_path,
    )
    saved_epoch_for_plot = epoch_for_save + 1
    plot_loss_curves(train_losses, val_losses, output_dir, log_path, saved_epoch_for_plot)
    return {
        "interrupted": False,
        "finished_reason": finished_reason,
        "checkpoint_path": saved_checkpoint_path,
        "saved_epoch": epoch_for_save,
    }

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Train a GIN model on RNA secondary structures.")
    parser.add_argument('--input_path', type=str, default=None, help='Path to the input CSV/TSV file containing RNA secondary structures.')
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
    parser.add_argument(
        '--diagnostic-aligment',
        dest='diagnostic_alignment',
        action='store_true',
        default=False,
        help='After each new best validation loss, run alignment diagnostics on dev/terts.csv and save the similarity matrix PNG.'
    )
    parser.add_argument(
        '--diagnostic-alignment',
        dest='diagnostic_alignment',
        action='store_true',
        help=argparse.SUPPRESS,
    )
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable verbose debug logging (alignment mode).')
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
    parser.add_argument('--alignment_temperature', type=float, default=0.1,
                        help='Temperature for the alignment contrastive InfoNCE loss (default: 0.1).')
    parser.add_argument('--alignment_max_negatives', type=int, default=5000,
                        help='Maximum additional negatives sampled per alignment batch (<=0 disables extra sampling).')
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
    parser.add_argument('--schedule', type=str, default=None,
                        help='Path to a JSON file describing sequential alignment training rounds.')
    args = parser.parse_args()

    if not math.isfinite(args.initial_eval_fraction) or args.initial_eval_fraction <= 0:
        raise ValueError("initial_eval_fraction must be a positive, finite value.")

    if not math.isfinite(args.f_sample_dataset) or not (0 < args.f_sample_dataset <= 1):
        raise ValueError("f_sample_dataset must be a positive, finite fraction in the interval (0, 1].")

    schedule_plan: Optional[dict] = None
    if args.schedule:
        expanded_schedule_path = os.path.expandvars(os.path.expanduser(args.schedule))
        if not os.path.isfile(expanded_schedule_path):
            raise FileNotFoundError(f"Schedule file not found: {expanded_schedule_path}")
        if args.training_mode != "alignment":
            raise ValueError("--schedule can only be used when training_mode is 'alignment'.")
        if args.input_path:
            raise ValueError("--input_path cannot be used together with --schedule.")
        if args.alignment_map_path:
            raise ValueError("--alignment_map_path cannot be used together with --schedule.")
        schedule_plan = _read_schedule(expanded_schedule_path)
        print("Warning: schedule provided; ignoring CLI patience, lr, num_epochs, and decay_rate.")
    else:
        if not args.input_path:
            raise ValueError("--input_path is required when no schedule is provided.")

    try:
        if ',' in args.hidden_dim:
            hidden_dim = [int(x.strip()) for x in args.hidden_dim.split(',')]
        else:
            hidden_dim = int(args.hidden_dim)
    except ValueError as exc:
        raise ValueError("hidden_dim must be an integer or comma-separated list of integers") from exc

    if args.batch_size < 1:
        raise ValueError("--batch_size must be a positive integer.")

    if args.num_workers is None:
        args.num_workers = max(1, os.cpu_count() // 2)

    random.seed(args.seed)

    device = args.device
    model = _create_model(args, hidden_dim)
    hidden_dims_log = hidden_dim if isinstance(hidden_dim, list) else [hidden_dim] * args.gin_layers

    if schedule_plan is None:
        dataset_path = args.input_path
        df, train_df, val_df, alignment_map, expanded_dataset_path = _prepare_dataset(
            args,
            dataset_path,
            args.alignment_map_path,
        )
        train_loader, val_loader, criterion, alignment_max_unaligned = _build_dataloaders_and_criterion(
            args,
            train_df,
            val_df,
            alignment_map,
        )

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        start_time = time.time()

        output_folder = os.path.join("output", args.model_id)
        os.makedirs(output_folder, exist_ok=True)

        log_path = os.path.join(output_folder, "train.log")
        log_setup(log_path)

        training_params = {
            "train_data_path": expanded_dataset_path,
            "train_data_samples": df.shape[0],
            "hidden_dims": hidden_dims_log,
            "output_dim": args.output_dim,
            "batch_size": args.batch_size,
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
            "debug": args.debug,
        }

        if args.training_mode == "alignment":
            training_params.update({
                "alignment_map_path": args.alignment_map_path,
                "alignment_margin": args.alignment_margin,
                "alignment_unaligned_per_graph": alignment_max_unaligned,
            })

        log_information(log_path, training_params, "Training params")

        default_checkpoint_path = os.path.join(output_folder, f"{args.model_id}.pth")
        training_outcome = train_model_with_early_stopping(
            model,
            args.model_id,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            num_epochs=args.num_epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            device=device,
            log_path=log_path,
            save_best_weights=args.save_best_weights,
            decay_rate=args.decay_rate,
            training_mode=args.training_mode,
            alignment_max_unaligned=alignment_max_unaligned,
            initial_eval_fraction=args.initial_eval_fraction,
            checkpoint_path=default_checkpoint_path,
            diagnostic_alignment=args.diagnostic_alignment,
            debug_enabled=args.debug,
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
    else:
        schedule_rounds = schedule_plan["rounds"]
        start_from_round = schedule_plan["start_from_round"]
        initial_checkpoint_path = schedule_plan["checkpoint"]
        rounds_to_execute = [cfg for cfg in schedule_rounds if cfg["round"] >= start_from_round]
        if not rounds_to_execute:
            raise ValueError(
                "No rounds to execute after applying 'start_from_round'."
            )
        base_output_dir = os.path.join("output", args.model_id)
        os.makedirs(base_output_dir, exist_ok=True)
        schedule_start_time = time.time()

        pending_checkpoint_path: Optional[str] = initial_checkpoint_path
        delete_after_load = False
        schedule_interrupted = False
        executed_rounds = 0

        for exec_idx, round_cfg in enumerate(rounds_to_execute):
            round_start_time = time.time()
            round_number = round_cfg["round"]
            round_label = f"round_{round_number:02d}"
            round_dir = os.path.join(base_output_dir, round_label)
            os.makedirs(round_dir, exist_ok=True)

            log_path = os.path.join(round_dir, "train.log")
            log_setup(log_path)

            raw_cfg = round_cfg.get("raw")
            if raw_cfg:
                log_information(log_path, dict(raw_cfg), "Schedule round config")

            if executed_rounds == 0:
                if pending_checkpoint_path:
                    _load_checkpoint_into_model(model, pending_checkpoint_path, device)
                pending_checkpoint_path = None
                delete_after_load = False
            else:
                if not pending_checkpoint_path:
                    raise RuntimeError(
                        f"Round {round_number} cannot start because no checkpoint was produced by the previous round."
                    )
                _load_checkpoint_into_model(model, pending_checkpoint_path, device)
                if delete_after_load and os.path.exists(pending_checkpoint_path):
                    os.remove(pending_checkpoint_path)
                delete_after_load = False
                pending_checkpoint_path = None

            df, train_df, val_df, alignment_map, expanded_dataset_path = _prepare_dataset(
                args,
                round_cfg["dataset_path"],
                round_cfg["alignment_map_path"],
            )
            train_loader, val_loader, criterion, alignment_max_unaligned = _build_dataloaders_and_criterion(
                args,
                train_df,
                val_df,
                alignment_map,
            )

            current_lr = round_cfg["lr"]
            current_decay = round_cfg["decay_rate"]
            current_patience = round_cfg["patience"]
            current_epochs = round_cfg["num_epochs"]
            keep_weights = round_cfg["keep_weights"]

            optimizer = optim.Adam(model.parameters(), lr=current_lr)

            training_params = {
                "round": round_number,
                "train_data_path": expanded_dataset_path,
                "train_data_samples": df.shape[0],
                "hidden_dims": hidden_dims_log,
                "output_dim": args.output_dim,
                "batch_size": args.batch_size,
                "num_epochs": current_epochs,
                "patience": current_patience,
                "lr": current_lr,
                "decay_rate": current_decay,
                "criterion": "AlignmentContrastiveLoss",
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
                "keep_weights": keep_weights,
                "debug": args.debug,
            }
            training_params.update({
                "alignment_map_path": round_cfg["alignment_map_path"],
                "alignment_margin": args.alignment_margin,
                "alignment_unaligned_per_graph": alignment_max_unaligned,
            })

            log_information(log_path, training_params, "Training params")

            checkpoint_path = os.path.join(round_dir, f"{args.model_id}_{round_label}.pth")
            round_outcome = train_model_with_early_stopping(
                model,
                args.model_id,
                train_loader,
                val_loader,
                optimizer,
                criterion,
                num_epochs=current_epochs,
                patience=current_patience,
                min_delta=args.min_delta,
                device=device,
                log_path=log_path,
                save_best_weights=args.save_best_weights,
                decay_rate=current_decay,
                training_mode=args.training_mode,
                alignment_max_unaligned=alignment_max_unaligned,
                initial_eval_fraction=args.initial_eval_fraction,
                checkpoint_path=checkpoint_path,
                diagnostic_alignment=args.diagnostic_alignment,
                debug_enabled=args.debug,
            )

            round_elapsed_minutes = (time.time() - round_start_time) / 60
            log_information(
                log_path,
                {"Execution time": f"{round_elapsed_minutes:.6f} minutes"},
                "Execution time",
            )

            if round_outcome.get("interrupted"):
                print(
                    f"Round {round_number} interrupted. Elapsed time: {round_elapsed_minutes:.6f} minutes"
                )
                pending_checkpoint_path = round_outcome.get("checkpoint_path")
                delete_after_load = False
                schedule_interrupted = True
                executed_rounds += 1
                break

            print(f"Finished round {round_number}. Execution time: {round_elapsed_minutes:.6f} minutes")

            pending_checkpoint_path = round_outcome.get("checkpoint_path")
            if not pending_checkpoint_path:
                raise RuntimeError(f"Round {round_number} did not produce a checkpoint to pass forward.")

            delete_after_load = not keep_weights
            if delete_after_load and exec_idx == len(rounds_to_execute) - 1 and pending_checkpoint_path:
                if os.path.exists(pending_checkpoint_path):
                    os.remove(pending_checkpoint_path)
                pending_checkpoint_path = None
                delete_after_load = False

            executed_rounds += 1

        total_schedule_minutes = (time.time() - schedule_start_time) / 60
        if schedule_interrupted:
            print(f"Schedule interrupted after {total_schedule_minutes:.6f} minutes")
        else:
            print(f"Schedule completed in {total_schedule_minutes:.6f} minutes")


if __name__ == "__main__":
    main()
