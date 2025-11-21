#!/usr/bin/env python3

import argparse
import io
import json
import os
import sys
import time
import warnings
from typing import List, Optional, Tuple

# Disable CUDA probing in worker processes when requested and silence init warnings there.
if os.environ.get("GINFINITY_DISABLE_CUDA_IN_WORKERS") == "1":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    warnings.filterwarnings(
        "ignore",
        message=r"CUDA initialization: .*",
        module=r"torch.__config__",
    )
# Always silence the noisy CUDA initialization warning; it does not affect CPU runs
# and keeps GPU runs cleaner when CUDA is healthy.
warnings.filterwarnings(
    "ignore",
    message=r"CUDA initialization: .*",
    module=r"torch.__config__",
)

import pandas as pd
import torch
from torch.multiprocessing import Pool, set_start_method, get_start_method
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from ginfinity.model.gin_model import GINModel
from ginfinity.utils import (
    FORGI_NODE_TYPES,
    setup_and_read_input,
    dotbracket_to_graph,
    graph_to_tensor,
    log_information,
    is_valid_dot_bracket,
)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
_cached_model = None  # per-process cache for CPU workers


def _ensure_spawn_start_method():
    """Force torch.multiprocessing to use the safe 'spawn' start method."""
    try:
        current = get_start_method(allow_none=True)
    except RuntimeError:
        current = None
    if current != "spawn":
        try:
            set_start_method("spawn", force=True)
        except RuntimeError:
            # Another component already set an incompatible method; leave as-is.
            pass


def _configure_multiprocessing(num_workers: int) -> None:
    """Set spawn start method and a safe sharing strategy for child processes."""
    if not num_workers or num_workers <= 1:
        return
    # Flag workers to suppress CUDA init to avoid repeated warnings
    os.environ.setdefault("GINFINITY_DISABLE_CUDA_IN_WORKERS", "1")
    _ensure_spawn_start_method()
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except (AttributeError, RuntimeError):
        # Some environments (e.g., stripped-down builds) may not support changing this.
        pass


def load_trained_model(model_path: str, device: str = "cpu") -> GINModel:
    model = GINModel.load_from_checkpoint(model_path, device)
    model.to(device)
    model.eval()
    return model


def _serialize_matrix(mat: torch.Tensor) -> str:
    """Serialize a 2D tensor (L x D) as a compact JSON string.

    Using JSON keeps row boundaries explicit and easy to parse downstream.
    We round to 6 decimals for stability and compactness.
    """
    arr = mat.detach().cpu().tolist()
    # Round for compactness without losing much precision
    rounded = [[round(float(x), 6) for x in row] for row in arr]
    return json.dumps(rounded, separators=(",", ":"))


def _serialize_data_object(data: Data) -> bytes:
    """Serialize a PyG Data object to bytes to avoid FD-based tensor sharing."""
    buffer = io.BytesIO()
    torch.save(data, buffer)
    return buffer.getvalue()


def _deserialize_data_object(blob: bytes) -> Data:
    buffer = io.BytesIO(blob)
    return torch.load(buffer, map_location="cpu", weights_only=False)


def _preprocess(args: Tuple[int, str, str, str, str, float, bool, bool, int]):
    """
    Worker: dot-bracket string -> torch_geometric Data
    args: (idx, uid, struct, log_path, graph_encoding, seq_weight, debug_flag, use_context_features, k_hops)
    """
    idx, uid, struct, log_path, graph_encoding, seq_weight, debug_preproc, use_context_features, k_hops = args
    start_time = time.perf_counter()
    stage_timings = None
    if debug_preproc:
        stage_timings = {
            "id": uid,
            "structure_length": len(struct),
            "graph_encoding": graph_encoding,
        }
        log_information(log_path, {**stage_timings, "status": "started"}, "preprocess_debug")
    try:
        if not is_valid_dot_bracket(struct):
            raise ValueError("Invalid dot-bracket")
    except ValueError:
        if stage_timings is not None:
            stage_timings["status"] = "invalid_structure"
            stage_timings["duration_s"] = round(time.perf_counter() - start_time, 3)
            log_information(log_path, stage_timings, "preprocess_debug")
        log_information(log_path, {"skipped_invalid": f"ID {uid}"})
        return None

    step_start = time.perf_counter()
    graph = dotbracket_to_graph(struct, graph_encoding=graph_encoding)
    graph_time = time.perf_counter() - step_start
    if stage_timings is not None:
        stage_timings["dotbracket_to_graph_s"] = round(graph_time, 3)

    tensor_start = time.perf_counter()
    data = graph_to_tensor(
        graph,
        seq_weight=seq_weight,
        graph_encoding=graph_encoding,
        use_context_features=use_context_features,
        k_hops=k_hops
    ) if graph is not None else None
    tensor_time = time.perf_counter() - tensor_start if graph is not None else 0.0
    if stage_timings is not None:
        stage_timings["graph_to_tensor_s"] = round(tensor_time, 3)

    if graph is None or data is None:
        if stage_timings is not None:
            stage_timings["status"] = "graph_build_failed" if graph is None else "tensor_failed"
            stage_timings["duration_s"] = round(time.perf_counter() - start_time, 3)
            log_information(log_path, stage_timings, "preprocess_debug")
        log_information(log_path, {"skipped_graph_fail": f"ID {uid}"})
        return None

    duration = time.perf_counter() - start_time
    if stage_timings is not None:
        stage_timings["duration_s"] = round(duration, 3)
        stage_timings["status"] = "ok"
        log_information(log_path, stage_timings, "preprocess_debug")
    if duration >= 5.0:
        log_information(
            log_path,
            {
                "id": uid,
                "structure_length": len(struct),
                "duration_s": round(duration, 3),
                "graph_encoding": graph_encoding,
            },
            "preprocess_slow",
        )

    return idx, uid, _serialize_data_object(data)


def _get_base_node_mask(data, device: Optional[torch.device] = None) -> torch.Tensor:
    """Return a boolean mask selecting base nodes for a PyG Data object."""
    mask = getattr(data, "base_node_mask", None)
    if mask is not None:
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask, dtype=torch.bool)
        else:
            mask = mask.to(dtype=torch.bool)
        if device is not None:
            mask = mask.to(device=device)
        return mask

    num_nodes = getattr(data, "num_nodes", None)
    if num_nodes is None and hasattr(data, "x") and isinstance(data.x, torch.Tensor):
        num_nodes = data.x.size(0)
    if num_nodes is None:
        num_nodes = int(getattr(data, "x", torch.empty(0)).size(0))

    num_base_nodes = getattr(data, "num_base_nodes", None)
    if num_base_nodes is not None and num_nodes is not None:
        mask = torch.zeros(int(num_nodes), dtype=torch.bool)
        limit = min(int(num_base_nodes), mask.numel())
        if limit > 0:
            mask[:limit] = True
        if device is not None:
            mask = mask.to(device=device)
        return mask

    x = getattr(data, "x", None)
    if isinstance(x, torch.Tensor) and x.dim() == 2:
        feature_dim = x.size(1)
        minimum_forgi_dim = 9 + len(FORGI_NODE_TYPES)  # paired/unpaired + loops + seq + base + forgi types
        if feature_dim >= minimum_forgi_dim:
            base_indicator_idx = feature_dim - len(FORGI_NODE_TYPES) - 1
            base_scores = x[:, base_indicator_idx]
            mask = base_scores > 0.5
            if device is not None:
                mask = mask.to(device=device)
            return mask

    fallback = torch.ones(int(num_nodes or 0), dtype=torch.bool)
    if device is not None:
        fallback = fallback.to(device=device)
    return fallback


def _filter_to_base_nodes(node_x: torch.Tensor, data) -> torch.Tensor:
    mask = _get_base_node_mask(data, device=node_x.device)
    if mask.numel() != node_x.size(0):
        mask = torch.ones(node_x.size(0), dtype=torch.bool, device=node_x.device)
    return node_x[mask]


def _cpu_node_embed(args: Tuple[int, str, object, str, str]):
    """
    Worker: single-graph per-node embeddings on CPU
    args: (idx, uid, data_blob_or_obj, model_path, log_path)
    returns: (idx, uid, serialized_matrix)
    """
    idx, uid, data_payload, model_path, _ = args
    global _cached_model
    if _cached_model is None:
        _cached_model = load_trained_model(model_path, "cpu")
    if isinstance(data_payload, (bytes, bytearray, memoryview)):
        data = _deserialize_data_object(data_payload)
    else:
        data = data_payload
    with torch.no_grad():
        node_x = _cached_model.get_node_embeddings(data)
        base_x = _filter_to_base_nodes(node_x, data)
    return idx, uid, _serialize_matrix(base_x)


def _split_batch_node_embeddings(node_x: torch.Tensor, batch: Batch) -> List[torch.Tensor]:
    """
    Split concatenated node embeddings (sum L_i x D) into list per-graph.
    Prefer Batch.ptr when available; otherwise use bincount on batch vector.
    """
    if hasattr(batch, "ptr") and batch.ptr is not None:
        idxs = batch.ptr.tolist()  # length = num_graphs + 1
        return [node_x[idxs[i]:idxs[i + 1]] for i in range(len(idxs) - 1)]
    else:
        counts = torch.bincount(batch.batch).tolist()
        out = []
        start = 0
        for c in counts:
            out.append(node_x[start:start + c])
            start += c
        return out


def generate_node_embeddings(
    input_df: pd.DataFrame,
    output_path: str,
    model_path: str,
    log_path: str,
    structure_column: str,
    id_column: str,
    device: str = "cpu",
    num_workers: int = 4,
    batch_size: int = 32,
    keep_cols: Optional[List[str]] = None,
    quiet: bool = False,
    graph_encoding_override: Optional[str] = None,
    seq_weight_override: Optional[float] = None,
    debug_preprocessing: bool = False,
):
    _configure_multiprocessing(num_workers)
    # Decide which columns to carry through
    final_keep = [id_column]
    if "seq_len" in input_df.columns:
        final_keep.append("seq_len")
    if keep_cols:
        final_keep.extend(keep_cols)

    total_start = time.perf_counter()

    metadata_encoding = 'standard'
    metadata_seq_weight = 0.0
    metadata_use_context_features = False
    metadata_k_hops = 2
    if model_path:
        temp_model = load_trained_model(model_path, 'cpu')
        if hasattr(temp_model, 'metadata'):
            metadata = temp_model.metadata
            metadata_encoding = metadata.get('graph_encoding', metadata_encoding)
            metadata_seq_weight = float(metadata.get('seq_weight', metadata_seq_weight) or 0.0)
            metadata_use_context_features = metadata.get('use_context_features', False)
            metadata_k_hops = metadata.get('k_hops', 2)
        del temp_model

    graph_encoding = (graph_encoding_override or metadata_encoding or 'standard').lower()
    if graph_encoding not in {'standard', 'forgi'}:
        raise ValueError(f"Unsupported graph encoding '{graph_encoding}'")

    if seq_weight_override is not None:
        seq_weight = float(seq_weight_override)
    else:
        seq_weight = float(metadata_seq_weight)
    seq_weight = max(0.0, min(1.0, seq_weight))

    use_context_features = metadata_use_context_features
    k_hops = metadata_k_hops

    run_context = {
        "total_rows": len(input_df),
        "graph_encoding": graph_encoding,
        "seq_weight": seq_weight,
        "device": device,
        "num_workers": num_workers,
        "batch_size": batch_size,
        "debug_preprocessing": debug_preprocessing,
    }
    log_information(log_path, run_context, "generate_node_embeddings_config")
    if not quiet:
        print(
            "[generate_node_embeddings] Starting run: "
            f"rows={run_context['total_rows']} device={device} "
            f"graph_encoding={graph_encoding} seq_weight={seq_weight}"
        )

    # 1) Preprocess rows -> Data
    tasks = [
        (
            idx,
            row[id_column],
            row[structure_column],
            log_path,
            graph_encoding,
            seq_weight,
            debug_preprocessing,
            use_context_features,
            k_hops,
        )
        for idx, row in input_df.iterrows()
    ]
    preproc = []
    preprocessing_start = time.perf_counter()
    if num_workers > 1:
        with Pool(num_workers) as pool:
            for res in tqdm(
                pool.imap_unordered(_preprocess, tasks),
                total=len(tasks),
                disable=quiet,
                desc="Preprocessing",
            ):
                if res is not None:
                    preproc.append(res)
    else:
        for t in tqdm(tasks, disable=quiet, desc="Preprocessing"):
            res = _preprocess(t)
            if res is not None:
                preproc.append(res)

    preprocessing_duration = time.perf_counter() - preprocessing_start
    preproc_summary = {
        "input_rows": len(tasks),
        "valid_graphs": len(preproc),
        "skipped": len(tasks) - len(preproc),
        "duration_s": round(preprocessing_duration, 3),
        "used_multiprocessing": num_workers > 1,
    }
    log_information(log_path, preproc_summary, "preprocessing_summary")
    if not quiet:
        print(
            "[generate_node_embeddings] Preprocessing complete: "
            f"valid={preproc_summary['valid_graphs']}/{len(tasks)} "
            f"skipped={preproc_summary['skipped']} "
            f"duration={preproc_summary['duration_s']}s"
        )

    if not preproc:
        print("No valid structures to process.")
        return

    meta_list = [(idx, uid) for idx, uid, _ in preproc]
    data_blobs = [blob for _, _, blob in preproc]

    # 2) Inference to get per-node embeddings
    results = []  # (idx, uid, serialized_matrix)
    inference_start = time.perf_counter()
    if device.lower() == "cpu":
        cpu_tasks = [
            (idx, uid, blob, model_path, log_path)
            for (idx, uid), blob in zip(meta_list, data_blobs)
        ]
        with Pool(num_workers) as pool:
            for idx, uid, emb_str in tqdm(
                pool.imap_unordered(_cpu_node_embed, cpu_tasks),
                total=len(cpu_tasks),
                disable=quiet,
                desc="Per-node embeddings (CPU)",
            ):
                results.append((idx, uid, emb_str))
    else:
        data_list = [_deserialize_data_object(blob) for blob in data_blobs]
        model = load_trained_model(model_path, device)
        pbar = tqdm(
            total=len(data_list),
            disable=quiet,
            desc="Per-node embeddings (GPU)",
            unit=" samples",
        )
        for start in range(0, len(data_list), batch_size):
            chunk = data_list[start : start + batch_size]
            metas = meta_list[start : start + batch_size]
            batch = Batch.from_data_list(chunk).to(device)
            with torch.no_grad():
                node_x = model.get_node_embeddings(batch)
            per_graph = _split_batch_node_embeddings(node_x, batch)
            for (idx, uid), mat, data in zip(metas, per_graph, chunk):
                base_mat = _filter_to_base_nodes(mat, data)
                results.append((idx, uid, _serialize_matrix(base_mat)))
            pbar.update(len(chunk))
        pbar.close()

    inference_duration = time.perf_counter() - inference_start
    inference_summary = {
        "graphs_processed": len(results),
        "duration_s": round(inference_duration, 3),
        "device": device,
    }
    log_information(log_path, inference_summary, "inference_summary")
    if not quiet:
        print(
            "[generate_node_embeddings] Inference complete: "
            f"graphs={inference_summary['graphs_processed']} "
            f"duration={inference_summary['duration_s']}s"
        )

    # 3) Assemble output - flat format (one row per node)
    assemble_start = time.perf_counter()
    rows = []
    for idx, uid, node_json in results:
        try:
            base = input_df.loc[idx]
        except KeyError:
            log_information(log_path, {"warning": f"Row {idx} missing after inference"})
            continue

        # Deserialize the node embeddings matrix (List of lists: L x D)
        node_matrix = json.loads(node_json)

        # Create one row per node
        for node_idx, embedding in enumerate(node_matrix):
            row = {
                id_column: base[id_column],
                "node_index": node_idx,
                "embedding_vector": ",".join(f"{v:.6f}" for v in embedding)
            }
            rows.append(row)

    out_df = pd.DataFrame(rows)

    # Column order: id_column, node_index, embedding_vector
    cols = [id_column, "node_index", "embedding_vector"]
    out_df = out_df[cols]

    out_df.to_csv(output_path, sep="\t", index=False, na_rep="NaN")
    assemble_duration = time.perf_counter() - assemble_start
    total_duration = time.perf_counter() - total_start
    output_summary = {
        "num_node_rows": len(out_df),
        "num_molecules": len(results),
        "duration_s": round(assemble_duration, 3),
        "output_path": output_path,
        "total_duration_s": round(total_duration, 3),
    }
    log_information(log_path, output_summary, "generate_node_embeddings")
    if not quiet:
        print(
            "[generate_node_embeddings] Wrote output: "
            f"node_rows={output_summary['num_node_rows']} "
            f"molecules={output_summary['num_molecules']} "
            f"duration={output_summary['duration_s']}s "
            f"total={output_summary['total_duration_s']}s"
        )
    print(f"Per-node embeddings (flat format) saved to {output_path}")


def preprocess_and_save(
    input_df: pd.DataFrame,
    graph_output_path: str,
    meta_output_path: str,
    log_path: str,
    structure_column: str,
    id_column: str,
    num_workers: int = 4,
    graph_chunk_size: Optional[int] = None,
    keep_cols: Optional[List[str]] = None,
    quiet: bool = False,
    graph_encoding_override: Optional[str] = None,
    seq_weight_override: Optional[float] = None,
    debug_preprocessing: bool = False,
    model_path: Optional[str] = None,
):
    """Preprocess structures and save graphs+metadata for later inference."""
    _configure_multiprocessing(num_workers)

    # Decide which columns to carry through
    final_keep = [id_column]
    if "seq_len" in input_df.columns:
        final_keep.append("seq_len")
    if keep_cols:
        final_keep.extend(keep_cols)

    total_start = time.perf_counter()

    # Load metadata from model if available
    metadata_encoding = 'standard'
    metadata_seq_weight = 0.0
    metadata_use_context_features = False
    metadata_k_hops = 2
    if model_path:
        temp_model = load_trained_model(model_path, 'cpu')
        if hasattr(temp_model, 'metadata'):
            metadata = temp_model.metadata
            metadata_encoding = metadata.get('graph_encoding', metadata_encoding)
            metadata_seq_weight = float(metadata.get('seq_weight', metadata_seq_weight) or 0.0)
            metadata_use_context_features = metadata.get('use_context_features', False)
            metadata_k_hops = metadata.get('k_hops', 2)
        del temp_model

    graph_encoding = (graph_encoding_override or metadata_encoding or 'standard').lower()
    if graph_encoding not in {'standard', 'forgi'}:
        raise ValueError(f"Unsupported graph encoding '{graph_encoding}'")

    if seq_weight_override is not None:
        seq_weight = float(seq_weight_override)
    else:
        seq_weight = float(metadata_seq_weight)
    seq_weight = max(0.0, min(1.0, seq_weight))

    use_context_features = metadata_use_context_features
    k_hops = metadata_k_hops

    run_context = {
        "total_rows": len(input_df),
        "graph_encoding": graph_encoding,
        "seq_weight": seq_weight,
        "num_workers": num_workers,
        "debug_preprocessing": debug_preprocessing,
        "graph_chunk_size": graph_chunk_size,
        "mode": "preprocess_only",
    }
    log_information(log_path, run_context, "preprocess_only_config")
    if not quiet:
        print(
            "[preprocess_only] Starting preprocessing: "
            f"rows={run_context['total_rows']} "
            f"graph_encoding={graph_encoding} seq_weight={seq_weight}"
        )

    # Preprocess rows -> Data
    tasks = [
        (
            idx,
            row[id_column],
            row[structure_column],
            log_path,
            graph_encoding,
            seq_weight,
            debug_preprocessing,
            use_context_features,
            k_hops,
        )
        for idx, row in input_df.iterrows()
    ]
    preproc = []
    preprocessing_start = time.perf_counter()
    if num_workers > 1:
        with Pool(num_workers) as pool:
            for res in tqdm(
                pool.imap_unordered(_preprocess, tasks),
                total=len(tasks),
                disable=quiet,
                desc="Preprocessing",
            ):
                if res is not None:
                    preproc.append(res)
    else:
        for t in tqdm(tasks, disable=quiet, desc="Preprocessing"):
            res = _preprocess(t)
            if res is not None:
                preproc.append(res)

    preprocessing_duration = time.perf_counter() - preprocessing_start
    preproc_summary = {
        "input_rows": len(tasks),
        "valid_graphs": len(preproc),
        "skipped": len(tasks) - len(preproc),
        "duration_s": round(preprocessing_duration, 3),
        "used_multiprocessing": num_workers > 1,
    }
    log_information(log_path, preproc_summary, "preprocessing_summary")
    if not quiet:
        print(
            "[preprocess_only] Preprocessing complete: "
            f"valid={preproc_summary['valid_graphs']}/{len(tasks)} "
            f"skipped={preproc_summary['skipped']} "
            f"duration={preproc_summary['duration_s']}s"
        )

    if not preproc:
        print("No valid structures to process.")
        return

    # Build graph dictionary and metadata
    graph_items = []
    meta_rows = []
    for idx, uid, data_blob in preproc:
        graph_items.append((uid, data_blob))
        try:
            base = input_df.loc[idx]
        except KeyError:
            log_information(log_path, {"warning": f"Row {idx} missing in metadata assembly"})
            continue
        meta = {c: base[c] for c in final_keep if c in base}
        meta_rows.append(meta)

    # Save graphs (optionally chunked)
    def _with_chunk_suffix(path: str, chunk_idx: int) -> str:
        base, ext = os.path.splitext(path)
        return f"{base}.part{chunk_idx:03d}{ext}"

    meta_map = {row[id_column]: row for row in meta_rows if id_column in row}
    save_start = time.perf_counter()
    total_graphs = len(graph_items)

    if graph_chunk_size and graph_chunk_size > 0:
        chunk = 0
        written = 0
        while written < total_graphs:
            slice_items = graph_items[written:written + graph_chunk_size]
            graphs = {uid: _deserialize_data_object(blob) for uid, blob in slice_items}
            meta_chunk = []
            missing_meta = []
            for uid, _ in slice_items:
                meta = meta_map.get(uid)
                if meta is not None:
                    meta_chunk.append(meta)
                else:
                    missing_meta.append(uid)
            g_path = _with_chunk_suffix(graph_output_path, chunk)
            m_path = _with_chunk_suffix(meta_output_path, chunk)
            torch.save(graphs, g_path)
            pd.DataFrame(meta_chunk).to_csv(m_path, sep="\t", index=False, na_rep="NaN")
            if not quiet:
                print(f"[preprocess_only] Saved chunk {chunk} ({len(graphs)} graphs) to {g_path}")
                if missing_meta:
                    print(f"[preprocess_only] Warning: missing metadata for {len(missing_meta)} graphs in chunk {chunk}")
            log_information(
                log_path,
                {
                    "chunk_index": chunk,
                    "graphs_in_chunk": len(graphs),
                    "meta_rows_in_chunk": len(meta_chunk),
                    "missing_meta": missing_meta,
                    "graph_output": g_path,
                    "meta_output": m_path,
                },
                "preprocess_only_chunk",
            )
            written += len(graphs)
            chunk += 1
        chunk_count = chunk
    else:
        graphs = {uid: _deserialize_data_object(blob) for uid, blob in graph_items}
        torch.save(graphs, graph_output_path)
        if not quiet:
            print(f"[preprocess_only] Saved {len(graphs)} graphs to {graph_output_path}")

        meta_df = pd.DataFrame(meta_rows)
        meta_df.to_csv(meta_output_path, sep="\t", index=False, na_rep="NaN")
        if not quiet:
            print(f"[preprocess_only] Saved metadata to {meta_output_path}")
        chunk_count = 1

    save_duration = time.perf_counter() - save_start
    total_duration = time.perf_counter() - total_start

    summary = {
        "graphs_saved": total_graphs,
        "chunks_written": chunk_count,
        "chunk_size": graph_chunk_size,
        "save_duration_s": round(save_duration, 3),
        "total_duration_s": round(total_duration, 3),
        "graph_output": graph_output_path,
        "meta_output": meta_output_path,
    }
    log_information(log_path, summary, "preprocess_only_summary")
    if not quiet:
        print(
            f"[preprocess_only] Complete: {summary['graphs_saved']} graphs "
            f"in {summary['total_duration_s']}s across {chunk_count} chunk(s)"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-node embeddings from either raw dot-bracket TSV/CSV or "
            "precomputed PyG graphs. Output is in flat format: one row per node "
            "with columns [id, node_index, embedding_vector]."
        )
    )

    # raw-TSV mode
    parser.add_argument("--input", help="Path to raw TSV/CSV with dot-bracket structures.")
    # graph-PT mode
    parser.add_argument("--graph-pt", help="Path to windows_graphs.pt")
    parser.add_argument("--meta-tsv", help="Path to windows_metadata.tsv")

    parser.add_argument("--output", required=True, help="Output TSV for per-node embeddings in flat format (one row per node).")

    parser.add_argument(
        "--model-path",
        required=False,
        default=None,
        help=(
            "(Optional) Path to pretrained GIN checkpoint. If omitted, uses the "
            "built-in default weights from ginfinity."
        ),
    )

    parser.add_argument("--id-column", required=True, help="Column name for unique IDs.")
    parser.add_argument(
        "--structure-column-name",
        default="secondary_structure",
        help="(raw) column name for dot-bracket.",
    )
    parser.add_argument("--keep-cols", default=None, help="Comma-separated list of extra columns to carry through.")
    parser.add_argument("--device", default="cpu", help="Device for inference: 'cpu' or 'cuda'.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker processes for CPU.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for GPU inference.")
    parser.add_argument(
        "--graph-encoding",
        choices=["standard", "forgi"],
        default=None,
        help="Override graph encoding for preprocessing. Defaults to the checkpoint metadata.",
    )
    parser.add_argument(
        "--seq-weight",
        type=float,
        default=None,
        help="Override sequence feature weight used during preprocessing (0-1). Defaults to the checkpoint metadata.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress bars and extra output.")
    parser.add_argument(
        "--debug-preprocessing",
        action="store_true",
        help="Log per-structure preprocessing timings to the run log for debugging.",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Only preprocess structures to PyG graphs and save. No inference.",
    )
    parser.add_argument(
        "--graph-output",
        help="(preprocess-only) Path to save preprocessed graphs (.pt file).",
    )
    parser.add_argument(
        "--meta-output",
        help="(preprocess-only) Path to save metadata (.tsv file).",
    )
    parser.add_argument(
        "--graph-chunk-size",
        type=int,
        default=None,
        help="(preprocess-only) Number of graphs per saved chunk. When set, writes multiple *.partXXX.pt and *.partXXX.tsv files.",
    )
    args = parser.parse_args()

    if args.model_path is None:
        # Resolve default weights relative to installed package
        script_dir = os.path.dirname(__file__)  # .../ginfinity/scripts
        package_dir = os.path.dirname(script_dir)
        default_weights = os.path.join(package_dir, "weights", "gin_weights_alignment_271025.pth")
        if not os.path.exists(default_weights):
            sys.exit(
                "ERROR: Default weights not found at %s. Please install ginfinity correctly or pass --model-path." % default_weights
            )
        args.model_path = default_weights
        if not args.quiet:
            print(
                "[generate_node_embeddings] No --model-path given, using built-in weights at:\n    %s\n"
                % args.model_path
            )

    # Preprocess-only mode
    if args.preprocess_only:
        if not args.input:
            sys.exit("ERROR: --preprocess-only requires --input (raw TSV/CSV with structures).")
        if not args.graph_output:
            sys.exit("ERROR: --preprocess-only requires --graph-output (path to save .pt file).")
        if not args.meta_output:
            sys.exit("ERROR: --preprocess-only requires --meta-output (path to save .tsv file).")

        df, log_path, propagate = setup_and_read_input(args, need_model=False)
        preprocess_and_save(
            input_df=df,
            graph_output_path=args.graph_output,
            meta_output_path=args.meta_output,
            log_path=log_path,
            structure_column=args.structure_column_name,
            id_column=args.id_column,
            num_workers=args.num_workers,
            graph_chunk_size=args.graph_chunk_size,
            keep_cols=propagate,
            quiet=args.quiet,
            graph_encoding_override=args.graph_encoding,
            seq_weight_override=args.seq_weight,
            debug_preprocessing=args.debug_preprocessing,
            model_path=args.model_path,
        )
        sys.exit(0)

    # Graph-PT mode
    if args.graph_pt and args.meta_tsv:
        graph_map = torch.load(args.graph_pt, weights_only=False)
        meta_df = pd.read_csv(args.meta_tsv, sep="\t")
        records = meta_df.to_dict(orient="records")

        # Support both windowed (window_id) and regular (id_column) formats
        if "window_id" in meta_df.columns:
            key_column = "window_id"
        elif args.id_column in meta_df.columns:
            key_column = args.id_column
        else:
            sys.exit(f"ERROR: Metadata must contain either 'window_id' or '{args.id_column}' column.")

        datas = [graph_map[r[key_column]] for r in records]

        log_path = os.path.splitext(args.output)[0] + ".log"
        open(log_path, "a").close()

        results = []  # (meta, serialized_matrix)

        # CPU path
        if args.device.lower() == "cpu":
            _configure_multiprocessing(args.num_workers)
            data_payloads = [
                _serialize_data_object(d) if not isinstance(d, (bytes, bytearray, memoryview)) else d
                for d in datas
            ]
            tasks = [
                (i, rec[args.id_column], data_payloads[i], args.model_path, log_path)
                for i, rec in enumerate(records)
            ]
            with Pool(args.num_workers) as pool:
                for idx, uid, node_json in tqdm(
                    pool.imap_unordered(_cpu_node_embed, tasks),
                    total=len(tasks),
                    disable=args.quiet,
                    desc="Per-node graph embeddings (CPU)",
                ):
                    results.append((records[idx], node_json))
        # GPU path
        else:
            model = load_trained_model(args.model_path, args.device)
            pbar = tqdm(
                total=len(datas),
                disable=args.quiet,
                desc="Per-node graph embeddings (GPU)",
                unit=" samples",
            )
            for start in range(0, len(datas), args.batch_size):
                chunk = datas[start : start + args.batch_size]
                chunk_md = records[start : start + args.batch_size]
                batch = Batch.from_data_list(chunk).to(args.device)
                with torch.no_grad():
                    node_x = model.get_node_embeddings(batch)
                per_graph = _split_batch_node_embeddings(node_x, batch)
                for mat, md, data in zip(per_graph, chunk_md, chunk):
                    base_mat = _filter_to_base_nodes(mat, data)
                    results.append((md, _serialize_matrix(base_mat)))
                pbar.update(len(chunk))
            pbar.close()

        # Assemble & write - flat format (one row per node)
        rows = []
        for meta, node_json in results:
            # Deserialize the node embeddings matrix (List of lists: L x D)
            node_matrix = json.loads(node_json)

            # Get the molecule ID from metadata
            mol_id = meta[args.id_column]

            # Create one row per node
            for node_idx, embedding in enumerate(node_matrix):
                row = {
                    args.id_column: mol_id,
                    "node_index": node_idx,
                    "embedding_vector": ",".join(f"{v:.6f}" for v in embedding)
                }
                rows.append(row)

        out_df = pd.DataFrame(rows)
        # Column order: id_column, node_index, embedding_vector
        cols = [args.id_column, "node_index", "embedding_vector"]
        out_df = out_df[cols]

        out_df.to_csv(args.output, sep="\t", index=False, na_rep="NaN")
        log_information(os.path.splitext(args.output)[0] + ".log", {"num_node_rows": len(out_df), "num_molecules": len(results)}, "generate_node_embeddings")
        print(f"Per-node embeddings (flat format) saved to {args.output}: {len(out_df)} node rows from {len(results)} molecules")
        sys.exit(0)

    # Otherwise raw TSV/CSV mode
    df, log_path, propagate = setup_and_read_input(args, need_model=True)
    generate_node_embeddings(
        input_df=df,
        output_path=args.output,
        model_path=args.model_path,
        log_path=log_path,
        structure_column=args.structure_column_name,
        id_column=args.id_column,
        device=args.device,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        keep_cols=propagate,
        quiet=args.quiet,
        graph_encoding_override=args.graph_encoding,
        seq_weight_override=args.seq_weight,
        debug_preprocessing=args.debug_preprocessing,
    )


if __name__ == "__main__":
    set_start_method("spawn", force=True)
    main()
