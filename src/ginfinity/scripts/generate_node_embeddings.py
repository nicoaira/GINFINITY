#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
from typing import List, Optional, Tuple

import pandas as pd
import torch
from torch.multiprocessing import Pool, set_start_method, get_start_method
from torch_geometric.data import Batch
from tqdm import tqdm

from ginfinity.model.gin_model import GINModel
from ginfinity.utils import (
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


def _preprocess(args: Tuple[int, str, str, str, str, float, bool]):
    """
    Worker: dot-bracket string -> torch_geometric Data
    args: (idx, uid, struct, log_path, graph_encoding, seq_weight, debug_flag)
    """
    idx, uid, struct, log_path, graph_encoding, seq_weight, debug_preproc = args
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
    data = graph_to_tensor(graph, seq_weight=seq_weight, graph_encoding=graph_encoding) if graph is not None else None
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

    return idx, uid, data


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
        if feature_dim >= 15:
            # Forgi encoding stores an explicit is_base indicator at index 7
            base_scores = x[:, 7]
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
    args: (idx, uid, data, model_path, log_path)
    returns: (idx, uid, serialized_matrix)
    """
    idx, uid, data, model_path, _ = args
    global _cached_model
    if _cached_model is None:
        _cached_model = load_trained_model(model_path, "cpu")
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
    if num_workers and num_workers > 1:
        _ensure_spawn_start_method()
    # Decide which columns to carry through
    final_keep = [id_column]
    if "seq_len" in input_df.columns:
        final_keep.append("seq_len")
    if keep_cols:
        final_keep.extend(keep_cols)

    total_start = time.perf_counter()

    metadata_encoding = 'standard'
    metadata_seq_weight = 0.0
    if model_path:
        temp_model = load_trained_model(model_path, 'cpu')
        if hasattr(temp_model, 'metadata'):
            metadata = temp_model.metadata
            metadata_encoding = metadata.get('graph_encoding', metadata_encoding)
            metadata_seq_weight = float(metadata.get('seq_weight', metadata_seq_weight) or 0.0)
        del temp_model

    graph_encoding = (graph_encoding_override or metadata_encoding or 'standard').lower()
    if graph_encoding not in {'standard', 'forgi'}:
        raise ValueError(f"Unsupported graph encoding '{graph_encoding}'")

    if seq_weight_override is not None:
        seq_weight = float(seq_weight_override)
    else:
        seq_weight = float(metadata_seq_weight)
    seq_weight = max(0.0, min(1.0, seq_weight))

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
    data_list = [data for _, _, data in preproc]

    # 2) Inference to get per-node embeddings
    results = []  # (idx, uid, serialized_matrix)
    inference_start = time.perf_counter()
    if device.lower() == "cpu":
        cpu_tasks = [
            (idx, uid, data, model_path, log_path)
            for (idx, uid), data in zip(meta_list, data_list)
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

    # 3) Assemble output
    assemble_start = time.perf_counter()
    rows = []
    for idx, uid, node_json in results:
        try:
            base = input_df.loc[idx]
        except KeyError:
            log_information(log_path, {"warning": f"Row {idx} missing after inference"})
            continue
        out = {c: base[c] for c in final_keep if c in base}
        out["node_embeddings"] = node_json
        rows.append(out)

    out_df = pd.DataFrame(rows)

    # Reorder: id, optional window_{start,end}, node_embeddings, then the rest
    cols = [id_column]
    if "window_start" in out_df.columns:
        cols.append("window_start")
    if "window_end" in out_df.columns:
        cols.append("window_end")
    cols.append("node_embeddings")
    others = [c for c in out_df.columns if c not in cols]
    out_df = out_df[cols + sorted(others)]

    out_df.to_csv(output_path, sep="\t", index=False, na_rep="NaN")
    assemble_duration = time.perf_counter() - assemble_start
    total_duration = time.perf_counter() - total_start
    output_summary = {
        "num_node_embeddings": len(out_df),
        "duration_s": round(assemble_duration, 3),
        "output_path": output_path,
        "total_duration_s": round(total_duration, 3),
    }
    log_information(log_path, output_summary, "generate_node_embeddings")
    if not quiet:
        print(
            "[generate_node_embeddings] Wrote output: "
            f"rows={output_summary['num_node_embeddings']} "
            f"duration={output_summary['duration_s']}s "
            f"total={output_summary['total_duration_s']}s"
        )
    print(f"Per-node embeddings saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-node embeddings (LxD) before pooling from either raw "
            "dot-bracket TSV/CSV or precomputed PyG graphs."
        )
    )

    # raw-TSV mode
    parser.add_argument("--input", help="Path to raw TSV/CSV with dot-bracket structures.")
    # graph-PT mode
    parser.add_argument("--graph-pt", help="Path to windows_graphs.pt")
    parser.add_argument("--meta-tsv", help="Path to windows_metadata.tsv")

    parser.add_argument("--output", required=True, help="Output TSV for per-node embeddings.")

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
    args = parser.parse_args()

    if args.model_path is None:
        # Resolve default weights relative to installed package
        script_dir = os.path.dirname(__file__)  # .../ginfinity/scripts
        package_dir = os.path.dirname(script_dir)
        default_weights = os.path.join(package_dir, "weights", "gin_weights_regression_180925.pth")
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

    # Graph-PT mode
    if args.graph_pt and args.meta_tsv:
        graph_map = torch.load(args.graph_pt, weights_only=False)
        meta_df = pd.read_csv(args.meta_tsv, sep="\t")
        records = meta_df.to_dict(orient="records")
        datas = [graph_map[r["window_id"]] for r in records]

        log_path = os.path.splitext(args.output)[0] + ".log"
        open(log_path, "a").close()

        results = []  # (meta, serialized_matrix)

        # CPU path
        if args.device.lower() == "cpu":
            if args.num_workers and args.num_workers > 1:
                _ensure_spawn_start_method()
            tasks = [
                (i, rec[args.id_column], datas[i], args.model_path, log_path)
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

        # Assemble & write
        rows = []
        for meta, node_json in results:
            row = meta.copy()
            row["node_embeddings"] = node_json
            rows.append(row)

        out_df = pd.DataFrame(rows)
        # reorder columns if present
        cols = []
        for c in ["window_id", args.id_column, "window_start", "window_end"]:
            if c in out_df.columns:
                cols.append(c)
        cols.append("node_embeddings")
        others = [c for c in out_df.columns if c not in cols]
        out_df = out_df[cols + others]

        out_df.to_csv(args.output, sep="\t", index=False, na_rep="NaN")
        log_information(os.path.splitext(args.output)[0] + ".log", {"num_node_embeddings": len(out_df)}, "generate_node_embeddings")
        print(f"Per-node embeddings saved to {args.output}")
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
