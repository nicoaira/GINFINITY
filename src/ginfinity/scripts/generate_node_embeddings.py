#!/usr/bin/env python3

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

import pandas as pd
import torch
from torch.multiprocessing import Pool, set_start_method
from torch_geometric.data import Batch
from tqdm import tqdm

from ginfinity.model.gin_model import GINModel
from ginfinity.utils import (
    setup_and_read_input,
    structure_to_data,
    log_information,
    is_valid_dot_bracket,
)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
_cached_model = None  # per-process cache for CPU workers


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


def _preprocess(args: Tuple[int, str, str, str, str, float]):
    """
    Worker: dot-bracket string -> torch_geometric Data
    args: (idx, uid, struct, log_path)
    """
    idx, uid, struct, log_path, graph_encoding, seq_weight = args
    try:
        if not is_valid_dot_bracket(struct):
            raise ValueError("Invalid dot-bracket")
    except ValueError:
        log_information(log_path, {"skipped_invalid": f"ID {uid}"})
        return None

    data = structure_to_data(
        struct,
        graph_encoding=graph_encoding,
        seq_weight=seq_weight,
    )
    if data is None:
        log_information(log_path, {"skipped_graph_fail": f"ID {uid}"})
        return None

    return idx, uid, data


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
    return idx, uid, _serialize_matrix(node_x)


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
    graph_encoding: str = "standard",
    seq_weight: float = 0.0,
):
    # Decide which columns to carry through
    final_keep = [id_column]
    if "seq_len" in input_df.columns:
        final_keep.append("seq_len")
    if keep_cols:
        final_keep.extend(keep_cols)

    # 1) Preprocess rows -> Data
    tasks = [
        (idx, row[id_column], row[structure_column], log_path, graph_encoding, seq_weight)
        for idx, row in input_df.iterrows()
    ]
    preproc = []
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

    if not preproc:
        print("No valid structures to process.")
        return

    meta_list = [(idx, uid) for idx, uid, _ in preproc]
    data_list = [data for _, _, data in preproc]

    # 2) Inference to get per-node embeddings
    results = []  # (idx, uid, serialized_matrix)
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
            for (idx, uid), mat in zip(metas, per_graph):
                results.append((idx, uid, _serialize_matrix(mat)))
            pbar.update(len(chunk))
        pbar.close()

    # 3) Assemble output
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
    log_information(log_path, {"num_node_embeddings": len(out_df)}, "generate_node_embeddings")
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
    parser.add_argument(
        "--graph-encoding",
        choices=["standard", "forgi"],
        default="standard",
        help="Graph encoding to use when converting dot-bracket structures.",
    )
    parser.add_argument(
        "--seq-weight",
        type=float,
        default=0.0,
        help="Relative weight for nucleotide one-hot features in node embeddings.",
    )
    parser.add_argument("--device", default="cpu", help="Device for inference: 'cpu' or 'cuda'.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker processes for CPU.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for GPU inference.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress bars and extra output.")
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
                for mat, md in zip(per_graph, chunk_md):
                    results.append((md, _serialize_matrix(mat)))
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
        graph_encoding=args.graph_encoding,
        seq_weight=args.seq_weight,
    )


if __name__ == "__main__":
    set_start_method("spawn", force=True)
    main()

