#!/usr/bin/env python3

import argparse
from typing import Optional
import pandas as pd
import torch
import os
import sys
from tqdm import tqdm
from torch.multiprocessing import Pool, set_start_method
from torch_geometric.data import Batch

from ginfinity.model.gin_model import GINModel
from ginfinity.utils import (
    setup_and_read_input,
    dotbracket_to_graph,
    graph_to_tensor,
    log_information,
    is_valid_dot_bracket
)

# Cache for CPU workers
_cached_model = None

def load_trained_model(model_path, device='cpu'):
    model = GINModel.load_from_checkpoint(model_path, device)
    model.to(device)
    model.eval()
    return model

def _cpu_embed(args):
    """
    Worker: single-graph inference on CPU via forward_once
    args: (idx, uid, data, model_path, log_path)
    """
    idx, uid, data, model_path, log_path = args
    global _cached_model
    if _cached_model is None:
        _cached_model = load_trained_model(model_path, 'cpu')
    with torch.no_grad():
        emb = _cached_model.forward_once(data)
    vec = emb.cpu().numpy().flatten()
    emb_str = ','.join(f'{x:.6f}' for x in vec)
    return idx, uid, emb_str

def _preprocess(args):
    """
    Worker: dot-bracket string → torch_geometric Data
    args: (idx, uid, struct, log_path)
    """
    idx, uid, struct, log_path, graph_encoding, seq_weight = args
    try:
        if not is_valid_dot_bracket(struct):
            raise ValueError("Invalid dot-bracket")
    except ValueError:
        log_information(log_path, {"skipped_invalid": f"ID {uid}"})
        return None

    graph = dotbracket_to_graph(struct, graph_encoding=graph_encoding)
    data  = graph_to_tensor(graph, seq_weight=seq_weight, graph_encoding=graph_encoding)

    if graph is None or data is None:
        log_information(log_path, {"skipped_graph_fail": f"ID {uid}"})
        return None

    return idx, uid, data


def generate_embeddings(
        input_df: pd.DataFrame,
        output_path: str,
        model_path: str,
        log_path: str,
        structure_column: str,
        id_column: str,
        device: str        = 'cpu',
        num_workers: int   = 4,
        batch_size: int    = 32,
        keep_cols: list     = None,
        quiet: bool        = False,
        graph_encoding_override: Optional[str] = None,
        seq_weight_override: Optional[float] = None,
):
    # Decide which columns to carry through
    final_keep = [id_column]
    if 'seq_len' in input_df.columns:
        final_keep.append('seq_len')
    if keep_cols:
        final_keep.extend(keep_cols)

    # Determine graph encoding + sequence weighting
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

    # 1) Preprocess all rows in parallel
    tasks = [
        (idx, row[id_column], row[structure_column], log_path, graph_encoding, seq_weight)
        for idx, row in input_df.iterrows()
    ]
    preproc = []
    if num_workers > 1:
        with Pool(num_workers) as pool:
            for res in tqdm(pool.imap_unordered(_preprocess, tasks), total=len(tasks), disable=quiet, desc="Preprocessing"):
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
    data_list = [data      for _, _, data in preproc]

    # 2) Inference
    embedding_results = []

    if device.lower() == 'cpu':
        cpu_tasks = [
            (idx, uid, data, model_path, log_path)
            for (idx, uid), data in zip(meta_list, data_list)
        ]
        with Pool(num_workers) as pool:
            for idx, uid, emb in tqdm(pool.imap_unordered(_cpu_embed, cpu_tasks), total=len(cpu_tasks), disable=quiet, desc="Embedding (CPU)"):
                embedding_results.append((idx, uid, emb))
    else:
        model = load_trained_model(model_path, device)
        total_batches = (len(data_list) + batch_size - 1) // batch_size
        pbar = tqdm(total=len(data_list), disable=quiet, desc="Embedding (GPU)", unit=" samples")
        for start in range(0, len(data_list), batch_size):
            chunk = data_list[start:start + batch_size]
            metas = meta_list[start:start + batch_size]
            batch = Batch.from_data_list(chunk).to(device)
            with torch.no_grad():
                out = model.forward_once(batch)
            emb_np = out.cpu().numpy()
            for (idx, uid), vec in zip(metas, emb_np):
                emb_str = ','.join(f'{x:.6f}' for x in vec.flatten())
                embedding_results.append((idx, uid, emb_str))
            pbar.update(len(chunk))
        pbar.close()

    # 3) Assemble and write output
    rows = []
    for idx, uid, emb_str in embedding_results:
        if not emb_str:
            log_information(log_path, {"skipped_empty": f"ID {uid}"})
            continue
        try:
            base = input_df.loc[idx]
        except KeyError:
            log_information(log_path, {"warning": f"Row {idx} missing after embedding"})
            continue
        out = {c: base[c] for c in final_keep if c in base}
        out['embedding_vector'] = emb_str
        rows.append(out)

    out_df = pd.DataFrame(rows)

    # Reorder: id, optional window_{start,end}, embedding_vector, then the rest
    cols = [id_column]
    if 'window_start' in out_df.columns:
        cols.append('window_start')
    if 'window_end' in out_df.columns:
        cols.append('window_end')
    cols.append('embedding_vector')
    others = [c for c in out_df.columns if c not in cols]
    out_df = out_df[cols + sorted(others)]

    out_df.to_csv(output_path, sep='\t', index=False, na_rep='NaN')
    log_information(log_path, {"num_embeddings": len(out_df)}, "generate_embeddings")
    print(f"Embeddings saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate high-quality embeddings from precomputed graphs or raw dot-bracket TSV."
    )

    # raw-TSV mode
    parser.add_argument('--input', help="Path to raw TSV/CSV with dot-bracket structures.")
    # graph-PT mode
    parser.add_argument('--graph-pt', help="Path to windows_graphs.pt")
    parser.add_argument('--meta-tsv', help="Path to windows_metadata.tsv")

    parser.add_argument('--output', required=True,
                        help="Output TSV for embeddings.")

    parser.add_argument('--model-path', required=False, default=None,
                        help="(Optional) Path to pretrained GIN checkpoint. If omitted, uses the built-in default weights from ginfinity.")

    parser.add_argument('--id-column', required=True,
                        help="Column name for unique IDs in raw or metadata TSV.")
    parser.add_argument('--structure-column-name', default="secondary_structure",
                        help="(raw) column name for dot-bracket.")
    parser.add_argument('--keep-cols', default=None,
                        help="Comma-separated list of extra columns to carry through.")
    parser.add_argument('--device', default='cpu',
                        help="Device for inference: 'cpu' or 'cuda'.")
    parser.add_argument('--num-workers', type=int, default=4,
                        help="Number of worker processes for CPU.")
    parser.add_argument('--batch-size', type=int, default=32,
                        help="Batch size for GPU inference.")
    parser.add_argument('--graph-encoding', choices=['standard', 'forgi'], default=None,
                        help="Override graph encoding for preprocessing. Defaults to the value stored in the checkpoint metadata.")
    parser.add_argument('--seq-weight', type=float, default=None,
                        help="Override sequence feature weight used during preprocessing (0-1). Defaults to the checkpoint metadata.")
    parser.add_argument('--quiet', action='store_true',
                        help="Suppress progress bars and extra output.")
    args = parser.parse_args()

    if args.model_path is None:
        # Determine the directory of this script, then go up one level to locate the "weights" folder in ginfinity.
        # __file__ is something like ".../site-packages/ginfinity/scripts/generate_embeddings.py"
        script_dir = os.path.dirname(__file__)                # .../site-packages/ginfinity/scripts
        package_dir = os.path.dirname(script_dir)             # .../site-packages/ginfinity
        default_weights = os.path.join(package_dir, "weights", "gin_weights_regression_180925.pth")

        if not os.path.exists(default_weights):
            sys.exit(f"ERROR: Default weights not found at {default_weights}. "
                     "Please either install ginfinity correctly or pass --model-path.")
        args.model_path = default_weights
        # Print a message so the user knows which weights are being used:
        if not args.quiet:
            print(f"[generate_embeddings] No --model-path given, using built-in weights at:\n    {args.model_path}\n")

    # If using precomputed graphs:
    if args.graph_pt and args.meta_tsv:
        # load with weights_only=False to allow PyG classes
        graph_map = torch.load(args.graph_pt, weights_only=False)
        meta_df   = pd.read_csv(args.meta_tsv, sep='\t')
        records   = meta_df.to_dict(orient='records')
        datas     = [graph_map[r['window_id']] for r in records]

        log_path = os.path.splitext(args.output)[0] + '.log'
        open(log_path, 'a').close()

        embedding_results = []

        # CPU path
        if args.device.lower() == 'cpu':
            tasks = [
                (i, rec[args.id_column], datas[i], args.model_path, log_path)
                for i, rec in enumerate(records)
            ]
            with Pool(args.num_workers) as pool:
                for idx, uid, emb in tqdm(
                    pool.imap_unordered(_cpu_embed, tasks),
                    total=len(tasks),
                    disable=args.quiet,
                    desc="Embedding graphs (CPU)"
                ):
                    embedding_results.append((records[idx], emb))
        # GPU path
        else:
            model = load_trained_model(args.model_path, args.device)
            pbar = tqdm(
                total=len(datas),
                disable=args.quiet,
                desc="Embedding graphs (GPU)",
                unit=" samples"
            )
            for start in range(0, len(datas), args.batch_size):
                chunk    = datas[start:start + args.batch_size]
                chunk_md = records[start:start + args.batch_size]
                batch    = Batch.from_data_list(chunk).to(args.device)
                with torch.no_grad():
                    out = model.forward_once(batch)
                emb_np = out.cpu().numpy()
                for vec, md in zip(emb_np, chunk_md):
                    emb_str = ','.join(f'{x:.6f}' for x in vec.flatten())
                    embedding_results.append((md, emb_str))
                pbar.update(len(chunk))
            pbar.close()

        # assemble & write
        rows = []
        for meta, emb_str in embedding_results:
            row = meta.copy()
            row['embedding_vector'] = emb_str
            rows.append(row)

        out_df = pd.DataFrame(rows)
        # reorder columns
        cols = []
        for c in ['window_id', args.id_column, 'window_start', 'window_end']:
            if c in out_df.columns:
                cols.append(c)
        cols.append('embedding_vector')
        others = [c for c in out_df.columns if c not in cols]
        out_df = out_df[cols + others]

        out_df.to_csv(args.output, sep='\t', index=False, na_rep='NaN')
        log_information(log_path, {"num_embeddings": len(out_df)}, "generate_embeddings")
        print(f"Embeddings saved to {args.output}")
        sys.exit(0)

    # Otherwise raw-TSV → embedding
    df, log_path, propagate = setup_and_read_input(args, need_model=True)
    generate_embeddings(
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
    )

if __name__ == "__main__":
    set_start_method('spawn', force=True)
    main()
