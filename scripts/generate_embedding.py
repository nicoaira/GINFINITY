#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import pandas as pd
import torch
from tqdm import tqdm
from torch.multiprocessing import Pool, set_start_method
from torch_geometric.data import Batch

from src.model.gin_model import GINModel
from src.utils import (
    dotbracket_to_graph,
    graph_to_tensor,
    dotbracket_to_forgi_graph,
    forgi_graph_to_tensor,
    read_input_data,
    get_structure_column_name,
    log_setup,
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

def _preprocess(args):
    """
    Worker: dot-bracket string → torch_geometric Data
    args: (idx, uid, struct, graph_encoding, log_path)
    """
    idx, uid, struct, graph_encoding, log_path = args
    try:
        if not is_valid_dot_bracket(struct):
            raise ValueError("Invalid dot-bracket")
    except ValueError:
        log_information(log_path, {"skipped_invalid": f"ID {uid}"})
        return None

    if graph_encoding == 'standard':
        graph = dotbracket_to_graph(struct)
        data  = graph_to_tensor(graph)
    else:
        graph = dotbracket_to_forgi_graph(struct)
        data  = forgi_graph_to_tensor(graph)

    if graph is None or data is None:
        log_information(log_path, {"skipped_graph_fail": f"ID {uid}"})
        return None

    return (idx, uid, data)

def _cpu_embed(args):
    """
    Worker: single-graph inference on CPU via forward_once
    args: (idx, uid, data, model_path, log_path)
    """
    idx, uid, data, model_path, log_path = args
    global _cached_model
    if _cached_model is None:
        _cached_model = load_trained_model(model_path, 'cpu')
    model = _cached_model

    with torch.no_grad():
        emb = model.forward_once(data)
    vec = emb.cpu().numpy().flatten()
    emb_str = ','.join(f'{x:.6f}' for x in vec)
    return (idx, uid, emb_str)

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
        keep_cols: str     = None
):
    # Setup logging
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    log_setup(log_path)

    # Decide which columns to carry through
    final_keep = [id_column]
    if 'seq_len' in input_df.columns:
        final_keep.append('seq_len')
    if keep_cols:
        for c in [c.strip() for c in keep_cols.split(',')]:
            if c in input_df.columns and c not in final_keep:
                final_keep.append(c)

    # Load model once on CPU to inspect metadata
    temp_model     = load_trained_model(model_path, 'cpu')
    graph_encoding = getattr(temp_model.metadata, 'graph_encoding', 'standard')
    del temp_model

    # 1) Preprocess all rows in parallel
    tasks = [
        (idx, row[id_column], row[structure_column], graph_encoding, log_path)
        for idx, row in input_df.iterrows()
    ]
    with Pool(num_workers) as pool:
        preproc = pool.map(_preprocess, tasks)
    preproc = [x for x in preproc if x is not None]

    if not preproc:
        print("No valid structures to process.")
        return

    meta_list = [(idx, uid) for idx, uid, _ in preproc]
    data_list = [data      for _, _, data in preproc]

    # 2) Inference
    embedding_results = []

    if device.lower() == 'cpu':
        # CPU: one worker per graph, using forward_once
        cpu_tasks = [
            (idx, uid, data, model_path, log_path)
            for (idx, uid), data in zip(meta_list, data_list)
        ]
        with Pool(num_workers) as pool:
            embedding_results = pool.map(_cpu_embed, cpu_tasks)
    else:
        # GPU: batched forward passes
        model = load_trained_model(model_path, device)
        for start in range(0, len(data_list), batch_size):
            chunk = data_list[start:start + batch_size]
            metas = meta_list[start:start + batch_size]
            batch = Batch.from_data_list(chunk).to(device)
            with torch.no_grad():
                try:
                    out = model(batch)
                except TypeError:
                    # fallback if .forward accepts only single graphs
                    out = torch.stack([model.forward_once(d.to(device)) for d in chunk], dim=0)
            emb_np = out.cpu().numpy()
            for (idx, uid), vec in zip(metas, emb_np):
                emb_str = ','.join(f'{x:.6f}' for x in vec.flatten())
                embedding_results.append((idx, uid, emb_str))

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

if __name__ == "__main__":
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(
        description="Generate GIN embeddings from RNA secondary structures."
    )
    parser.add_argument('--input',                type=str,   required=True)
    parser.add_argument('--output',               type=str,   required=True)
    parser.add_argument('--model-path',           type=str,   required=True)
    parser.add_argument('--id-column',            type=str,   required=True)
    parser.add_argument('--structure-column-name',type=str)
    parser.add_argument('--structure-column-num', type=int)
    parser.add_argument('--header',               type=str,   default='True',
                        choices=['True','False','true','false'])
    parser.add_argument('--keep-cols',            type=str)
    parser.add_argument('--device',               type=str,   default='cpu',
                        help="Device for inference: 'cpu' or 'cuda'")
    parser.add_argument('--num-workers',          type=int,   default=4)
    parser.add_argument('--batch-size',           type=int,   default=32)

    args = parser.parse_args()
    args.header = args.header.lower() == 'true'

    df = read_input_data(
        args.input,
        header=args.header,
        id_column_for_validation=args.id_column
    )
    struct_col = get_structure_column_name(
        df,
        args.header,
        col_name=args.structure_column_name,
        col_num=args.structure_column_num
    )

    log_path = os.path.splitext(args.output)[0] + '.log'
    generate_embeddings(
        input_df=df,
        output_path=args.output,
        model_path=args.model_path,
        log_path=log_path,
        structure_column=struct_col,
        id_column=args.id_column,
        device=args.device,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        keep_cols=args.keep_cols
    )
