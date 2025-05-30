#!/usr/bin/env python3

import argparse
import pandas as pd
import torch
from tqdm import tqdm
from torch.multiprocessing import Pool, set_start_method
from torch_geometric.data import Batch

from model.gin_model import GINModel
from utils import (
    setup_and_read_input,
    dotbracket_to_graph,
    graph_to_tensor,
    dotbracket_to_forgi_graph,
    forgi_graph_to_tensor,
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
        keep_cols: list     = None
):

    # Decide which columns to carry through
    final_keep = [id_column]
    if 'seq_len' in input_df.columns:
        final_keep.append('seq_len')
    final_keep.extend(keep_cols)

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
        description="Generate high-quality embeddings using a pretrained GIN model from RNA secondary structures " \
        "for comprehensive downstream analysis."
    )
    parser.add_argument('--input', type=str, required=True,
                        help="Path to the input TSV/CSV file containing RNA structures.")
    parser.add_argument('--output', type=str, default= "embeddings.tsv",
                        help="Name of the output TSV file to save the generated embeddings. Default is 'embeddings.tsv'.")
    parser.add_argument('--model-path', type=str, required=True,
                         help="Path to the pretrained GIN model checkpoint.")
    parser.add_argument('--id-column', type=str, required=True,
                         help="Column name in the input file that uniquely identifies each RNA structure.")
    parser.add_argument('--structure-column-name',type=str,
                         default="secondary_structure", help="Column name containing the RNA secondary structure in dot-bracket notation. Default is 'secondary_structure'.")
    parser.add_argument('--keep-cols', type=str, default=None,
                         help="Comma-separated list of additional column names to keep in the output. Default is None (only ID and embedding).")
    parser.add_argument('--device',  type=str,   default='cpu',
                        help="Device for inference: 'cpu' or 'cuda'")
    parser.add_argument('--num-workers', type=int, default=4,
                        help="Number of parallel worker processes to use for processing. Use values greater than 1 for parallel execution.")
    parser.add_argument('--batch-size', type=int, default=32,
                        help="Batch size for GPU inference. Default is 32. Ignored if device is 'cpu'.")

    args = parser.parse_args()

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
        keep_cols=propagate
    )
