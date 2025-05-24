#!/usr/bin/env python3

import sys, os
# TODO: Remove this when the module is properly installed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import time
import torch
import pandas as pd
from tqdm import tqdm
import argparse
from pathlib import Path

from src.model.gin_model import GINModel
from src.utils import (
    dotbracket_to_forgi_graph,
    forgi_graph_to_tensor,
    log_information,
    log_setup,
    dotbracket_to_graph,
    graph_to_tensor,
    read_input_data,
    get_structure_column_name,
    is_valid_dot_bracket
)

from torch.multiprocessing import Pool, set_start_method
from torch_geometric.data import Batch  # for batched inference

def load_trained_model(model_path, device='cpu'):
    model = GINModel.load_from_checkpoint(model_path, device)
    model.to(device)
    model.eval()
    return model

def get_gin_embedding(
        model,
        graph_encoding,
        structure,
        device
    ):
    try:
        if not is_valid_dot_bracket(structure):
            raise ValueError("Invalid dot bracket string")
    except ValueError as e:
        return [(None, "")]
    if graph_encoding == "standard":
        graph = dotbracket_to_graph(structure)
        tg = graph_to_tensor(graph)
    else:
        graph = dotbracket_to_forgi_graph(structure)
        tg = forgi_graph_to_tensor(graph)
    if graph is None or tg is None:
        return [(None, "")]
    tg = tg.to(device)
    with torch.no_grad():
        emb = model.forward_once(tg)
        return [(None, ','.join(f'{x:.6f}' for x in emb.cpu().numpy().flatten()))]

def generate_embedding_for_row(args_tuple):
    # (kept for backward compatibility; not used in batched mode)
    (
        original_row_index,
        unique_id,
        structure_string,
        model_path,
        graph_encoding_method,
        device_to_use,
    ) = args_tuple

    model_instance = load_trained_model(model_path, device_to_use)
    embedding_results = get_gin_embedding(
        model_instance,
        graph_encoding_method,
        structure_string,
        device_to_use
    )
    return [(original_row_index, unique_id, emb_str) for _, emb_str in embedding_results]

def generate_embeddings(
        input_df,
        output_path,
        model_path,
        log_path,
        structure_column,
        id_column,
        device='cpu',
        num_workers=4,
        keep_cols=None
):
    # Resolve columns to carry through
    final_keep_cols = [id_column]
    if 'seq_len' in input_df.columns:
        final_keep_cols.append('seq_len')
    if keep_cols:
        for col in [c.strip() for c in keep_cols.split(',')]:
            if col in input_df.columns and col not in final_keep_cols:
                final_keep_cols.append(col)

    # Load model once on CPU to get graph encoding metadata
    temp_model = load_trained_model(model_path, 'cpu')
    graph_encoding = temp_model.metadata.get('graph_encoding', 'standard')
    del temp_model

    # --- Begin: batch feed-forward implementation ---
    data_list = []
    meta_list = []
    for idx, row in input_df.iterrows():
        uid = row[id_column]
        struct = row[structure_column]
        try:
            if not is_valid_dot_bracket(struct):
                raise ValueError("Invalid dot-bracket string")
        except ValueError as e:
            print(f"Skipping ID {uid}: {e}")
            log_information(log_path, {"skipped_invalid_structure": f"ID {uid}"})
            continue

        if graph_encoding == 'standard':
            graph = dotbracket_to_graph(struct)
            data = graph_to_tensor(graph)
        else:
            graph = dotbracket_to_forgi_graph(struct)
            data = forgi_graph_to_tensor(graph)

        if graph is None or data is None:
            print(f"Skipping ID {uid}: graph conversion failed")
            log_information(log_path, {"skipped_graph_conversion": f"ID {uid}"})
            continue

        data_list.append(data)
        meta_list.append((idx, uid))

    if not data_list:
        print("No valid structures to process for embeddings.")
        log_information(log_path, {"info": "No valid structures"}, "generate_embeddings")
        return

    # Batch all graphs in one forward pass
    batch = Batch.from_data_list(data_list).to(device)
    model = load_trained_model(model_path, device)
    with torch.no_grad():
        # Depending on model API: .forward_once for single vs .forward for batch
        try:
            emb_tensor = model.forward(batch)
        except TypeError:
            # fallback to forward_once if batch API not supported
            emb_tensor = torch.stack([model.forward_once(d.to(device)) for d in data_list], dim=0)
    emb_np = emb_tensor.cpu().numpy()

    all_embedding_results = []
    for (orig_idx, uid), vec in zip(meta_list, emb_np):
        emb_str = ','.join(f'{x:.6f}' for x in vec.flatten())
        all_embedding_results.append((orig_idx, uid, emb_str))
    # --- End: batch feed-forward implementation ---

    # Assemble and write output
    output_rows = []
    for orig_idx, uid, emb_str in all_embedding_results:
        if emb_str == "":
            log_information(log_path, {"skipped_empty_embedding": f"ID {uid}"})
            continue
        try:
            base = input_df.loc[orig_idx]
        except KeyError:
            log_information(log_path, {"warning": f"Row {orig_idx} not found after embedding"})
            continue
        row_out = {c: base[c] for c in final_keep_cols if c in base}
        row_out['embedding_vector'] = emb_str
        output_rows.append(row_out)

    out_df = pd.DataFrame(output_rows)

    # Reorder: id, optional window columns, embedding_vector, then the rest
    cols = [id_column]
    if 'window_start' in out_df.columns:
        cols.append('window_start')
    if 'window_end' in out_df.columns:
        cols.append('window_end')
    cols.append('embedding_vector')
    others = [c for c in out_df.columns if c not in cols]
    out_df = out_df[cols + sorted(others)]

    out_df.to_csv(output_path, sep='\t', index=False, na_rep='NaN')
    print(f"Embeddings saved to {output_path}")
    log_information(log_path, {"num_embeddings_generated": len(out_df)}, "generate_embeddings")

if __name__ == "__main__":
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(
        description="Generate GIN embeddings from RNA secondary structures (batched inference)."
    )
    # Input/output arguments
    parser.add_argument('--input',               type=str, required=True,
                        help="Path to input TSV/CSV file.")
    parser.add_argument('--output',              type=str, required=True,
                        help="Path to output TSV file for embeddings.")
    parser.add_argument('--model-path',          type=str, required=True,
                        help="Path to the trained GIN model checkpoint.")
    # Columns
    parser.add_argument('--id-column',           type=str, required=True,
                        help="Name of the column with unique IDs.")
    parser.add_argument('--structure-column-name', type=str,
                        help="Name of the column with secondary structures.")
    parser.add_argument('--structure-column-num',  type=int,
                        help="0-based index of the structure column, if no header.")
    parser.add_argument('--header',              type=str, default='True', choices=['True','False','true','false'],
                        help="Whether the input file has a header row.")
    parser.add_argument('--keep-cols',           type=str,
                        help="Comma-separated list of additional columns to carry through.")
    # Inference settings
    parser.add_argument('--device',              type=str, default="cpu",
                        help="Device for model inference ('cpu' or 'cuda').")
    parser.add_argument('--num-workers',         type=int, default=4,
                        help="(Unused) Number of worker processes (retained for compatibility).")

    args = parser.parse_args()
    args.header = args.header.lower() == 'true'

    # Prepare logging and data
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    log_path = os.path.splitext(args.output)[0] + '.log'
    log_setup(log_path)

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

    generate_embeddings(
        input_df=df,
        output_path=args.output,
        model_path=args.model_path,
        log_path=log_path,
        structure_column=struct_col,
        id_column=args.id_column,
        device=args.device,
        num_workers=args.num_workers,
        keep_cols=args.keep_cols
    )

