#!/usr/bin/env python3

import random
import time
import torch
import pandas as pd
from tqdm import tqdm
import argparse
from src.model.gin_model import GINModel
from src.utils import (
    dotbracket_to_forgi_graph, forgi_graph_to_tensor,
    log_information, log_setup, dotbracket_to_graph,
    graph_to_tensor, generate_slices
)
import os
from torch.multiprocessing import Pool, set_start_method

# ── model loading & embedding routines unchanged ─────────────────────────────

def load_trained_model(model_path, device='cpu'):
    model = GINModel.load_from_checkpoint(model_path, device)
    model.to(device)
    model.eval()
    return model

def get_gin_embedding(model, graph_encoding, structure, device, L=None, keep_paired_neighbors=False):
    if graph_encoding == "standard":
        graph = dotbracket_to_graph(structure)
        tg = graph_to_tensor(graph)
    else:
        graph = dotbracket_to_forgi_graph(structure)
        tg = forgi_graph_to_tensor(graph)

    tg = tg.to(device)
    model.eval()
    with torch.no_grad():
        if L is not None:
            node_embs = model.get_node_embeddings(tg)
            sorted_nodes = sorted(graph.nodes())
            n = len(sorted_nodes)
            if n < L:
                return [(-1, "")]
            embeddings = []
            slices = generate_slices(graph, L, keep_paired_neighbors)
            for start_idx, subgraph_H in slices:
                subgraph_nodes = sorted(subgraph_H.nodes())
                node_indices = [sorted_nodes.index(node) for node in subgraph_nodes]
                if not node_indices:
                    continue
                sub_embs = node_embs[node_indices]
                batch = torch.zeros(len(sub_embs), dtype=torch.long, device=device)
                pooled = model.pooling(sub_embs, batch)
                sub_embedding = model.fc(pooled)
                emb_str = ','.join(f'{x:.6f}' for x in sub_embedding.cpu().numpy().flatten())
                embeddings.append((start_idx, emb_str))
            return embeddings or [(-1, "")]
        else:
            emb = model.forward_once(tg)
            return [(None, ','.join(f'{x:.6f}' for x in emb.cpu().numpy().flatten()))]

def validate_structure(structure):
    if not isinstance(structure, str):
        raise ValueError("Secondary structure must be a string in dot-bracket.")
    valid = "()[]{}<>AaBbCcDd."
    if any(ch not in valid for ch in structure):
        raise ValueError(f"Invalid structure chars: {set(structure)-set(valid)}")

def generate_embedding_for_row(args):
    idx, row, model, structure_column, device, graph_encoding, \
      subgraphs, L, keep_paired_neighbors = args

    structure = row[structure_column]
    validate_structure(structure)
    if subgraphs:
        emb_list = get_gin_embedding(model, graph_encoding, structure, device, L, keep_paired_neighbors)
        return [(idx, start, emb) for (start, emb) in emb_list]
    else:
        emb = get_gin_embedding(model, graph_encoding, structure, device)
        return [(idx, None, emb[0][1])]

# ── modified generate_embeddings with keep_cols support ──────────────────────

def generate_embeddings(
        input_df,
        output_path,
        model_path,
        log_path,
        structure_column,
        device='cpu',
        subgraphs=False,
        L=None,
        keep_paired_neighbors=False,
        num_workers=4,
        retries=0,
        keep_cols=None
):
    """
    keep_cols: list of input_df columns to retain (or None for all)
    """
    # resolve keep_cols
    if keep_cols is None:
        keep_cols = list(input_df.columns)
    else:
        missing = [c for c in keep_cols if c not in input_df.columns]
        if missing:
            raise ValueError(f"--keep-cols references unknown columns: {missing}")

    model = load_trained_model(model_path, device)
    graph_encoding = model.metadata['graph_encoding']

    # prepare jobs
    args_list = [
        (idx, row, model, structure_column, device, graph_encoding,
         subgraphs, L, keep_paired_neighbors)
        for idx, row in input_df.iterrows()
    ]

    # compute embeddings in parallel
    results = []
    with Pool(num_workers) as pool:
        try:
            for result in tqdm(pool.imap_unordered(generate_embedding_for_row, args_list),
                               total=len(input_df), desc="Processing Embeddings"):
                results.extend(result)
        finally:
            pool.close()
            pool.join()

    # build output rows
    new_rows = []
    for orig_idx, window_start, emb in results:
        if not (0 <= orig_idx < len(input_df)):
            continue
        base = input_df.iloc[orig_idx]
        row_dict = { c: base[c] for c in keep_cols }
        row_dict['window_start'] = window_start
        row_dict['window_end']   = (window_start + L - 1) if window_start is not None else None
        row_dict['embedding_vector'] = emb
        new_rows.append(row_dict)

    output_df = pd.DataFrame(new_rows)
    log_information(log_path, output_df.head(), "Output DataFrame head")
    output_df.to_csv(output_path, sep='\t', index=False)
    print(f"Embeddings saved to {output_path}")
    log_information(log_path, {"Embeddings saved path": output_path})

    # retry logic unchanged
    if not os.path.exists(output_path) and retries > 0:
        print(f"Output file not found; retrying ({retries})…")
        generate_embeddings(
            input_df, output_path, model_path, log_path,
            structure_column, device=device,
            subgraphs=subgraphs, L=L,
            keep_paired_neighbors=keep_paired_neighbors,
            num_workers=num_workers, retries=retries,
            keep_cols=keep_cols
        )

# ── I/O helpers unchanged ───────────────────────────────────────────────────

def read_input_data(input_path, samples, structure_column_num, header):
    """
    Read the input TSV/CSV into a DataFrame.
      - input_path: file path
      - samples:   number of random rows to sample (or None)
      - structure_column_num: required if header=False
      - header:    True if the file has a header row
    """
    sep_char = '\t' if input_path.endswith('.tsv') else ','

    if header:
        df = pd.read_csv(input_path, sep=sep_char)
    else:
        if structure_column_num is None:
            raise ValueError(
                "When header=False, structure_column_num must be specified."
            )
        df = pd.read_csv(input_path, sep=sep_char, header=None)

    if samples:
        df = df.sample(n=samples, random_state=42)

    return df

def get_structure_column_name(input_df, header, structure_column_name, structure_column_num):
    if header:
        if structure_column_name:
            return structure_column_name
        elif structure_column_num is not None:
            return input_df.columns[structure_column_num]
        else:
            return "secondary_structure"
    else:
        return input_df.columns[structure_column_num]

# ── main: add --keep-cols to parser and pass through ────────────────────────

if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(
        description="Generate embeddings from RNA structures via GIN."
    )
    parser.add_argument('--input',              type=str, required=True)
    parser.add_argument('--id-column',          type=str, required=True)
    parser.add_argument('--output',             type=str, help='Output TSV path')
    parser.add_argument('--model_path',         type=str, required=True)
    parser.add_argument('--structure_column_name', type=str)
    parser.add_argument('--structure_column_num',  type=int)
    parser.add_argument('--header',             type=str, default='True')
    parser.add_argument('--device',             type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--num_workers',        type=int, default=4)
    parser.add_argument('--subgraphs',          action='store_true')
    parser.add_argument('--L',                  type=int)
    parser.add_argument('--keep_paired_neighbors', action='store_true')
    parser.add_argument('--retries',            type=int, default=0)
    parser.add_argument(
        '--keep-cols', type=str,
        help="Comma-separated list of input columns to retain.  Default=all."
    )

    args = parser.parse_args()
    args.header = args.header.lower() == 'true'

    # read input
    df = read_input_data(args.input, None, args.structure_column_num, args.header)

    # validate id column
    if args.id_column not in df.columns:
        raise ValueError(f"--id-column '{args.id-column}' not in input columns {list(df.columns)}")
    if df[args.id_column].duplicated().any():
        raise ValueError(f"Values in '{args.id_column}' must be unique.")

    struct_col = get_structure_column_name(df, args.header, args.structure_column_name, args.structure_column_num)

    # parse keep-cols
    if args.keep_cols:
        keep_cols = [c.strip() for c in args.keep_cols.split(',')]
    else:
        keep_cols = None

    # prepare output path & log
    if not args.output:
        raise ValueError("Please supply --output")
    outdir = os.path.dirname(args.output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    log_path = os.path.splitext(args.output)[0] + '.log'
    log_setup(log_path)

    # run
    start = time.time()
    generate_embeddings(
        df, args.output, args.model_path, log_path, struct_col,
        device=args.device,
        subgraphs=args.subgraphs, L=args.L,
        keep_paired_neighbors=args.keep_paired_neighbors,
        num_workers=args.num_workers, retries=args.retries,
        keep_cols=keep_cols
    )
    elapsed = (time.time() - start)/60
    print(f"Finished in {elapsed:.2f} min.")
    log_information(log_path, {"execution_time_min": elapsed}, "Timing")
