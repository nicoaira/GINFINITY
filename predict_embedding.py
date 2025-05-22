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
            if len(sorted_nodes) < L:
                return [(-1, "")]
            embeddings = []
            slices = generate_slices(graph, L, keep_paired_neighbors)
            for start_idx, subgraph_H in slices:
                sub_nodes = sorted(subgraph_H.nodes())
                idxs = [sorted_nodes.index(n) for n in sub_nodes]
                if not idxs:
                    continue
                sub_embs = node_embs[idxs]
                batch = torch.zeros(len(sub_embs), dtype=torch.long, device=device)
                pooled = model.pooling(sub_embs, batch)
                emb = model.fc(pooled)
                emb_str = ','.join(f'{x:.6f}' for x in emb.cpu().numpy().flatten())
                embeddings.append((start_idx, emb_str))
            return embeddings or [(-1, "")]
        else:
            emb = model.forward_once(tg)
            return [(None, ','.join(f'{x:.6f}' for x in emb.cpu().numpy().flatten()))]

def validate_structure(structure):
    if not isinstance(structure, str):
        raise ValueError("Secondary structure must be a string.")
    valid = "()[]{}<>AaBbCcDd."
    bad = set(structure) - set(valid)
    if bad:
        raise ValueError(f"Invalid structure chars: {bad}")

def generate_embedding_for_row(args):
    idx, row, model, struct_col, device, enc, subgraphs, L, keep_pairs = args
    structure = row[struct_col]
    validate_structure(structure)
    if subgraphs:
        lst = get_gin_embedding(model, enc, structure, device, L, keep_pairs)
        return [(idx, start, emb) for (start, emb) in lst]
    else:
        emb = get_gin_embedding(model, enc, structure, device)
        return [(idx, None, emb[0][1])]

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
    # ─ ensure seq_len exists ───────────────────────────
    if 'seq_len' not in input_df.columns:
        input_df['seq_len'] = input_df[structure_column].str.len()

    # ─ resolve keep_cols, but force seq_len in it ──────
    if keep_cols is None:
        keep_cols = list(input_df.columns)
    else:
        # split, validate
        cols = [c.strip() for c in keep_cols.split(',')]
        missing = [c for c in cols if c not in input_df.columns]
        if missing:
            raise ValueError(f"--keep-cols references unknown columns: {missing}")
        keep_cols = cols
    if 'seq_len' not in keep_cols:
        keep_cols.append('seq_len')

    # ─ load model ───────────────────────────────────────
    model = load_trained_model(model_path, device)
    graph_encoding = model.metadata['graph_encoding']

    # ─ prepare arguments for each row ───────────────────
    args_list = [
        (idx, row, model, structure_column, device, graph_encoding,
         subgraphs, L, keep_paired_neighbors)
        for idx, row in input_df.iterrows()
    ]

    # ─ compute embeddings in parallel ───────────────────
    results = []
    with Pool(num_workers) as pool:
        try:
            for res in tqdm(pool.imap_unordered(generate_embedding_for_row, args_list),
                            total=len(input_df), desc="Embedding"):
                results.extend(res)
        finally:
            pool.close()
            pool.join()

    # ─ assemble output rows ─────────────────────────────
    out_rows = []
    for orig_idx, wstart, emb in results:
        if not (0 <= orig_idx < len(input_df)):
            continue
        base = input_df.iloc[orig_idx]
        row = { c: base[c] for c in keep_cols }
        row['window_start']    = wstart
        row['window_end']      = (wstart + L - 1) if wstart is not None else None
        row['embedding_vector'] = emb
        out_rows.append(row)

    output_df = pd.DataFrame(out_rows)
    log_information(log_path, output_df.head(), "Output head")
    output_df.to_csv(output_path, sep='\t', index=False)
    print(f"Embeddings saved to {output_path}")
    log_information(log_path, {"output": output_path})

    # ─ retry logic ──────────────────────────────────────
    if not os.path.exists(output_path) and retries > 0:
        print(f"Output missing, retrying ({retries})…")
        generate_embeddings(
            input_df, output_path, model_path, log_path,
            structure_column, device=device,
            subgraphs=subgraphs, L=L,
            keep_paired_neighbors=keep_paired_neighbors,
            num_workers=num_workers, retries=retries,
            keep_cols=','.join(keep_cols)
        )

def read_input_data(input_path, samples, structure_column_num, header):
    sep_char = '\t' if input_path.endswith('.tsv') else ','
    if header:
        df = pd.read_csv(input_path, sep=sep_char)
    else:
        if structure_column_num is None:
            raise ValueError("When header=False, must specify structure_column_num.")
        df = pd.read_csv(input_path, sep=sep_char, header=None)
    if samples:
        df = df.sample(n=samples, random_state=42)
    return df

def get_structure_column_name(df, header, col_name, col_num):
    if header:
        if col_name:   return col_name
        elif col_num is not None: return df.columns[col_num]
        else:          return "secondary_structure"
    else:
        return df.columns[col_num]

if __name__ == "__main__":
    try: set_start_method('spawn')
    except RuntimeError: pass

    parser = argparse.ArgumentParser(
        description="Generate embeddings from RNA structures"
    )
    parser.add_argument('--input',               type=str,  required=True)
    parser.add_argument('--id-column',           type=str,  required=True)
    parser.add_argument('--output',              type=str,  required=True)
    parser.add_argument('--model_path',          type=str,  required=True)
    parser.add_argument('--structure_column_name', type=str)
    parser.add_argument('--structure_column_num',  type=int)
    parser.add_argument('--header',              type=str,  default='True')
    parser.add_argument('--device',              type=str,  default="cpu")
    parser.add_argument('--num_workers',         type=int,  default=4)
    parser.add_argument('--subgraphs',           action='store_true')
    parser.add_argument('--L',                   type=int)
    parser.add_argument('--keep_paired_neighbors', action='store_true')
    parser.add_argument('--retries',             type=int,  default=0)
    parser.add_argument('--keep-cols',           type=str,
                        help="Comma-separated list of input columns to retain.")
    parser.add_argument('--mask-threshold',      type=float, default=0.3,
                    help="Minimum fraction of paired bases (brackets) required; rows below are dropped (default 0.3)")

    args = parser.parse_args()

    args.header = args.header.lower() == 'true'
    # read
    df = read_input_data(args.input, None,
                         args.structure_column_num, args.header)
    
     # ─── validate ID ──────────────────────────────────────
    if args.id_column not in df.columns:
        raise ValueError(f"--id-column '{args.id-column}' not in {list(df.columns)}")
    if df[args.id_column].duplicated().any():
        raise ValueError(f"Values in '{args.id_column}' must be unique.")

    # ─── determine structure column ───────────────────────
    struct_col = get_structure_column_name(
        df, args.header,
        args.structure_column_name,
        args.structure_column_num
    )

    # ─── NEW: filter low‐complexity windows ───────────────
    mt = args.mask_threshold
    if mt > 0:
        # count paired chars (anything ≠ '.') and compute fraction
        lengths = df[struct_col].astype(str).str.len().replace(0,1)
        paired  = df[struct_col].astype(str).map(lambda s: sum(1 for c in s if c != '.'))
        frac    = paired.div(lengths)
        mask    = frac >= mt
        removed = (~mask).sum()
        if removed:
            print(f"[filter] Dropped {removed} rows with paired‐base fraction < {mt}")
        df = df[mask].copy()

    # ─── make output dir & log ────────────────────────────
    outdir = os.path.dirname(args.output) or '.'
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.splitext(args.output)[0] + '.log'
    log_setup(log_path)

    # ─── run embeddings ───────────────────────────────────
    start = time.time()
    generate_embeddings(
        df, args.output, args.model_path, log_path, struct_col,
        device=args.device,
        subgraphs=args.subgraphs, L=args.L,
        keep_paired_neighbors=args.keep_paired_neighbors,
        num_workers=args.num_workers, retries=args.retries,
        keep_cols=args.keep_cols
    )
    elapsed = (time.time() - start)/60
    print(f"Done in {elapsed:.2f} min.")
    log_information(log_path, {"time_min": elapsed}, "Timing")