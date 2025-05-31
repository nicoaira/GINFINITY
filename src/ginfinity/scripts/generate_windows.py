#!/usr/bin/env python3

import argparse
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import os
import torch
import networkx as nx

from ginfinity.utils import (
    setup_and_read_input,
    dotbracket_to_graph,
    graph_to_tensor,
    log_information,
    is_valid_dot_bracket
)

def should_skip_window_due_to_low_complexity(window_sequence, mask_threshold):
    """
    Determines if a window should be skipped based on its fraction of paired bases.
    Assumes '(' and ')' are the primary pairing characters.
    """
    if mask_threshold <= 0:
        return False
    paired_bases = window_sequence.count('(') + window_sequence.count(')')
    total_bases = len(window_sequence)
    if total_bases == 0:
        return True
    return (paired_bases / total_bases) < mask_threshold

def generate_slices(G, L, keep_paired_neighbors=True):
    slices = []
    nodes = sorted(G.nodes())
    n = len(nodes)
    for start in range(n - L + 1):
        window_nodes = list(range(start, start + L))
        sub_nodes = set(window_nodes)
        if keep_paired_neighbors:
            for node in window_nodes:
                for nbr in G.neighbors(node):
                    if (G.edges[node, nbr].get('edge_type') == 'base_pair'
                        and nbr not in sub_nodes):
                        sub_nodes.add(nbr)
        H = G.subgraph(sub_nodes).copy()
        if keep_paired_neighbors:
            for node in list(H.nodes()):
                if node not in window_nodes:
                    for nbr in list(H.neighbors(node)):
                        if H.edges[node, nbr].get('edge_type') == 'adjacent':
                            H.remove_edge(node, nbr)
        slices.append((start, H))
    return slices

def process_structure_to_windows(
    structure_string,
    seq_len,
    other_kept_cols_data,
    window_size_L,
    keep_paired_neighbors_in_slice,
    mask_threshold_for_window
):
    """
    Generates windowed subgraphs from a single RNA structure.
    Returns list of (metadata_dict, networkx_subgraph) tuples.
    """
    if not is_valid_dot_bracket(structure_string):
        return []
    graph = dotbracket_to_graph(structure_string)
    if graph is None or len(graph) < window_size_L:
        return []

    out = []
    for start_idx, H in generate_slices(graph, window_size_L, keep_paired_neighbors_in_slice):
        subseq = structure_string[start_idx:start_idx + window_size_L]
        if not subseq:
            continue
        if should_skip_window_due_to_low_complexity(subseq, mask_threshold_for_window):
            continue
        meta = {
            **other_kept_cols_data,
            'window_start': start_idx,
            'window_end':   start_idx + window_size_L - 1,
            'seq_len':      seq_len
        }
        out.append((meta, H))
    return out

def process_structure_to_windows_wrapper(args):
    original_id, struct, seq_len, other, L, kp, mt, id_name = args
    other = {id_name: original_id, **other}
    return process_structure_to_windows(
        struct, seq_len, other, L, kp, mt
    )

def main():
    parser = argparse.ArgumentParser(
        description="Generate windowed subgraphs from RNA structures."
    )
    parser.add_argument('--input',       type=str, required=True)
    parser.add_argument('--output-dir',  type=str, default="windows_output")
    parser.add_argument('--id-column',   type=str, required=True)
    parser.add_argument('--structure-column-name', type=str, default="secondary_structure")
    parser.add_argument('--L',           type=int, required=True)
    parser.add_argument('--keep-paired-neighbors', action='store_true')
    parser.add_argument('--mask-threshold', type=float, default=0.0)
    parser.add_argument('--keep-cols',   type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--quiet',       action='store_true',
                        help="Suppress progress bars")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    graphs_pt   = os.path.join(args.output_dir, "windows_graphs.pt")
    meta_tsv    = os.path.join(args.output_dir, "windows_metadata.tsv")
    args.output = meta_tsv  # for setup_and_read_input

    df, log_path, propagate = setup_and_read_input(args, need_model=False)

    tasks = []
    for _, row in df.iterrows():
        struct = row[args.structure_column_name]
        if not isinstance(struct, str):
            print(f"Skipping {row[args.id_column]}: not a string")
            continue
        other = {c: row[c] for c in propagate if c in row}
        tasks.append((
            row[args.id_column],
            struct,
            len(struct),
            other,
            args.L,
            args.keep_paired_neighbors,
            args.mask_threshold,
            args.id_column
        ))

    all_windows = []
    if args.num_workers > 1:
        chunksize = max(1, len(tasks)//(args.num_workers*4))
        with Pool(args.num_workers) as pool:
            for result in tqdm(
                pool.imap_unordered(process_structure_to_windows_wrapper, tasks, chunksize),
                total=len(tasks),
                desc="Windowing",
                disable=args.quiet
            ):
                all_windows.extend(result)
    else:
        for t in tqdm(tasks, desc="Windowing", disable=args.quiet):
            all_windows.extend(process_structure_to_windows_wrapper(t))

    graph_map = {}
    meta_list = []
    for meta, H in all_windows:
        wid = f"{meta[args.id_column]}_{meta['window_start']}"
        # --- relabel to 0..N-1 ---
        H = nx.convert_node_labels_to_integers(H)
        data = graph_to_tensor(H)
        # --- validate edge_index ---
        max_idx = int(data.edge_index.max())
        num_nodes = data.num_nodes
        if max_idx >= num_nodes:
            raise RuntimeError(f"Bad window {wid}: edge_index.max()={max_idx} >= num_nodes={num_nodes}")
        graph_map[wid] = data
        meta['window_id'] = wid
        row = {k: meta[k] for k in ['window_id', args.id_column, 'window_start', 'window_end', 'seq_len'] + propagate if k in meta}
        meta_list.append(row)

    # save graphs
    torch.save(graph_map, graphs_pt)
    if not args.quiet:
        print(f"Saved {len(graph_map)} graphs to {graphs_pt}")

    # save metadata
    meta_df = pd.DataFrame(meta_list)
    leading = ['window_id', args.id_column, 'window_start', 'window_end', 'seq_len']
    others  = [c for c in meta_df.columns if c not in leading]
    meta_df = meta_df[leading + others]
    meta_df.to_csv(meta_tsv, sep='\t', index=False, na_rep='NaN')
    if not args.quiet:
        print(f"Saved metadata to {meta_tsv}")

    log_information(log_path, {"graphs":graphs_pt, "metadata":meta_tsv, "n_windows":len(meta_df)}, "Summary")

if __name__ == "__main__":
    main()
