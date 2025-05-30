#!/usr/bin/env python3

import sys
import os
import argparse
import pandas as pd
from tqdm import tqdm

from multiprocessing import Pool

from utils import (
    read_input_data,
    get_structure_column_name,
    dotbracket_to_graph,
    log_setup,
    log_information,
    is_valid_dot_bracket,
)

def should_skip_window_due_to_low_complexity(window_sequence, mask_threshold):
    """
    Determines if a window should be skipped based on its fraction of paired bases.
    Assumes '(' and ')' are the primary pairing characters.
    """
    if mask_threshold <= 0:  # No masking if threshold is zero or negative
        return False
    
    # Count standard paired bases. Extend this if other characters like [], {} are used for pairs.
    paired_bases = window_sequence.count('(') + window_sequence.count(')')
    
    total_bases = len(window_sequence)
    if total_bases == 0:
        return True  # Skip empty windows
    
    fraction_paired = paired_bases / total_bases
    return fraction_paired < mask_threshold

def generate_slices(G, L, keep_paired_neighbors=True):
    slices = []
    nodes = sorted(G.nodes())
    n = len(nodes)
    for start in range(n - L + 1):
        window_nodes = list(range(start, start + L))
        sub_nodes = set(window_nodes)
        if keep_paired_neighbors:
            for node in window_nodes:
                for neighbor in G.neighbors(node):
                    if G.edges[node, neighbor].get('edge_type') == 'base_pair' and neighbor not in window_nodes:
                        sub_nodes.add(neighbor)
        H = G.subgraph(sub_nodes).copy()
        if keep_paired_neighbors:
            for node in list(H.nodes()):
                if node not in window_nodes:
                    for neighbor in list(H.neighbors(node)):
                        if H.edges[node, neighbor].get('edge_type') == 'adjacent':
                            H.remove_edge(node, neighbor)
        slices.append((start, H))
    return slices

def process_structure_to_windows(
    original_id,
    structure_string,
    seq_len,
    other_kept_cols_data,
    window_size_L,
    keep_paired_neighbors_in_slice,
    mask_threshold_for_window
):
    """
    Generates windowed sequences from a single RNA structure.
    """
    # Validate dot‐bracket
    if not is_valid_dot_bracket(structure_string):
        # invalid → skip
        return []

    graph = dotbracket_to_graph(structure_string)
    if graph is None or len(graph.nodes()) < window_size_L:
        return []

    window_rows = []
    slices = generate_slices(graph, window_size_L, keep_paired_neighbors_in_slice)
    for start_idx, _ in slices:
        window_sequence = structure_string[start_idx:start_idx + window_size_L]
        if not window_sequence:
            continue
        if should_skip_window_due_to_low_complexity(window_sequence, mask_threshold_for_window):
            continue

        row = {
            **other_kept_cols_data,
            'window_start':     start_idx,
            'window_end':       start_idx + window_size_L - 1,
            'window_sequence':  window_sequence,
            'seq_len':          seq_len
        }
        window_rows.append(row)

    return window_rows

def process_structure_to_windows_wrapper(args_tuple):
    original_id, structure_string, seq_len, other_cols, window_L, keep_pairs, mask_thresh, id_col_name = args_tuple
    # ensure the ID is keyed correctly
    other_cols = {id_col_name: original_id, **other_cols}
    return process_structure_to_windows(
        original_id,
        structure_string,
        seq_len,
        other_cols,
        window_L,
        keep_pairs,
        mask_thresh
    )

def main():
    parser = argparse.ArgumentParser(
        description="Generate windowed sequences from RNA structures in a TSV/CSV file."
    )
    parser.add_argument('--input',                required=True)
    parser.add_argument('--output',               required=True)
    parser.add_argument('--id-column',            required=True)
    parser.add_argument('--structure-column-name',help="Overrides col‐num")
    parser.add_argument('--structure-column-num', type=int)
    parser.add_argument('--header',     default='True')
    parser.add_argument('--L',          type=int, required=True)
    parser.add_argument('--keep-paired-neighbors', action='store_true')
    parser.add_argument('--mask-threshold', type=float, default=0.0)
    parser.add_argument('--keep-cols',   default="")
    parser.add_argument('--num-workers', type=int, default=1)
    args = parser.parse_args()
    args.header = args.header.lower() == 'true'

    # logging setup
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    log_path = os.path.splitext(args.output)[0] + '.log'
    log_setup(log_path, print_log=False)
    log_information(log_path, vars(args), "Arguments", print_log=True)

    # read data
    df = read_input_data(args.input, header=args.header, id_column_for_validation=(args.id_column if args.header else None))
    struct_col = get_structure_column_name(df, args.header, args.structure_column_name, args.structure_column_num)
    id_col     = args.id_column

    # warnings if needed
    if args.header and id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' not in input columns {list(df.columns)}")
    if df[id_col].duplicated().any():
        log_information(log_path, {"warning": "duplicate IDs"}, "Warning")

    # determine extra cols to propagate
    propagate = []
    if args.keep_cols:
        for col in [c.strip() for c in args.keep_cols.split(',')]:
            if args.header and col in df.columns:
                propagate.append(col)
            else:
                print(f"Warning: keep-col '{col}' not found or invalid.")

    # build tasks
    tasks = []
    for _, row in df.iterrows():
        struct = row[struct_col]
        if not isinstance(struct, str):
            print(f"Skipping {row[id_col]}: not a string")
            continue
        other = {c: row[c] for c in propagate if c in row}
        tasks.append((
            row[id_col],
            struct,
            len(struct),
            other,
            args.L,
            args.keep_paired_neighbors,
            args.mask_threshold,
            args.id_column
        ))

    all_rows = []
    if tasks:
        if args.num_workers > 1:
            # compute a reasonable chunksize
            chunksize = max(1, len(tasks) // (args.num_workers * 4))
            with Pool(args.num_workers) as pool:
                for result in tqdm(
                    pool.imap_unordered(process_structure_to_windows_wrapper, tasks, chunksize=chunksize),
                    total=len(tasks),
                    desc="Windowing structures"
                ):
                    all_rows.extend(result)
        else:
            for args_tuple in tqdm(tasks, desc="Windowing structures (serial)"):
                all_rows.extend(process_structure_to_windows_wrapper(args_tuple))
    else:
        print("No valid structures found.")

    out_df = pd.DataFrame(all_rows)
    leading = [args.id_column, 'window_start', 'window_end', 'window_sequence', 'seq_len']
    for col in leading:
        if col not in out_df.columns:
            out_df[col] = pd.NA
    others = sorted(c for c in out_df.columns if c not in leading)
    out_df = out_df[leading + others]
    out_df.to_csv(args.output, sep='\t', index=False, na_rep='NaN')
    print(f"Windowed sequences saved to {args.output}")
    log_information(log_path, {"output": args.output, "n_windows": len(out_df)}, "Summary")

if __name__ == "__main__":
    main()
