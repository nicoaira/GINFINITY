#!/usr/bin/env python3
import sys, os
# TODO: Remove this when the module is properly installed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import argparse
import os
import pandas as pd
from tqdm import tqdm
from torch.multiprocessing import Pool, set_start_method # Added

from src.utils import (
    read_input_data,
    get_structure_column_name,
    dotbracket_to_graph,
    generate_slices,
    log_setup,
    log_information,
    is_valid_dot_bracket, 
    should_skip_window_due_to_low_complexity
)

def process_structure_to_windows(
    original_id,
    structure_string,
    seq_len, # Renamed from original_seq_len
    other_kept_cols_data, 
    window_size_L,
    keep_paired_neighbors_in_slice,
    mask_threshold_for_window
):
    """
    Generates windowed sequences from a single RNA structure.
    Validates the structure and applies low-complexity masking to windows.
    """
    try:
        # Changed to use is_valid_dot_bracket from src.utils
        if not is_valid_dot_bracket(structure_string):
            raise ValueError("Invalid dot-bracket string")
    except ValueError as e:
        print(f"Skipping structure ID {original_id} due to invalid characters: {e}")
        return []

    # For generate_slices, we need a NetworkX graph.
    # dotbracket_to_graph from src.utils creates a suitable graph.
    # This graph is only used for defining the slices, not for GIN model input directly here.
    graph = dotbracket_to_graph(structure_string)
    if graph is None: # dotbracket_to_graph might return None for malformed structures
        print(f"Skipping structure ID {original_id} due to graph conversion failure.")
        return []

    if len(graph.nodes()) < window_size_L:
        # Optionally, one could output the full sequence if it's shorter than L,
        # but current generate_slices behavior implies skipping.
        # For now, we return no windows if too short for any L-window.
        return []

    window_rows = []
    # generate_slices yields (start_index, subgraph_object)
    # The subgraph_object itself is not directly used here, only its start_index
    # to extract the sequence string for the window.
    slices = generate_slices(graph, window_size_L, keep_paired_neighbors_in_slice)

    for start_idx, _subgraph_H in slices:
        window_sequence = structure_string[start_idx : start_idx + window_size_L]

        if not window_sequence: # Should not happen if start_idx and L are correct
            continue

        # Changed to use should_skip_window_due_to_low_complexity from src.utils
        if should_skip_window_due_to_low_complexity(window_sequence, mask_threshold_for_window):
            continue

        window_data = {
            **other_kept_cols_data, # Add original kept columns
            'window_start': start_idx,
            'window_end': start_idx + window_size_L - 1,
            'window_sequence': window_sequence,
            'seq_len': seq_len # Renamed here
        }
        window_rows.append(window_data)

    return window_rows

def process_structure_to_windows_wrapper(args_tuple):
    """Helper function to unpack arguments for process_structure_to_windows for use with Pool."""
    (
        original_id,
        structure_string,
        seq_len,
        other_kept_cols_data,
        window_size_L,
        keep_paired_neighbors_in_slice,
        mask_threshold_for_window,
        id_column_name # Added to correctly structure other_kept_cols_data
    ) = args_tuple

    # Reconstruct other_cols_data as it was in the serial version
    # other_kept_cols_data already contains the necessary data, but id needs to be set with the specific id_column_name
    # The id_column_name itself is passed in other_kept_cols_data if it was in keep-cols.
    # What's crucial is that original_id is correctly associated with the id_column_name key.
    
    # Create a base for data to be passed, ensuring the ID is correctly keyed
    final_other_kept_cols_data = {id_column_name: original_id}
    final_other_kept_cols_data.update(other_kept_cols_data)


    return process_structure_to_windows(
        original_id,
        structure_string,
        seq_len,
        final_other_kept_cols_data, # Pass the reconstructed dict
        window_size_L,
        keep_paired_neighbors_in_slice,
        mask_threshold_for_window
    )

def main():
    parser = argparse.ArgumentParser(
        description="Generate windowed sequences from RNA structures in a TSV/CSV file."
    )
    parser.add_argument('--input', type=str, required=True, help="Path to input TSV/CSV file.")
    parser.add_argument('--output', type=str, required=True, help="Path to output TSV file for windows.")
    parser.add_argument('--id-column', type=str, required=True, help="Name of the column containing unique IDs.")
    parser.add_argument('--structure-column-name', type=str, help="Name of the column containing RNA secondary structures. Overrides --structure-column-num.")
    parser.add_argument('--structure-column-num', type=int, help="0-based index of the structure column (if no header or name not used).")
    parser.add_argument('--header', type=str, default='True', help="Whether the input file has a header row ('True' or 'False').")
    parser.add_argument('--L', type=int, required=True, help="Window size for slicing.")
    parser.add_argument('--keep-paired-neighbors', action='store_true', help="Include paired neighbors outside the window in slices (affects graph structure for slicing).")
    parser.add_argument('--mask-threshold', type=float, default=0.0, help="Minimum fraction of paired bases per window; windows below this are skipped (0 or negative means no masking).")
    parser.add_argument('--keep-cols', type=str, default="", help="Comma-separated list of additional columns from the input to retain in the output.")
    parser.add_argument('--num-workers', type=int, default=1, help="Number of worker processes for parallel window generation (default: 1).") # Added
    
    args = parser.parse_args()
    args.header = args.header.lower() == 'true'

    # Setup logging
    outdir = os.path.dirname(args.output) or '.'
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.splitext(args.output)[0] + '.log'
    log_setup(log_path, print_log=False) # Initial setup
    log_information(log_path, vars(args), "Arguments", print_log=True)


    # Read input data
    input_df = read_input_data(
        args.input,
        header=args.header,
        id_column_for_validation=args.id_column if args.header else None
    )

    # Determine structure column
    structure_col_name_actual = get_structure_column_name(
        input_df,
        args.header,
        col_name=args.structure_column_name,
        col_num=args.structure_column_num
    )
    # If no header, structure_col_name_actual is an int (index)
    # We need the string name for df access if headerless, pandas assigns numerical names
    if not args.header:
        structure_col_name_for_df_access = structure_col_name_actual
    else:
        structure_col_name_for_df_access = structure_col_name_actual


    # Validate ID column
    id_col_for_df_access = args.id_column
    if args.header:
        if args.id_column not in input_df.columns:
            raise ValueError(f"ID column '{args.id_column}' not found in input file columns: {list(input_df.columns)}")
    else: # No header, id_column is a name we assign to the column index
        try:
            # If id_column was given as a number string by user for headerless, convert
            id_col_idx = int(args.id_column)
            if id_col_idx >= len(input_df.columns):
                 raise ValueError(f"ID column index {id_col_idx} is out of bounds.")
            id_col_for_df_access = input_df.columns[id_col_idx] # Use pandas default numerical name
        except ValueError:
            # if id_column was not a number string, this is an issue for headerless
            raise ValueError("For headerless input, --id-column should typically be a column index.")


    if input_df[id_col_for_df_access].duplicated().any():
        print(f"Warning: Duplicate values found in ID column '{id_col_for_df_access}'. Output will retain these duplicates if they generate windows.")
        log_information(log_path, {"warning": f"Duplicate values found in ID column '{id_col_for_df_access}'."})


    # Determine columns to keep
    propagate_col_names = []
    if args.keep_cols:
        additional_keep_cols = [c.strip() for c in args.keep_cols.split(',')]
        for col_name_or_idx_str in additional_keep_cols:
            if args.header:
                if col_name_or_idx_str in input_df.columns:
                    if col_name_or_idx_str not in propagate_col_names:
                        propagate_col_names.append(col_name_or_idx_str)
                else:
                    print(f"Warning: Requested keep_col '{col_name_or_idx_str}' not found in input columns. It will be ignored.")
            else: # No header
                try:
                    col_idx = int(col_name_or_idx_str)
                    if col_idx < len(input_df.columns):
                        actual_col_name_in_df = input_df.columns[col_idx]
                        if actual_col_name_in_df not in propagate_col_names:
                             propagate_col_names.append(actual_col_name_in_df)
                    else:
                        print(f"Warning: Requested keep_col index '{col_idx}' is out of bounds. It will be ignored.")
                except ValueError:
                     print(f"Warning: Requested keep_col '{col_name_or_idx_str}' is not a valid index for headerless input. It will be ignored.")


    all_window_rows = []
    tasks_for_pool = []

    for _, row in input_df.iterrows():
        original_id = row[id_col_for_df_access]
        structure = row[structure_col_name_for_df_access]
        
        if not isinstance(structure, str):
            print(f"Skipping ID {original_id}: structure is not a string (found {type(structure)}).")
            log_information(log_path, {"skipped_non_string_structure": f"ID {original_id}"})
            continue

        original_len = len(structure)

        # Prepare data from other columns to be kept.
        # This data is specific to this row.
        other_cols_data_for_this_row = {col_name: row[col_name] for col_name in propagate_col_names if col_name in row}
        # The ID itself (original_id) will be handled by process_structure_to_windows_wrapper
        # to ensure it's correctly named as args.id_column in the final output structure.
        # However, process_structure_to_windows expects the ID column data within other_kept_cols_data
        # with the key being args.id_column.

        task_args = (
            original_id,
            structure,
            original_len,
            other_cols_data_for_this_row, # Pass the data for other columns
            args.L,
            args.keep_paired_neighbors,
            args.mask_threshold,
            args.id_column # Pass the target name for the ID column
        )
        tasks_for_pool.append(task_args)

    if tasks_for_pool:
        if args.num_workers > 1:
            try:
                set_start_method('spawn', force=True)
            except RuntimeError as e:
                print(f"Note: Could not set multiprocessing start method to 'spawn': {e}")
                pass
            
            with Pool(args.num_workers) as pool:
                for window_list_result in tqdm(
                    pool.imap_unordered(process_structure_to_windows_wrapper, tasks_for_pool),
                    total=len(tasks_for_pool),
                    desc="Processing structures into windows (parallel)"
                ):
                    all_window_rows.extend(window_list_result)
        else: # Serial execution if num_workers is 1 or less
            for task_args in tqdm(tasks_for_pool, desc="Processing structures into windows (serial)"):
                window_list_result = process_structure_to_windows_wrapper(task_args)
                all_window_rows.extend(window_list_result)
    else:
        print("No valid structures found to process.")
        log_information(log_path, {"info": "No valid structures to process for windowing."})

    output_df = pd.DataFrame(all_window_rows)
    
    # Reorder columns for consistency: ID, window_start, window_end, window_sequence, seq_len, then others
    leading_cols = [args.id_column, 'window_start', 'window_end', 'window_sequence', 'seq_len'] # Renamed here
    # Get other columns that were actually generated
    other_output_cols = [col for col in output_df.columns if col not in leading_cols]
    final_col_order = leading_cols + sorted(other_output_cols) # Sort other cols alphabetically
    
    # Ensure all leading_cols are present before reordering, add if missing (e.g. if no windows produced)
    for lc in leading_cols:
        if lc not in output_df.columns:
            output_df[lc] = None # or pd.NA

    output_df = output_df[final_col_order]


    output_df.to_csv(args.output, sep='\t', index=False, na_rep='NaN')
    print(f"Windowed sequences saved to {args.output}")
    log_information(log_path, {"output_path": args.output, "num_windows_generated": len(output_df)}, "Output Summary")

if __name__ == "__main__":
    main()
