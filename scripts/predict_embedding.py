#!/usr/bin/env python3
import sys, os
# TODO: Remove this when the module is properly installed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import time
import torch
import pandas as pd
from tqdm import tqdm
import argparse
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
import os
from torch.multiprocessing import Pool, set_start_method


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
    """
    Generates GIN embeddings for an RNA structure.
    Assumes the input 'structure' is the exact sequence to be embedded (either full or a pre-generated window).
    """
    # Validate the overall structure first
    try:
        if not is_valid_dot_bracket(structure): 
            raise ValueError("Invalid dot bracket string")
    except ValueError as e:
        return [(None, "")] # Indicate error/skip

    if graph_encoding == "standard":
        graph = dotbracket_to_graph(structure)
        tg = graph_to_tensor(graph)
    else:
        graph = dotbracket_to_forgi_graph(structure)
        tg = forgi_graph_to_tensor(graph)
    
    if graph is None or tg is None: # Graph conversion failed
        return [(None, "")]

    tg = tg.to(device)
    model.eval()
    with torch.no_grad():
        emb = model.forward_once(tg)
        return [(None, ','.join(f'{x:.6f}' for x in emb.cpu().numpy().flatten()))]

def generate_embedding_for_row(args_tuple):
    """
    Wrapper to generate embedding for a single row of data.
    Designed for use with multiprocessing Pool.

    Args:
        args_tuple: A tuple containing all necessary arguments:
            (unique_id, structure_string, model_instance, graph_encoding_method, device_to_use)
    """
    (
        unique_id,
        structure_string,
        model_instance,
        graph_encoding_method,
        device_to_use,
    ) = args_tuple

    embedding_results = get_gin_embedding(
        model_instance,
        graph_encoding_method,
        structure_string,
        device_to_use
    )

    return [(unique_id, start_idx, emb_str) for start_idx, emb_str in embedding_results]

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
    # Resolve keep_cols
    final_keep_cols = [id_column] 
    if 'seq_len' not in final_keep_cols and 'seq_len' in input_df.columns:
        final_keep_cols.append('seq_len') 
        
    if keep_cols:
        specified_cols = [c.strip() for c in keep_cols.split(',')]
        for col in specified_cols:
            if col not in input_df.columns:
                print(f"Warning: Requested keep_col '{col}' not found in input DataFrame. It will be ignored.")
                log_information(log_path, {"warning": f"Keep_col '{col}' not found and ignored."})
            elif col not in final_keep_cols:
                final_keep_cols.append(col)
    
    # Load model
    model = load_trained_model(model_path, device)
    graph_encoding = model.metadata['graph_encoding']

    # Prepare arguments for each row for multiprocessing
    args_list_for_pool = []
    for original_idx, row_series in input_df.iterrows():
        current_id = row_series[id_column]
        structure = row_series[structure_column]

        if not isinstance(structure, str):
            print(f"Skipping ID {current_id} (DataFrame index {original_idx}): structure is not a string.")
            log_information(log_path, {"skipped_invalid_structure_type": f"ID {current_id}"})
            continue

        args_list_for_pool.append((
            current_id,
            structure,
            model,
            graph_encoding,
            device
        ))

    # Compute embeddings in parallel
    all_embedding_results = []
    if args_list_for_pool: 
        with Pool(num_workers) as pool:
            try:
                for single_row_results in tqdm(
                    pool.imap_unordered(generate_embedding_for_row, args_list_for_pool),
                    total=len(args_list_for_pool),
                    desc="Generating Embeddings"
                ):
                    all_embedding_results.extend(single_row_results)
            finally:
                pool.close()
                pool.join()
    else:
        print("No valid structures found to process for embeddings.")
        log_information(log_path, {"info": "No valid structures to process."})

    # Assemble output rows
    output_rows_list = []
    input_df_id_indexed = input_df.set_index(id_column)

    for processed_id, window_start_or_none, emb_str in all_embedding_results:
        if emb_str == "": 
            log_information(log_path, {"skipped_empty_embedding": f"ID {processed_id}"})
            continue
        
        try:
            original_row_data = input_df_id_indexed.loc[processed_id]
        except KeyError:
            print(f"Warning: ID {processed_id} from embedding results not found in original DataFrame. Skipping.")
            log_information(log_path, {"warning": f"ID {processed_id} not found in original DF after processing."})
            continue

        output_row = {col: original_row_data[col] for col in final_keep_cols if col in original_row_data}
        output_row[id_column] = processed_id 
        
        output_row['embedding_vector'] = emb_str
        output_rows_list.append(output_row)

    output_df = pd.DataFrame(output_rows_list)
    
    # Define column order for output
    leading_cols = [id_column, 'embedding_vector']
    
    other_output_cols = [col for col in final_keep_cols if col not in leading_cols and col in output_df.columns]
    
    window_cols_present = []
    if 'window_start' in other_output_cols:
        window_cols_present.append('window_start')
        other_output_cols.remove('window_start')
    if 'window_end' in other_output_cols:
        window_cols_present.append('window_end')
        other_output_cols.remove('window_end')

    final_col_order = [id_column] + window_cols_present + ['embedding_vector'] + sorted(other_output_cols)

    for col in final_col_order:
        if col not in output_df.columns:
            output_df[col] = None
            
    output_df = output_df[final_col_order]

    log_information(log_path, {"output_head_sample": output_df.head().to_string()}, "Output Sample")
    output_df.to_csv(output_path, sep='\t', index=False, na_rep='NaN')
    print(f"Embeddings saved to {output_path}")
    log_information(log_path, {"output_saved_to": output_path, "num_embeddings_generated": len(output_df)})

if __name__ == "__main__":
    try:
        set_start_method('spawn', force=True) 
    except RuntimeError as e:
        print(f"Note: Could not set multiprocessing start method to 'spawn': {e}")
        pass 

    parser = argparse.ArgumentParser(
        description="Generate GIN embeddings from RNA secondary structures (full or pre-windowed)."
    )
    # --- Input/Output Arguments ---
    parser.add_argument('--input', type=str, required=True, help="Path to input TSV/CSV file. This file should contain structures, either full or pre-windowed.")
    parser.add_argument('--output', type=str, required=True, help="Path to output TSV file for embeddings.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained GIN model (.pth file or checkpoint)." )
    
    # --- Column Specification Arguments ---
    parser.add_argument('--id-column', type=str, required=True, help="Name of the column containing unique sequence/structure identifiers.")
    parser.add_argument('--structure-column-name', type=str, help="Name of the column containing RNA secondary structures. Overrides --structure-column-num. Assumed to be 'secondary_structure' if not specified and header is present.")
    parser.add_argument('--structure-column-num', type=int, help="0-based index of the structure column. Required if input has no header and --structure-column-name is not provided.")
    parser.add_argument('--header', type=str, default='True', choices=['True', 'False', 'true', 'false'], help="Whether the input file has a header row (default: True).")
    parser.add_argument('--keep-cols', type=str, help="Comma-separated list of additional columns from the input file to retain in the output file.")

    # --- Embedding Generation Arguments ---
    parser.add_argument('--device', type=str, default="cpu", help="Device to use for model inference ('cpu' or 'cuda', default: cpu).")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of worker processes for parallel embedding generation (default: 4).")
    
    args = parser.parse_args()
    args.header = args.header.lower() == 'true'

    # Setup logging
    outdir = os.path.dirname(args.output) or '.'
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.splitext(args.output)[0] + '.log'
    log_setup(log_path, print_log=False)
    log_information(log_path, vars(args), "Effective Arguments", print_log=True)

    input_df = read_input_data(
        args.input,
        header=args.header,
        id_column_for_validation=args.id_column if args.header else None 
    )

    structure_col_actual = get_structure_column_name(
        input_df,
        args.header,
        col_name=args.structure_column_name,
        col_num=args.structure_column_num,
        default_name='secondary_structure'
    )
    if args.header and structure_col_actual not in input_df.columns:
        raise ValueError(f"Structure column '{structure_col_actual}' not found in input file with header. Columns: {list(input_df.columns)}")
    if not args.header and not isinstance(structure_col_actual, int):
        if structure_col_actual not in input_df.columns:
            raise ValueError(f"Structure column index {structure_col_actual} is invalid for headerless input.")

    id_col_actual = args.id_column
    if args.header:
        if id_col_actual not in input_df.columns:
            raise ValueError(f"ID column '{id_col_actual}' not found in input file. Columns: {list(input_df.columns)}")
    else:
        try:
            col_idx_for_id = int(id_col_actual) if id_col_actual.isdigit() else id_col_actual
            if col_idx_for_id not in input_df.columns:
                raise ValueError(f"ID column '{id_col_actual}' (interpreted as index/name {col_idx_for_id}) not found in headerless input. Available columns: {list(input_df.columns)}")
        except ValueError:
            raise ValueError(f"ID column '{id_col_actual}' is problematic for headerless input. Provide a valid column index/name.")

    if input_df[id_col_actual].duplicated().any():
        print(f"Warning: Duplicate values found in ID column '{id_col_actual}'. This may lead to incorrect data merging for embeddings.")
        log_information(log_path, {"critical_warning": f"Duplicate IDs in '{id_col_actual}'"})

    start_time = time.time()
    generate_embeddings(
        input_df=input_df,
        output_path=args.output,
        model_path=args.model_path,
        log_path=log_path,
        structure_column=structure_col_actual, 
        id_column=id_col_actual,          
        device=args.device,
        num_workers=args.num_workers,
        keep_cols=args.keep_cols
    )
    elapsed_time_minutes = (time.time() - start_time) / 60
    print(f"Processing completed in {elapsed_time_minutes:.2f} minutes.")
    log_information(log_path, {"total_time_minutes": round(elapsed_time_minutes, 2)}, "Performance")
