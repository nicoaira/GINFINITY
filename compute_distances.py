#!/usr/bin/env python3

import argparse
import pandas as pd
import torch
from itertools import combinations, product
from tqdm import tqdm
import concurrent.futures

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute squared Euclidean distances between rows’ embedding vectors."
    )
    parser.add_argument("--input", required=True, help="Path to the input TSV file.")
    parser.add_argument("--output", required=True, help="Path to the output TSV file.")
    parser.add_argument("--embedding-col", default="embedding_vector",
                        help="Name of the column that contains the embedding vector (default: embedding_vector).")
    parser.add_argument("--keep-cols", default="exon_id",
                        help="Comma-separated list of column names to keep (e.g., exon_id,transcript_id). Default is 'exon_id'.")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of worker threads to use (default: 1).")
    parser.add_argument("--device", default="cpu",
                        help="PyTorch device to perform computations on (e.g., cpu, cuda:0).")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Number of pairs to compute in each batch (default: 1000).")
    parser.add_argument("--mode", type=int, default=1, choices=[1, 2],
                        help="Mode of distance calculation: 1=all-vs-all (default), 2=one-vs-all.")
    parser.add_argument("--id-column", default="exon_id",
                        help="Name of the column that identifies the query group in mode=2 (default: exon_id).")
    parser.add_argument("--query",
                        help="Value in --id-column to use for one-vs-all comparisons in mode=2.")

    return parser.parse_args()


def process_batch(pair_indices, embeddings_tensor, df, columns_to_keep, device):
    """
    Process a batch of row pairs.
    Parameters:
      pair_indices: list of tuples (i, j) of row indices.
      embeddings_tensor: torch.Tensor of shape [n, d] containing all embeddings.
      df: original pandas DataFrame.
      columns_to_keep: list of the specified columns to be kept.
      device: torch.device.
      
    Returns:
      pandas DataFrame corresponding to the processed batch, containing the specified
      non-embedding columns (with suffixes _1 and _2) and an additional 'distance' column.
    """
    # Unzip the list of pairs into two lists: indices for first and second row in each pair.
    idx_1, idx_2 = zip(*pair_indices) if pair_indices else ([], [])
    
    # Select the embedding vectors for the indices and move to the desired device
    v1 = embeddings_tensor[list(idx_1)].to(device)
    v2 = embeddings_tensor[list(idx_2)].to(device)
    
    # Compute squared Euclidean distance in a vectorized manner
    distances = torch.sum((v1 - v2) ** 2, dim=1).cpu().numpy()
    
    # Retrieve the specified columns and add suffix _1 for first rows and _2 for second rows
    rows1 = df.iloc[list(idx_1)][columns_to_keep].copy().add_suffix("_1").reset_index(drop=True)
    rows2 = df.iloc[list(idx_2)][columns_to_keep].copy().add_suffix("_2").reset_index(drop=True)
    
    # Combine both dataframes and add the distances column
    batch_df = pd.concat([rows1, rows2], axis=1)
    batch_df["distance"] = distances
    
    return batch_df


def batch_generator(pair_gen, batch_size):
    """
    Yields lists (batches) of pair indices of length at most batch_size.
    """
    batch = []
    for pair in pair_gen:
        batch.append(pair)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    args = parse_args()

    # Load the TSV file
    df = pd.read_csv(args.input, sep="\t")
    n_rows = len(df)
    if n_rows < 2:
        raise ValueError("Need at least two rows to compute pairwise distances.")

    # Parse the embedding column.
    # Assumes that each cell in the embedding column is a comma-separated list of numbers.
    embeddings_list = df[args.embedding_col].apply(lambda s: [float(x) for x in s.split(",")])
    embeddings_tensor = torch.tensor(embeddings_list.tolist(), dtype=torch.float32)

    # Process --keep-cols argument, split by comma, and verify that each provided column exists in the dataframe.
    columns_to_keep = [col.strip() for col in args.keep_cols.split(",")]
    missing_cols = [col for col in columns_to_keep if col not in df.columns]
    if missing_cols:
        raise ValueError(f"The following columns are missing in the input file: {', '.join(missing_cols)}")

    device = torch.device(args.device)

    # Determine pairwise indices based on mode
    if args.mode == 1:
        # ALL VS ALL
        pair_indices_generator = combinations(range(n_rows), 2)
        total_pairs = n_rows * (n_rows - 1) // 2
    else:
        # ONE VS ALL
        if not args.query:
            raise ValueError("--query must be provided when --mode=2.")

        if args.id_column not in df.columns:
            raise ValueError(f"The --id-column '{args.id_column}' does not exist in the dataframe.")
        
        # Rows matching query
        mask_query = (df[args.id_column] == args.query)
        idx_query = df[mask_query].index
        if len(idx_query) == 0:
            raise ValueError(f"No rows found where {args.id_column} == {args.query}.")

        # Rows not matching query
        mask_others = (df[args.id_column] != args.query)
        idx_others = df[mask_others].index

        # We'll compute distances for the cartesian product: each row in idx_query vs each in idx_others
        pair_indices_generator = product(idx_query, idx_others)
        total_pairs = len(idx_query) * len(idx_others)

    # Container for all resulting DataFrame chunks
    result_chunks = []

    # Split the pairs into batches
    batches = list(batch_generator(pair_indices_generator, args.batch_size))
    num_batches = len(batches)

    # Setup tqdm progress bar
    pbar = tqdm(total=total_pairs, desc="Processing pairs", unit="pair")

    # Process batches in parallel or sequentially depending on num-workers
    if args.num_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(process_batch, batch, embeddings_tensor, df, columns_to_keep, device)
                       for batch in batches]

            for future in concurrent.futures.as_completed(futures):
                batch_df = future.result()
                result_chunks.append(batch_df)
                pbar.update(len(batch_df))
    else:
        for batch in batches:
            batch_df = process_batch(batch, embeddings_tensor, df, columns_to_keep, device)
            result_chunks.append(batch_df)
            pbar.update(len(batch_df))

    pbar.close()

    # Concatenate the result chunks and output to TSV
    result_df = pd.concat(result_chunks, ignore_index=True)
    result_df.to_csv(args.output, sep="\t", index=False)
    print(f"Finished processing {total_pairs} pairs. Output written to {args.output}")


if __name__ == "__main__":
    main()
