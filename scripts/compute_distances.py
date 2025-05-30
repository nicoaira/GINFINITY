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
    parser.add_argument(
        "--input", required=True,
        help="Path to the input TSV file."
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to the output TSV file."
    )
    parser.add_argument(
        "--embedding-col", default="embedding_vector",
        help="Name of the column that contains the embedding vector."
    )
    parser.add_argument(
        "--keep-cols", default=None,
        help=(
            "Comma-separated list of columns to carry through into the output "
            "(e.g. transcript_id,other_meta). If not provided, uses --id-column."
        )
    )
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help="Number of worker threads to use."
    )
    parser.add_argument(
        "--device", default="cpu",
        help="PyTorch device for computation (e.g. cpu, cuda:0)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000,
        help="How many pairs to process per batch."
    )
    parser.add_argument(
        "--mode", type=int, default=1, choices=[1, 2],
        help="1 = all-vs-all; 2 = one-vs-all (needs --query)."
    )
    parser.add_argument(
        "--id-column", default="exon_id",
        help="Name of the column that identifies each row (used in mode=2)."
    )
    parser.add_argument(
        "--query",
        help="Value in --id-column to compare against all others (required if --mode=2)."
    )
    return parser.parse_args()

def process_batch(pair_indices, embeddings_tensor, df, columns_to_keep, device):
    idx_1, idx_2 = zip(*pair_indices) if pair_indices else ([], [])
    v1 = embeddings_tensor[list(idx_1)].to(device)
    v2 = embeddings_tensor[list(idx_2)].to(device)
    distances = torch.sum((v1 - v2) ** 2, dim=1).cpu().numpy()

    rows1 = df.iloc[list(idx_1)][columns_to_keep].copy().add_suffix("_1").reset_index(drop=True)
    rows2 = df.iloc[list(idx_2)][columns_to_keep].copy().add_suffix("_2").reset_index(drop=True)
    batch_df = pd.concat([rows1, rows2], axis=1)
    batch_df["distance"] = distances
    return batch_df

def batch_generator(pair_gen, batch_size):
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

    # If user didn't specify keep-cols, default to the id column
    if not args.keep_cols:
        args.keep_cols = args.id_column

    # Load the input DataFrame
    df = pd.read_csv(args.input, sep="\t")

    # Parse keep-cols and verify they exist
    columns_to_keep = [col.strip() for col in args.keep_cols.split(",")]
    missing = [c for c in columns_to_keep if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input: {', '.join(missing)}")

    # Parse embeddings into a tensor
    embeddings_list = df[args.embedding_col].apply(lambda s: [float(x) for x in s.split(",")])
    embeddings_tensor = torch.tensor(embeddings_list.tolist(), dtype=torch.float32)

    device = torch.device(args.device)

    # Decide which pairs to compute
    n = len(df)
    if args.mode == 1:
        pair_indices = combinations(range(n), 2)
        total_pairs = n * (n - 1) // 2
    else:
        if not args.query:
            raise ValueError("--query must be provided when --mode=2.")
        if args.id_column not in df.columns:
            raise ValueError(f"--id-column '{args.id_column}' not found in input.")
        mask_q = df[args.id_column] == args.query
        idx_q = df[mask_q].index
        if len(idx_q) == 0:
            raise ValueError(f"No rows where {args.id_column} == {args.query}")
        idx_o = df[~mask_q].index
        pair_indices = product(idx_q, idx_o)
        total_pairs = len(idx_q) * len(idx_o)

    # Break into batches
    batches = list(batch_generator(pair_indices, args.batch_size))

    # Process with a progress bar
    pbar = tqdm(total=total_pairs, desc="Processing pairs", unit="pair")
    results = []
    if args.num_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as exec:
            futures = [exec.submit(process_batch, b, embeddings_tensor, df, columns_to_keep, device)
                       for b in batches]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())
                pbar.update(len(f.result()))
    else:
        for b in batches:
            batch_df = process_batch(b, embeddings_tensor, df, columns_to_keep, device)
            results.append(batch_df)
            pbar.update(len(batch_df))
    pbar.close()

    # Concatenate and write out
    out_df = pd.concat(results, ignore_index=True)
    out_df.to_csv(args.output, sep="\t", index=False)
    print(f"Finished processing {total_pairs} pairs. Output written to {args.output}")

if __name__ == "__main__":
    main()
