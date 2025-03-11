import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import argparse
import os
from multiprocessing import cpu_count
from utils import calculate_distances  # Added import

def sample_rows(input_path, n=1000):
    df = pd.read_csv(input_path, sep='\t')
    sampled_df = df.sample(n=n, random_state=42).reset_index(drop=True)
    return sampled_df

# Removed: calculate_distance_batch and calculate_distances functions

def generate_pairs(sampled_df, distances):
    pairs = []
    for i, j, distance in distances:
        row_i = sampled_df.iloc[i]
        row_j = sampled_df.iloc[j]
        pair = {
            'id_1': row_i['rnacentral_id'],
            'id_2': row_j['rnacentral_id'],
            'distance': distance,
            'structure_1': row_i['secondary_structure'],
            'structure_2': row_j['secondary_structure'],
            'sequence_1': row_i['sequence'],
            'sequence_2': row_j['sequence']
        }
        pairs.append(pair)
    return pd.DataFrame(pairs)

def main(input_path, n, metric, num_workers, batch_size):
    # Sample rows
    sampled_df = sample_rows(input_path, n)

    # Extract embedding vectors
    embeddings = sampled_df['embedding_vector'].apply(lambda x: np.array([float(i) for i in x.split(',')])).tolist()

    # Calculate distances
    distances = calculate_distances(embeddings, metric, num_workers, batch_size)

    # Generate pairs
    pairs_df = generate_pairs(sampled_df, distances)

    # Output file
    output_path = os.path.splitext(input_path)[0] + '_pairs.tsv'
    pairs_df.to_csv(output_path, sep='\t', index=False)
    print(f"Pairs saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample rows from a TSV file, generate pairwise combinations, and calculate distances or similarities between embedding vectors.")
    parser.add_argument('--input', type=str, required=True, help='Path to the input TSV file.')
    parser.add_argument('--n', type=int, default=1000, help='Number of rows to sample (default: 1000).')
    parser.add_argument('--metric', type=str, choices=['squared', 'cosine'], default='squared', help='Distance metric to use (default: squared).')
    parser.add_argument('--num_workers', type=int, default=cpu_count(), help='Number of worker processes to use for multiprocessing (default: number of CPU cores).')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for distance calculations (default: 1000).')
    args = parser.parse_args()

    main(args.input, args.n, args.metric, args.num_workers, args.batch_size)
