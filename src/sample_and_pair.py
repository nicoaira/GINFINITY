import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import argparse
import os
from multiprocessing import Pool, cpu_count

def sample_rows(input_path, n=1000):
    df = pd.read_csv(input_path, sep='\t')
    sampled_df = df.sample(n=n, random_state=42).reset_index(drop=True)
    return sampled_df

def calculate_distance_batch(args):
    batch, embeddings_tensor, metric = args
    results = []
    for i, j in batch:
        if metric == 'cosine':
            distance = 1 - torch.nn.functional.cosine_similarity(embeddings_tensor[i], embeddings_tensor[j], dim=0).item()
        else:  # squared distance
            distance = torch.sum((embeddings_tensor[i] - embeddings_tensor[j]) ** 2).item()
        results.append((i, j, distance))
    return results

def calculate_distances(embeddings, metric='squared', num_workers=1, batch_size=1000):
    embeddings_tensor = torch.tensor(np.array(embeddings), dtype=torch.float32)
    num_embeddings = embeddings_tensor.shape[0]
    
    total_pairs = num_embeddings * (num_embeddings - 1) // 2
    pairs = [(i, j) for i in range(num_embeddings) for j in range(i + 1, num_embeddings)]
    
    # Split pairs into batches
    batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
    args_list = [(batch, embeddings_tensor, metric) for batch in batches]
    
    distances = []
    with Pool(num_workers) as pool:
        with tqdm(total=total_pairs, desc="Calculating distances") as pbar:
            for result in pool.imap_unordered(calculate_distance_batch, args_list):
                distances.extend(result)
                pbar.update(len(result))
    
    return distances

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
