#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from dask import dataframe as dd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def compute_pair_metric(args):
    """
    Compute G(A,B) for one exon pair with separate numerator/denominator exponents.
    Args tuple: (group, alpha1, beta1, alpha2, beta2, gamma)
    """
    group, alpha1, beta1, alpha2, beta2, gamma = args
    L = group['len1'].iat[0]
    M = group['len2'].iat[0]
    M_mat = np.zeros((L, M), dtype=float)
    denom = np.zeros((L, M), dtype=float)

    for _, row in group.iterrows():
        s1, e1 = int(row['window_start_1']), int(row['window_end_1'])
        s2, e2 = int(row['window_start_2']), int(row['window_end_2'])
        r = row['rank']
        f_num = 1.0 / (r ** alpha1)
        f_den = 1.0 / (r ** alpha2)
        for i in range(s1, e1 + 1):
            for j in range(s2, e2 + 1):
                delta = abs((i - s1) - (j - s2))
                w_num = np.exp(-beta1 * delta)
                w_den = np.exp(-beta2 * delta)
                M_mat[i, j] += f_num * w_num
                denom[i, j] += f_den * w_den

    mask = denom > 0
    M_mat[mask] /= denom[mask] ** gamma
    # Global metric: sum of weighted contributions (no normalization by matrix size)
    G = M_mat.sum()
    return (group['exon_id_1'].iat[0], group['exon_id_2'].iat[0], G)


def main():
    parser = argparse.ArgumentParser(
        description="Compute exon-pair similarity metric with separate numerator/denominator parameters"
    )
    parser.add_argument('--input',      required=True,
                        help='Path to sorted distances TSV')
    parser.add_argument('--percentile', type=float, default=0.01,
                        help='Top percentile to keep (default 0.01)')
    parser.add_argument('--num-workers',type=int,   default=1,
                        help='Number of parallel workers')
    parser.add_argument('--alpha1',     type=float, default=1.0,
                        help='Numerator rank-decay exponent (default 1.0)')
    parser.add_argument('--beta1',      type=float, default=0.1,
                        help='Numerator off-diagonal decay (default 0.1)')
    parser.add_argument('--alpha2',     type=float, default=1.0,
                        help='Denominator rank-decay exponent (default 1.0)')
    parser.add_argument('--beta2',      type=float, default=0.1,
                        help='Denominator off-diagonal decay (default 0.1)')
    parser.add_argument('--gamma',      type=float, default=0.5,
                        help='Redundancy penalty exponent (default 0.5)')
    parser.add_argument('--output',     default='exon_pair_metrics.tsv',
                        help='Output TSV file')
    args = parser.parse_args()

    # 1) Read and slice top-percentile
    ddf = dd.read_csv(
        args.input, sep='\t',
        usecols=[
            'exon_id_1','window_start_1','window_end_1','exon_sequence_1',
            'exon_id_2','window_start_2','window_end_2','exon_sequence_2','distance'
        ], dtype={'exon_id_1':str,'exon_id_2':str}
    )
    total_rows = ddf.shape[0].compute()
    top_n = max(1, int(total_rows * (args.percentile/100.0)))
    df = ddf.head(top_n, compute=True)

    # 2) Prepare DataFrame
    df = pd.DataFrame(df)
    df['len1'] = df['exon_sequence_1'].str.len()
    df['len2'] = df['exon_sequence_2'].str.len()
    df = df[[
        'exon_id_1','window_start_1','window_end_1',
        'exon_id_2','window_start_2','window_end_2','len1','len2'
    ]]
    df['rank'] = np.arange(1, len(df) + 1)

    # 3) Unify exon order
    def unify(row):
        if row['exon_id_1'] <= row['exon_id_2']:
            return row
        return pd.Series({
            'exon_id_1':      row['exon_id_2'],
            'window_start_1': row['window_start_2'],
            'window_end_1':   row['window_end_2'],
            'exon_id_2':      row['exon_id_1'],
            'window_start_2': row['window_start_1'],
            'window_end_2':   row['window_end_1'],
            'len1':           row['len2'],
            'len2':           row['len1'],
            'rank':           row['rank']
        })
    df = df.apply(unify, axis=1)

    # 4) Group
    df['pair'] = list(zip(df['exon_id_1'], df['exon_id_2']))
    groups = [grp for _, grp in df.groupby('pair')]

    # 5) Compute in parallel
    tasks = [(
        grp,
        args.alpha1, args.beta1,
        args.alpha2, args.beta2,
        args.gamma
    ) for grp in groups]
    results = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(compute_pair_metric, t): t for t in tasks}
        for f in tqdm(as_completed(futures), total=len(futures),
                      desc="Computing exon-pair metrics"):
            results.append(f.result())

    # 6) Write output
    out = pd.DataFrame(results, columns=['exon_id_1','exon_id_2','metric'])
    out = out.sort_values('metric', ascending=False)
    out.to_csv(args.output, sep='\t', index=False)
    print(f"Done. Results in {args.output}")

if __name__ == '__main__':
    main()
