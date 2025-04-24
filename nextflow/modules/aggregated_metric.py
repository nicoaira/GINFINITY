#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from dask import dataframe as dd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def compute_pair_metric(args):
    """
    Compute G(A,B) for one exon pair (or contig) with separate numerator/denominator exponents.
    args = (group_df, alpha1, beta1, alpha2, beta2, gamma)
    """
    group, alpha1, beta1, alpha2, beta2, gamma = args
    L = int(group['len1'].iat[0])
    M = int(group['len2'].iat[0])

    M_mat = np.zeros((L, M), dtype=float)
    denom = np.zeros((L, M), dtype=float)

    for _, row in group.iterrows():
        s1, e1 = int(row['window_start_1']), int(row['window_end_1'])
        s2, e2 = int(row['window_start_2']), int(row['window_end_2'])
        r = float(row['rnk'])
        f_num = r ** (-alpha1)
        f_den = r ** (-alpha2)
        for i in range(s1, e1 + 1):
            for j in range(s2, e2 + 1):
                delta = abs((i - s1) - (j - s2))
                w_num = np.exp(-beta1 * delta)
                w_den = np.exp(-beta2 * delta)
                M_mat[i, j] += f_num * w_num
                denom[i, j] += f_den * w_den

    mask = denom > 0
    M_mat[mask] /= denom[mask] ** gamma

    G = M_mat.sum()
    # scale down by 10E4 = 10 * 10^4 = 100000
    return G / 1e5

def find_contigs(df_grp):
    """
    Given windows for one exon pair, return list of DataFrames for each connected contig.
    """
    n = len(df_grp)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    idxs = np.argsort(df_grp['window_start_1'].values)
    for ii in range(n):
        i = idxs[ii]
        s1_i, e1_i = df_grp.at[i, 'window_start_1'], df_grp.at[i, 'window_end_1']
        s2_i, e2_i = df_grp.at[i, 'window_start_2'], df_grp.at[i, 'window_end_2']
        for jj in range(ii+1, n):
            j = idxs[jj]
            if df_grp.at[j, 'window_start_1'] > e1_i:
                break
            if (df_grp.at[j, 'window_start_2'] <= e2_i and
                s2_i <= df_grp.at[j, 'window_end_2']):
                union(i, j)

    comps = {}
    for i in range(n):
        root = find(i)
        comps.setdefault(root, []).append(i)
    return [df_grp.iloc(idxs).reset_index(drop=True) for idxs in comps.values()]

def main():
    parser = argparse.ArgumentParser(
        description="Compute exon-pair similarity metric with 'global' or 'contigs' aggregation"
    )
    parser.add_argument('--input',      required=True, help='Path to sorted distances TSV')
    parser.add_argument('--percentile', type=float, default=0.01, help='Top percentile to keep')
    parser.add_argument('--mode',       choices=['global','contigs'], default='global',
                        help="Aggregation mode")
    parser.add_argument('--num-workers',type=int, default=1, help='Parallel workers')
    parser.add_argument('--alpha1',     type=float, default=1.0, help='Numerator rank-decay exponent')
    parser.add_argument('--beta1',      type=float, default=0.1, help='Numerator off-diagonal decay')
    parser.add_argument('--alpha2',     type=float, default=1.0, help='Denominator rank-decay exponent')
    parser.add_argument('--beta2',      type=float, default=0.1, help='Denominator off-diagonal decay')
    parser.add_argument('--gamma',      type=float, default=0.5, help='Redundancy penalty exponent')
    parser.add_argument('--output',     default='exon_pair_metrics.tsv', help='Output TSV file')
    args = parser.parse_args()

    # 1) Read & select top-percentile
    ddf = dd.read_csv(args.input, sep='\t',
        usecols=['exon_id_1','window_start_1','window_end_1','exon_sequence_1',
                 'exon_id_2','window_start_2','window_end_2','exon_sequence_2','distance'],
        dtype={'exon_id_1':str,'exon_id_2':str})
    total = ddf.shape[0].compute()
    top_n = max(1, int(total * (args.percentile/100.0)))
    df = ddf.head(top_n, compute=True)

    # 2) Prepare DataFrame
    df = pd.DataFrame(df)
    df['len1'] = df['exon_sequence_1'].str.len()
    df['len2'] = df['exon_sequence_2'].str.len()
    df = df[['exon_id_1','window_start_1','window_end_1',
             'exon_id_2','window_start_2','window_end_2','len1','len2']]
    df['rnk'] = np.arange(1, len(df)+1)

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
            'rnk':            row['rnk']
        })
    df = df.apply(unify, axis=1)

    # 4) Group & compute
    df['pair'] = list(zip(df['exon_id_1'], df['exon_id_2']))
    out = []

    for (e1,e2), grp in df.groupby('pair'):
        grp = grp.reset_index(drop=True)
        contigs = [grp] if args.mode=='global' else find_contigs(grp)
        tasks = [(sub, args.alpha1, args.beta1, args.alpha2, args.beta2, args.gamma) for sub in contigs]
        with ProcessPoolExecutor(max_workers=args.num_workers) as exe:
            for sub, G in zip(contigs, exe.map(compute_pair_metric, tasks)):
                if args.mode=='global':
                    out.append((e1, e2, G))
                else:
                    cs1, ce1 = sub.window_start_1.min(), sub.window_end_1.max()
                    cs2, ce2 = sub.window_start_2.min(), sub.window_end_2.max()
                    out.append((e1, cs1, ce1, e2, cs2, ce2, G))

    # 5) Write
    if args.mode=='global':
        df_out = pd.DataFrame(out, columns=['exon_id_1','exon_id_2','metric'])
    else:
        df_out = pd.DataFrame(out, columns=[
            'exon_id_1','contig_start_1','contig_end_1',
            'exon_id_2','contig_start_2','contig_end_2','metric'])
    df_out.sort_values('metric', ascending=False, inplace=True)
    df_out.to_csv(args.output, sep='\t', index=False)
    print(f"Done. Results ({args.mode}) in {args.output}")

if __name__ == '__main__':
    main()