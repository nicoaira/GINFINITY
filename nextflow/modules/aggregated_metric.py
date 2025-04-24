#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
from dask import dataframe as dd

def compute_pair_metric(args):
    group, alpha1, beta1, alpha2, beta2, gamma = args
    L = int(group['len1'].iat[0]); M = int(group['len2'].iat[0])
    M_mat = np.zeros((L, M)); denom = np.zeros((L, M))
    for _, row in group.iterrows():
        s1,e1 = int(row['window_start_1']), int(row['window_end_1'])
        s2,e2 = int(row['window_start_2']), int(row['window_end_2'])
        r = float(row['rnk'])
        f_num, f_den = r**(-alpha1), r**(-alpha2)
        for i in range(s1, e1+1):
            for j in range(s2, e2+1):
                delta = abs((i-s1)-(j-s2))
                M_mat[i,j] += f_num * np.exp(-beta1*delta)
                denom[i,j] += f_den * np.exp(-beta2*delta)
    mask = denom>0
    M_mat[mask] /= denom[mask]**gamma
    return int(round(M_mat.sum()/1e5))

def find_contigs(df_grp):
    n = len(df_grp); parent=list(range(n))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[rb]=ra
    idxs = np.argsort(df_grp['window_start_1'].values)
    for ii in range(n):
        i=idxs[ii]
        s1_i,e1_i = df_grp.at[i,'window_start_1'],df_grp.at[i,'window_end_1']
        s2_i,e2_i = df_grp.at[i,'window_start_2'],df_grp.at[i,'window_end_2']
        for jj in range(ii+1,n):
            j=idxs[jj]
            if df_grp.at[j,'window_start_1']>e1_i: break
            if (df_grp.at[j,'window_start_2']<=e2_i and
                s2_i<=df_grp.at[j,'window_end_2']):
                union(i,j)
    comps={}
    for i in range(n):
        r=find(i); comps.setdefault(r,[]).append(i)
    return [df_grp.iloc[v].reset_index(drop=True) for v in comps.values()]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--percentile', type=float, default=0.01)
    p.add_argument('--mode', choices=['global','contigs'], default='global')
    p.add_argument('--num-workers', type=int, default=1)
    p.add_argument('--alpha1', type=float, default=1.0)
    p.add_argument('--beta1', type=float, default=0.1)
    p.add_argument('--alpha2', type=float, default=1.0)
    p.add_argument('--beta2', type=float, default=0.1)
    p.add_argument('--gamma', type=float, default=0.5)
    p.add_argument('--output', default='exon_pair_metrics.tsv')
    p.add_argument('--output-unaggregated', action='store_true')
    args=p.parse_args()

    # 1) load & filter
    ddf = dd.read_csv(args.input, sep='\t',
        usecols=[
            'exon_id_1','window_start_1','window_end_1','exon_sequence_1',
            'exon_id_2','window_start_2','window_end_2','exon_sequence_2',
            'distance'
        ], dtype={'exon_id_1':str,'exon_id_2':str})
    total = ddf.shape[0].compute()
    top_n = max(1, int(total * args.percentile/100.0))
    df = ddf.head(top_n, compute=True).reset_index(drop=True)

    # 2) annotate windows
    df['window_distance'] = df['distance']
    df['window_rank']     = np.arange(1, len(df)+1)
    df['len1']            = df['exon_sequence_1'].str.len()
    df['len2']            = df['exon_sequence_2'].str.len()
    df = df[[
        'exon_id_1','window_start_1','window_end_1',
        'exon_id_2','window_start_2','window_end_2',
        'len1','len2','window_distance','window_rank'
    ]]
    df['rnk'] = df['window_rank']
    if args.output_unaggregated:
        df_unagg = df.copy()

    # 3) unify exon order
    def unify(r):
        if r['exon_id_1'] <= r['exon_id_2']:
            return r
        return pd.Series({
            'exon_id_1':       r['exon_id_2'],
            'window_start_1':  r['window_start_2'],
            'window_end_1':    r['window_end_2'],
            'exon_id_2':       r['exon_id_1'],
            'window_start_2':  r['window_start_1'],
            'window_end_2':    r['window_end_1'],
            'len1':            r['len2'],
            'len2':            r['len1'],
            'window_distance': r['window_distance'],
            'window_rank':     r['window_rank'],
            'rnk':             r['rnk']
        })
    df = df.apply(unify, axis=1)
    if args.output_unaggregated:
        df_unagg = df_unagg.apply(unify, axis=1)

    # 4) group & collapse contigs
    df['pair'] = list(zip(df['exon_id_1'], df['exon_id_2']))
    aggregated = []; unagg = []; old_id = 0

    for (e1,e2), grp in df.groupby('pair'):
        grp = grp.reset_index(drop=True)
        contigs = [grp] if args.mode=='global' else find_contigs(grp)
        for sub in contigs:
            old_id += 1
            nwin = len(sub)
            G = compute_pair_metric((sub, args.alpha1, args.beta1,
                                     args.alpha2, args.beta2, args.gamma))
            if args.mode == 'global':
                aggregated.append((e1, e2, old_id, nwin, G))
            else:
                cs1,ce1 = sub.window_start_1.min(), sub.window_end_1.max()
                cs2,ce2 = sub.window_start_2.min(), sub.window_end_2.max()
                aggregated.append((e1, cs1, ce1, e2, cs2, ce2, old_id, nwin, G))
            if args.output_unaggregated:
                for _, r in sub.iterrows():
                    unagg.append({
                        **r[['exon_id_1','window_start_1','window_end_1',
                             'exon_id_2','window_start_2','window_end_2',
                             'window_distance','window_rank']].to_dict(),
                        'old_contig_id': old_id
                    })

    # 5) build & sort aggregated
    if args.mode=='global':
        cols_in = ['exon_id_1','exon_id_2','old_contig_id','n_collapsed_windows','metric']
    else:
        cols_in = [
            'exon_id_1','contig_start_1','contig_end_1',
            'exon_id_2','contig_start_2','contig_end_2',
            'old_contig_id','n_collapsed_windows','metric'
        ]
    df_agg = pd.DataFrame(aggregated, columns=cols_in)
    df_agg.sort_values('metric', ascending=False, inplace=True)
    df_agg['contig_id']   = np.arange(1, len(df_agg)+1)
    df_agg['contig_rank'] = df_agg['contig_id']

    # write aggregated
    if args.mode=='global':
        out_cols = ['exon_id_1','exon_id_2','contig_id','contig_rank',
                    'n_collapsed_windows','metric']
    else:
        out_cols = [
            'exon_id_1','contig_start_1','contig_end_1',
            'exon_id_2','contig_start_2','contig_end_2',
            'contig_id','contig_rank','n_collapsed_windows','metric'
        ]
    df_agg[out_cols].to_csv(args.output, sep='\t', index=False)
    print(f"Written aggregated to {args.output}")

    # 6) write unaggregated, sorted by descending contig_metric
    if args.output_unaggregated:
        base, ext = os.path.splitext(args.output)
        fn = f"{base}.unaggregated{ext}"
        mapping = df_agg.set_index('old_contig_id')[['contig_id','contig_rank',
                                                     'n_collapsed_windows','metric']].to_dict('index')
        rows = []
        for rec in unagg:
            m = mapping[rec['old_contig_id']]
            rec.update({
                'contig_id':           m['contig_id'],
                'contig_rank':         m['contig_rank'],
                'n_collapsed_windows': m['n_collapsed_windows'],
                'contig_metric':       m['metric']
            })
            rec.pop('old_contig_id')
            rows.append(rec)
        df_un = pd.DataFrame(rows)
        # **new change**: sort by contig_metric descending
        df_un.sort_values('contig_metric', ascending=False, inplace=True)
        df_un.to_csv(fn, sep='\t', index=False)
        print(f"Written unaggregated to {fn}")

if __name__=='__main__':
    main()
