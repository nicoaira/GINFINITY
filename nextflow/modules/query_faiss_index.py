#!/usr/bin/env python3

import argparse
import pandas as pd
import faiss
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description="Query a FAISS index to get top-K neighbors, outputting full _1/_2 schema"
    )
    parser.add_argument('--input',        required=True,
                        help="Same embeddings.tsv used to build the index")
    parser.add_argument('--id-column',    required=True,
                        help="Name of the transcript/window ID column")
    parser.add_argument('--query',        required=True,
                        help="ID value to treat as “query”")
    parser.add_argument('--index-path',   required=True,
                        help="Path to the faiss index file")
    parser.add_argument('--mapping-path', required=True,
                        help="DB metadata file (from build_faiss_index.py)")
    parser.add_argument('--top-k',        type=int, default=100,
                        help="How many nearest neighbors per query window")
    parser.add_argument('--output',       required=True,
                        help="Path to write distances.tsv")
    args = parser.parse_args()

    # 1) Load everything
    df_all    = pd.read_csv(args.input, sep='\t')
    db_meta   = pd.read_csv(args.mapping_path, sep='\t')
    index     = faiss.read_index(args.index_path)

    # 2) Parse vectors
    vecs = np.stack(
        df_all['embedding_vector']
          .str.split(',')
          .map(lambda xs: [float(x) for x in xs])
    )

    # 3) Identify query windows and their meta
    mask_q = df_all[args.id_column] == args.query
    q_meta = df_all.loc[mask_q, [args.id_column, 'window_start', 'window_end', 'seq_len']] \
                   .to_dict('records')
    q_vecs = vecs[mask_q]

    # 4) Search
    D, I = index.search(q_vecs, args.top_k)

    # 5) Build output rows with <col>_1 (query) and <col>_2 (neighbor)
    rows = []
    for qi, (dists, idxs) in enumerate(zip(D, I)):
        qrec = q_meta[qi]
        for dist, dbi in zip(dists, idxs):
            nrec = db_meta.iloc[dbi]
            rows.append({
                f"{args.id_column}_1":    qrec[args.id_column],
                "window_start_1":         qrec["window_start"],
                "window_end_1":           qrec["window_end"],
                "seq_len_1":              qrec["seq_len"],

                f"{args.id_column}_2":    nrec[args.id_column],
                "window_start_2":         nrec["window_start"],
                "window_end_2":           nrec["window_end"],
                "seq_len_2":              nrec["seq_len"],

                "distance": float(dist)
            })
    out_df = pd.DataFrame(rows)

    # 6) Write exactly distances.tsv that aggregated_score expects
    out_df.to_csv(args.output, sep='\t', index=False)

if __name__ == '__main__':
    main()
