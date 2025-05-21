#!/usr/bin/env python3

import argparse
import pandas as pd
import faiss
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description="Build a FAISS index from embeddings.tsv (including seq_len)"
    )
    parser.add_argument('--input',        required=True,
                        help="Path to embeddings.tsv")
    parser.add_argument('--id-column',    required=True,
                        help="Name of the transcript/window ID column")
    parser.add_argument('--query',        required=True,
                        help="ID value to treat as “query”")
    parser.add_argument('--index-path',   required=True,
                        help="Where to write the faiss index file")
    parser.add_argument('--mapping-path', required=True,
                        help="Where to write the DB metadata (with seq_len)")
    args = parser.parse_args()

    # 1) Load embeddings (must include seq_len!)
    df = pd.read_csv(args.input, sep='\t')
    if 'seq_len' not in df.columns:
        raise ValueError("embeddings.tsv missing seq_len column")

    # 2) Split out query vs database
    mask_q   = df[args.id_column] == args.query
    db_df    = df.loc[~mask_q, [args.id_column, 'window_start', 'window_end', 'seq_len']].reset_index(drop=True)
    db_vecs  = np.stack(
        df.loc[~mask_q, 'embedding_vector']
          .str.split(',')
          .map(lambda xs: [float(x) for x in xs])
    )

    # 3) Build and save a simple L2 FAISS index
    dim   = db_vecs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(db_vecs)
    faiss.write_index(index, args.index_path)

    # 4) Save the DB‐side metadata (so that index row i → db_df.iloc[i])
    db_df.to_csv(args.mapping_path, sep='\t', index=False)

if __name__ == '__main__':
    main()
