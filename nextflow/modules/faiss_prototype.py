# prototype_faiss.py
import faiss, numpy as np, pandas as pd

df = pd.read_csv('../results_nf_transcript/embeddings.tsv', sep='\t')
# parse vectors into array shape (N, D)
vecs = np.stack(df.embedding_vector.str.split(',').map(lambda xs: [float(x) for x in xs]))
# boolean mask of your query windows
is_query = df.transcript_id.eq('ENST00000811669.1')
db_vecs    = vecs[~is_query]
query_vecs = vecs[is_query]

# build an exact L2 index (IndexFlatL2); for larger scale switch to IVF/PQ or HNSW
index = faiss.IndexFlatL2(db_vecs.shape[1])
index.add(db_vecs)

# search top 100 nearest neighbors per query
D, I = index.search(query_vecs, 300)

# I is an array shape (n_query, K) of row-indices into db_vecs
# map back to original df rows:
db_meta = df.loc[~is_query, ['transcript_id','window_start','window_end']].reset_index(drop=True)
out = []
for qi, (dist_row, idx_row) in enumerate(zip(D, I)):
    qmeta = df.loc[is_query].iloc[qi, :][['transcript_id','window_start','window_end']]
    for dist, dbi in zip(dist_row, idx_row):
        nnei = db_meta.iloc[dbi]
        out.append({
            'q_id': qmeta.transcript_id,
            'q_start': qmeta.window_start,
            'q_end':   qmeta.window_end,
            'n_id': nnei.transcript_id,
            'n_start': nnei.window_start,
            'n_end':   nnei.window_end,
            'distance': float(dist)
        })

pd.DataFrame(out).to_csv('faiss_distances.tsv', sep='\t', index=False)
