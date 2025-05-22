#!/usr/bin/env python3
import argparse
import csv
import numpy as np
import faiss

def main():
    p = argparse.ArgumentParser(
        description="Build a FAISS index from embeddings.tsv (streaming, low‐memory)"
    )
    p.add_argument('--input',        required=True,
                   help="Path to embeddings.tsv")
    p.add_argument('--id-column',    required=True,
                   help="Name of the transcript/window ID column")
    p.add_argument('--query',        required=True,
                   help="ID value to treat as “query”")
    p.add_argument('--index-path',   required=True,
                   help="Where to write the faiss index file")
    p.add_argument('--mapping-path', required=True,
                   help="Where to write the DB metadata (with seq_len)")
    p.add_argument('--chunk-size',   type=int, default=10000,
                   help="Number of vectors to buffer before adding to index")
    args = p.parse_args()

    # Open mapping file and write header
    with open(args.mapping_path, 'w', newline='') as map_f:
        writer = csv.writer(map_f, delimiter='\t')
        writer.writerow([args.id_column, 'window_start', 'window_end', 'seq_len'])

        # Stream‐read the embeddings TSV
        with open(args.input, newline='') as in_f:
            reader = csv.DictReader(in_f, delimiter='\t')
            # Determine embedding dimension from first non-query row
            for first in reader:
                if first[args.id_column] == args.query:
                    continue
                emb0 = np.fromstring(first['embedding_vector'], sep=',', dtype='float32')
                dim = emb0.shape[0]
                # initialize FAISS index
                index = faiss.IndexFlatL2(dim)
                # start buffer with this first vector
                buffer_vecs  = [emb0]
                buffer_meta  = [
                    (first[args.id_column],
                     first['window_start'],
                     first['window_end'],
                     first['seq_len'])
                ]
                break
            else:
                raise ValueError(f"Query {args.query} not found in {args.input}")

            # Process remaining rows
            for row in reader:
                if row[args.id_column] == args.query:
                    continue
                vec = np.fromstring(row['embedding_vector'], sep=',', dtype='float32')
                buffer_vecs.append(vec)
                buffer_meta.append((
                    row[args.id_column],
                    row['window_start'],
                    row['window_end'],
                    row['seq_len']
                ))

                # once buffer is full, add to FAISS and flush metadata
                if len(buffer_vecs) >= args.chunk_size:
                    arr = np.stack(buffer_vecs)
                    index.add(arr)
                    for m in buffer_meta:
                        writer.writerow(m)
                    buffer_vecs.clear()
                    buffer_meta.clear()

            # Add any leftovers
            if buffer_vecs:
                arr = np.stack(buffer_vecs)
                index.add(arr)
                for m in buffer_meta:
                    writer.writerow(m)

    # Finally write the FAISS index to disk
    faiss.write_index(index, args.index_path)
    print(f"Wrote FAISS index → {args.index_path}")
    print(f"Wrote mapping TSV → {args.mapping_path}")

if __name__ == '__main__':
    main()
