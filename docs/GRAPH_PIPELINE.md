# Distributed graph pipeline

GINFINITY supports a two-stage data path while retaining the direct
`encode_many` interface:

```text
RNA TSV or Python records
        |
        v
CPU: GraphBuilder.build_shard
        |
        v
*.safetensors + *.json
        |
        v
GPU: Ginfinity.encode_graphs
        |
        v
per-nucleotide embeddings
```

Each graph shard is independent. A scheduler can retry, relocate, or process
it without coordinating mutable state with another worker. The package itself
does not depend on a particular scheduler or object store.

## Artifact contract

The Safetensors file contains only numeric arrays:

| tensor | dtype | shape |
|---|---|---|
| `node_features` | `float32` | `(total_nodes, 7)` |
| `edge_index` | `int32` | `(2, total_edges)` |
| `edge_types` | `uint8` | `(total_edges,)` |
| `node_ptr` | `int64` | `(records + 1,)` |
| `edge_ptr` | `int64` | `(records + 1,)` |

The JSON sidecar stores identifiers, sequences, structures, counts, and the
versioned graph specification. Dense edge one-hot vectors are reconstructed
on the GPU, reducing intermediate storage and transfer volume.

For the bundled model, node columns `0–3` are the `A/C/G/U` one-hot values,
column `4` is the paired/unpaired indicator, and columns `5–6` are sine/cosine
relative-position values. Compact edge codes identify forward/reverse backbone,
base-pair, and two-nucleotide skip edges. The exact mapping is included in the
sidecar's `graph_spec` object.

The graph-specification fingerprint is a constant compatibility identifier
for a particular feature/edge contract. Loading compares that identifier to
the encoder's expected value. It does not calculate a hash per graph.

## Work partitioning

Use a shard as the persistent scheduling unit and a microbatch as the temporary
model-execution unit. `partition_records(records, max_records=b,
max_nodes=...)` creates deterministic input groups for independent CPU jobs.
At encoding time, use both
`max_batch_nodes` and `max_batch_edges` to control accelerator memory.

Store each shard at a unique deterministic path. Write the tensor file and
sidecar before marking the corresponding task complete. Downstream jobs should
consume only completed pairs.

## Validation levels

Default loading checks the format, graph specification, tensor set, dtypes,
shapes, offsets, counts, and global edge bounds. It does not perform a second
content-hash pass.

For less trusted transfer paths, write with `checksum=True` and read with
`verify_checksum=True`. For deeper numerical validation, select
`validation="full"`. Both options consume additional CPU time and are disabled
by default.

## Reproducibility

Record these values beside every embedding output:

- GINFINITY package and model versions.
- Model checkpoint SHA-256.
- Graph specification SHA-256.
- Input graph-shard path and record identifiers.
- Device and node/edge microbatch limits.

Repeated execution with the same device and batching layout is deterministic
on the supported CPU path. Changing microbatch boundaries can change the last
few floating-point bits because numerical kernels may use a different operation
layout; resulting embeddings remain numerically equivalent.
