# API reference

## `RNA`

`RNA(identifier, sequence, structure, start=None, end=None)` is an immutable
validated input record. Normalization and structural validation happen before
graph construction. `start` and `end` are optional 0-based half-open
coordinates into the normalized sequence (`sequence[start:end]` is the
core window). Both must be omitted for a full-molecule record.

`RNA.from_mapping(row, identifier_column=..., sequence_column=...,
structure_column=..., start_column=None, end_column=None)` creates one
record from a mapping. A mapping that lists several windows must use
`RNA.many_from_mapping(...)`, which emits one `RNA` per window and suffixes
identifiers as `{id}:{start}-{end}`.

## `read_rna_table`

```python
read_rna_table(
    path,
    *,
    identifier_column="transcript_id",
    sequence_column="sequence",
    structure_column="secondary_structure",
    start_column="start",
    end_column="end",
    delimiter="\t",
) -> list[RNA]
```

Reads records in file order. Required columns may appear in any order and the
table may contain additional columns. The delimiter must be one character.
Every row passes through the normal `RNA` validation contract.

`start` and `end` are optional. When those columns are present, each row
may hold one window or parallel comma-separated windows. Each window
becomes its own record with identifier `{id}:{start}-{end}`. Pass
`start_column=None, end_column=None` to ignore window columns.

## `Ginfinity.load`

```python
Ginfinity.load(
    device="cpu",
    *,
    allow_nondeterministic_cuda=False,
    model_dir=None,
) -> Ginfinity
```

Loads and SHA-256-verifies the packaged checkpoint before restricted,
weights-only deserialization.
Inference defaults to `float16`; pass `full_precision=True` for `float32`
model inference.
`model_dir` is intended for controlled deployment mirrors and must contain
`encoder.pt`, `model.json`, and `alignment.json`.

## `Ginfinity.encode`

```python
encode(
    rna: RNA,
    *,
    keep_paired_neighbours=False,
    context_hops=1,
) -> numpy.ndarray
```

Returns an `(L, 128)` C-contiguous `float16` matrix with unit-length rows by
default. `embedding_dtype` accepts a NumPy dtype or dtype name (`float16`,
`float32`, or `float64`).
For a sliced record `L` is the core window length, not the source molecule.
`keep_paired_neighbours` and `context_hops` have the same meaning as on
`encode_many`.

## `Ginfinity.encode_many`

```python
encode_many(
    records: Sequence[RNA],
    *,
    max_batch_nodes=60000,
    max_batch_edges=300000,
    keep_paired_neighbours=False,
    context_hops=1,
)
    -> list[numpy.ndarray]
```

Preserves input order and rejects duplicate identifiers. Batches are bounded by
total nucleotide and edge counts rather than record count.

`keep_paired_neighbours` retains nucleotides outside a window when they are
base-paired with a core nucleotide. `context_hops` is the depth of that
neighbourhood (hop 1 is the partner). Context nodes participate in GINE
message passing and are discarded; each returned array contains only core
rows. See [sliced graphs](SLICED_GRAPHS.md).

## Graph construction and persistence

`GraphSpec` defines the model-compatible node and edge representation. Its
SHA-256 is calculated from this small release-time contract; it is not a
content hash calculated separately for each graph.

```python
builder = GraphBuilder()             # uses the bundled GraphSpec
builder = GraphBuilder(
    keep_paired_neighbours=True, context_hops=3)
graph = builder.build(rna)           # one Graph
graphs = builder.build_many(records) # list[Graph]
shard = builder.build_shard(records) # one GraphShard
```

A sliced `Graph` keeps the full source `sequence`/`structure`. Selected
nodes are described by `residue_index` (0-based source coordinates) and
`node_roles` (`NODE_ROLE_CORE` or `NODE_ROLE_CONTEXT`). The role array is
not part of `node_features`.

`GraphShard` stores concatenated compact arrays plus per-record offsets and
metadata. Persist it without Python-object serialization:

```python
save_graph_shard(shard, "part-00000.safetensors", checksum=False)
shard = load_graph_shard(
    "part-00000.safetensors",
    expected_spec=encoder.graph_spec,
    verify_checksum=False,
    validation="metadata",
)
```

`checksum=True` adds one whole-shard SHA-256. It is optional. Set
`validation="full"` to add a linear scan for finite node values and edges
crossing graph boundaries.

Partition an input stream before dispatching independent CPU jobs:

```python
parts = partition_records(records, max_records=128, max_nodes=60_000)
```

The function yields tuples in input order and closes a part when either limit
would be exceeded. It does not construct graphs itself.

## `Ginfinity.encode_graph` and `Ginfinity.encode_graphs`

```python
encode_graph(graph: Graph) -> numpy.ndarray

encode_graphs(
    graphs: Sequence[Graph] | GraphShard,
    *,
    max_batch_nodes=60000,
    max_batch_edges=300000,
) -> list[numpy.ndarray]
```

The encoder rejects graphs whose `GraphSpec` differs from `encoder.graph_spec`.
Different microbatch boundaries can change the last few floating-point bits;
applications should compare outputs numerically, not byte-for-byte, across
different batching layouts.

## `Ginfinity.info`

Returns a copy of the verified model metadata dictionary.

## `default_alignment_parameters`

Returns a dictionary accepted by
`ginfinity_sw.ScoringParameters(**parameters)`. The values are tied to the
bundled model version; do not mix them with another checkpoint.

## Exceptions

- `InputValidationError`: invalid sequence, structure, or identifier.
- `GraphValidationError`: malformed graph or shard artifact.
- `GraphCompatibilityError`: graph specification does not match the model.
- `ModelIntegrityError`: missing artifact, unexpected hash, incompatible model
  format, or metadata/checkpoint mismatch.
- `ValueError`: invalid device policy or duplicate batch identifier.
