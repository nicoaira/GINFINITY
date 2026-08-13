# API reference

## `RNA`

`RNA(identifier, sequence, structure)` is an immutable validated input record.
Normalization and structural validation happen before graph construction.

`RNA.from_mapping(row, identifier_column=..., sequence_column=...,
structure_column=...)` creates the same record from a mapping whose field names
are selected by the caller.

## `read_rna_table`

```python
read_rna_table(
    path,
    *,
    identifier_column="transcript_id",
    sequence_column="sequence",
    structure_column="secondary_structure",
    delimiter="\t",
) -> list[RNA]
```

Reads records in file order. Required columns may appear in any order and the
table may contain additional columns. The delimiter must be one character.
Every row passes through the normal `RNA` validation contract.

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
`model_dir` is intended for controlled deployment mirrors and must contain
`encoder.pt`, `model.json`, and `alignment.json`.

## `Ginfinity.encode`

```python
encode(rna: RNA) -> numpy.ndarray
```

Returns an `(L, 128)` C-contiguous `float32` matrix with unit-length rows.

## `Ginfinity.encode_many`

```python
encode_many(
    records: Sequence[RNA],
    *,
    max_batch_nodes=60000,
    max_batch_edges=300000,
)
    -> list[numpy.ndarray]
```

Preserves input order and rejects duplicate identifiers. Batches are bounded by
total nucleotide and edge counts rather than record count.

## Graph construction and persistence

`GraphSpec` defines the model-compatible node and edge representation. Its
SHA-256 is calculated from this small release-time contract; it is not a
content hash calculated separately for each graph.

```python
builder = GraphBuilder()             # uses the bundled GraphSpec
graph = builder.build(rna)           # one Graph
graphs = builder.build_many(records) # list[Graph]
shard = builder.build_shard(records) # one GraphShard
```

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
