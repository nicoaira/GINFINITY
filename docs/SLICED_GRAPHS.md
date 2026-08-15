# Sliced graphs

A sliced graph is the GINFINITY graph of one window on a longer RNA, plus
optional structural context around base pairs that cross the window boundary.
The GINE still runs on every selected node. After message passing, only the
nucleotides inside the window are returned.

Use this when you want an embedding for a motif, domain, or other interval
without discarding the pairing information that sits just outside that
interval.

## Coordinates

`start` and `end` are **0-based half-open** coordinates, the same convention
as a Python slice:

```text
sequence[start:end]     # core nucleotides
```

The first nucleotide is position `0`. Both bounds are required. The window
must be non-empty and must lie inside the sequence.

## Table input

`start` and `end` are optional columns. A table without them is encoded as
full molecules, as before.

One window:

```text
transcript_id	sequence	secondary_structure	start	end
AANU01100842.1/307-440	CUUAUGAAGUCUUCCUUUCAGUUCAGAAGAAAUGGAAUUCGCUCUCCAACUUCAGGAAACUGAAAUAAAGAGUUGCUUGGAUUUAGUGUUCACCUUUACCAUAAAAUGGAUUUGCUAACACUGCCACCCUGCUUUGAUAGCGAAUAAAGCAAAAAGGGCUUCUGUCGUGAGUGGCACACGUAGGGCAACUCGAUUGCUCUUCGUGCGGAAUCGACAUCAAGAGAUUUCGGAAGCAUAAUUUUUUGACAUUCGGGCAGCUGGUGAUCGUUGGUCCCGGCGCCCUUCUUUUUUUCUGUCUCAAGUCAGAUGAAUUUUUCUGGUGAGUUAGGUGUUAGUUUUGUAAGUGGAUGUAAGAUUUAUGUUAAUCCUUUUUAUUUGAAGUUGCGUAGCUAUCUGCGUGAACCGCAGAUGACUAAAUUAGCAGGGUAUUUAAC	......((.((((..(((((((((.((((...((((........)))).))))..)).)))))))..)))).))....((...(((((((........((((...)))).......)))))))))((((((((......((((...(((.((((((((((((((((......))))((((.(((((((.....))))))).))))(((((((..........))))))))))))((((.((((..((((((((((.(((((.((((...))))))))))))).........((((.((((..(((((........))))))))).))))................)))))))))).))))......))))))).)))....)))).(((.((((((((.....)))))))).)))....)))))))).......	50	113
```

The core of that row is `sequence[50:113]` (63 nucleotides). In the
dot-bracket string the window sits between the two `!` marks:

```text
......((.((((..(((((((((.((((...((((........)))).)!)))..)).)))))))..)))).))....((...(((((((........((((...))))....!...)))))))))((((((((......
```

Several windows on the same molecule are written as parallel
comma-separated lists. Whitespace around the commas is ignored. Each window
becomes its own graph and its own embedding, with identifier
`{transcript_id}:{start}-{end}`:

```text
transcript_id	sequence	secondary_structure	start	end
AANU01100842.1/307-440	…	…	50,70	113,120
```

This row yields two graphs: `[50, 113)` and `[70, 120)`.

Column names are configurable (`--start-column`, `--end-column`, or the
same arguments on `read_rna_table`). Use `--no-slices` if the file has
`start`/`end` columns that are not windows.

## Crossing-pair context

By default a slice keeps only the core window. Backbone neighbours just
outside the cut are **not** kept.

`keep_paired_neighbours=True` changes that for **base pairs only**. If core
nucleotide A is paired with B, and B lies outside `[start, end)`, B is added
as context.

`context_hops` is the depth of that neighbourhood:

| `context_hops` | Nodes kept beyond the core |
|---|---|
| `1` (default) | only B, the crossing-pair partner |
| `2` | B and every graph neighbour of B |
| `3` | those nodes and *their* graph neighbours |

Hop 1 uses pairing edges only. Further hops follow every edge the model
graph actually has: backbone, pairing, and skip-2.

Example: A (core) is paired with B (outside). B is covalently adjacent to C.
C is covalently adjacent to D and base-paired with E.

```text
keep_paired_neighbours=True, context_hops=3

core     A
hop 1    B
hop 2    C
hop 3    D and E
```

## Node roles

Every graph node is tagged as `core` or `context`:

- `NODE_ROLE_CORE` (`0`) — inside `[start, end)`
- `NODE_ROLE_CONTEXT` (`1`) — a retained neighbour

This tag is **not** a GINE feature. It is stored beside the graph
(`Graph.node_roles`, shard tensor `node_roles`) so the encoder can drop
context after message passing. Node features remain the 7-column
full-molecule encoding: one-hot base, paired/unpaired, sine/cosine
relative position on the **source** molecule.

`Graph.residue_index` maps each selected node back to its 0-based position
in that source molecule. Nodes are stored in 5′→3′ order.

Returned embeddings contain **only core rows**, still in 5′→3′ order, with
shape `(end - start, 128)`.

## Python

```python
from ginfinity import Ginfinity, GraphBuilder, RNA, read_rna_table

encoder = Ginfinity.load()

# One window on a Python record. The identifier is left unchanged.
rna = RNA(
    "example",
    "GGGAAACCCUUUUGGG",
    "......(((....)))",
    start=9,
    end=16,
)
core_only = encoder.encode(rna)
core_with_context = encoder.encode(
    rna, keep_paired_neighbours=True, context_hops=3)
print(core_only.shape)          # (7, 128)
print(core_with_context.shape)  # (7, 128)

# Inspect the graph that the encoder would run.
graph = GraphBuilder(
    keep_paired_neighbours=True, context_hops=3).build(rna)
print(graph.core_span)       # (9, 16)
print(graph.core_count)      # 7
print(graph.node_count)      # 7 core + retained neighbours
print(graph.residue_index)   # source coordinates of every selected node
print(graph.node_roles)      # 0 = core, 1 = context

# A table row with several windows expands to several records.
records = read_rna_table("structures.tsv")
outputs = encoder.encode_many(
    records, keep_paired_neighbours=True, context_hops=3)
```

`RNA.from_mapping(..., start_column="start", end_column="end")` accepts a
single window. Several windows on one mapping use
`RNA.many_from_mapping(...)`.

## Command line

```bash
ginfinity embed \
  --input structures.tsv \
  --output embeddings.npz \
  --keep-paired-neighbours \
  --context-hops 3
```

`--context-hops` implies `--keep-paired-neighbours`. Hop `1` keeps only the
crossing-pair partners.

The staged pipeline is the same. Context is baked into the shard; the
encoder reads `node_roles` and writes only core embeddings.

```bash
ginfinity build-graphs \
  --input structures.tsv \
  --output graphs-00000.safetensors \
  --keep-paired-neighbours \
  --context-hops 3

ginfinity embed-graphs \
  --input graphs-00000.safetensors \
  --output embeddings-00000.npz
```

NPZ keys for table-derived slices are `{id}:{start}-{end}`. The manifest
records `start`, `end`, and `core_length` for each window.

## Worked example

```text
sequence   GGGAAACCCUUUUGGG
structure  ......(((....)))
positions  0123456789012345
window                 [9, 16)
```

The core is `UUUUGGG` (positions 9–15). Three pairs cross the left cut:
`6–15`, `7–14`, `8–13`.

- no extra flags → nodes `{9…15}`
- `keep_paired_neighbours=True` → also `{6, 7, 8}`
- `context_hops=2` → also `{4, 5}` (backbone / skip-2 of 6–8)
- `context_hops=3` → also `{2, 3}`

The GINE runs on the selected set. The embedding has 7 rows, one per core
nucleotide.

## Shard tensors

Sliced shards add two optional arrays next to the existing ones:

| tensor | dtype | shape | role |
|---|---|---|---|
| `residue_index` | `int32` | `(total_nodes,)` | source coordinate of each node |
| `node_roles` | `uint8` | `(total_nodes,)` | `0` core, `1` context |

Shards written before this feature omit both arrays. The loader treats every
node as core. `node_ptr` counts selected nodes, which can be smaller than
the source sequence length.
