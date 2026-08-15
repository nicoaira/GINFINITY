# GINFINITY 🚀 : Graph-based RNA Structure Embedding Generator

GINFINITY generates per-nucleotide embeddings from RNA sequences and
dot-bracket secondary structures with a Graph Isomorphism Network. The
embeddings can be used for alignment, clustering, classification, and other
downstream analyses.

## Install

From PyPI:

```bash
python -m pip install ginfinity
```

From the tagged GitHub release:

```bash
python -m pip install \
  "git+https://github.com/nicoaira/GINFINITY.git@v1.0.1"
```

Using conda:

```bash
conda install -c nicolas.aira -c conda-forge ginfinity
```

For a local checkout:

```bash
python -m pip install .
```

## Python API

```python
from ginfinity import Ginfinity, RNA

encoder = Ginfinity.load()
rna = RNA(
    identifier="example",
    sequence="ACGUACGU",
    structure="((....))",
)

embedding = encoder.encode(rna)
print(embedding.shape)  # (8, 128)
```

Embeddings are returned as L2-normalized `float32` NumPy arrays in 5′→3′
sequence order. Reuse a loaded `Ginfinity` instance across requests.

Batch inference amortizes graph construction and model overhead:

```python
records = [
    RNA("rna-1", "ACGUACGU", "((....))"),
    RNA("rna-2", "AGAGAAUCCCU", "((.(....)))"),
]
outputs = encoder.encode_many(records)
```

### Separate graph construction from embedding

For distributed CPU/GPU pipelines, build a persistent graph shard without
loading the model checkpoint:

```python
from ginfinity import GraphBuilder, RNA, partition_records, save_graph_shard

records = [
    RNA("rna-1", "ACGUACGU", "((....))"),
    RNA("rna-2", "AGAGAAUCCCU", "((.(....)))"),
]
builder = GraphBuilder()
for index, part in enumerate(partition_records(
        records, max_records=128, max_nodes=60_000)):
    shard = builder.build_shard(part)
    save_graph_shard(shard, f"graphs-{index:05d}.safetensors")
```

Load and encode that shard on another machine:

```python
from ginfinity import Ginfinity, load_graph_shard

encoder = Ginfinity.load(
    device="cuda", allow_nondeterministic_cuda=True)
shard = load_graph_shard(
    "graphs-00000.safetensors", expected_spec=encoder.graph_spec)
embeddings = encoder.encode_graphs(
    shard, max_batch_nodes=60_000, max_batch_edges=300_000)
```

Graph shards are scheduling and storage units. The encoder may divide one
shard into smaller in-memory microbatches according to node and edge limits.
The graph specification fingerprint is one model-versioned compatibility
value shared by every graph; GINFINITY does not hash every graph. Full shard
content checksums are available but opt-in.

## Command line

Create `structures.tsv` with the RNA identifier, sequence, and dot-bracket
secondary structure. The two complete records below are
`AANU01100842.1/307-440` and `AAZO01007178.1/21512-21382` from RF00548.
Additional columns are allowed.

```text
transcript_id	sequence	secondary_structure
AANU01100842.1/307-440	CUUAUGAAGUCUUCCUUUCAGUUCAGAAGAAAUGGAAUUCGCUCUCCAACUUCAGGAAACUGAAAUAAAGAGUUGCUUGGAUUUAGUGUUCACCUUUACCAUAAAAUGGAUUUGCUAACACUGCCACCCUGCUUUGAUAGCGAAUAAAGCAAAAAGGGCUUCUGUCGUGAGUGGCACACGUAGGGCAACUCGAUUGCUCUUCGUGCGGAAUCGACAUCAAGAGAUUUCGGAAGCAUAAUUUUUUGACAUUCGGGCAGCUGGUGAUCGUUGGUCCCGGCGCCCUUCUUUUUUUCUGUCUCAAGUCAGAUGAAUUUUUCUGGUGAGUUAGGUGUUAGUUUUGUAAGUGGAUGUAAGAUUUAUGUUAAUCCUUUUUAUUUGAAGUUGCGUAGCUAUCUGCGUGAACCGCAGAUGACUAAAUUAGCAGGGUAUUUAAC	......((.((((..(((((((((.((((...((((........)))).))))..)).)))))))..)))).))....((...(((((((........((((...)))).......)))))))))((((((((......((((...(((.((((((((((((((((......))))((((.(((((((.....))))))).))))(((((((..........))))))))))))((((.((((..((((((((((.(((((.((((...))))))))))))).........((((.((((..(((((........))))))))).))))................)))))))))).))))......))))))).)))....)))).(((.((((((((.....)))))))).)))....)))))))).......
AAZO01007178.1/21512-21382	AUUCGGAAAAAAAUUUCAACGGAUAUAAAAUACGUUAAUUCAAAUCAUUUUAAACUUUAUCCGUUUUGAAAUAUUAUUUAGAGAUUUUAACCGAAGGAUUUAACGUUUAUGUAAAUUGUUAAAUGAAAGAAUAACCGUUUGAACUUUUAAUAAAAGGGUGCUUGCUGCUUAGAUAGCACAUAGGUCCAAACGGGCCUGUGGGGUAUGGUUAUCAAGACAUAUCCAGCACGUAAUUUUUGUACCUGUAGGGCUGUUGACUGCCAAUAUGCAUCGACGCCCUCGUAAUCGACCUCAAUGAUAAUAAUCUCUUAUACCUUUCGCACUUUCCACGUGUACCUGGCAAUUACAAUUUAAUUAUCGGGUGAGUAGGUUCUUUUUUUGGCAAUCGGGUUGCUAAUUAUUAUCCCCCCUGGAAGGUGUUAACUUCAUCC	.(((((...((((((((((((((((((((((..(((......))).)))))).....))))))))...............))))))))..)))))..((((((((.((.....)).))))))))(((((..............(((((....)))))((((((((((......)))))(((((((((....)))))))))(((((((..........))))))))))))..................((((.(((((.(((......))))))))))))(((....))).......(((....)))........)))))((((((((((....(((((((.(((((.....))))).)))))))....((........((((((((...)))))))).......))....))))))))))...........
```

Generate embeddings and an integrity manifest:

```bash
ginfinity embed \
  --input structures.tsv \
  --output embeddings.npz \
  --manifest embeddings.manifest.json
```

Export the model-specific parameters for the separate aligner CLI:

```bash
ginfinity alignment-config --output alignment.json
```

Inspect the bundled model metadata without loading input data:

```bash
ginfinity info
```

The same stages can be executed as independent jobs:

```bash
ginfinity build-graphs \
  --input structures.tsv \
  --output graphs-00000.safetensors

ginfinity embed-graphs \
  --input graphs-00000.safetensors \
  --output embeddings-00000.npz \
  --device cuda \
  --allow-nondeterministic-cuda
```

Use `--checksum` while building and `--verify-checksum` while loading only
when end-to-end content hashing is required. Normal loading performs format,
graph-specification, dtype, shape, offset, and bounds validation without a
second full-file checksum pass.

### Custom column names

Column names and order are configurable in both Python and the CLI. For a CSV
with columns `name,bases,dot_bracket,source`:

```python
from ginfinity import read_rna_table

records = read_rna_table(
    "structures.csv",
    identifier_column="name",
    sequence_column="bases",
    structure_column="dot_bracket",
    delimiter=",",
)
```

```bash
ginfinity embed \
  --input structures.csv \
  --output embeddings.npz \
  --id-column name \
  --sequence-column bases \
  --structure-column dot_bracket \
  --delimiter ,
```

`RNA.from_mapping()` accepts the same three column-name arguments when records
already exist as Python mappings. The column flags are also available on
`ginfinity build-graphs`.

## Sliced graphs

Optional `start` and `end` columns select a window on a longer molecule.
Coordinates are 0-based and half-open, like `sequence[start:end]`. One row
may list several windows as parallel comma-separated lists; each window
becomes its own graph.

```text
transcript_id	sequence	secondary_structure	start	end
example	GGGAAACCCUUUUGGG	......(((....)))	9	16
example	GGGAAACCCUUUUGGG	......(((....)))	9,6	16,12
```

By default only the window is kept. `--keep-paired-neighbours` also keeps a
nucleotide outside the window when it is base-paired with one inside it.
`--context-hops N` expands that neighbourhood: hop 1 is the crossing-pair
partner, further hops follow backbone, pairing, and skip-2 edges.

```python
from ginfinity import Ginfinity, RNA

encoder = Ginfinity.load()
rna = RNA(
    "example",
    "GGGAAACCCUUUUGGG",
    "......(((....)))",
    start=9,
    end=16,
)
embedding = encoder.encode(
    rna, keep_paired_neighbours=True, context_hops=3)
print(embedding.shape)  # (7, 128) — core nucleotides only
```

```bash
ginfinity embed \
  --input structures.tsv \
  --output embeddings.npz \
  --keep-paired-neighbours \
  --context-hops 3
```

Context nucleotides take part in GINE message passing and are discarded
afterwards. They are never written into the embedding matrix. See
[Sliced graphs](https://github.com/nicoaira/GINFINITY/blob/main/docs/SLICED_GRAPHS.md)
for the coordinate convention, node-role metadata, and a worked example.

## Input contract

- Sequence length: 1–4,096 nucleotides.
- Accepted bases: `A`, `C`, `G`, `U`; `T` is normalized to `U`.
- Structure: equally long, balanced, properly nested dot-bracket using `.()`.
- Secondary structure is supplied by the caller and is used unchanged.
- Empty and duplicate IDs, ambiguity codes, malformed brackets, pseudoknots,
  and length mismatches are rejected before model execution.

The package does not predict secondary structure. Embedding quality depends on
the quality and biological relevance of the structure supplied by the caller.

## Device policy

CPU is the default and deterministic deployment path:

```python
encoder = Ginfinity.load(device="cpu")
```

CUDA is available for throughput-sensitive workloads:

```python
encoder = Ginfinity.load(device="cuda", allow_nondeterministic_cuda=True)
```

CUDA floating-point output is not promised to be bit-identical across drivers
or hardware. Use CPU when stable byte-for-byte output is required.

## Documentation

- [API reference](https://github.com/nicoaira/GINFINITY/blob/main/docs/API.md)
- [Sliced graphs](https://github.com/nicoaira/GINFINITY/blob/main/docs/SLICED_GRAPHS.md)
- [Distributed graph pipeline](https://github.com/nicoaira/GINFINITY/blob/main/docs/GRAPH_PIPELINE.md)
- [Operations guide](https://github.com/nicoaira/GINFINITY/blob/main/docs/OPERATIONS.md)
- [Publishing guide](https://github.com/nicoaira/GINFINITY/blob/main/docs/PUBLISHING.md)
- [Changelog](https://github.com/nicoaira/GINFINITY/blob/main/CHANGELOG.md)

## Versioning

GINFINITY follows semantic versioning. Model weights, input semantics, and
default alignment parameters are versioned together. Pin a Git tag or commit
in automated deployments.

## Licensing & Commercial Use

This repository uses a dual-licensing structure:

- **Code & Scripts:** Licensed under
  [PolyForm Noncommercial 1.0.0](https://github.com/nicoaira/GINFINITY/blob/main/LICENSE).
- **Model Weights, Checkpoints & Embeddings:** Licensed under
  [CC BY-NC 4.0](https://github.com/nicoaira/GINFINITY/blob/main/LICENSE-WEIGHTS).

### 🔒 Non-Commercial & Academic Use

You are free to use, modify, and study this software and model for
non-commercial, personal, academic, or educational purposes, subject to the
applicable license terms.

### 🏢 Commercial Licensing

Any commercial use—including using GINFINITY to deliver commercial CRO
services, operate internal enterprise pipelines, or provide hosted APIs—requires
a separate commercial license from the copyright holder.
