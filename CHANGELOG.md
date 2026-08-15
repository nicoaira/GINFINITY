# Changelog

## 1.1.0 - 2026-08-15

- Added sliced graphs: optional `start`/`end` windows on an RNA, including
  several comma-separated windows per table row.
- Added `keep_paired_neighbours` and `context_hops` so crossing-pair
  partners and a bounded neighbourhood can take part in GINE message
  passing. Returned embeddings keep only the core window.
- Stored `residue_index` and `node_roles` on graphs and on shards that
  actually contain a window or context. Role metadata is not a GINE
  feature. Full-molecule shards stay readable by 1.0.x; older shards
  without these tensors still load as all-core graphs.

## 1.0.1 - 2026-08-13

- Republished the 1.0.0 encoder and APIs from one git tag so PyPI and the
  personal Anaconda channel ship the same source tree.
- Publish conda packages from the GitHub release tag.

## 1.0.0 - 2026-08-12

- First production release.
- Added standalone PyPI and Anaconda distribution metadata and publishing
  workflows.
- Dual licensing: PolyForm Noncommercial 1.0.0 for code and CC BY-NC 4.0 for
  trained weights, checkpoints, and embeddings.
- Bundled SHA-256-verified all-data GINE checkpoint.
- Restricted weights-only checkpoint loading.
- Public versioned graph construction and Safetensors shard interchange API.
- Prebuilt-graph encoding with independent node and edge microbatch limits.
- Stateless `build-graphs` and `embed-graphs` pipeline commands.
- Configurable identifier, sequence, and structure columns for table input.
- Per-nucleotide 128-dimensional embedding API and batch API.
- Strict RNA and dot-bracket validation.
- Deterministic CPU default and explicit CUDA policy.
- TSV-to-NPZ command-line interface with integrity manifest.
- Model-specific alignment parameters exposed for `ginfinity-sw`.
