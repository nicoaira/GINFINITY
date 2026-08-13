# Changelog

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
