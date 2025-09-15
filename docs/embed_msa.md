# Embedding-based RNA MSA (`embed_msa.py`)

This document explains how the embedding-based MSA pipeline works internally and how to use it in practice. The script aligns RNAs using per-position node embeddings with a T‑Coffee/ProbCons-style approach. It is sparse and scalable, uses affine-gap dynamic programming (DP), and supports optional structure-aware scoring from dot-bracket or explicit pairing indices.

File: `src/ginfinity/scripts/embed_msa.py`

## Overview

Given a TSV with a column of node embeddings per RNA (shape L×D per sequence), the tool:

1. Loads and L2-normalizes per-position embeddings.
2. Computes pairwise cosine similarities and calibrates them into match log-odds.
3. Runs a soft pair-HMM (Forward–Backward in log space) with affine gaps to obtain match posterior probabilities P(i~j).
4. Keeps a sparse alignment library via top‑K pruning and a minimum probability threshold.
5. Optionally applies 1–2 T‑Coffee-style consistency rounds across sequences.
6. Builds a guide tree (Neighbor Joining or UPGMA) from pairwise evidence.
7. Progressively merges profiles using a profile–profile DP, scoring columns by mean embedding dot product and a simple structure-compatibility bonus. Gap opens can be modulated inside stem-like columns.
8. (Optional) Performs iterative refinement (hook available; conservative by default).
9. Writes MSA in FASTA and Stockholm, a TSV mapping names to aligned strings, and diagnostics (pairwise expected scores, optional posterior plots, metadata).

The design keeps data sparse (top‑K pruning, coordinate lists) and uses float32 throughout. Numba accelerates DP kernels when available.

## Inputs

- Required TSV columns:
  - `Name`: sequence identifier.
  - `node_embeddings`: JSON-encoded list of lists with shape L×D (outer list length = sequence length; each inner list length = embedding dimension). Example cell:
    ```json
    [[0.12, -0.44, ...], [0.09, 0.31, ...], ...]
    ```
- Optional TSV columns (structural priors):
  - `DotBracket`: a dot-bracket string of length L.
  - `PairedIndices`: JSON-encoded list of length L; each element is the partner index or -1. If present, it is preferred over DotBracket.

Robust parsing: invalid rows are skipped with warnings. All sequences must share the same embedding dimension D.

## Outputs

- `out_prefix.fasta`: FASTA alignment. Residue symbols are dot-bracket characters (`().[]{}.`) when structural info is provided, otherwise `X`. Gaps are `-`.
- `out_prefix.sto`: Stockholm alignment with the same symbols.
- `out_prefix.aln.tsv`: two-column TSV mapping `Name` → aligned string.
- `out_prefix.diagnostics/`:
  - `expected_scores.tsv`: square matrix (N×N) of expected alignment scores (Σ P(i~j)·S(i,j)).
  - Optional `pair_A_B.png` heatmaps of sparse posteriors for a few pairs (if `--plot-diagnostics` and matplotlib available).
  - `run_meta.json`: parameters, counts, and timing info.

Note: actual nucleotides are not used; alignment is over embedding positions. If `DotBracket` or `PairedIndices` is present, the aligned strings contain the corresponding dot-bracket characters per sequence; otherwise they contain `X` for residues and `-` for gaps.

## CLI Usage

Basic example:

```
python src/ginfinity/scripts/embed_msa.py \
  --input premirnas_node_embeds.tsv \
  --name-col Name \
  --embeds-col node_embeddings \
  --dotbracket-col DotBracket \
  --paired-col PairedIndices \
  --out-prefix out/premirnas \
  --topk 20 \
  --consistency-rounds 1 \
  --alpha 6.0 --beta 0.0 \
  --gap-open -10.0 --gap-extend -0.5 \
  --stem-gap-bonus -2.0 \
  --use-local \
  --tree nj \
  --refine-iters 2 \
  --num-workers 8 \
  --max-pairs 2000 \
  --seed 42 \
  --plot-diagnostics
```

Arguments:

- `--input`: TSV file path or `dummy` to generate toy data.
- `--name-col`, `--embeds-col`: column names (defaults: `Name`, `node_embeddings`).
- `--dotbracket-col`, `--paired-col`: optional columns for structural priors.
- `--out-prefix`: prefix for outputs.
- `--topk`: top-K pruning per row and per column in posterior matrices (default 20).
- `--consistency-rounds`: number of T‑Coffee consistency rounds (default 1; set 0 to disable).
- `--alpha`, `--beta`: calibration parameters to map cosine similarity s∈[-1,1] via p=σ(α·s+β), log-odds = logit(p). If omitted, defaults to α=5.0, β=0.0 with a warning.
- `--gap-open`, `--gap-extend`: affine gap costs (negative scores).
- `--stem-gap-bonus`: additional penalty added to gap-open inside majority-stem columns (e.g., -2.0 makes opening gaps in stems more expensive).
- `--use-local`: use local scoring in pairwise soft-DP (Smith–Waterman style) for posterior estimation.
- `--tree`: guide tree method (`nj` or `upgma`).
- `--refine-iters`: number of iterative refinement passes (lightweight placeholder; default 0).
- `--num-workers`: threads for pairwise computation.
- `--max-pairs`: cap total number of pairwise computations. If exceeded, uses embedding-mean kNN selection.
- `--seed`: RNG seed for reproducibility.
- `--plot-diagnostics`: if set, writes posterior heatmaps for a few pairs (requires matplotlib).

Dummy mode for quick testing:
```
python src/ginfinity/scripts/embed_msa.py --input dummy --out-prefix /tmp/embed_msa_dummy
```

## Algorithmic Details

### 1) Loading and Normalization

- TSV is read with pandas; JSON fields are parsed robustly (`node_embeddings` must be L×D). Empty or malformed rows are skipped.
- Each position’s embedding is L2-normalized so cosine similarity reduces to a dot product.

### 2) Pairwise Posteriors via Soft DP

- Pair selection: if the number of all pairs exceeds `--max-pairs`, choose kNN per sequence using mean embedding cosine similarity.
- Similarity matrix: `S = Ea @ Eb^T` where Ea, Eb are L2-normalized (La×D, Lb×D).
- Calibration: `p = σ(α·S + β)`; `ℓ = logit(p)` as match log-odds. Defaults α=5.0, β=0.0 if not given.
- Pair-HMM: three states (Match M, Insertion X, Deletion Y) with affine gaps (`gap_open`, `gap_extend`), computed in log-space.
  - Forward DP computes total log-probabilities.
  - Backward DP is used to obtain posteriors; we use `P(i~j) ∝ exp(M_f[i+1,j+1] + M_b[i+1,j+1] − Z)`. Local mode allows restarts (approximation in backward).
- Sparsification: keep top‑K in each row and each column, intersected, and drop entries with `P < 1e-4`.
- Expected alignment score: `E = Σ P(i~j)·S(i,j)`; stored for diagnostics.

Note: Optional banding is not implemented yet. The library remains sparse via top‑K pruning.

### 3) T‑Coffee Consistency Transform

- Initial library `LAB` contains sparse P_AB.
- One or more rounds update each pair A,B via:
  - `P̃_AB(i,j) = (1−λ)·P_AB(i,j) + λ·(1/|C|)·Σ_k P_AC(i,k)·P_CB(k,j)` with λ=0.5.
- After each round, re-prune to top‑K and drop tiny weights.

### 4) Guide Tree

- Distance `D[a,b] = 1 − mean(P_AB)` for the kept entries; defaults to 1 for empty.
- NJ or UPGMA implementations produce a binary merge structure (sufficient for progressive guidance; branch lengths are not used).

### 5) Progressive Profile–Profile Alignment

- Each profile column stores:
  - `μ_C`: normalized mean of member embeddings (ignoring gaps), shape (D,).
  - `stem_fraction ∈ [0,1]`: fraction of members that are paired if pairing info exists; otherwise 0.
- Column–column score:
  - `S_col(C,D) = μ_C·μ_D + β_struct·compat(C,D)` with `β_struct = 0.2` and `compat=1` if both majority-stem or both majority-loop, else 0.
- Affine gaps with context:
  - Gap open in a majority-stem column adds `stem_gap_bonus` to `gap_open` (i.e., more negative discourages gaps in stems).
- Profiles are merged following the guide tree.

### 6) Iterative Refinement

- Hook provided; current implementation is conservative (no structural change). The scaffolding is present to add tree-split realignment with SP-score acceptance.

### 7) Outputs and Diagnostics

- MSA in FASTA and Stockholm, per-name TSV.
- Diagnostics include expected scores, optional posterior heatmaps, and a JSON of run parameters and timing.

## Performance & Scalability

- Uses float32 and numba-accelerated DP kernels when numba is available.
- Pairwise stage is multithreaded (`--num-workers`).
- Sparse library uses coordinate lists (i,j,p) with `int32` and `float32` to minimize memory.
- For large N, limit pairwise computations with `--max-pairs`. kNN selection uses mean-pooled embedding vectors.

## Error Handling & Edge Cases

- Invalid or empty embeddings: row skipped with a warning.
- Mismatched embedding dimensions: run exits with an error.
- Very short sequences (<5) are supported; consider setting `--consistency-rounds 0` if poor posteriors are observed.
- Missing structure columns: structural bonuses default to 0.

## Reproducibility

- Set `--seed` to fix RNG state for dummy data and any randomized steps.

## Extending the Pipeline

- Add diagonal banding to pairwise DP for long sequences.
- Improve backward local-mode normalization for sharper posteriors.
- Implement real iterative refinement by splitting the guide tree and realigning sub-MSAs.
 

## Troubleshooting

- “No valid records found”: Ensure TSV has the required columns and JSON parses to a 2D list with consistent inner lengths.
- Poor or empty alignments:
  - Try increasing `--topk` or `--alpha` (sharper posteriors).
  - Disable consistency (`--consistency-rounds 0`) for tiny datasets.
  - Use `--use-local` if sequences share only subsections.
- Slow pairwise stage:
  - Increase `--num-workers` and lower `--max-pairs`.
  - Reduce embedding dimension D if possible.
- Dissimilar sequences all appear distant:
  - Adjust calibration `--alpha`/`--beta` or verify embeddings are normalized (script does this automatically).

## Examples

Dummy data run:
```
python src/ginfinity/scripts/embed_msa.py --input dummy --out-prefix /tmp/embed_msa_dummy --plot-diagnostics
```

Real data run (with structure):
```
python src/ginfinity/scripts/embed_msa.py \
  --input premirnas_node_embeds.tsv \
  --name-col Name --embeds-col node_embeddings \
  --dotbracket-col DotBracket --paired-col PairedIndices \
  --out-prefix out/premirnas --topk 30 --consistency-rounds 1 \
  --alpha 6.0 --beta 0.0 --gap-open -10 --gap-extend -0.5 \
  --stem-gap-bonus -2.0 --tree nj --num-workers 8 --plot-diagnostics
```
