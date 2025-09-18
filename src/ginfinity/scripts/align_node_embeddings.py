#!/usr/bin/env python3

import argparse
import json
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def read_table_auto(path: str) -> pd.DataFrame:
    """Read TSV/CSV with simple separator heuristics."""
    if path.endswith(".tsv"):
        return pd.read_csv(path, sep="\t", low_memory=False)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_csv(path, sep=None, engine="python")


def parse_embeddings(cell: str) -> np.ndarray:
    """
    Parse the JSON-encoded LxD matrix into a numpy array (float32).
    """
    try:
        arr = json.loads(cell)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse node_embeddings JSON: {e}") from e
    mat = np.asarray(arr, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError("node_embeddings must be a 2D array [L x D].")
    return mat


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute pairwise cosine similarity between rows of A (L1 x D) and B (L2 x D).
    Returns an (L1 x L2) matrix with values in [-1, 1].
    """
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"Embedding dims mismatch: {A.shape[1]} vs {B.shape[1]}")
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)
    return A_norm @ B_norm.T


def needleman_wunsch_affine(score: np.ndarray, gap_open: float, gap_extend: float) -> Tuple[float, List[Tuple[Optional[int], Optional[int]]]]:
    """
    Global alignment (Needleman–Wunsch) with affine gaps (Gotoh).

    score: (L1 x L2) similarity matrix; gap penalties should be negative.
    Returns best score and alignment path as (i,j) pairs with None for gaps.
    """
    L1, L2 = score.shape
    neg_inf = np.float32(-1e9)
    H = np.full((L1 + 1, L2 + 1), neg_inf, dtype=np.float32)  # overall best
    E = np.full((L1 + 1, L2 + 1), neg_inf, dtype=np.float32)  # gap in B (up)
    F = np.full((L1 + 1, L2 + 1), neg_inf, dtype=np.float32)  # gap in A (left)
    # tracebacks
    TH = np.zeros((L1 + 1, L2 + 1), dtype=np.uint8)  # 0=diag,1=E,2=F
    TE = np.zeros((L1 + 1, L2 + 1), dtype=np.uint8)  # 0=from H,1=from E
    TF = np.zeros((L1 + 1, L2 + 1), dtype=np.uint8)  # 0=from H,1=from F

    H[0, 0] = 0.0
    # initialize first row/col for global affine
    for i in range(1, L1 + 1):
        H[i, 0] = gap_open + (i - 1) * gap_extend
        E[i, 0] = neg_inf
        F[i, 0] = neg_inf
        TH[i, 0] = 1  # comes from E notionally (gap in B)
    for j in range(1, L2 + 1):
        H[0, j] = gap_open + (j - 1) * gap_extend
        E[0, j] = neg_inf
        F[0, j] = neg_inf
        TH[0, j] = 2  # comes from F notionally (gap in A)

    for i in range(1, L1 + 1):
        for j in range(1, L2 + 1):
            # update gap states
            e_from_h = H[i - 1, j] + gap_open
            e_from_e = E[i - 1, j] + gap_extend
            if e_from_h >= e_from_e:
                E[i, j] = e_from_h
                TE[i, j] = 0
            else:
                E[i, j] = e_from_e
                TE[i, j] = 1

            f_from_h = H[i, j - 1] + gap_open
            f_from_f = F[i, j - 1] + gap_extend
            if f_from_h >= f_from_f:
                F[i, j] = f_from_h
                TF[i, j] = 0
            else:
                F[i, j] = f_from_f
                TF[i, j] = 1

            diag = H[i - 1, j - 1] + score[i - 1, j - 1]
            # choose best among diag, E, F
            if diag >= E[i, j] and diag >= F[i, j]:
                H[i, j] = diag
                TH[i, j] = 0
            elif E[i, j] >= F[i, j]:
                H[i, j] = E[i, j]
                TH[i, j] = 1
            else:
                H[i, j] = F[i, j]
                TH[i, j] = 2

    # traceback
    i, j = L1, L2
    path: List[Tuple[Optional[int], Optional[int]]] = []
    state = TH[i, j]  # 0=diag,1=E,2=F
    while i > 0 or j > 0:
        if state == 0:  # diag
            if i == 0 or j == 0:
                break
            path.append((i - 1, j - 1))
            i -= 1
            j -= 1
            state = TH[i, j]
        elif state == 1:  # E: gap in B (up moves)
            if i == 0:
                break
            path.append((i - 1, None))
            prev = TE[i, j]
            i -= 1
            state = 0 if prev == 0 else 1
        else:  # state == 2, F: gap in A (left moves)
            if j == 0:
                break
            path.append((None, j - 1))
            prev = TF[i, j]
            j -= 1
            state = 0 if prev == 0 else 2
    path.reverse()
    return float(H[L1, L2]), path


def smith_waterman_affine(score: np.ndarray, gap_open: float, gap_extend: float) -> Tuple[float, List[Tuple[Optional[int], Optional[int]]]]:
    """
    Local alignment (Smith–Waterman) with affine gaps.
    Returns best local score and path.
    """
    L1, L2 = score.shape
    neg_inf = np.float32(-1e9)
    H = np.zeros((L1 + 1, L2 + 1), dtype=np.float32)
    E = np.full((L1 + 1, L2 + 1), neg_inf, dtype=np.float32)
    F = np.full((L1 + 1, L2 + 1), neg_inf, dtype=np.float32)
    TH = np.full((L1 + 1, L2 + 1), 3, dtype=np.uint8)  # 0=diag,1=E,2=F,3=stop(0)
    TE = np.zeros((L1 + 1, L2 + 1), dtype=np.uint8)
    TF = np.zeros((L1 + 1, L2 + 1), dtype=np.uint8)

    best = 0.0
    bi = bj = 0

    for i in range(1, L1 + 1):
        for j in range(1, L2 + 1):
            e_from_h = H[i - 1, j] + gap_open
            e_from_e = E[i - 1, j] + gap_extend
            if e_from_h >= e_from_e:
                E[i, j] = e_from_h
                TE[i, j] = 0
            else:
                E[i, j] = e_from_e
                TE[i, j] = 1

            f_from_h = H[i, j - 1] + gap_open
            f_from_f = F[i, j - 1] + gap_extend
            if f_from_h >= f_from_f:
                F[i, j] = f_from_h
                TF[i, j] = 0
            else:
                F[i, j] = f_from_f
                TF[i, j] = 1

            diag = H[i - 1, j - 1] + score[i - 1, j - 1]
            val = max(0.0, diag, E[i, j], F[i, j])
            H[i, j] = val
            if val == 0.0:
                TH[i, j] = 3
            elif val == diag:
                TH[i, j] = 0
            elif val == E[i, j]:
                TH[i, j] = 1
            else:
                TH[i, j] = 2

            if val > best:
                best = val
                bi, bj = i, j

    # traceback
    i, j = bi, bj
    path: List[Tuple[Optional[int], Optional[int]]] = []
    while i > 0 and j > 0 and TH[i, j] != 3 and H[i, j] > 0:
        tb = TH[i, j]
        if tb == 0:
            path.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif tb == 1:
            path.append((i - 1, None))
            prev = TE[i, j]
            i -= 1
            if prev == 0:
                # came from H
                # stay in loop; next TH will decide
                pass
            else:
                # came from E, keep tb=1
                pass
        elif tb == 2:
            path.append((None, j - 1))
            prev = TF[i, j]
            j -= 1
            if prev == 0:
                pass
            else:
                pass
        else:
            break
    path.reverse()
    return float(best), path


def alignment_to_tsv(
    path: List[Tuple[Optional[int], Optional[int]]],
    score_matrix: np.ndarray,
    s1: Optional[str] = None,
    s2: Optional[str] = None,
) -> str:
    """
    Convert an alignment path to a TSV string.
    Columns: step, i_index, j_index, cell_score[, char1, char2]
    If s1/s2 provided, outputs aligned characters (gap='-').
    """
    base_header = "step\ti_index\tj_index\tcell_score"
    if s1 is not None and s2 is not None:
        lines = [base_header + "\tchar1\tchar2"]
    else:
        lines = [base_header]

    len1 = len(s1) if s1 is not None else 0
    len2 = len(s2) if s2 is not None else 0

    for k, (i, j) in enumerate(path):
        cell = "NaN"
        if i is not None and j is not None:
            cell = f"{score_matrix[i, j]:.6f}"
        part = f"{k}\t{'' if i is None else i}\t{'' if j is None else j}\t{cell}"
        if s1 is not None and s2 is not None:
            c1 = '-' if i is None else (s1[i] if i < len1 else '?')
            c2 = '-' if j is None else (s2[j] if j < len2 else '?')
            part = part + f"\t{c1}\t{c2}"
        lines.append(part)
    return "\n".join(lines)


def save_matrix_tsv(matrix: np.ndarray, path: str):
    """
    Save a 2D matrix as TSV with 0-based indices as headers.
    """
    L1, L2 = matrix.shape
    header = ["i/j"] + [str(j) for j in range(L2)]
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for i in range(L1):
            row = [str(i)] + [f"{matrix[i, j]:.6f}" for j in range(L2)]
            f.write("\t".join(row) + "\n")


def save_matrix_png(matrix: np.ndarray, path: str, title: Optional[str] = None):
    """
    Save a heatmap PNG of the matrix using a diverging colormap.
    Color scale is fixed to [-1, 1] for cosine similarity.
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required to write PNGs. Please install it (e.g., 'pip install matplotlib')."
        ) from e

    L1, L2 = matrix.shape

    # Sensible figure sizing without exploding memory on long sequences
    def _size(n):
        return max(4.0, min(12.0, 0.08 * n))
    fig_w = _size(L2)
    fig_h = _size(L1)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    im = ax.imshow(
        matrix,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
    )
    ax.set_xlabel("RNA2 node index")
    ax.set_ylabel("RNA1 node index")
    if title:
        ax.set_title(title)

    # Tick density management
    def set_ticks(ax, length, which="x"):
        max_ticks = 20
        if length <= max_ticks:
            ticks = np.arange(length)
        else:
            ticks = np.linspace(0, length - 1, num=max_ticks, dtype=int)
        if which == "x":
            ax.set_xticks(ticks)
        else:
            ax.set_yticks(ticks)

    set_ticks(ax, L2, "x")
    set_ticks(ax, L1, "y")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("cosine similarity")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Align two RNAs using node embeddings: compute pairwise cosine "
            "similarity matrix and perform dynamic programming alignment."
        )
    )
    parser.add_argument("--input", required=True, help="Path to TSV/CSV with node embeddings.")
    parser.add_argument("--id-column", required=True, help="Column name for unique RNA IDs.")
    parser.add_argument("--rna1", required=True, help="ID value of the first RNA.")
    parser.add_argument("--rna2", required=True, help="ID value of the second RNA.")
    # Optional base (sequence) embeddings
    parser.add_argument(
        "--base-input",
        default=None,
        help=(
            "Optional path to TSV/CSV that contains base embeddings (per-position sequence embeddings). "
            "If omitted, attempts to read base embeddings from --input if the column is present."
        ),
    )
    parser.add_argument(
        "--base-embeds-col",
        default="base_embeddings",
        help="Column name for base (sequence) embeddings if provided.",
    )
    parser.add_argument(
        "--seq-weight",
        type=float,
        default=0.0,
        help=
        "Weight for base (sequence) similarity in [0,1]. 0=only structural (node) embeddings; 1=only base.",
    )
    # Gap penalties (affine). Maintain deprecated --gap as alias to --gap-open.
    parser.add_argument("--gap-open", type=float, default=-1.0, help="Gap opening penalty (negative).")
    parser.add_argument("--gap-extend", type=float, default=-1.0, help="Gap extension penalty (negative).")
    parser.add_argument("--gap", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--mode",
        choices=["global", "local"],
        default="global",
        help="Alignment type: global (Needleman–Wunsch) or local (Smith–Waterman).",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "Prefix for output files. If not provided, uses '<input_basename>__<rna1>__vs__<rna2>'. "
            "Writes '<prefix>.matrix.tsv', '<prefix>.matrix.png', '<prefix>.alignment.tsv' and, if --structure-column-name is given, '<prefix>.structures.txt'."
        ),
    )
    parser.add_argument(
        "--plot-matrix",
        action="store_true",
        help="If set, writes a PNG heatmap of the matrix alongside the TSV.",
    )
    # Optional: pull dot-bracket structures directly from the input table
    parser.add_argument(
        "--structure-column-name",
        default=None,
        help=(
            "If provided, uses this column from --input to emit aligned dot-bracket structures."
        ),
    )
    parser.add_argument(
        "--save-components",
        action="store_true",
        help="If set and base embeddings are used, also write base/struct component matrices.",
    )
    args = parser.parse_args()

    if not (0.0 <= float(args.seq_weight) <= 1.0):
        raise ValueError("--seq-weight must be in [0,1].")

    df = read_table_auto(args.input)
    if args.id_column not in df.columns:
        raise ValueError(f"Required column '{args.id_column}' not found in input.")
    if "node_embeddings" not in df.columns:
        raise ValueError("Input does not contain a 'node_embeddings' column.")

    rows1 = df[df[args.id_column] == args.rna1]
    rows2 = df[df[args.id_column] == args.rna2]
    if len(rows1) == 0:
        raise ValueError(f"No row found where {args.id_column} == {args.rna1}")
    if len(rows2) == 0:
        raise ValueError(f"No row found where {args.id_column} == {args.rna2}")
    if len(rows1) > 1:
        raise ValueError(f"Multiple rows found for {args.id_column} == {args.rna1}; expected exactly one.")
    if len(rows2) > 1:
        raise ValueError(f"Multiple rows found for {args.id_column} == {args.rna2}; expected exactly one.")

    A_struct = parse_embeddings(rows1.iloc[0]["node_embeddings"])
    B_struct = parse_embeddings(rows2.iloc[0]["node_embeddings"])
    sim_struct = cosine_similarity_matrix(A_struct, B_struct)

    sim = sim_struct
    used_base = False
    A_base = B_base = None
    if args.seq_weight > 0.0:
        # Try to get base embeddings either from --base-input or from same df
        base_df = None
        if args.base_input:
            base_df = read_table_auto(args.base_input)
        else:
            base_df = df
        if args.base_embeds_col not in base_df.columns:
            print(
                f"[warn] Base embeddings column '{args.base_embeds_col}' not found; continuing with structural only."
            )
        else:
            br1 = base_df[base_df[args.id_column] == args.rna1]
            br2 = base_df[base_df[args.id_column] == args.rna2]
            if len(br1) == 1 and len(br2) == 1:
                A_base = parse_embeddings(br1.iloc[0][args.base_embeds_col])
                B_base = parse_embeddings(br2.iloc[0][args.base_embeds_col])
                if A_base.shape[0] != A_struct.shape[0] or B_base.shape[0] != B_struct.shape[0]:
                    trimmed = False
                    if A_base.shape[0] == A_struct.shape[0] + 2 and B_base.shape[0] == B_struct.shape[0] + 2:
                        A_base = A_base[1:-1]
                        B_base = B_base[1:-1]
                        trimmed = True
                        print("[info] Trimmed BOS/EOS from base embeddings to match structural length.")
                    if not trimmed:
                        print(
                            "[warn] Length mismatch between base and structural embeddings; skipping base weighting."
                        )
                if A_base.shape[0] == A_struct.shape[0] and B_base.shape[0] == B_struct.shape[0]:
                    sim_base = cosine_similarity_matrix(A_base, B_base)
                    w = float(args.seq_weight)
                    sim = (1.0 - w) * sim_struct + w * sim_base
                    used_base = True
                    print(
                        f"[info] Using base embeddings with seq_weight={w:.3f}; struct_shape={sim_struct.shape}, base_shape={sim_base.shape}"
                    )
            else:
                print("[warn] Could not find unique base embeddings rows for both RNAs; skipping base weighting.")

    # Alignment
    # Handle deprecated --gap alias
    if args.gap is not None:
        print("[align] --gap is deprecated; use --gap-open and --gap-extend. Treating --gap as --gap-open.")
        args.gap_open = args.gap
    if args.gap_extend is None:
        args.gap_extend = args.gap_open

    if args.mode == "global":
        best_score, path = needleman_wunsch_affine(sim, args.gap_open, args.gap_extend)
    else:
        best_score, path = smith_waterman_affine(sim, args.gap_open, args.gap_extend)

    # Outputs
    if args.output_prefix is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output_prefix = f"{base}__{args.rna1}__vs__{args.rna2}"
    matrix_out = args.output_prefix + ".matrix.tsv"
    matrix_png = args.output_prefix + ".matrix.png"
    align_out = args.output_prefix + ".alignment.tsv"
    struct_txt_out = args.output_prefix + ".structures.txt"

    os.makedirs(os.path.dirname(matrix_out) or ".", exist_ok=True)

    save_matrix_tsv(sim, matrix_out)
    if args.plot_matrix:
        save_matrix_png(sim, matrix_png, title=f"Cosine similarity (combined): {args.rna1} vs {args.rna2}")
    if used_base and args.save_components:
        base_path = args.output_prefix + ".matrix.base.tsv"
        struct_path = args.output_prefix + ".matrix.struct.tsv"
        save_matrix_tsv(sim_struct, struct_path)
        if 'sim_base' in locals():
            save_matrix_tsv(sim_base, base_path)

    # Optional: aligned dot-bracket output from input table
    s1 = s2 = None
    if args.structure_column_name:
        if args.structure_column_name not in df.columns:
            raise ValueError(
                f"Structure column '{args.structure_column_name}' not found in input data."
            )
        s1 = str(rows1.iloc[0][args.structure_column_name])
        s2 = str(rows2.iloc[0][args.structure_column_name])
        if len(s1) != A_struct.shape[0]:
            print(
                f"[warning] Length mismatch for RNA1: structure={len(s1)} vs embeddings={A_struct.shape[0]}"
            )
        if len(s2) != B_struct.shape[0]:
            print(
                f"[warning] Length mismatch for RNA2: structure={len(s2)} vs embeddings={B_struct.shape[0]}"
            )

    with open(align_out, "w") as f:
        f.write(f"# mode=\"{args.mode}\"\n")
        f.write(f"# gap_open=\"{args.gap_open}\"\n")
        f.write(f"# gap_extend=\"{args.gap_extend}\"\n")
        f.write(f"# rna1=\"{args.rna1}\", rna2=\"{args.rna2}\"\n")
        f.write(f"# total_alignment_score=\"{best_score:.6f}\"\n")
        if used_base:
            f.write(f"# seq_weight=\"{args.seq_weight}\"\n")
        if s1 is not None and s2 is not None:
            f.write(f"# aligned_structures_present=\"true\"\n")
        f.write(alignment_to_tsv(path, sim) if s1 is None else alignment_to_tsv(path, sim, s1, s2))

    # If structures provided, also emit a compact TXT with the two aligned strings
    if s1 is not None and s2 is not None:
        aligned1_chars = []
        aligned2_chars = []
        len1 = len(s1)
        len2 = len(s2)
        for i, j in path:
            c1 = '-' if i is None else (s1[i] if i < len1 else '?')
            c2 = '-' if j is None else (s2[j] if j < len2 else '?')
            aligned1_chars.append(c1)
            aligned2_chars.append(c2)
        aligned1 = ''.join(aligned1_chars)
        aligned2 = ''.join(aligned2_chars)
        with open(struct_txt_out, 'w') as f:
            f.write(f"{args.rna1}\t{aligned1}\n")
            f.write(f"{args.rna2}\t{aligned2}\n")

    print(f"Scoring matrix written to {matrix_out}")
    if args.plot_matrix:
        print(f"Matrix heatmap written to {matrix_png}")
    print(f"Alignment written to {align_out}")
    if s1 is not None and s2 is not None:
        print(f"Structure alignment written to {struct_txt_out}")
    print(f"Total alignment score: {best_score:.6f}")


if __name__ == "__main__":
    main()
