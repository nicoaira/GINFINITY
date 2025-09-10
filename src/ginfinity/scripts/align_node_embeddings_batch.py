#!/usr/bin/env python3

import argparse
import json
import os
import re
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - fallback when tqdm missing
    class tqdm:  # minimal shim
        def __init__(self, total=None, desc=None, unit=None):
            self.total = total
        def update(self, n):
            pass
        def close(self):
            pass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def read_embeddings_table(path: str) -> pd.DataFrame:
    if path.endswith(".tsv"):
        df = pd.read_csv(path, sep="\t", low_memory=False)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, sep=None, engine="python")
    if "node_embeddings" not in df.columns:
        raise ValueError("Input does not contain a 'node_embeddings' column.")
    return df


def parse_node_embeddings(cell: str) -> np.ndarray:
    try:
        arr = json.loads(cell)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse node_embeddings JSON: {e}") from e
    mat = np.asarray(arr, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError("node_embeddings must be a 2D array [L x D].")
    return mat


def sanitize_pair_name(a: str, b: str) -> str:
    s = f"{a}__vs__{b}"
    return re.sub(r"[^A-Za-z0-9_.\-]+", "_", s)


# -----------------------------------------------------------------------------
# Scoring + alignment
# -----------------------------------------------------------------------------

def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"Embedding dims mismatch: {A.shape[1]} vs {B.shape[1]}")
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)
    return A_norm @ B_norm.T


def needleman_wunsch_affine(score: np.ndarray, gap_open: float, gap_extend: float) -> Tuple[float, List[Tuple[Optional[int], Optional[int]]]]:
    L1, L2 = score.shape
    neg_inf = np.float32(-1e9)
    H = np.full((L1 + 1, L2 + 1), neg_inf, dtype=np.float32)
    E = np.full((L1 + 1, L2 + 1), neg_inf, dtype=np.float32)
    F = np.full((L1 + 1, L2 + 1), neg_inf, dtype=np.float32)
    TH = np.zeros((L1 + 1, L2 + 1), dtype=np.uint8)  # 0=diag,1=E,2=F
    TE = np.zeros((L1 + 1, L2 + 1), dtype=np.uint8)  # 0=from H,1=from E
    TF = np.zeros((L1 + 1, L2 + 1), dtype=np.uint8)  # 0=from H,1=from F

    H[0, 0] = 0.0
    for i in range(1, L1 + 1):
        H[i, 0] = gap_open + (i - 1) * gap_extend
        TH[i, 0] = 1
    for j in range(1, L2 + 1):
        H[0, j] = gap_open + (j - 1) * gap_extend
        TH[0, j] = 2

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
            if diag >= E[i, j] and diag >= F[i, j]:
                H[i, j] = diag
                TH[i, j] = 0
            elif E[i, j] >= F[i, j]:
                H[i, j] = E[i, j]
                TH[i, j] = 1
            else:
                H[i, j] = F[i, j]
                TH[i, j] = 2

    i, j = L1, L2
    path: List[Tuple[Optional[int], Optional[int]]] = []
    state = TH[i, j]
    while i > 0 or j > 0:
        if state == 0:
            if i == 0 or j == 0:
                break
            path.append((i - 1, j - 1))
            i -= 1
            j -= 1
            state = TH[i, j]
        elif state == 1:
            if i == 0:
                break
            path.append((i - 1, None))
            prev = TE[i, j]
            i -= 1
            state = 0 if prev == 0 else 1
        else:
            if j == 0:
                break
            path.append((None, j - 1))
            prev = TF[i, j]
            j -= 1
            state = 0 if prev == 0 else 2
    path.reverse()
    return float(H[L1, L2]), path


def smith_waterman_affine(score: np.ndarray, gap_open: float, gap_extend: float) -> Tuple[float, List[Tuple[Optional[int], Optional[int]]]]:
    L1, L2 = score.shape
    neg_inf = np.float32(-1e9)
    H = np.zeros((L1 + 1, L2 + 1), dtype=np.float32)
    E = np.full((L1 + 1, L2 + 1), neg_inf, dtype=np.float32)
    F = np.full((L1 + 1, L2 + 1), neg_inf, dtype=np.float32)
    TH = np.full((L1 + 1, L2 + 1), 3, dtype=np.uint8)
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
        elif tb == 2:
            path.append((None, j - 1))
            prev = TF[i, j]
            j -= 1
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
    L1, L2 = matrix.shape
    header = ["i/j"] + [str(j) for j in range(L2)]
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for i in range(L1):
            row = [str(i)] + [f"{matrix[i, j]:.6f}" for j in range(L2)]
            f.write("\t".join(row) + "\n")


def save_matrix_png(matrix: np.ndarray, path: str, title: Optional[str] = None):
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required to write PNGs. Please install it (e.g., 'pip install matplotlib')."
        ) from e
    L1, L2 = matrix.shape
    def _size(n):
        return max(4.0, min(12.0, 0.08 * n))
    fig_w = _size(L2)
    fig_h = _size(L1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto", interpolation="nearest", origin="upper")
    ax.set_xlabel("RNA2 node index")
    ax.set_ylabel("RNA1 node index")
    if title:
        ax.set_title(title)
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


# -----------------------------------------------------------------------------
# Batch processing
# -----------------------------------------------------------------------------

def pair_batcher(n: int, batch_size: int) -> Iterable[List[Tuple[int, int]]]:
    # Generate all i<j pairs and yield batches
    batch: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            batch.append((i, j))
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def _process_pair_task(task_args) -> Dict[str, object]:
    (
        i,
        j,
        id1,
        id2,
        A,
        B,
        s1,
        s2,
        gap_open,
        gap_extend,
        mode,
        write_alignment,
        write_matrix,
        plot_matrix,
        output_dir,
    ) = task_args

    sim = cosine_similarity_matrix(A, B)
    if mode == "global":
        best_score, path = needleman_wunsch_affine(sim, gap_open, gap_extend)
    else:
        best_score, path = smith_waterman_affine(sim, gap_open, gap_extend)

    pair_name = sanitize_pair_name(id1, id2)
    pair_dir = os.path.join(output_dir, pair_name)
    if write_alignment or write_matrix or plot_matrix:
        os.makedirs(pair_dir, exist_ok=True)

    if write_alignment:
        align_out = os.path.join(pair_dir, f"{pair_name}.alignment.tsv")
        with open(align_out, "w") as f:
            f.write(f"# mode=\"{mode}\"\n")
            f.write(f"# gap_open=\"{gap_open}\"\n")
            f.write(f"# gap_extend=\"{gap_extend}\"\n")
            f.write(f"# rna1=\"{id1}\", rna2=\"{id2}\"\n")
            f.write(f"# total_alignment_score=\"{best_score:.6f}\"\n")
            if s1 is not None and s2 is not None:
                f.write(f"# aligned_structures_present=\"true\"\n")
            f.write(alignment_to_tsv(path, sim, s1, s2))
        if s1 is not None and s2 is not None:
            struct_out = os.path.join(pair_dir, f"{pair_name}.structures.txt")
            aligned1_chars = []
            aligned2_chars = []
            len1 = len(s1)
            len2 = len(s2)
            for pi, pj in path:
                c1 = '-' if pi is None else (s1[pi] if pi < len1 else '?')
                c2 = '-' if pj is None else (s2[pj] if pj < len2 else '?')
                aligned1_chars.append(c1)
                aligned2_chars.append(c2)
            with open(struct_out, 'w') as f:
                f.write(f"{id1}\t{''.join(aligned1_chars)}\n")
                f.write(f"{id2}\t{''.join(aligned2_chars)}\n")

    if write_matrix:
        m_tsv = os.path.join(pair_dir, f"{pair_name}.matrix.tsv")
        save_matrix_tsv(sim, m_tsv)
    if plot_matrix:
        m_png = os.path.join(pair_dir, f"{pair_name}.matrix.png")
        save_matrix_png(sim, m_png, title=f"Cosine similarity: {id1} vs {id2}")

    return {
        "id1": id1,
        "id2": id2,
        "n1": int(A.shape[0]),
        "n2": int(B.shape[0]),
        "score": float(best_score),
        "mode": mode,
        "gap_open": float(gap_open),
        "gap_extend": float(gap_extend),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Batch-align all row pairs from a node-embeddings table using cosine similarity "
            "and dynamic programming (CPU)."
        )
    )
    parser.add_argument("--input", required=True, help="Path to TSV/CSV with node embeddings.")
    parser.add_argument("--id-column", required=True, help="Column name for unique RNA IDs.")
    parser.add_argument("--output-dir", required=True, help="Directory to write outputs.")
    # Gap penalties (affine). Keep deprecated --gap as alias to --gap-open
    parser.add_argument("--gap-open", type=float, default=-1.0, help="Gap opening penalty (negative).")
    parser.add_argument("--gap-extend", type=float, default=-1.0, help="Gap extension penalty (negative).")
    parser.add_argument("--gap", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=["global", "local"], default="global", help="Alignment type.")
    parser.add_argument("--batch-size", type=int, default=16, help="Pairs processed per batch.")
    parser.add_argument("--structure-column-name", default=None, help="Optional column with dot-bracket strings in --input.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of worker processes for pair alignment.")
    parser.add_argument("--write-alignment", action="store_true", help="Write per-pair alignment TSVs (and structures if available).")
    parser.add_argument("--write-matrix", action="store_true", help="Write per-pair matrix TSV.")
    parser.add_argument("--plot-matrix", action="store_true", help="Also write per-pair PNG heatmap of the matrix.")
    parser.add_argument("--summary", default="summary.tsv", help="Filename for the summary TSV inside --output-dir.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = read_embeddings_table(args.input)
    if args.id_column not in df.columns:
        raise ValueError(f"ID column '{args.id_column}' not found in input.")
    if args.structure_column_name and args.structure_column_name not in df.columns:
        raise ValueError(f"Structure column '{args.structure_column_name}' not found in input.")

    # Load and cache per-row embeddings and structures
    ids: List[str] = []
    mats: List[np.ndarray] = []
    structs: List[Optional[str]] = []
    for _, row in df.iterrows():
        ids.append(str(row[args.id_column]))
        M = parse_node_embeddings(row["node_embeddings"])
        mats.append(M.astype(np.float32, copy=False))
        if args.structure_column_name:
            structs.append(str(row[args.structure_column_name]))
        else:
            structs.append(None)

    n = len(ids)
    if n < 2:
        print("Nothing to do: fewer than 2 rows.")
        return

    # Summary rows
    summary_rows: List[Dict[str, object]] = []

    # Progress bar over total pairs
    total_pairs = n * (n - 1) // 2
    pbar = tqdm(total=total_pairs, desc="Aligning pairs", unit="pair")

    # Handle deprecated --gap alias
    if args.gap is not None:
        print("[align-batch] --gap is deprecated; use --gap-open and --gap-extend. Treating --gap as --gap-open.")
        args.gap_open = args.gap
    if args.gap_extend is None:
        args.gap_extend = args.gap_open

    # Process in batches (optionally parallel)
    if args.num_workers > 1:
        executor = ProcessPoolExecutor(max_workers=args.num_workers)
        try:
            for batch in pair_batcher(n, args.batch_size):
                futures = []
                for i, j in batch:
                    futures.append(
                        executor.submit(
                            _process_pair_task,
                            (
                                i,
                                j,
                                ids[i],
                                ids[j],
                                mats[i],
                                mats[j],
                                structs[i],
                                structs[j],
                                args.gap_open,
                                args.gap_extend,
                                args.mode,
                                args.write_alignment,
                                args.write_matrix,
                                args.plot_matrix,
                                args.output_dir,
                            ),
                        )
                    )
                for fut in as_completed(futures):
                    summary_rows.append(fut.result())
                    pbar.update(1)
        finally:
            executor.shutdown()
    else:
        for batch in pair_batcher(n, args.batch_size):
            for i, j in batch:
                res = _process_pair_task(
                    (
                        i,
                        j,
                        ids[i],
                        ids[j],
                        mats[i],
                        mats[j],
                        structs[i],
                        structs[j],
                        args.gap_open,
                        args.gap_extend,
                        args.mode,
                        args.write_alignment,
                        args.write_matrix,
                        args.plot_matrix,
                        args.output_dir,
                    )
                )
                summary_rows.append(res)
                pbar.update(1)

    pbar.close()

    # Write summary
    summary_df = pd.DataFrame(summary_rows)
    out_path = os.path.join(args.output_dir, args.summary)
    summary_df.to_csv(out_path, sep="\t", index=False)
    print(f"Processed {len(summary_rows)} pair(s). Summary written to {out_path}")


if __name__ == "__main__":
    main()
