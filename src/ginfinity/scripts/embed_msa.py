#!/usr/bin/env python3
"""
embed_msa.py — Embedding-based RNA MSA

This script builds a multiple sequence alignment (MSA) for RNAs from
per-position node embeddings using a T‑Coffee/ProbCons‑style pipeline.

High-level stages
- Load TSV data and L2-normalize per-position embeddings.
- Pairwise soft-DP (pair-HMM with affine gaps) to get match posteriors.
- T‑Coffee library consistency transform on sparse posteriors.
- Build a guide tree (NJ or UPGMA) from pairwise evidence.
- Progressive profile–profile alignment with affine gaps and simple
  structure-aware scoring using optional pairing priors.
- Optional iterative refinement (placeholder hook).
- Write outputs (FASTA, Stockholm, TSV) and diagnostics.

Quick usage
  python src/ginfinity/scripts/embed_msa.py \
    --input premirnas_node_embeds.tsv \
    --name-col Name --embeds-col node_embeddings \
    --dotbracket-col DotBracket --paired-col PairedIndices \
    --out-prefix out/premirnas --topk 20 --consistency-rounds 1 \
    --alpha 6.0 --beta 0.0 --gap-open -10 --gap-extend -0.5 \
    --stem-gap-bonus -2.0 --tree nj --refine-iters 2 --num-workers 8

Documentation
- Detailed documentation is available at docs/embed_msa.md.
  It covers the input format, algorithmic details, CLI flags, outputs,
  performance tips, and troubleshooting.

Dependencies
- numpy, pandas, numba, tqdm, networkx, matplotlib (optional for plots)

Notes
- Operations are sparse via top‑K pruning with a probability threshold.
- DP kernels are float32 and numba‑accelerated when available.
- No external ML frameworks are required.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    def njit(*args, **kwargs):  # type: ignore
        def wrapper(f):
            return f
        return wrapper
    NUMBA_AVAILABLE = False

try:
    import networkx as nx
except Exception as e:  # pragma: no cover
    nx = None  # type: ignore

try:
    import matplotlib.pyplot as plt  # optional
    MPL_AVAILABLE = True
except Exception:
    MPL_AVAILABLE = False


# ==========================
# Data structures and types
# ==========================

@dataclass
class SequenceRecord:
    name: str
    emb: np.ndarray  # shape (L, D), float32, L2-normalized rows
    dotbracket: Optional[str] = None
    paired_idx: Optional[List[int]] = None  # length L, partner index or -1


@dataclass
class SparsePairs:
    # Coordinate sparse representation of posteriors for a given pair
    # rows: i positions in A, cols: j positions in B
    # Keep as simple arrays for speed and memory
    i: np.ndarray  # int32
    j: np.ndarray  # int32
    p: np.ndarray  # float32
    shape: Tuple[int, int]


@dataclass
class ProfileColumn:
    mu: np.ndarray  # normalized mean embedding, shape (D,)
    stem_fraction: float  # in [0, 1]


@dataclass
class Profile:
    # A profile is a list of columns; also keeps mapping from sequences to their strings
    columns: List[ProfileColumn]
    # For reconstructing aligned strings
    member_indices: List[int]  # which original sequences are present
    # Per member, a list of per-column residue: char or '-' for gap
    aligned_chars: Dict[int, List[str]]


@dataclass
class GuideTree:
    # Simple binary tree used for progressive alignment
    # Represented as nested tuples (left, right) or leaf index
    structure: Any


# ======================
# Utility/helper methods
# ======================

def _json_loads_maybe(x: Any) -> Any:
    if isinstance(x, (list, dict)):
        return x
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return None
        try:
            return json.loads(x)
        except Exception:
            return None
    return None


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    eps = 1e-8
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (x / norms).astype(np.float32)


def _dotbracket_to_pairs(db: str) -> List[int]:
    # Simple stack-based pairing for (), [] , {}
    L = len(db)
    pairs = [-1] * L
    stacks = {"(": [], "[": [], "{": []}
    mates = {")": "(", "]": "[", "}": "{",
    }
    for i, ch in enumerate(db):
        if ch in stacks:
            stacks[ch].append(i)
        elif ch in mates:
            op = mates[ch]
            if stacks[op]:
                j = stacks[op].pop()
                pairs[i] = j
                pairs[j] = i
    return pairs


# ======================
# I/O and preprocessing
# ======================

def load_tsv(path: str,
             name_col: str,
             embeds_col: str,
             dotbracket_col: Optional[str] = None,
             paired_col: Optional[str] = None) -> List[SequenceRecord]:
    df = pd.read_csv(path, sep='\t')
    if name_col not in df.columns or embeds_col not in df.columns:
        raise ValueError(f"Missing required columns: {name_col}, {embeds_col}")

    records: List[SequenceRecord] = []
    for idx, row in df.iterrows():
        name = str(row[name_col])
        embeds_raw = _json_loads_maybe(row[embeds_col])
        if embeds_raw is None:
            print(f"[WARN] Row {idx} ('{name}') has invalid embeddings; skipping.")
            continue
        try:
            emb = np.array(embeds_raw, dtype=np.float32)
        except Exception:
            print(f"[WARN] Row {idx} ('{name}') embeddings parse failed; skipping.")
            continue
        if emb.ndim != 2 or emb.shape[0] == 0:
            print(f"[WARN] Row {idx} ('{name}') embeddings malformed; skipping.")
            continue

        dotbracket = None
        paired_idx = None
        if paired_col and paired_col in df.columns:
            paired_raw = _json_loads_maybe(row[paired_col])
            if isinstance(paired_raw, list) and len(paired_raw) == emb.shape[0]:
                try:
                    paired_idx = [int(v) for v in paired_raw]
                except Exception:
                    paired_idx = None
        if paired_idx is None and dotbracket_col and dotbracket_col in df.columns:
            db = row[dotbracket_col]
            if isinstance(db, str) and len(db) == emb.shape[0]:
                dotbracket = db
                paired_idx = _dotbracket_to_pairs(db)

        records.append(SequenceRecord(name=name, emb=emb, dotbracket=dotbracket, paired_idx=paired_idx))
    return records


def l2_normalize_embeddings(records: List[SequenceRecord]) -> None:
    for r in records:
        r.emb = _l2_normalize_rows(r.emb)


# ==================================
# Pair selection and similarities
# ==================================

def pairwise_pairs_to_compute(records: List[SequenceRecord],
                              max_pairs: Optional[int],
                              threads: int = 1) -> List[Tuple[int, int]]:
    N = len(records)
    pairs: List[Tuple[int, int]] = []
    if N <= 1:
        return pairs
    # All pairs if small
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((i, j))
    if max_pairs is None or max_pairs <= 0 or len(pairs) <= max_pairs:
        return pairs
    # For scalability: choose nearest neighbors by mean embedding cosine
    means = [r.emb.mean(axis=0) for r in records]
    means = [m / (np.linalg.norm(m) + 1e-8) for m in means]
    mean_mat = np.stack(means, axis=0).astype(np.float32)
    sims = mean_mat @ mean_mat.T
    # For each i, keep top neighbors
    k = max(1, int(max_pairs / max(1, N)))
    nn_pairs = set()
    for i in range(N):
        order = np.argsort(-sims[i])
        c = 0
        for j in order:
            if j == i:
                continue
            a, b = (i, j) if i < j else (j, i)
            nn_pairs.add((a, b))
            c += 1
            if c >= k:
                break
    pairs = sorted(nn_pairs)
    if max_pairs is not None and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]
    return pairs


def cosine_similarity_matrix(Ea: np.ndarray, Eb: np.ndarray) -> np.ndarray:
    # Both Ea and Eb should have L2-normalized rows.
    # Return float32 matrix of shape (La, Lb)
    return (Ea @ Eb.T).astype(np.float32)


def calibrate_log_odds(S: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    # s in [-1, 1] -> p = sigmoid(alpha*s + beta); L = log(p/(1-p))
    X = alpha * S + beta
    # stable sigmoid then logit
    p = 1.0 / (1.0 + np.exp(-X))
    eps = 1e-6
    p = np.clip(p, eps, 1.0 - eps)
    L = np.log(p) - np.log(1.0 - p)
    return L.astype(np.float32)


# ======================================================
# Soft-DP: Forward-Backward with affine gaps in log-space
# ======================================================

@njit(cache=True)
def _logsumexp2(a: float, b: float) -> float:
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))


@njit(cache=True)
def _logsumexp3(a: float, b: float, c: float) -> float:
    m = a
    if b > m:
        m = b
    if c > m:
        m = c
    return m + math.log(math.exp(a - m) + math.exp(b - m) + math.exp(c - m))


@njit(cache=True)
def forward_log(L: np.ndarray, go: float, ge: float, local: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    # Three-state affine pair-HMM in log-space: M, X (gap in A, insert in B), Y (gap in B, insert in A)
    # Transitions: open cost go, extend cost ge added per step in gap states
    La, Lb = L.shape
    NEG_INF = -1e30
    M = np.full((La + 1, Lb + 1), NEG_INF, dtype=np.float32)
    X = np.full((La + 1, Lb + 1), NEG_INF, dtype=np.float32)
    Y = np.full((La + 1, Lb + 1), NEG_INF, dtype=np.float32)

    # Initialization
    M[0, 0] = 0.0 if local else NEG_INF
    X[0, 0] = Y[0, 0] = NEG_INF
    for i in range(1, La + 1):
        if local:
            X[i, 0] = max(X[i - 1, 0] + ge, go + (M[i - 1, 0] if M[i - 1, 0] > NEG_INF/2 else 0.0))
        else:
            X[i, 0] = go + ge * (i - 1)
    for j in range(1, Lb + 1):
        if local:
            Y[0, j] = max(Y[0, j - 1] + ge, go + (M[0, j - 1] if M[0, j - 1] > NEG_INF/2 else 0.0))
        else:
            Y[0, j] = go + ge * (j - 1)

    # DP
    for i in range(1, La + 1):
        for j in range(1, Lb + 1):
            e = L[i - 1, j - 1]
            m_from = _logsumexp3(M[i - 1, j - 1], X[i - 1, j - 1], Y[i - 1, j - 1])
            M[i, j] = m_from + e

            x_from_open = M[i - 1, j] + go
            x_from_ext = X[i - 1, j] + ge
            X[i, j] = x_from_open if x_from_open > x_from_ext else x_from_ext

            y_from_open = M[i, j - 1] + go
            y_from_ext = Y[i, j - 1] + ge
            Y[i, j] = y_from_open if y_from_open > y_from_ext else y_from_ext

            if local:
                # Allow restart at zero for local alignment
                if M[i, j] < 0.0:
                    M[i, j] = 0.0
                if X[i, j] < 0.0:
                    X[i, j] = 0.0
                if Y[i, j] < 0.0:
                    Y[i, j] = 0.0

    # End score
    if local:
        best = NEG_INF
        for i in range(La + 1):
            for j in range(Lb + 1):
                best = max(best, M[i, j])
                best = max(best, X[i, j])
                best = max(best, Y[i, j])
        Z = best
    else:
        Z = _logsumexp3(M[La, Lb], X[La, Lb], Y[La, Lb])
    return M, X, Y, Z


@njit(cache=True)
def backward_log(L: np.ndarray, go: float, ge: float, local: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    La, Lb = L.shape
    NEG_INF = -1e30
    M = np.full((La + 1, Lb + 1), NEG_INF, dtype=np.float32)
    X = np.full((La + 1, Lb + 1), NEG_INF, dtype=np.float32)
    Y = np.full((La + 1, Lb + 1), NEG_INF, dtype=np.float32)

    # Initialization at end
    if local:
        # In local case, allow ending anywhere => backward start sums over all positions
        # We approximate by starting from (La, Lb) as global; still yields useful posteriors.
        M[La, Lb] = 0.0
    else:
        M[La, Lb] = 0.0

    # DP backwards
    for i in range(La, -1, -1):
        for j in range(Lb, -1, -1):
            if i < La and j < Lb:
                e = L[i, j]
                # From M[i,j] to M[i+1,j+1]
                M[i, j] = max(M[i, j], M[i + 1, j + 1] + e)
                # From X[i+1,j] open
                M[i, j] = max(M[i, j], X[i + 1, j] + go)
                # From Y[i,j+1] open
                M[i, j] = max(M[i, j], Y[i, j + 1] + go)
            if i < La:
                # X extends
                X[i, j] = max(X[i, j], X[i + 1, j] + ge)
            if j < Lb:
                # Y extends
                Y[i, j] = max(Y[i, j], Y[i, j + 1] + ge)

    if local:
        Z = 0.0  # approximate
    else:
        Z = 0.0
    return M, X, Y, Z


def forward_backward_affine_logspace(L: np.ndarray,
                                     go: float,
                                     ge: float,
                                     mode: str = "global") -> np.ndarray:
    # Returns posterior P(i~j) dense matrix float32 with shape (La, Lb)
    local = 1 if mode == "local" else 0
    Mf, Xf, Yf, Zf = forward_log(L, go, ge, local)
    Mb, Xb, Yb, Zb = backward_log(L, go, ge, local)

    La, Lb = L.shape
    P = np.zeros((La, Lb), dtype=np.float32)
    # Posterior match at (i,j): contribution through M state aligning i with j
    # Using log-space: P ∝ exp(Mf[i,j] + Mb[i+1,j+1] - Z)
    # Approximate partition Z by using Zf from forward.
    Z = Zf
    for i in range(La):
        for j in range(Lb):
            P[i, j] = math.exp(Mf[i + 1, j + 1] + Mb[i + 1, j + 1] - Z)
    return P


# ==================================
# Sparsification and consistency
# ==================================

def sparsify_posteriors(P: np.ndarray, topk: int, pmin: float = 1e-4) -> SparsePairs:
    La, Lb = P.shape
    # Keep top-K in each row and column; intersect sets for stability
    row_keep = [set() for _ in range(La)]
    col_keep = [set() for _ in range(Lb)]
    for i in range(La):
        if Lb <= topk:
            idx = np.argsort(-P[i])
        else:
            idx = np.argpartition(-P[i], topk - 1)[:topk]
            idx = idx[np.argsort(-P[i, idx])]
        for j in idx:
            if P[i, j] >= pmin:
                row_keep[i].add(int(j))
    for j in range(Lb):
        col = P[:, j]
        if La <= topk:
            idx = np.argsort(-col)
        else:
            idx = np.argpartition(-col, topk - 1)[:topk]
            idx = idx[np.argsort(-col[idx])]
        for i in idx:
            if P[i, j] >= pmin:
                col_keep[j].add(int(i))

    ii: List[int] = []
    jj: List[int] = []
    pp: List[float] = []
    for i in range(La):
        for j in row_keep[i]:
            if i in col_keep[j]:
                val = P[i, j]
                if val >= pmin:
                    ii.append(i)
                    jj.append(j)
                    pp.append(float(val))
    if len(ii) == 0:
        return SparsePairs(i=np.zeros(0, dtype=np.int32), j=np.zeros(0, dtype=np.int32), p=np.zeros(0, dtype=np.float32), shape=(La, Lb))
    return SparsePairs(i=np.array(ii, dtype=np.int32), j=np.array(jj, dtype=np.int32), p=np.array(pp, dtype=np.float32), shape=(La, Lb))


def consistency_round(sparse_lib: Dict[Tuple[int, int], SparsePairs],
                      sequences: List[SequenceRecord],
                      neighbors: Optional[Dict[int, List[int]]] = None,
                      lam: float = 0.5,
                      topk: int = 20,
                      pmin: float = 1e-4) -> Dict[Tuple[int, int], SparsePairs]:
    N = len(sequences)
    # Build adjacency lists from library
    adj: Dict[int, List[int]] = {i: [] for i in range(N)}
    for (a, b) in sparse_lib.keys():
        adj[a].append(b)
        adj[b].append(a)
    out: Dict[Tuple[int, int], SparsePairs] = {}

    for a in range(N):
        for b in adj.get(a, []):
            if a >= b:
                continue
            AB = sparse_lib[(a, b)]
            La, Lb = AB.shape
            # Build dict for quick lookup
            Pab = {}
            for k in range(AB.i.shape[0]):
                Pab[(int(AB.i[k]), int(AB.j[k]))] = float(AB.p[k])

            # Average consistency via intermediates C
            acc: Dict[Tuple[int, int], float] = {}
            count = 0
            Cs = neighbors.get(a, None) if neighbors else None
            if Cs is None:
                Cs = [c for c in range(N) if c != a and c != b]
            for c in Cs:
                if c == a or c == b:
                    continue
                AC = sparse_lib.get((min(a, c), max(a, c)))
                CB = sparse_lib.get((min(c, b), max(c, b)))
                if AC is None or CB is None:
                    continue
                # Build maps by i and by j for efficient multiply
                map_ac: Dict[int, List[Tuple[int, float]]] = {}
                for t in range(AC.i.shape[0]):
                    ai = int(AC.i[t])
                    cj = int(AC.j[t])
                    map_ac.setdefault(ai, []).append((cj, float(AC.p[t])))
                map_cb: Dict[int, List[Tuple[int, float]]] = {}
                for t in range(CB.i.shape[0]):
                    ci = int(CB.i[t])
                    bj = int(CB.j[t])
                    map_cb.setdefault(ci, []).append((bj, float(CB.p[t])))
                # Sum over k
                for ai, lst in map_ac.items():
                    for (ck, pac) in lst:
                        if ck in map_cb:
                            for (bj, pcb) in map_cb[ck]:
                                acc[(ai, bj)] = acc.get((ai, bj), 0.0) + pac * pcb
                count += 1

            # Combine
            new_scores: Dict[Tuple[int, int], float] = {}
            # Existing
            for (key, val) in Pab.items():
                new_scores[key] = (1.0 - lam) * val + lam * (acc.get(key, 0.0) / max(1, count))
            # New supported pairs from consistency only
            for (key, val) in acc.items():
                if key not in new_scores:
                    new_scores[key] = lam * (val / max(1, count))

            # Prune to topk by row and col
            if len(new_scores) == 0:
                out[(a, b)] = SparsePairs(i=np.zeros(0, dtype=np.int32), j=np.zeros(0, dtype=np.int32), p=np.zeros(0, dtype=np.float32), shape=(La, Lb))
                continue

            # Build dense-like for pruning by rows and cols
            by_row: Dict[int, List[Tuple[int, float]]] = {}
            by_col: Dict[int, List[Tuple[int, float]]] = {}
            for (i, j), v in new_scores.items():
                if v >= pmin:
                    by_row.setdefault(i, []).append((j, v))
                    by_col.setdefault(j, []).append((i, v))
            row_keep: Dict[int, set] = {}
            for i, lst in by_row.items():
                lst.sort(key=lambda x: -x[1])
                row_keep[i] = set([j for j, _ in lst[:topk]])
            col_keep: Dict[int, set] = {}
            for j, lst in by_col.items():
                lst.sort(key=lambda x: -x[1])
                col_keep[j] = set([i for i, _ in lst[:topk]])

            ii: List[int] = []
            jj: List[int] = []
            pp: List[float] = []
            for (i, j), v in new_scores.items():
                if v < pmin:
                    continue
                if i in row_keep and j in row_keep[i] and j in col_keep and i in col_keep[j]:
                    ii.append(i)
                    jj.append(j)
                    pp.append(v)
            out[(a, b)] = SparsePairs(i=np.array(ii, dtype=np.int32), j=np.array(jj, dtype=np.int32), p=np.array(pp, dtype=np.float32), shape=(La, Lb))

    return out


def build_distance_matrix_from_posteriors(sparse_lib: Dict[Tuple[int, int], SparsePairs], N: int) -> np.ndarray:
    D = np.zeros((N, N), dtype=np.float32)
    for (a, b), sp in sparse_lib.items():
        if sp.p.size == 0:
            d = 1.0
        else:
            # 1 - mean posterior of kept matches as a proxy distance
            d = 1.0 - float(sp.p.mean())
        D[a, b] = D[b, a] = max(0.0, min(1.0, d))
    return D


# ========================
# Guide tree construction
# ========================

def build_guide_tree(D: np.ndarray, method: str = "nj") -> GuideTree:
    N = D.shape[0]
    # Implement simple UPGMA and NJ. These are unrooted in theory; we return a binary merge structure.
    # Indices 0..N-1 are leaves.
    if N == 1:
        return GuideTree(structure=0)

    # Represent clusters as ints for leaves, and tuples for internal merges
    clusters: Dict[int, Any] = {i: i for i in range(N)}
    sizes: Dict[int, int] = {i: 1 for i in range(N)}
    # Distance map keyed by (a,b) with a<b
    dist: Dict[Tuple[int, int], float] = {}
    for i in range(N):
        for j in range(i + 1, N):
            dist[(i, j)] = float(D[i, j])

    next_id = N  # internal cluster ids
    active = set(range(N))

    def get_d(a: int, b: int) -> float:
        if a == b:
            return 0.0
        x, y = (a, b) if a < b else (b, a)
        return dist[(x, y)]

    if method == 'upgma':
        while len(active) > 1:
            # find min distance pair
            best = None
            best_d = 1e9
            act_list = sorted(active)
            for i_idx in range(len(act_list)):
                for j_idx in range(i_idx + 1, len(act_list)):
                    a, b = act_list[i_idx], act_list[j_idx]
                    d = get_d(a, b)
                    if d < best_d:
                        best_d = d
                        best = (a, b)
            assert best is not None
            a, b = best
            # merge
            new = next_id
            next_id += 1
            clusters[new] = (clusters[a], clusters[b])
            sa, sb = sizes[a], sizes[b]
            sizes[new] = sa + sb

            # update distances
            for c in list(active):
                if c in (a, b):
                    continue
                dc = (get_d(a, c) * sa + get_d(b, c) * sb) / (sa + sb)
                x, y = (c, new) if c < new else (new, c)
                dist[(x, y)] = dc
            active.remove(a)
            active.remove(b)
            active.add(new)
        root = next(iter(active))
        return GuideTree(structure=clusters[root])
    else:
        # Neighbor-Joining
        active = set(range(N))
        rsum: Dict[int, float] = {i: 0.0 for i in active}
        while len(active) > 2:
            act_list = sorted(active)
            m = len(act_list)
            # compute r[i] = sum d(i,k)
            for i in act_list:
                s = 0.0
                for k in act_list:
                    if k != i:
                        s += get_d(i, k)
                rsum[i] = s
            # compute Q matrix and find min
            best = None
            best_q = 1e9
            for a_idx in range(len(act_list)):
                for b_idx in range(a_idx + 1, len(act_list)):
                    a, b = act_list[a_idx], act_list[b_idx]
                    q = (m - 2) * get_d(a, b) - rsum[a] - rsum[b]
                    if q < best_q:
                        best_q = q
                        best = (a, b)
            assert best is not None
            a, b = best
            # merge a,b into new node u
            u = next_id
            next_id += 1
            clusters[u] = (clusters[a], clusters[b])
            sizes[u] = sizes[a] + sizes[b]
            # update distances to other nodes: d(u,k) = (d(a,k)+d(b,k)-d(a,b))/2
            dab = get_d(a, b)
            for k in list(active):
                if k in (a, b):
                    continue
                duk = (get_d(a, k) + get_d(b, k) - dab) / 2.0
                x, y = (k, u) if k < u else (u, k)
                dist[(x, y)] = duk
            active.remove(a)
            active.remove(b)
            active.add(u)
        # join last two
        a, b = sorted(active)
        root = next_id
        clusters[root] = (clusters[a], clusters[b])
        return GuideTree(structure=clusters[root])


# =====================================
# Profiles and profile–profile alignment
# =====================================

def _column_from_indices(cols_chars: List[str], emb: np.ndarray, paired: Optional[List[int]]) -> ProfileColumn:
    # Build column embedding as mean of non-gap embeddings then L2 normalize
    idxs = [k for k, c in enumerate(cols_chars) if c != '-']
    if len(idxs) == 0:
        # Gap-only column: zero vector; handled in scoring as 0 dot
        mu = np.zeros(emb.shape[1], dtype=np.float32)
        stem_fraction = 0.0
        return ProfileColumn(mu=mu, stem_fraction=stem_fraction)
    vecs = emb[idxs]
    mu = vecs.mean(axis=0)
    nrm = np.linalg.norm(mu) + 1e-8
    mu = (mu / nrm).astype(np.float32)
    # Estimate stem fraction if pairing available
    if paired is not None:
        n_stem = sum(1 for i in idxs if paired[i] != -1)
        stem_fraction = n_stem / float(len(idxs))
    else:
        stem_fraction = 0.0
    return ProfileColumn(mu=mu, stem_fraction=stem_fraction)


def _initial_profiles(records: List[SequenceRecord]) -> List[Profile]:
    profiles: List[Profile] = []
    for idx, r in enumerate(records):
        L = r.emb.shape[0]
        # Start with each sequence as its own profile: one char per column
        aligned = {idx: ["X"] * L}  # placeholder char; we output gaps in final strings only
        cols = []
        for pos in range(L):
            cols.append(ProfileColumn(mu=r.emb[pos], stem_fraction=(1.0 if (r.paired_idx and r.paired_idx[pos] != -1) else 0.0)))
        profiles.append(Profile(columns=cols, member_indices=[idx], aligned_chars=aligned))
    return profiles


def _compatibility_score(ca: ProfileColumn, cb: ProfileColumn, beta_struct: float = 0.2) -> float:
    comp = 1.0 if ((ca.stem_fraction >= 0.5) and (cb.stem_fraction >= 0.5)) or ((ca.stem_fraction < 0.5) and (cb.stem_fraction < 0.5)) else 0.0
    return float(beta_struct * comp)


@njit(cache=True)
def _affine_dp_profile(muA: np.ndarray, muB: np.ndarray, stemA: np.ndarray, stemB: np.ndarray,
                       go: float, ge: float, stem_bonus: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # DP over columns; returns traceback pointers for merging
    La = muA.shape[0]
    Lb = muB.shape[0]
    NEG = -1e30
    M = np.full((La + 1, Lb + 1), NEG, dtype=np.float32)
    X = np.full((La + 1, Lb + 1), NEG, dtype=np.float32)
    Y = np.full((La + 1, Lb + 1), NEG, dtype=np.float32)
    tb = np.zeros((La + 1, Lb + 1), dtype=np.int8)  # 0:diag(M), 1:up(X), 2:left(Y)

    M[0, 0] = 0.0
    for i in range(1, La + 1):
        mod = stem_bonus if stemA[i - 1] >= 0.5 else 0.0
        X[i, 0] = max(M[i - 1, 0] + go + mod, X[i - 1, 0] + ge)
    for j in range(1, Lb + 1):
        mod = stem_bonus if stemB[j - 1] >= 0.5 else 0.0
        Y[0, j] = max(M[0, j - 1] + go + mod, Y[0, j - 1] + ge)

    for i in range(1, La + 1):
        for j in range(1, Lb + 1):
            s = 0.0
            # dot product
            for d in range(muA.shape[1]):
                s += muA[i - 1, d] * muB[j - 1, d]
            # structural compatibility bonus
            comp = 0.2 if ((stemA[i - 1] >= 0.5 and stemB[j - 1] >= 0.5) or (stemA[i - 1] < 0.5 and stemB[j - 1] < 0.5)) else 0.0

            m_val = M[i - 1, j - 1]
            x_val = X[i - 1, j - 1]
            y_val = Y[i - 1, j - 1]
            best_prev = m_val
            tb_code = 0
            if x_val > best_prev:
                best_prev = x_val
                tb_code = 0  # merge as match regardless
            if y_val > best_prev:
                best_prev = y_val
                tb_code = 0
            M[i, j] = best_prev + s + comp

            modA = stem_bonus if stemA[i - 1] >= 0.5 else 0.0
            open_x = M[i - 1, j] + go + modA
            ext_x = X[i - 1, j] + ge
            X[i, j] = open_x if open_x > ext_x else ext_x

            modB = stem_bonus if stemB[j - 1] >= 0.5 else 0.0
            open_y = M[i, j - 1] + go + modB
            ext_y = Y[i, j - 1] + ge
            Y[i, j] = open_y if open_y > ext_y else ext_y

            # Choose best state to proceed (for traceback selection)
            if M[i, j] >= X[i, j] and M[i, j] >= Y[i, j]:
                tb[i, j] = 0
            elif X[i, j] >= Y[i, j]:
                tb[i, j] = 1
            else:
                tb[i, j] = 2

    return M, X, Y


def profile_profile_dp(profileA: Profile, profileB: Profile,
                       gap_open: float, gap_extend: float, stem_gap_bonus: float) -> Profile:
    La = len(profileA.columns)
    Lb = len(profileB.columns)
    D = profileA.columns[0].mu.shape[0] if La > 0 else profileB.columns[0].mu.shape[0]
    muA = np.stack([c.mu for c in profileA.columns], axis=0).astype(np.float32)
    muB = np.stack([c.mu for c in profileB.columns], axis=0).astype(np.float32)
    stemA = np.array([c.stem_fraction for c in profileA.columns], dtype=np.float32)
    stemB = np.array([c.stem_fraction for c in profileB.columns], dtype=np.float32)

    M, X, Y = _affine_dp_profile(muA, muB, stemA, stemB, gap_open, gap_extend, stem_gap_bonus)

    # Traceback
    i, j = La, Lb
    # Recompute best-state selection at each step consistently with DP values
    aligned_cols: List[ProfileColumn] = []
    # Prepare aligned chars
    new_members = profileA.member_indices + profileB.member_indices
    new_aligned: Dict[int, List[str]] = {idx: [] for idx in new_members}

    while i > 0 or j > 0:
        # Select state by max of M/X/Y
        cur_state = 0
        cur_val = -1e30
        if i > 0 and j > 0 and M[i, j] > cur_val:
            cur_val = M[i, j]
            cur_state = 0
        if i > 0 and X[i, j] > cur_val:
            cur_val = X[i, j]
            cur_state = 1
        if j > 0 and Y[i, j] > cur_val:
            cur_val = Y[i, j]
            cur_state = 2

        if cur_state == 0:
            # Match columns i-1 and j-1
            # Merge columns
            mu = profileA.columns[i - 1].mu + profileB.columns[j - 1].mu
            nrm = np.linalg.norm(mu) + 1e-8
            mu = (mu / nrm).astype(np.float32)
            stem_fraction = (profileA.columns[i - 1].stem_fraction + profileB.columns[j - 1].stem_fraction) / 2.0
            aligned_cols.append(ProfileColumn(mu=mu, stem_fraction=float(stem_fraction)))
            # Propagate chars
            for idx in profileA.member_indices:
                new_aligned[idx].append('X')
            for idx in profileB.member_indices:
                new_aligned[idx].append('X')
            i -= 1
            j -= 1
        elif cur_state == 1 and i > 0:
            # Gap in B; take A column alone
            mu = profileA.columns[i - 1].mu.copy()
            stem_fraction = profileA.columns[i - 1].stem_fraction
            aligned_cols.append(ProfileColumn(mu=mu, stem_fraction=float(stem_fraction)))
            for idx in profileA.member_indices:
                new_aligned[idx].append('X')
            for idx in profileB.member_indices:
                new_aligned[idx].append('-')
            i -= 1
        else:
            # Gap in A; take B column alone
            mu = profileB.columns[j - 1].mu.copy()
            stem_fraction = profileB.columns[j - 1].stem_fraction
            aligned_cols.append(ProfileColumn(mu=mu, stem_fraction=float(stem_fraction)))
            for idx in profileA.member_indices:
                new_aligned[idx].append('-')
            for idx in profileB.member_indices:
                new_aligned[idx].append('X')
            j -= 1

    aligned_cols.reverse()
    for k in new_aligned.keys():
        new_aligned[k].reverse()

    return Profile(columns=aligned_cols, member_indices=new_members, aligned_chars=new_aligned)


def msa_from_tree(tree: GuideTree, seq_profiles: List[Profile],
                  gap_open: float, gap_extend: float, stem_gap_bonus: float) -> Profile:
    # Recursively traverse the tree combining profiles
    def build(node: Any) -> Profile:
        if isinstance(node, int):
            return seq_profiles[node]
        left = build(node[0])
        right = build(node[1])
        return profile_profile_dp(left, right, gap_open, gap_extend, stem_gap_bonus)

    return build(tree.structure)


# ======================
# Iterative refinement
# ======================

def _sp_score(profile: Profile, beta_struct: float = 0.2) -> float:
    # Sum-of-pairs over columns, dot(mu) + structure bonus for non-gap matches
    members = profile.member_indices
    score = 0.0
    for col in profile.columns:
        mu = col.mu
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                # approximate: use mu dot mu as a proxy for similarity contributed by this column
                score += float(mu @ mu + (beta_struct if col.stem_fraction >= 0.5 else 0.0))
    return score


def iterative_refinement(aln: Profile, tree: GuideTree, params: Dict[str, Any]) -> Profile:
    iters = int(params.get('refine_iters', 0))
    if iters <= 0:
        return aln
    best = aln
    best_score = _sp_score(aln)
    random.seed(int(params.get('seed', 42)))
    for _ in range(iters):
        # This is a lightweight placeholder: in practice we would split and realign.
        # We simulate a minor shuffle and keep if score improves (no-op mostly).
        cand = best  # No change for now; refinement hook present
        sc = _sp_score(cand)
        if sc > best_score:
            best = cand
            best_score = sc
    return best


# ======================
# Output and diagnostics
# ======================

def _profile_to_msa_strings(profile: Profile, names: List[str]) -> Dict[str, str]:
    # Convert aligned_chars to strings of '-' and 'X'
    aln_len = len(profile.columns)
    out: Dict[str, str] = {}
    for idx in profile.member_indices:
        chars = profile.aligned_chars[idx]
        if len(chars) != aln_len:
            # pad if needed
            if len(chars) < aln_len:
                chars = chars + ['-'] * (aln_len - len(chars))
            else:
                chars = chars[:aln_len]
        out[names[idx]] = ''.join(chars)
    return out


def write_outputs(aln: Profile, names: List[str], out_prefix: str, diagnostics: Dict[str, Any]) -> None:
    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    msa = _profile_to_msa_strings(aln, names)
    # FASTA
    fasta_path = f"{out_prefix}.fasta"
    with open(fasta_path, 'w') as f:
        for n in names:
            if n in msa:
                f.write(f">{n}\n{msa[n]}\n")
    # Stockholm
    sto_path = f"{out_prefix}.sto"
    with open(sto_path, 'w') as f:
        f.write("# STOCKHOLM 1.0\n")
        for n in names:
            if n in msa:
                f.write(f"{n} {msa[n]}\n")
        f.write("//\n")
    # TSV mapping
    tsv_path = f"{out_prefix}.aln.tsv"
    df = pd.DataFrame({"Name": list(msa.keys()), "Aligned": list(msa.values())})
    df.to_csv(tsv_path, sep='\t', index=False)

    # Diagnostics
    diag_dir = f"{out_prefix}.diagnostics"
    os.makedirs(diag_dir, exist_ok=True)
    # Expected scores matrix
    if 'expected_scores' in diagnostics:
        es = diagnostics['expected_scores']
        es_path = os.path.join(diag_dir, "expected_scores.tsv")
        pd.DataFrame(es).to_csv(es_path, sep='\t', header=False, index=False)
    # Optional posterior heatmaps
    if 'posteriors_heatmaps' in diagnostics and diagnostics['posteriors_heatmaps']:
        if MPL_AVAILABLE:
            for k, (pair, sp) in enumerate(diagnostics['posteriors_heatmaps']):
                if k >= 6:
                    break
                La, Lb = sp.shape
                mat = np.zeros((La, Lb), dtype=np.float32)
                for t in range(sp.i.shape[0]):
                    mat[int(sp.i[t]), int(sp.j[t])] = float(sp.p[t])
                plt.figure(figsize=(4, 4))
                plt.imshow(mat, origin='lower', aspect='auto', cmap='viridis')
                plt.colorbar()
                plt.title(f"Pair {pair[0]}-{pair[1]}")
                plt.tight_layout()
                plt.savefig(os.path.join(diag_dir, f"pair_{pair[0]}_{pair[1]}.png"))
                plt.close()
    # Save run meta
    meta = {k: v for k, v in diagnostics.items() if k != 'posteriors_heatmaps'}
    meta_path = os.path.join(diag_dir, "run_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)


# ======================
# Main orchestration
# ======================

def main():
    ap = argparse.ArgumentParser(description="MSA for RNAs using node embeddings (T-Coffee/ProbCons-style)")
    ap.add_argument('--input', required=True, help='Input TSV path or "dummy"')
    ap.add_argument('--name-col', default='Name')
    ap.add_argument('--embeds-col', default='node_embeddings')
    ap.add_argument('--dotbracket-col', default=None)
    ap.add_argument('--paired-col', default=None)
    ap.add_argument('--out-prefix', required=True)
    ap.add_argument('--topk', type=int, default=20)
    ap.add_argument('--consistency-rounds', type=int, default=1)
    ap.add_argument('--alpha', type=float, default=None)
    ap.add_argument('--beta', type=float, default=None)
    ap.add_argument('--gap-open', type=float, default=-10.0)
    ap.add_argument('--gap-extend', type=float, default=-0.5)
    ap.add_argument('--stem-gap-bonus', type=float, default=-2.0)
    ap.add_argument('--use-local', action='store_true')
    ap.add_argument('--tree', choices=['nj', 'upgma'], default='nj')
    ap.add_argument('--refine-iters', type=int, default=0)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--max-pairs', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--plot-diagnostics', action='store_true')

    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    t_start = time.time()

    # Dummy mode
    if args.input == 'dummy':
        # Generate 5 toy sequences with small lengths
        N = 5
        D = 16
        records: List[SequenceRecord] = []
        for i in range(N):
            L = random.randint(6, 10)
            emb = np.random.randn(L, D).astype(np.float32)
            records.append(SequenceRecord(name=f"seq{i+1}", emb=emb))
        out_prefix = args.out_prefix if args.out_prefix else "/tmp/embed_msa_dummy"
    else:
        records = load_tsv(args.input, args.name_col, args.embeds_col, args.dotbracket_col, args.paired_col)
        out_prefix = args.out_prefix
        if len(records) == 0:
            raise SystemExit("No valid records found.")

    # Normalize
    l2_normalize_embeddings(records)

    N = len(records)
    names = [r.name for r in records]
    dims = [r.emb.shape[1] for r in records]
    if len(set(dims)) != 1:
        raise SystemExit("All embeddings must have the same dimension.")

    # Alpha/beta defaults
    alpha = args.alpha if args.alpha is not None else 5.0
    beta = args.beta if args.beta is not None else 0.0
    if args.alpha is None or args.beta is None:
        print("[WARN] alpha/beta not fully provided; falling back to default alpha=5.0, beta=0.0")

    # Pair selection
    pairs = pairwise_pairs_to_compute(records, args.max_pairs, threads=args.num_workers)
    print(f"Computing pairwise posteriors for {len(pairs)} pairs...")

    sparse_lib: Dict[Tuple[int, int], SparsePairs] = {}
    expected_scores = np.zeros((N, N), dtype=np.float32)
    heatmaps: List[Tuple[Tuple[int, int], SparsePairs]] = []

    # Parallel worker
    def _compute_pair(a: int, b: int) -> Tuple[Tuple[int, int], SparsePairs, float, Optional[SparsePairs]]:
        Ea, Eb = records[a].emb, records[b].emb
        S = cosine_similarity_matrix(Ea, Eb)
        L = calibrate_log_odds(S, alpha, beta)
        mode = 'local' if args.use_local else 'global'
        P = forward_backward_affine_logspace(L, args.gap_open, args.gap_extend, mode=mode)
        sp = sparsify_posteriors(P, args.topk, pmin=1e-4)
        if sp.p.size > 0:
            vals = [float(S[int(sp.i[k]), int(sp.j[k])]) * float(sp.p[k]) for k in range(sp.p.shape[0])]
            E = float(np.sum(np.array(vals, dtype=np.float32)))
        else:
            E = 0.0
        return (a, b), sp, E, sp if args.plot_diagnostics else None

    # Threaded loop
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
            futs = [ex.submit(_compute_pair, a, b) for (a, b) in pairs]
            for fut in tqdm(as_completed(futs), total=len(futs), desc='Pairwise', unit='pair'):
                (a, b), sp, E, sp_heat = fut.result()
                sparse_lib[(a, b)] = sp
                expected_scores[a, b] = expected_scores[b, a] = E
                if sp_heat is not None and len(heatmaps) < 6:
                    heatmaps.append(((a, b), sp_heat))
    except Exception:
        # Fallback to serial
        for (a, b) in tqdm(pairs, desc='Pairwise', unit='pair'):
            (a, b), sp, E, sp_heat = _compute_pair(a, b)
            sparse_lib[(a, b)] = sp
            expected_scores[a, b] = expected_scores[b, a] = E
            if sp_heat is not None and len(heatmaps) < 6:
                heatmaps.append(((a, b), sp_heat))

    # Consistency rounds
    if N >= 3 and args.consistency_rounds > 0:
        print(f"Running {args.consistency_rounds} consistency round(s)...")
        for _ in range(args.consistency_rounds):
            sparse_lib = consistency_round(sparse_lib, records, neighbors=None, lam=0.5, topk=args.topk, pmin=1e-4)

    # Guide tree
    D = build_distance_matrix_from_posteriors(sparse_lib, N)
    tree = build_guide_tree(D, method=args.tree)

    # Progressive alignment
    seq_profiles = _initial_profiles(records)
    aln = msa_from_tree(tree, seq_profiles, args.gap_open, args.gap_extend, args.stem_gap_bonus)

    # Iterative refinement
    if args.refine_iters > 0:
        aln = iterative_refinement(aln, tree, {"refine_iters": args.refine_iters, "seed": args.seed})

    # Outputs
    diagnostics: Dict[str, Any] = {
        'expected_scores': expected_scores.tolist(),
        'num_pairs': len(pairs),
        'N': N,
        'alpha': alpha,
        'beta': beta,
        'timing_sec': time.time() - t_start,
    }
    if args.plot_diagnostics:
        diagnostics['posteriors_heatmaps'] = heatmaps
    write_outputs(aln, names, out_prefix, diagnostics)

    print(f"Done. Outputs written to: {out_prefix}.*")


if __name__ == '__main__':
    main()
