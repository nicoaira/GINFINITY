#!/usr/bin/env python3
"""
Optimize embed_msa.py hyperparameters with Optuna.

This script launches multiple trials in parallel, each running
`src/ginfinity/scripts/embed_msa.py` with a specific parameter set.
The objective scores how well the MSA aligns the specified regions of
two RNAs (hsa-mir-103a-1 and hsa-mir-103a-2), as described by a TSV
file with 1-indexed Start/End coordinates.

Scoring: For k in [Start1..End1] and corresponding k' in [Start2..End2]
with k' aligned to k via a 1-offset in the example provided, the
objective adds +1 if the two bases are aligned in the same MSA column,
otherwise -1.

Results are stored in a SQLite Optuna storage for later exploration.

Example:
  ginfinity-optimize-msa \
    --threads 16 \
    --n-trials 100 \
    --storage sqlite:///optuna_embed_msa.db
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import pandas as pd


def parse_regions_tsv(path: Path) -> Dict[str, Tuple[int, int]]:
    regions: Dict[str, Tuple[int, int]] = {}
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split()  # tolerate spaces or tabs
        # Basic validation
        if len(header) < 3 or header[0] != "Name":
            # Try tab-separated fallback
            header = f.readline().strip().split("\t")
            if len(header) < 3 or header[0] != "Name":
                raise ValueError("Invalid regions TSV header. Expect columns: Name\tStart\tEnd")
        else:
            # rewind one line
            f.seek(0)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                parts = line.split("\t")
            if len(parts) < 3:
                continue
            name, s, e = parts[0], parts[1], parts[2]
            try:
                start = int(s)
                end = int(e)
            except Exception:
                continue
            regions[name] = (start, end)
    if not regions:
        raise ValueError(f"No regions parsed from {path}")
    return regions


def load_aln_tsv(path: Path) -> Dict[str, str]:
    # Expect columns: Name, Aligned
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        if len(header) < 2:
            raise ValueError(f"Malformed alignment TSV: {path}")
        name_idx = header.index("Name") if "Name" in header else 0
        aligned_idx = header.index("Aligned") if "Aligned" in header else 1
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) <= max(name_idx, aligned_idx):
                continue
            out[parts[name_idx]] = parts[aligned_idx]
    return out


def build_pos_to_col_map(aligned: str) -> Dict[int, int]:
    # Map 1-indexed base positions to MSA column indices (0-based)
    pos_to_col: Dict[int, int] = {}
    pos = 0
    for col, ch in enumerate(aligned):
        if ch != '-':
            pos += 1
            pos_to_col[pos] = col
    return pos_to_col


def compute_score(msa: Dict[str, str], regions: Dict[str, Tuple[int, int]],
                  name1: str, name2: str) -> int:
    if name1 not in msa or name2 not in msa:
        raise ValueError(f"Required names not in MSA: {name1}, {name2}")
    if name1 not in regions or name2 not in regions:
        raise ValueError(f"Regions missing for: {name1} or {name2}")
    a1, a2 = regions[name1]
    b1, b2 = regions[name2]
    if (a2 - a1) != (b2 - b1):
        # Lengths must match to have 1-to-1 mapping across intervals
        raise ValueError("Region lengths differ between the two RNAs")

    s_aln = msa[name1]
    t_aln = msa[name2]
    if len(s_aln) != len(t_aln):
        raise ValueError("MSA row lengths differ; invalid alignment TSV")

    s_map = build_pos_to_col_map(s_aln)
    t_map = build_pos_to_col_map(t_aln)

    score = 0
    # 1-indexed positions; expected mapping is (a1+k) aligned with (b1+k)
    L = (a2 - a1)
    for k in range(0, L + 1):
        p1 = a1 + k
        p2 = b1 + k
        c1 = s_map.get(p1, None)
        c2 = t_map.get(p2, None)
        if c1 is not None and c2 is not None and c1 == c2:
            score += 1
        else:
            score -= 1
    return score


def make_embed_args(args: argparse.Namespace, out_prefix: Path,
                    refine_iters: int, alpha: float, beta: float,
                    gap_open: float, gap_extend: float) -> List[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "embed_msa.py"),
        "--input", args.input,
        "--name-col", args.name_col,
        "--embeds-col", args.embeds_col,
        "--dotbracket-col", args.dotbracket_col,
        "--paired-col", args.paired_col,
        "--topk", str(args.topk),
        "--consistency-rounds", str(args.consistency_rounds),
        "--use-local",
        "--tree", args.tree,
        "--num-workers", "1",
        "--max-pairs", str(args.max_pairs),
        "--seed", str(args.seed),
        "--out-prefix", str(out_prefix),
        "--refine-iters", str(refine_iters),
        "--alpha", str(alpha),
        "--beta", str(beta),
        "--gap-open", str(gap_open),
        "--gap-extend", str(gap_extend),
    ]
    if args.plot_diagnostics:
        cmd.append("--plot-diagnostics")
    return cmd


def _pairs_to_dotbracket(pairs: List[int]) -> str:
    L = len(pairs)
    out = ["."] * L
    for i in range(L):
        j = pairs[i]
        if j == -1:
            out[i] = '.'
        elif j > i:
            out[i] = '('
        else:
            out[i] = ')'
    return ''.join(out)


def _parse_json_list(s: str) -> Optional[List[int]]:
    try:
        v = json.loads(s)
        if isinstance(v, list):
            out = []
            for x in v:
                try:
                    out.append(int(x))
                except Exception:
                    return None
            return out
    except Exception:
        return None
    return None


def load_dotbrackets(input_path: Path, name_col: str, dotbracket_col: str, paired_col: str) -> Dict[str, Optional[str]]:
    df = pd.read_csv(input_path, sep='\t')
    have_db = dotbracket_col in df.columns
    have_pair = paired_col in df.columns
    result: Dict[str, Optional[str]] = {}
    for _, row in df.iterrows():
        name = str(row[name_col])
        db: Optional[str] = None
        if have_db:
            val = row[dotbracket_col]
            if isinstance(val, str) and len(val) > 0 and not pd.isna(val):
                db = val
        if db is None and have_pair:
            val = row[paired_col]
            if isinstance(val, str) and len(val) > 0 and not pd.isna(val):
                lst = _parse_json_list(val)
                if lst is not None:
                    db = _pairs_to_dotbracket(lst)
        result[name] = db
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Optuna optimization for embed_msa.py")
    # Fixed embed_msa defaults requested by the user
    ap.add_argument("--input", default="/home/nicolas/ginfinity_applications/premirnas/premirans_node_embeds.tsv")
    ap.add_argument("--name-col", default="Name")
    ap.add_argument("--embeds-col", default="node_embeddings")
    ap.add_argument("--dotbracket-col", default="DotBracket")
    ap.add_argument("--paired-col", default="PairedIndices")
    ap.add_argument("--topk", type=int, default=80)
    ap.add_argument("--consistency-rounds", type=int, default=30)
    ap.add_argument("--tree", choices=["nj", "upgma"], default="nj")
    ap.add_argument("--max-pairs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot-diagnostics", action="store_true", default=True)

    # Optimization controls
    ap.add_argument("--threads", type=int, default=1, help="Parallel trials to run")
    ap.add_argument("--n-trials", type=int, default=50, help="Total trials to run")
    ap.add_argument("--storage", default="sqlite:///optuna_embed_msa.db", help="Optuna storage URL (sqlite:///<path>.db)")
    ap.add_argument("--study-name", default="embed_msa_opt", help="Optuna study name")
    ap.add_argument("--regions-tsv", default=str(Path.cwd() / "hsa-mir-103a-regions.tsv"), help="TSV with Name, Start, End")
    ap.add_argument("--outdir", default=str(Path.cwd() / "output" / "optuna_embed_msa"), help="Base directory for trial outputs")
    # Keep outputs by default; allow disabling with --discard-outputs
    ap.add_argument("--keep-outputs", dest="keep_outputs", action="store_true", default=True, help="Keep per-trial outputs on disk (default: True)")
    ap.add_argument("--discard-outputs", dest="keep_outputs", action="store_false", help="Discard per-trial outputs after scoring")
    # Save a summary CSV/JSON for all trials
    ap.add_argument("--save-summary", action="store_true", default=True, help="Write trials.csv and best_params.json to study directory")

    # Names to evaluate
    ap.add_argument("--name-a", default="hsa-mir-103a-1")
    ap.add_argument("--name-b", default="hsa-mir-103a-2")

    args = ap.parse_args()

    try:
        import optuna
    except Exception as e:  # pragma: no cover
        raise SystemExit("Optuna is required. Install with: pip install optuna>=3.6")

    regions_path = Path(args.regions_tsv).resolve()
    regions = parse_regions_tsv(regions_path)

    outbase = Path(args.outdir).resolve()
    outbase.mkdir(parents=True, exist_ok=True)

    # Print dot-bracket slices for the two RNAs used in optimization
    try:
        input_path = Path(args.input).resolve()
        db_map = load_dotbrackets(input_path, args.name_col, args.dotbracket_col, args.paired_col)
        for nm in (args.name_a, args.name_b):
            if nm not in db_map or db_map[nm] is None:
                print(f"[INFO] DotBracket not available for {nm} — cannot print slice.")
                continue
            if nm not in regions:
                print(f"[WARN] Regions not found for {nm} in {regions_path}")
                continue
            start, end = regions[nm]
            db = db_map[nm]
            # Convert to 0-based slice; Python end is exclusive
            s = max(1, start)
            e = max(s, end)
            sl = db[s-1:e]
            print(f"[REGION] {nm} {s}-{e} (len {len(sl)}):")
            print(sl)
    except Exception as e:
        print(f"[WARN] Failed to print dot-bracket slices: {e}")

    # Objective
    def objective(trial: "optuna.trial.Trial") -> float:
        refine_iters = trial.suggest_categorical("refine_iters", [4, 16, 32, 64, 128])
        alpha = trial.suggest_float("alpha", 1.0, 12.0)
        beta = trial.suggest_float("beta", -2.0, 1.0)
        gap_open = trial.suggest_float("gap_open", -5.0, -1.0)
        gap_extend = trial.suggest_float("gap_extend", -5.0, -1.0)

        # Unique output prefix per trial
        tdir = outbase / f"{args.study_name}" / f"trial_{trial.number}_{uuid.uuid4().hex[:8]}"
        out_prefix = tdir / "msa"
        tdir.mkdir(parents=True, exist_ok=True)

        cmd = make_embed_args(args, out_prefix, refine_iters, alpha, beta, gap_open, gap_extend)

        start = time.time()
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
            retcode = proc.returncode
        except Exception as e:
            trial.set_user_attr("error", f"spawn_failed: {e}")
            if not args.keep_outputs:
                shutil.rmtree(tdir, ignore_errors=True)
            return -1e9

        # Persist raw stdout/stderr for the trial
        try:
            (tdir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
            (tdir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
        except Exception:
            pass

        trial.set_user_attr("outdir", str(tdir))
        trial.set_user_attr("stdout_tail", proc.stdout[-500:])
        trial.set_user_attr("stderr_tail", proc.stderr[-500:])

        if retcode != 0:
            trial.set_user_attr("error", f"embed_msa_failed: code={retcode}")
            if not args.keep_outputs:
                shutil.rmtree(tdir, ignore_errors=True)
            return -1e9

        aln_path = Path(str(out_prefix) + ".aln.tsv")
        try:
            msa = load_aln_tsv(aln_path)
            score = compute_score(msa, regions, args.name_a, args.name_b)
        except Exception as e:
            trial.set_user_attr("error", f"scoring_failed: {e}")
            if not args.keep_outputs:
                shutil.rmtree(tdir, ignore_errors=True)
            return -1e9
        finally:
            trial.set_user_attr("elapsed_sec", round(time.time() - start, 3))

        # Save a manifest for reproducibility
        try:
            manifest = {
                "trial_number": trial.number,
                "params": {
                    "refine_iters": refine_iters,
                    "alpha": alpha,
                    "beta": beta,
                    "gap_open": gap_open,
                    "gap_extend": gap_extend,
                },
                "command": cmd,
                "elapsed_sec": round(time.time() - start, 3),
                "score": float(score),
            }
            (tdir / "trial_meta.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        except Exception:
            pass

        if not args.keep_outputs:
            shutil.rmtree(tdir, ignore_errors=True)

        # Maximize correct alignments
        return float(score)

    # Create study with SQLite storage for parallel execution
    study = optuna.create_study(direction="maximize", study_name=args.study_name, storage=args.storage, load_if_exists=True)
    study.optimize(objective, n_trials=args.n_trials, n_jobs=max(1, args.threads))

    best = study.best_trial
    print("Best value:", best.value)
    print("Best params:")
    print(json.dumps(best.params, indent=2))
    print("Storage:", args.storage)
    print("Study:", args.study_name)

    # Save trials summary and best params under the study directory
    if args.save_summary:
        try:
            study_dir = outbase / args.study_name
            study_dir.mkdir(parents=True, exist_ok=True)
            # DataFrame of all trials (including user attrs)
            df = study.trials_dataframe(attr_keys=("outdir", "stdout_tail", "stderr_tail", "error", "elapsed_sec"))
            df.to_csv(study_dir / "trials.csv", index=False)
            # Best params as JSON for quick lookup
            (study_dir / "best_params.json").write_text(json.dumps(best.params, indent=2), encoding="utf-8")
            # Also write a small README
            (study_dir / "README.txt").write_text(
                "This directory contains Optuna study summaries and per-trial outputs (if kept).\n"
                "- trials.csv: all trials with params, values, states, and user attrs\n"
                "- best_params.json: parameter set of the best trial\n",
                encoding="utf-8",
            )
        except Exception as e:
            print(f"[WARN] Failed to write trials summary: {e}")


if __name__ == "__main__":
    main()
