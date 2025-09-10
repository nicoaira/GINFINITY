#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Launch pairwise alignments for all names in a TSV using align_node_embeddings.py.

Default behavior:
  - Uses all unique unordered pairs (A,B) with A != B (i.e., no duplicates like (B,A))
  - No self-comparisons
  - Parallel execution with configurable number of jobs
  - Output prefixes like: <output_dir>/<rna1>_vs_<rna2>

You can change behavior with flags:
  --ordered                Use ordered pairs (A,B) and (B,A)
  --include-self           Include self-pairs (A,A)
  --jobs                   Parallel workers (default: CPU count)
  --dry-run                Print commands but do not execute
  --strip-prefix           Optional string to strip from rna names for output prefix only

Example:
  python run_all_pairwise_alignments.py \
    --input /home/nicolas/ginfinity_applications/premirnas/premirans_node_embeds.tsv \
    --id-column Name \
    --align-script src/ginfinity/scripts/align_node_embeddings.py \
    --structures-path /home/nicolas/ginfinity_applications/premirnas/merged_premirna_data.csv \
    --dotbracket-column DotBracket \
    --structures-id-column Name \
    --gap -0.25 \
    --mode local \
    --output-dir output

This will generate commands like:
  python src/ginfinity/scripts/align_node_embeddings.py \
    --input <tsv> --id-column Name --rna1 <A> --rna2 <B> --gap -0.25 --mode local \
    --output-prefix output/<A>_vs_<B> \
    --structures-path <csv> --dotbracket-column DotBracket --structures-id-colum Name
"""

import os
import sys
import csv
import argparse
import itertools
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

def read_names_from_tsv(tsv_path: str, id_column: str) -> list[str]:
    # Increase CSV field size in case some columns contain huge blobs
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(10**9)

    names = []
    with open(tsv_path, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{tsv_path} appears to be empty.")
        try:
            idx = header.index(id_column)
        except ValueError:
            raise ValueError(
                f"Column '{id_column}' not found in header. Columns: {header}"
            )
        for row in reader:
            if not row:
                continue
            if idx >= len(row):
                continue
            name = row[idx].strip()
            if name:
                names.append(name)
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq

def sanitize_for_path(s: str) -> str:
    # Avoid path-breaking characters; hyphens are fine
    return "".join(ch if ch not in r'\/:*?"<>|' else "_" for ch in s)

def build_output_prefix(out_dir: str, rna1: str, rna2: str, strip_prefix: str | None) -> str:
    r1 = rna1
    r2 = rna2
    if strip_prefix:
        if r1.startswith(strip_prefix):
            r1 = r1[len(strip_prefix):]
        if r2.startswith(strip_prefix):
            r2 = r2[len(strip_prefix):]
    r1 = sanitize_for_path(r1)
    r2 = sanitize_for_path(r2)
    return os.path.join(out_dir, f"{r1}_vs_{r2}")

def build_command(
    align_script: str,
    input_tsv: str,
    id_column: str,
    rna1: str,
    rna2: str,
    gap: float,
    mode: str,
    output_prefix: str,
    structures_path: str,
    dotbracket_column: str,
    structures_id_column: str,
    python_exec: str = "python",
) -> list[str]:
    # Note: "--structures-id-colum" matches the flag given in your example (without the 'n').
    # If your script actually expects "--structures-id-column", change the flag below accordingly.
    return [
        python_exec, align_script,
        "--input", input_tsv,
        "--id-column", id_column,
        "--rna1", rna1,
        "--rna2", rna2,
        "--gap", str(gap),
        "--mode", mode,
        "--output-prefix", output_prefix,
        "--structures-path", structures_path,
        "--dotbracket-column", dotbracket_column,
        "--structures-id-colum", structures_id_column,  # keep as in your example
    ]

def run_cmd(cmd: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return proc.returncode, proc.stderr if proc.returncode != 0 else proc.stdout
    except Exception as e:
        return 1, f"Exception: {e}"

def main():
    parser = argparse.ArgumentParser(description="Launch pairwise alignments for all Name values in a TSV.")
    parser.add_argument("--input", required=True, help="Path to the TSV with a 'Name' (or chosen) column.")
    parser.add_argument("--id-column", default="Name", help="Column name containing RNA IDs (default: Name).")
    parser.add_argument("--align-script", required=True, help="Path to src/ginfinity/scripts/align_node_embeddings.py")
    parser.add_argument("--structures-path", required=True, help="Path to CSV with structures (e.g., merged_premirna_data.csv)")
    parser.add_argument("--dotbracket-column", default="DotBracket", help="DotBracket column name in structures CSV (default: DotBracket)")
    parser.add_argument("--structures-id-column", default="Name", help="ID column in structures CSV (default: Name).")
    parser.add_argument("--gap", type=float, default=-0.25, help="Gap penalty (default: -0.25)")
    parser.add_argument("--mode", choices=["local", "global"], default="local", help="Alignment mode (default: local)")
    parser.add_argument("--output-dir", default="output", help="Directory to write outputs (default: output)")
    parser.add_argument("--ordered", action="store_true", help="Use ordered pairs (A,B) and (B,A). Default: unique unordered.")
    parser.add_argument("--include-self", action="store_true", help="Include self-pairs (A,A). Default: exclude.")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4, help="Parallel workers (default: CPU count)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--python-exec", default="python", help="Python executable to run the align script (default: python)")
    parser.add_argument("--strip-prefix", default=None, help="Optional prefix to strip from rna names in output prefixes (e.g., 'hsa-').")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of pairwise jobs (useful for testing).")

    args = parser.parse_args()

    names = read_names_from_tsv(args.input, args.id_column)
    if not names:
        raise SystemExit("No names found in the specified column.")

    os.makedirs(args.output_dir, exist_ok=True)

    # Build pair list
    if args.ordered:
        pairs_iter = itertools.product(names, names) if args.include-self else (
            (a, b) for a in names for b in names if a != b
        )
    else:
        # Unordered unique pairs
        if args.include_self:
            # combinations_with_replacement gives (A,A), (A,B), (B,B)...
            pairs_iter = itertools.combinations_with_replacement(names, 2)
        else:
            pairs_iter = itertools.combinations(names, 2)

    pairs = list(pairs_iter)
    if args.limit is not None:
        pairs = pairs[:args.limit]

    # Prepare commands
    jobs = []
    for rna1, rna2 in pairs:
        out_prefix = build_output_prefix(args.output_dir, rna1, rna2, args.strip_prefix)
        cmd = build_command(
            align_script=args.align_script,
            input_tsv=args.input,
            id_column=args.id_column,
            rna1=rna1,
            rna2=rna2,
            gap=args.gap,
            mode=args.mode,
            output_prefix=out_prefix,
            structures_path=args.structures_path,
            dotbracket_column=args.dotbracket_column,
            structures_id_column=args.structures_id_column,
            python_exec=args.python_exec,
        )
        jobs.append(cmd)

    print(f"Found {len(names)} unique names -> {len(jobs)} pairwise jobs.", file=sys.stderr)

    if args.dry_run:
        for cmd in jobs:
            print(shlex.join(cmd))
        return

    # Run in parallel
    failures = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = {ex.submit(run_cmd, cmd): cmd for cmd in jobs}
        for fut in as_completed(futures):
            cmd = futures[fut]
            rc, out = fut.result()
            if rc != 0:
                failures += 1
                sys.stderr.write(f"[FAIL] {shlex.join(cmd)}\n{out}\n")
            else:
                # Optional: print minimal progress; comment out if too chatty
                sys.stdout.write(f"[OK] {shlex.join(cmd)}\n")

    if failures:
        sys.stderr.write(f"Completed with {failures} failures.\n")
        sys.exit(1)
    else:
        sys.stderr.write("All jobs completed successfully.\n")

if __name__ == "__main__":
    main()

