#!/usr/bin/env python3

import argparse
import json
import math
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def _serialize_matrix(mat: torch.Tensor) -> str:
    """Serialize a 2D tensor (L x D) as a compact JSON string with rounding."""
    arr = mat.detach().cpu().tolist()
    rounded = [[round(float(x), 6) for x in row] for row in arr]
    return json.dumps(rounded, separators=(",", ":"))


def _load_rinalmo(model_name: str = "giga-v1", device: str = "cpu"):
    try:
        from rinalmo.pretrained import get_pretrained_model  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "RiNALMo not installed. Please 'pip install rinalmo' in this Python environment."
        ) from e
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"RiNALMo import failed: {e.__class__.__name__}: {e}") from e

    model, alphabet = get_pretrained_model(model_name=model_name)
    dev = torch.device(device)
    model = model.to(device=dev)
    model.eval()
    return model, alphabet, dev


def generate_base_embeddings(
    df: pd.DataFrame,
    id_column: str,
    sequence_column: str,
    output_path: str,
    keep_cols: Optional[List[str]] = None,
    model_name: str = "giga-v1",
    device: str = "cpu",
    batch_size: int = 8,
    use_amp: bool = True,
    trim_special: bool = True,
    quiet: bool = False,
):
    if id_column not in df.columns:
        raise ValueError(f"Missing id column '{id_column}' in input.")
    if sequence_column not in df.columns:
        raise ValueError(f"Missing sequence column '{sequence_column}' in input.")

    model, alphabet, dev = _load_rinalmo(model_name, device)

    # Decide which columns to carry through
    final_keep = [id_column]
    if keep_cols:
        final_keep.extend(keep_cols)

    # Prepare batches
    records = df[[id_column, sequence_column] + [c for c in (keep_cols or []) if c in df.columns]].to_dict(
        orient="records"
    )

    rows_out = []

    pbar = tqdm(total=len(records), disable=quiet, desc="RiNALMo base embeddings", unit="seq")
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        ids = [str(r[id_column]) for r in batch]
        seqs = [str(r[sequence_column]) for r in batch]

        # Tokenize
        try:
            toks = alphabet.batch_tokenize(seqs)
        except Exception as e:
            raise SystemExit(f"Tokenization failed: {e}")
        tokens = torch.tensor(toks, dtype=torch.int64, device=dev)

        with torch.no_grad():
            if use_amp and dev.type == "cuda":
                with torch.cuda.amp.autocast(True):
                    outputs = model(tokens)
            else:
                outputs = model(tokens)

        reps = outputs["representation"]  # (B, L, D)
        if not torch.is_tensor(reps) or reps.ndim != 3:
            raise SystemExit("Unexpected RiNALMo output: missing 'representation' (B,L,D)")

        for i, uid in enumerate(ids):
            mat = reps[i]
            # Trim BOS/EOS if present (common for LMs)
            Lr = int(mat.shape[0])
            Ls = len(seqs[i])
            if trim_special and Lr == Ls + 2:
                mat = mat[1:-1]
            elif trim_special and Lr != Ls and Lr > Ls and Ls > 0:
                # Conservative fallback: center-crop to sequence length
                start_idx = max(0, (Lr - Ls) // 2)
                end_idx = min(Lr, start_idx + Ls)
                if end_idx - start_idx == Ls:
                    mat = mat[start_idx:end_idx]
            # L2-normalize rows for cosine use downstream? Keep raw here; aligners normalize.
            out = {c: batch[i][c] for c in final_keep if c in batch[i]}
            out[id_column] = uid
            out["base_embeddings"] = _serialize_matrix(mat)
            out["seq_len"] = int(mat.shape[0])
            rows_out.append(out)
        pbar.update(len(batch))
    pbar.close()

    out_df = pd.DataFrame(rows_out)
    # Reorder for readability: id, seq_len, base_embeddings, then others
    cols = [id_column]
    if "seq_len" in out_df.columns:
        cols.append("seq_len")
    cols.append("base_embeddings")
    others = [c for c in out_df.columns if c not in cols]
    out_df = out_df[cols + sorted(others)]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_df.to_csv(output_path, sep="\t", index=False, na_rep="NaN")
    if not quiet:
        print(f"Base embeddings saved to {output_path}")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Generate per-base RiNALMo embeddings from an input TSV/CSV with sequences. "
            "Outputs a TSV with a 'base_embeddings' JSON column (LxD per sequence)."
        )
    )
    ap.add_argument("--input", required=True, help="Input TSV/CSV with sequences.")
    ap.add_argument("--output", required=True, help="Output TSV path.")
    ap.add_argument("--id-column", required=True, help="Column name containing unique IDs.")
    ap.add_argument("--sequence-column-name", default="sequence", help="Column with RNA sequence.")
    ap.add_argument("--keep-cols", default=None, help="Comma-separated extra columns to carry through.")
    ap.add_argument("--model-name", default="giga-v1", help="RiNALMo model name (e.g., giga-v1).")
    ap.add_argument("--device", default="cpu", help="Device: cpu or cuda:0")
    ap.add_argument("--batch-size", type=int, default=8, help="Batch size for inference.")
    ap.add_argument("--no-amp", action="store_true", help="Disable AMP on CUDA.")
    ap.add_argument("--no-trim-special", action="store_true", help="Do not trim BOS/EOS special tokens (keep model length).")
    ap.add_argument("--quiet", action="store_true", help="Suppress progress bar.")
    args = ap.parse_args()

    # Read with auto-separator detection for robustness
    if args.input.endswith(".tsv"):
        df = pd.read_csv(args.input, sep="\t")
    elif args.input.endswith(".csv"):
        df = pd.read_csv(args.input)
    else:
        df = pd.read_csv(args.input, sep=None, engine="python")

    keep_cols: Optional[List[str]] = None
    if args.keep_cols:
        keep_cols = [c.strip() for c in args.keep_cols.split(",") if c.strip()]

    generate_base_embeddings(
        df=df,
        id_column=args.id_column,
        sequence_column=args.sequence_column_name,
        output_path=args.output,
        keep_cols=keep_cols,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        use_amp=not args.no_amp,
        trim_special=not args.no_trim_special,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
