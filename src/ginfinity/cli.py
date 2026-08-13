"""Command-line embedding interface."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .api import Ginfinity, default_alignment_parameters
from .graph import (GraphBuilder, graph_metadata_path, load_graph_shard,
                    save_graph_shard)
from .table import read_rna_table


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_records(args: argparse.Namespace):
    return read_rna_table(
        args.input,
        identifier_column=args.id_column,
        sequence_column=args.sequence_column,
        structure_column=args.structure_column,
        delimiter=args.delimiter,
    )


def _embed(args: argparse.Namespace) -> int:
    started = time.time()
    records = _read_records(args)
    encoder = Ginfinity.load(
        device=args.device,
        allow_nondeterministic_cuda=args.allow_nondeterministic_cuda)
    outputs = encoder.encode_many(
        records,
        max_batch_nodes=args.max_batch_nodes,
        max_batch_edges=args.max_batch_edges,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, **{record.identifier: value
                        for record, value in zip(records, outputs)})
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest = {
        "status": "complete",
        "ginfinity_version": __version__,
        "model_version": encoder.info()["model_version"],
        "checkpoint_sha256": encoder.info()["checkpoint_sha256"],
        "input": str(args.input),
        "input_sha256": _sha256(args.input),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "device": args.device,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "records": [{"identifier": record.identifier,
                     "length": record.length,
                     "shape": list(value.shape)}
                    for record, value in zip(records, outputs)],
        "elapsed_seconds": time.time() - started,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output),
                      "manifest": str(manifest_path),
                      "records": len(records)}))
    return 0


def _build_graphs(args: argparse.Namespace) -> int:
    started = time.time()
    records = _read_records(args)
    shard = GraphBuilder().build_shard(records)
    metadata_path = args.metadata or graph_metadata_path(args.output)
    save_graph_shard(
        shard,
        args.output,
        metadata_path=metadata_path,
        checksum=args.checksum,
    )
    print(json.dumps({
        "output": str(args.output),
        "metadata": str(metadata_path),
        "records": shard.record_count,
        "nodes": shard.node_count,
        "edges": shard.edge_count,
        "graph_spec_sha256": shard.spec.sha256,
        "checksum": args.checksum,
        "elapsed_seconds": time.time() - started,
    }))
    return 0


def _embed_graphs(args: argparse.Namespace) -> int:
    started = time.time()
    encoder = Ginfinity.load(
        device=args.device,
        allow_nondeterministic_cuda=args.allow_nondeterministic_cuda,
    )
    shard = load_graph_shard(
        args.input,
        metadata_path=args.metadata,
        expected_spec=encoder.graph_spec,
        verify_checksum=args.verify_checksum,
        validation="full" if args.full_validation else "metadata",
    )
    outputs = encoder.encode_graphs(
        shard,
        max_batch_nodes=args.max_batch_nodes,
        max_batch_edges=args.max_batch_edges,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **{identifier: value
           for identifier, value in zip(shard.identifiers, outputs)},
    )
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest = {
        "status": "complete",
        "ginfinity_version": __version__,
        "model_version": encoder.info()["model_version"],
        "checkpoint_sha256": encoder.info()["checkpoint_sha256"],
        "graph_spec_sha256": shard.spec.sha256,
        "input": str(args.input),
        "input_metadata": str(
            args.metadata or graph_metadata_path(args.input)),
        "output": str(args.output),
        "device": args.device,
        "records": [
            {"identifier": identifier, "length": length,
             "shape": list(value.shape)}
            for identifier, length, value
            in zip(shard.identifiers, shard.lengths, outputs)
        ],
        "elapsed_seconds": time.time() - started,
    }
    if args.checksum:
        manifest["output_sha256"] = _sha256(args.output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "manifest": str(manifest_path),
        "records": shard.record_count,
    }))
    return 0


def _add_table_columns(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--id-column", default="transcript_id")
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--structure-column", default="secondary_structure")
    parser.add_argument("--delimiter", default="\t")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ginfinity")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    info = subparsers.add_parser("info", help="show verified model metadata")
    info.add_argument("--device", default="cpu")
    alignment = subparsers.add_parser(
        "alignment-config", help="export parameters for ginfinity-sw")
    alignment.add_argument("--output", type=Path)
    embed = subparsers.add_parser("embed", help="encode a TSV of RNA records")
    embed.add_argument("--input", type=Path, required=True)
    embed.add_argument("--output", type=Path, required=True)
    embed.add_argument("--manifest", type=Path)
    embed.add_argument("--device", default="cpu")
    embed.add_argument("--allow-nondeterministic-cuda", action="store_true")
    embed.add_argument("--max-batch-nodes", type=int, default=60_000)
    embed.add_argument("--max-batch-edges", type=int, default=300_000)
    _add_table_columns(embed)
    build_graphs = subparsers.add_parser(
        "build-graphs", help="build a persistent graph shard from an RNA TSV")
    build_graphs.add_argument("--input", type=Path, required=True)
    build_graphs.add_argument("--output", type=Path, required=True)
    build_graphs.add_argument("--metadata", type=Path)
    build_graphs.add_argument("--checksum", action="store_true")
    _add_table_columns(build_graphs)
    embed_graphs = subparsers.add_parser(
        "embed-graphs", help="encode a previously built graph shard")
    embed_graphs.add_argument("--input", type=Path, required=True)
    embed_graphs.add_argument("--metadata", type=Path)
    embed_graphs.add_argument("--output", type=Path, required=True)
    embed_graphs.add_argument("--manifest", type=Path)
    embed_graphs.add_argument("--device", default="cpu")
    embed_graphs.add_argument(
        "--allow-nondeterministic-cuda", action="store_true")
    embed_graphs.add_argument("--max-batch-nodes", type=int, default=60_000)
    embed_graphs.add_argument("--max-batch-edges", type=int, default=300_000)
    embed_graphs.add_argument("--verify-checksum", action="store_true")
    embed_graphs.add_argument("--full-validation", action="store_true")
    embed_graphs.add_argument("--checksum", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "info":
            print(json.dumps(Ginfinity.load(device=args.device).info(), indent=2))
            return 0
        if args.command == "alignment-config":
            text = json.dumps({
                "scoring_parameters": default_alignment_parameters()},
                indent=2) + "\n"
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text)
            else:
                print(text, end="")
            return 0
        if args.command == "embed":
            return _embed(args)
        if args.command == "build-graphs":
            return _build_graphs(args)
        return _embed_graphs(args)
    except Exception as error:
        print(f"ginfinity: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
