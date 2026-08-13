import json

import numpy as np

from ginfinity.cli import main


def test_cli_writes_embeddings_and_integrity_manifest(tmp_path):
    source = tmp_path / "molecules.tsv"
    source.write_text(
        "transcript_id\tsequence\tsecondary_structure\n"
        "rna-1\tACGUACGU\t((....))\n"
        "rna-2\tGGAACCUU\t........\n")
    output = tmp_path / "embeddings.npz"
    manifest = tmp_path / "manifest.json"
    assert main(["embed", "--input", str(source), "--output", str(output),
                 "--manifest", str(manifest)]) == 0
    with np.load(output) as archive:
        assert archive["rna-1"].shape == (8, 128)
        assert archive["rna-2"].shape == (8, 128)
    got = json.loads(manifest.read_text())
    assert got["status"] == "complete"
    assert got["records"][0]["identifier"] == "rna-1"
    assert len(got["checkpoint_sha256"]) == 64


def test_cli_exports_aligner_configuration(tmp_path):
    output = tmp_path / "alignment.json"
    assert main(["alignment-config", "--output", str(output)]) == 0
    got = json.loads(output.read_text())
    assert got["scoring_parameters"]["sigma"] == 1.0


def test_embed_cli_accepts_configurable_columns_and_extra_fields(tmp_path):
    source = tmp_path / "structures.csv"
    source.write_text(
        "name,bases,fold,family\n"
        "custom-id,ACGU,(()),example\n")
    output = tmp_path / "embeddings.npz"
    assert main([
        "embed", "--input", str(source), "--output", str(output),
        "--id-column", "name", "--sequence-column", "bases",
        "--structure-column", "fold", "--delimiter", ",",
    ]) == 0
    with np.load(output) as archive:
        assert archive["custom-id"].shape == (4, 128)


def test_cli_can_build_then_embed_a_graph_shard(tmp_path):
    source = tmp_path / "molecules.tsv"
    source.write_text(
        "transcript_id\tsequence\tsecondary_structure\n"
        "rna-1\tACGUACGU\t((....))\n"
        "rna-2\tGGAACCUU\t........\n")
    graphs = tmp_path / "graphs.safetensors"
    graph_metadata = tmp_path / "graphs.json"
    assert main([
        "build-graphs", "--input", str(source), "--output", str(graphs),
        "--metadata", str(graph_metadata), "--checksum",
    ]) == 0
    embeddings = tmp_path / "embeddings.npz"
    manifest = tmp_path / "embeddings.json"
    assert main([
        "embed-graphs", "--input", str(graphs),
        "--metadata", str(graph_metadata), "--output", str(embeddings),
        "--manifest", str(manifest), "--verify-checksum",
        "--full-validation", "--max-batch-nodes", "8",
    ]) == 0
    with np.load(embeddings) as archive:
        assert archive["rna-1"].shape == (8, 128)
        assert archive["rna-2"].shape == (8, 128)
    got = json.loads(manifest.read_text())
    assert got["status"] == "complete"
    assert len(got["graph_spec_sha256"]) == 64
