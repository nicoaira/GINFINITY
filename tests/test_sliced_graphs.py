import json

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file

from ginfinity import (NODE_ROLE_CONTEXT, NODE_ROLE_CORE, Ginfinity,
                       GraphBuilder, RNA, load_graph_shard, save_graph_shard)
from ginfinity.cli import main
from ginfinity.graph import GRAPH_SHARD_FORMAT, GRAPH_SHARD_FORMAT_VERSION


# ....(((....)))..  crossing pairs at 6-15, 7-14, 8-13
_SEQUENCE = "GGGAAACCCUUUUGGG"
_STRUCTURE = "......(((....)))"


def _sliced(**kwargs) -> RNA:
    return RNA("stem", _SEQUENCE, _STRUCTURE, **kwargs)


def test_slice_without_neighbours_keeps_only_the_window():
    graph = GraphBuilder().build(_sliced(start=9, end=16))
    np.testing.assert_array_equal(
        graph.residue_index, np.arange(9, 16, dtype=np.int32))
    assert graph.core_count == 7
    assert graph.node_count == 7
    assert np.all(graph.node_roles == NODE_ROLE_CORE)
    assert graph.node_features.shape == (7, 7)
    assert graph.core_span == (9, 16)


def test_keep_paired_neighbours_adds_crossing_pair_partners():
    graph = GraphBuilder(keep_paired_neighbours=True).build(
        _sliced(start=9, end=16))
    np.testing.assert_array_equal(
        graph.residue_index,
        np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.int32))
    np.testing.assert_array_equal(
        graph.node_roles,
        np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8))
    np.testing.assert_array_equal(
        graph.core_positions, np.arange(9, 16, dtype=np.int32))


def test_context_hops_expand_through_backbone_and_pairs():
    builder = GraphBuilder(keep_paired_neighbours=True, context_hops=3)
    graph = builder.build(_sliced(start=9, end=16))
    np.testing.assert_array_equal(
        graph.residue_index,
        np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                 dtype=np.int32))
    context = graph.residue_index[graph.node_roles == NODE_ROLE_CONTEXT]
    np.testing.assert_array_equal(
        context, np.array([2, 3, 4, 5, 6, 7, 8], dtype=np.int32))


def test_context_hops_one_is_only_the_crossing_pair_partner():
    hops1 = GraphBuilder(keep_paired_neighbours=True, context_hops=1).build(
        _sliced(start=9, end=16))
    hops2 = GraphBuilder(keep_paired_neighbours=True, context_hops=2).build(
        _sliced(start=9, end=16))
    np.testing.assert_array_equal(
        hops1.residue_index[hops1.node_roles == NODE_ROLE_CONTEXT],
        np.array([6, 7, 8], dtype=np.int32))
    np.testing.assert_array_equal(
        hops2.residue_index[hops2.node_roles == NODE_ROLE_CONTEXT],
        np.array([4, 5, 6, 7, 8], dtype=np.int32))


def test_sliced_node_features_match_the_full_molecule_and_exclude_roles():
    full = GraphBuilder().build(_sliced())
    sliced = GraphBuilder(keep_paired_neighbours=True, context_hops=2).build(
        _sliced(start=9, end=16))
    assert sliced.node_features.shape[1] == full.node_features.shape[1] == 7
    np.testing.assert_array_equal(
        sliced.node_features, full.node_features[sliced.residue_index])


def test_adjacent_nucleotides_outside_the_window_are_not_kept_by_default():
    graph = GraphBuilder().build(_sliced(start=9, end=16))
    assert 8 not in set(int(value) for value in graph.residue_index)


def test_encode_returns_only_core_rows_in_sequence_order():
    encoder = Ginfinity.load()
    record = _sliced(start=9, end=16)
    embedding = encoder.encode(record, keep_paired_neighbours=True, context_hops=3)
    assert embedding.shape == (7, 128)
    assert embedding.dtype == np.float16
    np.testing.assert_allclose(np.linalg.norm(embedding, axis=1), 1.0, atol=1e-6)


def test_context_nodes_change_core_embeddings_but_are_not_returned():
    encoder = Ginfinity.load()
    record = _sliced(start=9, end=16)
    without = encoder.encode(record)
    with_context = encoder.encode(
        record, keep_paired_neighbours=True, context_hops=3)
    assert without.shape == with_context.shape == (7, 128)
    assert not np.allclose(without, with_context, atol=1e-6)


def test_multiple_windows_produce_independent_graphs():
    records = RNA.many_from_mapping(
        {
            "transcript_id": "stem",
            "sequence": _SEQUENCE,
            "secondary_structure": _STRUCTURE,
            "start": "6,9",
            "end": "12, 16",
        })
    graphs = GraphBuilder(keep_paired_neighbours=True).build_many(records)
    assert [graph.identifier for graph in graphs] == ["stem:6-12", "stem:9-16"]
    assert [graph.core_span for graph in graphs] == [(6, 12), (9, 16)]
    encoder = Ginfinity.load()
    outputs = encoder.encode_many(records, keep_paired_neighbours=True)
    assert [value.shape[0] for value in outputs] == [6, 7]


def test_sliced_shard_round_trip_preserves_roles(tmp_path):
    records = [_sliced(start=9, end=16)]
    original = GraphBuilder(keep_paired_neighbours=True, context_hops=2).build_shard(
        records)
    path = tmp_path / "graphs.safetensors"
    save_graph_shard(original, path)
    with safe_open(str(path), framework="np") as handle:
        assert "residue_index" in handle.keys()
        assert "node_roles" in handle.keys()
    restored = load_graph_shard(path, validation="full")
    np.testing.assert_array_equal(restored.node_roles, original.node_roles)
    np.testing.assert_array_equal(restored.residue_index, original.residue_index)
    assert restored.core_counts == (7,)
    assert restored.lengths[0] > restored.core_counts[0]


def test_legacy_shard_without_roles_loads_as_all_core(tmp_path):
    original = GraphBuilder().build_shard([_sliced()])
    tensor_path = tmp_path / "legacy.safetensors"
    metadata_path = tmp_path / "legacy.json"
    save_file(
        {
            "node_features": original.node_features,
            "edge_index": original.edge_index,
            "edge_types": original.edge_types,
            "node_ptr": original.node_ptr,
            "edge_ptr": original.edge_ptr,
        },
        str(tensor_path),
        metadata={
            "format": GRAPH_SHARD_FORMAT,
            "format_version": str(GRAPH_SHARD_FORMAT_VERSION),
            "graph_spec_sha256": original.spec.sha256,
        },
    )
    metadata_path.write_text(json.dumps({
        "format": GRAPH_SHARD_FORMAT,
        "format_version": GRAPH_SHARD_FORMAT_VERSION,
        "graph_spec": original.spec.to_dict(),
        "graph_spec_sha256": original.spec.sha256,
        "tensor_file": tensor_path.name,
        "record_count": original.record_count,
        "node_count": original.node_count,
        "edge_count": original.edge_count,
        "identifiers": list(original.identifiers),
        "sequences": list(original.sequences),
        "structures": list(original.structures),
    }))
    restored = load_graph_shard(tensor_path)
    np.testing.assert_array_equal(restored.residue_index, original.residue_index)
    assert np.all(restored.node_roles == NODE_ROLE_CORE)


def test_user_window_example_has_sixty_three_core_nucleotides():
    sequence = (
        "CUUAUGAAGUCUUCCUUUCAGUUCAGAAGAAAUGGAAUUCGCUCUCCAACUUCAGGAAACUGAAAUA"
        "AAGAGUUGCUUGGAUUUAGUGUUCACCUUUACCAUAAAAUGGAUUUGCUAACACUGCCACCCUGCUU"
        "UGAUAGCGAAUAAAGCAAAAAGGGCUUCUGUCGUGAGUGGCACACGUAGGGCAACUCGAUUGCUCUU"
        "CGUGCGGAAUCGACAUCAAGAGAUUUCGGAAGCAUAAUUUUUUGACAUUCGGGCAGCUGGUGAUCGU"
        "UGGUCCCGGCGCCCUUCUUUUUUUCUGUCUCAAGUCAGAUGAAUUUUUCUGGUGAGUUAGGUGUUAG"
        "UUUUGUAAGUGGAUGUAAGAUUUAUGUUAAUCCUUUUUAUUUGAAGUUGCGUAGCUAUCUGCGUGAA"
        "CCGCAGAUGACUAAAUUAGCAGGGUAUUUAAC")
    structure = (
        "......((.((((..(((((((((.((((...((((........)))).))))..)).))))))).."
        ")))).))....((...(((((((........((((...)))).......)))))))))((((((((."
        ".....((((...(((.((((((((((((((((......))))((((.(((((((.....)))))))"
        ".))))(((((((..........))))))))))))((((.((((..((((((((((.(((((.(((("
        "...))))))))))))).........((((.((((..(((((........))))))))).))))..."
        ".............)))))))))).))))......))))))).)))....)))).(((.(((((((("
        ".....)))))))).)))....)))))))).......")
    record = RNA(
        "AANU01100842.1/307-440", sequence, structure, start=50, end=113)
    graph = GraphBuilder(keep_paired_neighbours=True).build(record)
    assert graph.core_count == 63
    assert graph.core_span == (50, 113)
    assert graph.node_count > graph.core_count


def test_context_hops_requires_positive_depth():
    with pytest.raises(ValueError, match="context_hops"):
        GraphBuilder(context_hops=0)


def test_cli_embed_writes_one_array_per_window(tmp_path):
    source = tmp_path / "molecules.tsv"
    source.write_text(
        "transcript_id\tsequence\tsecondary_structure\tstart\tend\n"
        f"stem\t{_SEQUENCE}\t{_STRUCTURE}\t9,6\t16,12\n")
    output = tmp_path / "embeddings.npz"
    manifest = tmp_path / "manifest.json"
    assert main([
        "embed", "--input", str(source), "--output", str(output),
        "--manifest", str(manifest),
        "--keep-paired-neighbours", "--context-hops", "2",
    ]) == 0
    with np.load(output) as archive:
        assert set(archive.files) == {"stem:9-16", "stem:6-12"}
        assert archive["stem:9-16"].shape == (7, 128)
        assert archive["stem:6-12"].shape == (6, 128)
    got = json.loads(manifest.read_text())
    by_id = {row["identifier"]: row for row in got["records"]}
    assert by_id["stem:9-16"]["start"] == 9
    assert by_id["stem:9-16"]["end"] == 16
    assert by_id["stem:9-16"]["core_length"] == 7


def test_cli_no_slices_ignores_window_columns(tmp_path):
    source = tmp_path / "molecules.tsv"
    source.write_text(
        "transcript_id\tsequence\tsecondary_structure\tstart\tend\n"
        f"stem\t{_SEQUENCE}\t{_STRUCTURE}\t9\t16\n")
    output = tmp_path / "embeddings.npz"
    assert main([
        "embed", "--input", str(source), "--output", str(output),
        "--no-slices",
    ]) == 0
    with np.load(output) as archive:
        assert archive["stem"].shape == (16, 128)
