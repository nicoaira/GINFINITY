import json

import numpy as np
import pytest

import ginfinity.graph as graph_module
from safetensors import safe_open

from ginfinity import (Ginfinity, GraphBuilder, GraphCompatibilityError,
                        GraphSpec, GraphValidationError, RNA,
                        load_graph_shard, partition_records, save_graph_shard)


def _records():
    return [
        RNA("rna-1", "ACGUACGU", "((....))"),
        RNA("rna-2", "GGAACCUU", "........"),
    ]


def test_graph_builder_exposes_compact_model_versioned_arrays():
    graph = GraphBuilder().build(_records()[0])
    assert graph.node_features.shape == (8, 7)
    assert graph.node_features.dtype == np.float32
    assert graph.edge_index.shape == (2, 30)
    assert graph.edge_index.dtype == np.int32
    assert graph.edge_types.shape == (30,)
    assert graph.edge_types.dtype == np.uint8
    np.testing.assert_array_equal(graph.residue_index, np.arange(8, dtype=np.int32))
    assert graph.node_count == 8
    assert graph.core_count == 8
    assert graph.spec.sha256 == GraphSpec.bundled().sha256
    assert len(graph.spec.sha256) == 64


def test_graph_shard_round_trip_uses_metadata_validation_by_default(tmp_path):
    original = GraphBuilder().build_shard(_records())
    tensor_path = tmp_path / "graphs.safetensors"
    _, metadata_path = save_graph_shard(original, tensor_path)
    metadata = json.loads(metadata_path.read_text())
    assert "tensor_sha256" not in metadata
    restored = load_graph_shard(tensor_path, validation="full")
    assert restored.identifiers == original.identifiers
    assert restored.sequences == original.sequences
    assert restored.structures == original.structures
    assert restored.spec == original.spec
    np.testing.assert_array_equal(restored.node_features, original.node_features)
    np.testing.assert_array_equal(restored.edge_index, original.edge_index)
    np.testing.assert_array_equal(restored.edge_types, original.edge_types)
    np.testing.assert_array_equal(restored.node_ptr, original.node_ptr)
    np.testing.assert_array_equal(restored.edge_ptr, original.edge_ptr)
    np.testing.assert_array_equal(restored.residue_index, original.residue_index)
    np.testing.assert_array_equal(restored.node_roles, original.node_roles)


def test_full_molecule_shards_omit_optional_node_metadata(tmp_path):
    tensor_path = tmp_path / "graphs.safetensors"
    save_graph_shard(GraphBuilder().build_shard(_records()), tensor_path)
    with safe_open(str(tensor_path), framework="np") as handle:
        assert set(handle.keys()) == {
            "node_features", "edge_index", "edge_types", "node_ptr", "edge_ptr"}


def test_optional_shard_checksum_detects_content_change(tmp_path):
    tensor_path = tmp_path / "graphs.safetensors"
    save_graph_shard(
        GraphBuilder().build_shard(_records()), tensor_path, checksum=True)
    payload = bytearray(tensor_path.read_bytes())
    payload[-1] ^= 1
    tensor_path.write_bytes(payload)
    with pytest.raises(GraphValidationError, match="checksum"):
        load_graph_shard(tensor_path, verify_checksum=True)


def test_default_shard_path_does_not_hash_tensor_content(tmp_path, monkeypatch):
    def unexpected_hash(_path):
        raise AssertionError("content hashing must remain opt-in")

    monkeypatch.setattr(graph_module, "_sha256", unexpected_hash)
    tensor_path = tmp_path / "graphs.safetensors"
    save_graph_shard(GraphBuilder().build_shard(_records()), tensor_path)
    assert load_graph_shard(tensor_path).record_count == 2


def test_saved_staged_encoding_equals_direct_encoding_across_microbatches(tmp_path):
    records = _records()
    encoder = Ginfinity.load()
    direct = encoder.encode_many(records)
    tensor_path = tmp_path / "graphs.safetensors"
    save_graph_shard(GraphBuilder().build_shard(records), tensor_path)
    restored = load_graph_shard(
        tensor_path, expected_spec=encoder.graph_spec)
    staged = encoder.encode_graphs(
        restored, max_batch_nodes=8, max_batch_edges=30)
    for direct_value, staged_value in zip(direct, staged):
        np.testing.assert_allclose(
            staged_value, direct_value, rtol=1e-5, atol=3e-7)


def test_encoder_rejects_an_incompatible_graph_specification():
    incompatible = GraphSpec(
        struct_feature="B", positional=True, edge_dim=10,
        extra_edges=("skip2",))
    graph = GraphBuilder(incompatible).build(_records()[0])
    with pytest.raises(GraphCompatibilityError, match="incompatible"):
        Ginfinity.load().encode_graph(graph)


def test_graph_microbatch_limits_reject_one_oversized_graph():
    graph = GraphBuilder().build(_records()[0])
    with pytest.raises(ValueError, match="max_batch_edges"):
        Ginfinity.load().encode_graphs([graph], max_batch_edges=29)


def test_record_partitioning_respects_record_and_node_limits():
    records = [RNA(str(index), "ACGU", "....") for index in range(5)]
    parts = list(partition_records(
        records, max_records=3, max_nodes=8))
    assert [[record.identifier for record in part] for part in parts] == [
        ["0", "1"], ["2", "3"], ["4"]]
