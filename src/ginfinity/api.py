"""Stable public embedding API."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ._model import EncoderConfig, GINEEncoder
from ._validation import RNA
from .graph import (NODE_ROLE_CORE, Graph, GraphBuilder,
                    GraphCompatibilityError, GraphShard, GraphSpec)


class ModelIntegrityError(RuntimeError):
    """A packaged model artifact failed compatibility or integrity checks."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resource_directory() -> Path:
    return Path(__file__).resolve().parent / "data"


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ModelIntegrityError(f"cannot read model metadata {path}: {error}") from error


def _embedding_dtype(value: np.dtype | str) -> np.dtype:
    try:
        dtype = np.dtype(value)
    except TypeError as error:
        raise ValueError(f"unsupported embedding dtype {value!r}") from error
    if dtype.kind != "f":
        raise ValueError(f"embedding dtype must be floating-point, got {dtype}")
    return dtype


def default_alignment_parameters() -> dict[str, float]:
    """Return model-versioned parameters for the separate SW package."""
    data = _load_json(_resource_directory() / "alignment.json")
    return dict(data["scoring_parameters"])


class Ginfinity:
    """Loaded GINFINITY encoder ready for repeated inference."""

    def __init__(self, model: GINEEncoder, metadata: dict, device: str,
                 graph_spec: GraphSpec, *, full_precision: bool) -> None:
        self._model = model.eval()
        self._metadata = metadata
        self._graph_spec = graph_spec
        self.device = device
        self.full_precision = full_precision

    @classmethod
    def load(cls, device: str = "cpu", *,
             allow_nondeterministic_cuda: bool = False,
             model_dir: str | Path | None = None,
             full_precision: bool = False) -> "Ginfinity":
        if device != "cpu" and not device.startswith("cuda"):
            raise ValueError("device must be 'cpu' or a CUDA device")
        if device.startswith("cuda"):
            if not allow_nondeterministic_cuda:
                raise ValueError(
                    "CUDA requires allow_nondeterministic_cuda=True")
            if not torch.cuda.is_available():
                raise ValueError("CUDA was requested but is unavailable")
        root = Path(model_dir) if model_dir is not None else _resource_directory()
        metadata_path = root / "model.json"
        checkpoint = root / "encoder.pt"
        metadata = _load_json(metadata_path)
        if not checkpoint.is_file():
            raise ModelIntegrityError(f"missing checkpoint {checkpoint}")
        actual_hash = _sha256(checkpoint)
        if actual_hash != metadata.get("checkpoint_sha256"):
            raise ModelIntegrityError("checkpoint SHA-256 mismatch")
        if metadata.get("format_version") != 1:
            raise ModelIntegrityError("unsupported model format")
        try:
            payload = torch.load(
                checkpoint, map_location=device, weights_only=True)
            config = EncoderConfig.from_dict(payload["cfg"])
            if config != EncoderConfig.from_dict(metadata["encoder_config"]):
                raise ModelIntegrityError(
                    "checkpoint and metadata architecture mismatch")
            graph_spec = GraphSpec.from_dict(metadata["graph_spec"])
            expected_graph_spec = GraphSpec.from_encoder_config(config)
            if (graph_spec.sha256 != expected_graph_spec.sha256
                    or metadata.get("graph_spec_sha256") != graph_spec.sha256):
                raise ModelIntegrityError(
                    "model and graph specification mismatch")
            model = GINEEncoder(config)
            model.load_state_dict(payload["state_dict"], strict=True)
        except ModelIntegrityError:
            raise
        except Exception as error:
            raise ModelIntegrityError(
                f"checkpoint could not be loaded: {error}") from error
        if model.parameter_count != metadata.get("parameter_count"):
            raise ModelIntegrityError("parameter-count mismatch")
        model = model.to(device)
        if not full_precision:
            model = model.half()
        return cls(model, metadata, device, graph_spec,
                   full_precision=full_precision)

    @property
    def embedding_dimension(self) -> int:
        return self._model.cfg.out_dim

    @property
    def graph_spec(self) -> GraphSpec:
        """The graph contract accepted by this encoder."""
        return self._graph_spec

    def info(self) -> dict:
        return json.loads(json.dumps(self._metadata))

    def encode(
        self,
        record: RNA,
        *,
        keep_paired_neighbours: bool = False,
        context_hops: int = 1,
        embedding_dtype: np.dtype | str = np.float16,
    ) -> np.ndarray:
        return self.encode_many(
            [record],
            keep_paired_neighbours=keep_paired_neighbours,
            context_hops=context_hops,
            embedding_dtype=embedding_dtype,
        )[0]

    def encode_many(
        self,
        records: Sequence[RNA],
        *,
        max_batch_nodes: int = 60_000,
        max_batch_edges: int = 300_000,
        keep_paired_neighbours: bool = False,
        context_hops: int = 1,
        embedding_dtype: np.dtype | str = np.float16,
    ) -> list[np.ndarray]:
        """Build graphs and encode records through the staged public path.

        For sliced records, context nucleotides participate in message
        passing and are discarded after encoding. Returned arrays contain
        only core nucleotides, in 5′→3′ order.
        """
        records = list(records)
        if not records:
            return []
        shard = GraphBuilder(
            self._graph_spec,
            keep_paired_neighbours=keep_paired_neighbours,
            context_hops=context_hops,
        ).build_shard(records)
        return self.encode_graphs(
            shard,
            max_batch_nodes=max_batch_nodes,
            max_batch_edges=max_batch_edges,
            embedding_dtype=embedding_dtype,
        )

    def encode_graph(
        self, graph: Graph, *, embedding_dtype: np.dtype | str = np.float16
    ) -> np.ndarray:
        """Encode one prebuilt graph."""
        return self.encode_graphs([graph], embedding_dtype=embedding_dtype)[0]

    def encode_graphs(
        self,
        graphs: Sequence[Graph] | GraphShard,
        *,
        max_batch_nodes: int = 60_000,
        max_batch_edges: int = 300_000,
        embedding_dtype: np.dtype | str = np.float16,
    ) -> list[np.ndarray]:
        """Encode prebuilt graphs, dynamically microbatching a persistent shard."""
        if isinstance(graphs, GraphShard):
            shard = graphs
        else:
            graph_list = list(graphs)
            if not graph_list:
                return []
            shard = GraphShard.from_graphs(graph_list)
        if shard.spec.sha256 != self._graph_spec.sha256:
            raise GraphCompatibilityError(
                "graphs were built with a specification incompatible with "
                "this encoder")
        if max_batch_nodes <= 0 or max_batch_edges <= 0:
            raise ValueError("batch node and edge limits must be positive")
        embedding_dtype = _embedding_dtype(embedding_dtype)
        lengths = shard.lengths
        edge_counts = shard.edge_counts
        if max(lengths) > max_batch_nodes:
            raise ValueError(
                "max_batch_nodes is smaller than the longest graph")
        if max(edge_counts) > max_batch_edges:
            raise ValueError(
                "max_batch_edges is smaller than the largest graph")
        outputs: list[np.ndarray] = []
        start = 0
        while start < shard.record_count:
            stop = start
            nodes = 0
            edges = 0
            while stop < shard.record_count:
                next_nodes = lengths[stop]
                next_edges = edge_counts[stop]
                if stop > start and (
                        nodes + next_nodes > max_batch_nodes
                        or edges + next_edges > max_batch_edges):
                    break
                nodes += next_nodes
                edges += next_edges
                stop += 1
            outputs.extend(self._run_graph_shard(
                shard.slice(start, stop), embedding_dtype))
            start = stop
        return outputs

    @torch.inference_mode()
    def _run_graph_shard(
        self, shard: GraphShard, embedding_dtype: np.dtype
    ) -> list[np.ndarray]:
        model_dtype = next(self._model.parameters()).dtype
        node = torch.from_numpy(shard.node_features).to(
            device=self.device, dtype=model_dtype)
        edge_index = torch.from_numpy(shard.edge_index).to(
            device=self.device, dtype=torch.long)
        edge_types = torch.from_numpy(shard.edge_types).to(
            device=self.device, dtype=torch.long)
        edge_attributes = torch.nn.functional.one_hot(
            edge_types, num_classes=self._graph_spec.edge_dim).to(
                dtype=model_dtype)
        embedding = self._model(
            node,
            edge_index,
            edge_attributes,
        ).to(dtype=torch.float32).cpu().numpy().astype(np.float64)
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = embedding / np.maximum(norms, 1e-12)
        outputs: list[np.ndarray] = []
        for index in range(shard.record_count):
            node_start = int(shard.node_ptr[index])
            node_stop = int(shard.node_ptr[index + 1])
            core = shard.node_roles[node_start:node_stop] == NODE_ROLE_CORE
            outputs.append(np.ascontiguousarray(
                embedding[node_start:node_stop][core], dtype=embedding_dtype))
        return outputs
