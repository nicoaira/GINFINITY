"""Versioned RNA graph construction and shard interchange API."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Literal, Mapping, Sequence

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

from ._validation import RNA

GRAPH_SHARD_FORMAT = "ginfinity-graph-shard"
GRAPH_SHARD_FORMAT_VERSION = 1
_EDGE_TYPES = {
    "backbone_forward": 0,
    "backbone_reverse": 1,
    "base_pair_forward": 2,
    "base_pair_reverse": 3,
    "skip2_forward": 4,
    "skip2_reverse": 5,
}

# Per-node provenance for sliced graphs. These values are not GINE features.
NODE_ROLE_CORE = np.uint8(0)
NODE_ROLE_CONTEXT = np.uint8(1)
_SHARD_REQUIRED_TENSORS = {
    "node_features", "edge_index", "edge_types", "node_ptr", "edge_ptr",
}
_SHARD_OPTIONAL_TENSORS = {"residue_index", "node_roles"}


class GraphValidationError(ValueError):
    """A graph or graph shard violates the public interchange contract."""


class GraphCompatibilityError(GraphValidationError):
    """A graph was built with a specification incompatible with the encoder."""


def _canonical_json(value: Mapping) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class GraphSpec:
    """The model-versioned feature and edge contract used to build graphs."""

    format_version: int = 1
    struct_feature: str = "A"
    positional: bool = True
    edge_dim: int = 10
    extra_edges: tuple[str, ...] = ("skip2",)
    _fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra_edges", tuple(self.extra_edges))
        if self.format_version != GRAPH_SHARD_FORMAT_VERSION:
            raise GraphValidationError(
                f"unsupported graph specification version {self.format_version}")
        if self.struct_feature not in {"A", "B"}:
            raise GraphValidationError(
                f"unsupported structure feature {self.struct_feature!r}")
        unknown = sorted(set(self.extra_edges) - {"skip2"})
        if unknown:
            raise GraphValidationError(
                "unsupported extra edge type(s): " + ", ".join(unknown))
        required_edge_dim = 6 if "skip2" in self.extra_edges else 4
        if self.edge_dim < required_edge_dim:
            raise GraphValidationError(
                f"edge_dim={self.edge_dim} cannot represent all configured edges")
        object.__setattr__(self, "_fingerprint", hashlib.sha256(
            _canonical_json(self.to_dict())).hexdigest())

    @property
    def node_feature_dim(self) -> int:
        return 4 + (1 if self.struct_feature == "A" else 3) + (
            2 if self.positional else 0)

    @property
    def edge_types(self) -> dict[str, int]:
        names = list(_EDGE_TYPES)[:4]
        if "skip2" in self.extra_edges:
            names.extend(("skip2_forward", "skip2_reverse"))
        return {name: _EDGE_TYPES[name] for name in names}

    def to_dict(self) -> dict:
        return {
            "format_version": self.format_version,
            "struct_feature": self.struct_feature,
            "positional": self.positional,
            "node_feature_dimension": self.node_feature_dim,
            "edge_feature_dimension": self.edge_dim,
            "edge_types": self.edge_types,
            "extra_edges": list(self.extra_edges),
        }

    @property
    def sha256(self) -> str:
        """A precomputed-contract identifier, not a per-graph content hash."""
        return self._fingerprint

    @classmethod
    def from_dict(cls, value: Mapping) -> "GraphSpec":
        spec = cls(
            format_version=int(value.get("format_version", 1)),
            struct_feature=str(value["struct_feature"]),
            positional=bool(value["positional"]),
            edge_dim=int(value.get(
                "edge_feature_dimension", value.get("edge_dim", 10))),
            extra_edges=tuple(value.get("extra_edges", ())),
        )
        if ("node_feature_dimension" in value
                and int(value["node_feature_dimension"])
                != spec.node_feature_dim):
            raise GraphValidationError("node feature dimension is inconsistent")
        if "edge_types" in value and dict(value["edge_types"]) != spec.edge_types:
            raise GraphValidationError("edge type mapping is inconsistent")
        return spec

    @classmethod
    def from_encoder_config(cls, value: Mapping | object) -> "GraphSpec":
        def field(name: str):
            return value[name] if isinstance(value, Mapping) else getattr(value, name)

        return cls(
            struct_feature=str(field("struct_feature")),
            positional=bool(field("positional")),
            edge_dim=int(field("edge_dim")),
            extra_edges=tuple(field("extra_edges")),
        )

    @classmethod
    def bundled(cls) -> "GraphSpec":
        """Read the graph contract without loading the model checkpoint."""
        metadata_path = Path(__file__).resolve().parent / "data" / "model.json"
        try:
            metadata = json.loads(metadata_path.read_text())
            spec = cls.from_dict(metadata["graph_spec"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise GraphValidationError(
                f"cannot read bundled graph specification: {error}") from error
        if metadata.get("graph_spec_sha256") != spec.sha256:
            raise GraphValidationError(
                "bundled graph specification fingerprint mismatch")
        return spec


@dataclass(frozen=True, slots=True)
class Graph:
    """One validated RNA graph with local, zero-based edge indices.

    ``sequence`` and ``structure`` are always the full source molecule.
    ``residue_index`` maps each graph node back to that molecule. For a
    sliced graph the node set may be smaller than the sequence: core
    nucleotides plus any retained context. ``node_roles`` is provenance
    metadata and is not part of the GINE feature matrix.
    """

    identifier: str
    sequence: str
    structure: str
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_types: np.ndarray
    spec: GraphSpec
    residue_index: np.ndarray
    node_roles: np.ndarray

    def __post_init__(self) -> None:
        if (not isinstance(self.identifier, str)
                or not isinstance(self.sequence, str)
                or not isinstance(self.structure, str)):
            raise GraphValidationError("invalid graph record metadata")
        length = len(self.sequence)
        if not self.identifier or length == 0 or len(self.structure) != length:
            raise GraphValidationError("invalid graph record metadata")
        if (self.residue_index.dtype != np.int32 or self.residue_index.ndim != 1
                or self.residue_index.size == 0):
            raise GraphValidationError("residue_index must be a non-empty int32 vector")
        node_count = int(self.residue_index.shape[0])
        if (self.node_roles.dtype != np.uint8
                or self.node_roles.shape != (node_count,)):
            raise GraphValidationError("node_roles must match residue_index")
        if (self.node_features.dtype != np.float32
                or self.node_features.shape
                != (node_count, self.spec.node_feature_dim)):
            raise GraphValidationError("invalid node feature array")
        if (self.edge_index.dtype != np.int32 or self.edge_index.ndim != 2
                or self.edge_index.shape[0] != 2):
            raise GraphValidationError("edge_index must have shape (2, E) and int32 dtype")
        if (self.edge_types.dtype != np.uint8 or self.edge_types.ndim != 1
                or self.edge_types.shape[0] != self.edge_index.shape[1]):
            raise GraphValidationError("edge_types must have shape (E,) and uint8 dtype")
        if self.edge_index.size and (
                int(self.edge_index.min()) < 0
                or int(self.edge_index.max()) >= node_count):
            raise GraphValidationError("edge index outside graph node range")
        if self.edge_types.size and int(self.edge_types.max()) >= self.spec.edge_dim:
            raise GraphValidationError("edge type outside graph feature range")
        if (int(self.residue_index.min()) < 0
                or int(self.residue_index.max()) >= length):
            raise GraphValidationError("residue index outside source sequence")
        if (node_count > 1
                and not bool(np.all(np.diff(self.residue_index) > 0))):
            raise GraphValidationError("residue_index must be strictly increasing")
        allowed = {int(NODE_ROLE_CORE), int(NODE_ROLE_CONTEXT)}
        if set(int(value) for value in np.unique(self.node_roles)) - allowed:
            raise GraphValidationError("unknown node role")
        if not bool(np.any(self.node_roles == NODE_ROLE_CORE)):
            raise GraphValidationError("graph has no core nodes")

    @property
    def length(self) -> int:
        """Length of the source molecule, not the selected node count."""
        return len(self.sequence)

    @property
    def node_count(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def core_count(self) -> int:
        return int(np.count_nonzero(self.node_roles == NODE_ROLE_CORE))

    @property
    def core_mask(self) -> np.ndarray:
        return self.node_roles == NODE_ROLE_CORE

    @property
    def core_positions(self) -> np.ndarray:
        """0-based source coordinates of core nodes, in 5′→3′ order."""
        return self.residue_index[self.core_mask]

    @property
    def core_span(self) -> tuple[int, int]:
        """Half-open ``[start, end)`` covering the contiguous core window."""
        core = self.core_positions
        return int(core[0]), int(core[-1]) + 1

    @property
    def edge_count(self) -> int:
        return self.edge_index.shape[1]


@dataclass(frozen=True, slots=True)
class GraphShard:
    """A persistent scheduling unit containing one or more RNA graphs."""

    identifiers: tuple[str, ...]
    sequences: tuple[str, ...]
    structures: tuple[str, ...]
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_types: np.ndarray
    node_ptr: np.ndarray
    edge_ptr: np.ndarray
    spec: GraphSpec
    residue_index: np.ndarray
    node_roles: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.identifiers)
        if count == 0:
            raise GraphValidationError("a graph shard cannot be empty")
        if (not all(isinstance(value, str) and value
                    for value in self.identifiers)
                or not all(isinstance(value, str) and value
                           for value in self.sequences)
                or not all(isinstance(value, str)
                           for value in self.structures)):
            raise GraphValidationError("invalid graph shard record metadata")
        if len(set(self.identifiers)) != count:
            raise GraphValidationError("duplicate identifiers in graph shard")
        if len(self.sequences) != count or len(self.structures) != count:
            raise GraphValidationError("graph shard metadata count mismatch")
        if self.node_ptr.dtype != np.int64 or self.node_ptr.shape != (count + 1,):
            raise GraphValidationError("node_ptr must have shape (B + 1,) and int64 dtype")
        if self.edge_ptr.dtype != np.int64 or self.edge_ptr.shape != (count + 1,):
            raise GraphValidationError("edge_ptr must have shape (B + 1,) and int64 dtype")
        if (self.node_ptr[0] != 0 or self.edge_ptr[0] != 0
                or np.any(np.diff(self.node_ptr) <= 0)
                or np.any(np.diff(self.edge_ptr) < 0)):
            raise GraphValidationError("invalid graph shard offsets")
        node_count = int(self.node_ptr[-1])
        edge_count = int(self.edge_ptr[-1])
        if (self.node_features.dtype != np.float32
                or self.node_features.shape
                != (node_count, self.spec.node_feature_dim)):
            raise GraphValidationError("invalid shard node feature array")
        if (self.edge_index.dtype != np.int32
                or self.edge_index.shape != (2, edge_count)):
            raise GraphValidationError("invalid shard edge index array")
        if (self.edge_types.dtype != np.uint8
                or self.edge_types.shape != (edge_count,)):
            raise GraphValidationError("invalid shard edge type array")
        if (self.residue_index.dtype != np.int32
                or self.residue_index.shape != (node_count,)):
            raise GraphValidationError("invalid shard residue_index array")
        if (self.node_roles.dtype != np.uint8
                or self.node_roles.shape != (node_count,)):
            raise GraphValidationError("invalid shard node_roles array")
        if self.edge_index.size and (
                int(self.edge_index.min()) < 0
                or int(self.edge_index.max()) >= node_count):
            raise GraphValidationError("edge index outside shard node range")
        if self.edge_types.size and int(self.edge_types.max()) >= self.spec.edge_dim:
            raise GraphValidationError("edge type outside shard feature range")
        if any(len(sequence) != len(structure)
               for sequence, structure in zip(self.sequences, self.structures)):
            raise GraphValidationError("sequence/structure length mismatch in shard")
        allowed = {int(NODE_ROLE_CORE), int(NODE_ROLE_CONTEXT)}
        if set(int(value) for value in np.unique(self.node_roles)) - allowed:
            raise GraphValidationError("unknown node role")
        for index in range(count):
            node_start, node_stop = (
                int(self.node_ptr[index]), int(self.node_ptr[index + 1]))
            residue = self.residue_index[node_start:node_stop]
            roles = self.node_roles[node_start:node_stop]
            source_length = len(self.sequences[index])
            if residue.size == 0:
                raise GraphValidationError("graph has no nodes")
            if int(residue.min()) < 0 or int(residue.max()) >= source_length:
                raise GraphValidationError("residue index outside source sequence")
            if residue.size > 1 and not bool(np.all(np.diff(residue) > 0)):
                raise GraphValidationError("residue_index must be strictly increasing")
            if not bool(np.any(roles == NODE_ROLE_CORE)):
                raise GraphValidationError("graph has no core nodes")

    @property
    def record_count(self) -> int:
        return len(self.identifiers)

    @property
    def node_count(self) -> int:
        return int(self.node_ptr[-1])

    @property
    def edge_count(self) -> int:
        return int(self.edge_ptr[-1])

    @property
    def lengths(self) -> tuple[int, ...]:
        """Selected node counts per graph, including context nodes."""
        return tuple(int(value) for value in np.diff(self.node_ptr))

    @property
    def core_counts(self) -> tuple[int, ...]:
        return tuple(
            int(np.count_nonzero(
                self.node_roles[int(self.node_ptr[index]):
                                int(self.node_ptr[index + 1])]
                == NODE_ROLE_CORE))
            for index in range(self.record_count)
        )

    @property
    def edge_counts(self) -> tuple[int, ...]:
        return tuple(int(value) for value in np.diff(self.edge_ptr))

    @classmethod
    def from_graphs(cls, graphs: Sequence[Graph]) -> "GraphShard":
        graphs = list(graphs)
        if not graphs:
            raise GraphValidationError("cannot create a shard without graphs")
        spec = graphs[0].spec
        if any(graph.spec.sha256 != spec.sha256 for graph in graphs):
            raise GraphCompatibilityError(
                "all graphs in a shard must use the same graph specification")
        node_ptr = np.zeros(len(graphs) + 1, dtype=np.int64)
        edge_ptr = np.zeros(len(graphs) + 1, dtype=np.int64)
        node_ptr[1:] = np.cumsum([graph.node_count for graph in graphs])
        edge_ptr[1:] = np.cumsum([graph.edge_count for graph in graphs])
        if int(node_ptr[-1]) > np.iinfo(np.int32).max:
            raise GraphValidationError(
                "graph shard exceeds the int32 node-index capacity; split it")
        edge_parts = [
            graph.edge_index + np.int32(node_ptr[index])
            for index, graph in enumerate(graphs)
        ]
        return cls(
            identifiers=tuple(graph.identifier for graph in graphs),
            sequences=tuple(graph.sequence for graph in graphs),
            structures=tuple(graph.structure for graph in graphs),
            node_features=np.ascontiguousarray(np.concatenate(
                [graph.node_features for graph in graphs], axis=0)),
            edge_index=np.ascontiguousarray(np.concatenate(edge_parts, axis=1)),
            edge_types=np.ascontiguousarray(np.concatenate(
                [graph.edge_types for graph in graphs], axis=0)),
            node_ptr=node_ptr,
            edge_ptr=edge_ptr,
            spec=spec,
            residue_index=np.ascontiguousarray(np.concatenate(
                [graph.residue_index for graph in graphs], axis=0)),
            node_roles=np.ascontiguousarray(np.concatenate(
                [graph.node_roles for graph in graphs], axis=0)),
        )

    def slice(self, start: int, stop: int) -> "GraphShard":
        """Return a contiguous range of graphs with rebased indices."""
        if not 0 <= start < stop <= self.record_count:
            raise IndexError("invalid graph shard slice")
        node_start, node_stop = int(self.node_ptr[start]), int(self.node_ptr[stop])
        edge_start, edge_stop = int(self.edge_ptr[start]), int(self.edge_ptr[stop])
        node_ptr = np.ascontiguousarray(
            self.node_ptr[start:stop + 1] - node_start, dtype=np.int64)
        edge_ptr = np.ascontiguousarray(
            self.edge_ptr[start:stop + 1] - edge_start, dtype=np.int64)
        edge_index = np.ascontiguousarray(
            self.edge_index[:, edge_start:edge_stop] - np.int32(node_start),
            dtype=np.int32,
        )
        return GraphShard(
            identifiers=self.identifiers[start:stop],
            sequences=self.sequences[start:stop],
            structures=self.structures[start:stop],
            node_features=np.ascontiguousarray(
                self.node_features[node_start:node_stop]),
            edge_index=edge_index,
            edge_types=np.ascontiguousarray(
                self.edge_types[edge_start:edge_stop]),
            node_ptr=node_ptr,
            edge_ptr=edge_ptr,
            spec=self.spec,
            residue_index=np.ascontiguousarray(
                self.residue_index[node_start:node_stop]),
            node_roles=np.ascontiguousarray(
                self.node_roles[node_start:node_stop]),
        )

    def validate_values(self) -> None:
        """Run optional linear-time numerical and graph-isolation checks."""
        if not np.isfinite(self.node_features).all():
            raise GraphValidationError("non-finite node features in graph shard")
        for index in range(self.record_count):
            edge_start, edge_stop = int(self.edge_ptr[index]), int(self.edge_ptr[index + 1])
            if edge_start == edge_stop:
                continue
            node_start, node_stop = int(self.node_ptr[index]), int(self.node_ptr[index + 1])
            edges = self.edge_index[:, edge_start:edge_stop]
            if int(edges.min()) < node_start or int(edges.max()) >= node_stop:
                raise GraphValidationError("edge crosses graph boundaries")


class GraphBuilder:
    """Deterministically convert validated RNA records into model-ready graphs.

    ``keep_paired_neighbours`` retains nucleotides outside a requested
    window when they are base-paired with a core nucleotide. ``context_hops``
    is the depth of that neighbourhood: hop 1 is the crossing-pair partner;
    further hops follow every graph edge (backbone, pairing, and skip-2).
    """

    def __init__(
        self,
        spec: GraphSpec | None = None,
        *,
        keep_paired_neighbours: bool = False,
        context_hops: int = 1,
    ) -> None:
        if context_hops < 1:
            raise ValueError("context_hops must be >= 1")
        self.spec = spec if spec is not None else GraphSpec.bundled()
        self.keep_paired_neighbours = bool(keep_paired_neighbours)
        self.context_hops = int(context_hops)

    def build(self, record: RNA) -> Graph:
        graph = self._build_full(record)
        if not record.sliced:
            return graph
        return _extract_slice(
            graph,
            start=record.start,
            end=record.end,
            keep_paired_neighbours=self.keep_paired_neighbours,
            context_hops=self.context_hops,
        )

    def _build_full(self, record: RNA) -> Graph:
        length = record.length
        base_lookup = {base: index for index, base in enumerate("ACGU")}
        bases = np.zeros((length, 4), dtype=np.float32)
        bases[np.arange(length),
              [base_lookup[base] for base in record.sequence]] = 1
        if self.spec.struct_feature == "A":
            structural = np.fromiter(
                (char != "." for char in record.structure), dtype=np.float32,
                count=length)[:, None]
        else:
            state_lookup = {"(": 0, ".": 1, ")": 2}
            structural = np.zeros((length, 3), dtype=np.float32)
            structural[np.arange(length),
                       [state_lookup[char] for char in record.structure]] = 1
        features = [bases, structural]
        if self.spec.positional:
            relative = (np.arange(length, dtype=np.float32)
                        / max(length - 1, 1))[:, None]
            features.append(np.concatenate(
                [np.sin(np.pi * relative), np.cos(np.pi * relative)], axis=1))

        partners = _pair_table(record.structure)
        source: list[int] = []
        destination: list[int] = []
        edge_types: list[int] = []
        backbone = list(range(length - 1))
        source.extend(backbone)
        destination.extend(index + 1 for index in backbone)
        edge_types.extend([_EDGE_TYPES["backbone_forward"]] * len(backbone))
        source.extend(index + 1 for index in backbone)
        destination.extend(backbone)
        edge_types.extend([_EDGE_TYPES["backbone_reverse"]] * len(backbone))
        openings = np.nonzero(partners > np.arange(length))[0]
        closings = partners[openings]
        source.extend(int(value) for value in openings)
        destination.extend(int(value) for value in closings)
        edge_types.extend([_EDGE_TYPES["base_pair_forward"]] * len(openings))
        source.extend(int(value) for value in closings)
        destination.extend(int(value) for value in openings)
        edge_types.extend([_EDGE_TYPES["base_pair_reverse"]] * len(openings))
        if "skip2" in self.spec.extra_edges:
            for index in range(length - 2):
                source.extend((index, index + 2))
                destination.extend((index + 2, index))
                edge_types.extend((
                    _EDGE_TYPES["skip2_forward"],
                    _EDGE_TYPES["skip2_reverse"],
                ))
        edge_index = (
            np.asarray([source, destination], dtype=np.int32)
            if source else np.zeros((2, 0), dtype=np.int32)
        )
        residue_index = np.arange(length, dtype=np.int32)
        node_roles = np.full(length, NODE_ROLE_CORE, dtype=np.uint8)
        return Graph(
            identifier=record.identifier,
            sequence=record.sequence,
            structure=record.structure,
            node_features=np.ascontiguousarray(
                np.concatenate(features, axis=1), dtype=np.float32),
            edge_index=np.ascontiguousarray(edge_index),
            edge_types=np.ascontiguousarray(
                np.asarray(edge_types, dtype=np.uint8)),
            spec=self.spec,
            residue_index=np.ascontiguousarray(residue_index),
            node_roles=np.ascontiguousarray(node_roles),
        )

    def build_many(self, records: Iterable[RNA]) -> list[Graph]:
        return [self.build(record) for record in records]

    def build_shard(self, records: Iterable[RNA]) -> GraphShard:
        return GraphShard.from_graphs(self.build_many(records))


def partition_records(
    records: Iterable[RNA],
    *,
    max_records: int,
    max_nodes: int | None = None,
) -> Iterator[tuple[RNA, ...]]:
    """Partition records into deterministic scheduling units without building."""
    if max_records <= 0:
        raise ValueError("max_records must be positive")
    if max_nodes is not None and max_nodes <= 0:
        raise ValueError("max_nodes must be positive")
    pending: list[RNA] = []
    nodes = 0
    for record in records:
        if max_nodes is not None and record.length > max_nodes:
            raise ValueError(
                f"record {record.identifier!r} exceeds max_nodes={max_nodes}")
        if pending and (
                len(pending) >= max_records
                or (max_nodes is not None and nodes + record.length > max_nodes)):
            yield tuple(pending)
            pending = []
            nodes = 0
        pending.append(record)
        nodes += record.length
    if pending:
        yield tuple(pending)


def _adjacency(edge_index: np.ndarray, node_count: int) -> list[list[int]]:
    neighbours: list[list[int]] = [[] for _ in range(node_count)]
    if edge_index.size == 0:
        return neighbours
    for source, destination in zip(edge_index[0], edge_index[1]):
        neighbours[int(source)].append(int(destination))
    return neighbours


def _select_slice_nodes(
    graph: Graph,
    start: int,
    end: int,
    *,
    keep_paired_neighbours: bool,
    context_hops: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted residue indices and matching core/context roles."""
    core = set(range(start, end))
    selected = set(core)
    if keep_paired_neighbours:
        partners = _pair_table(graph.structure)
        frontier = []
        for index in core:
            partner = int(partners[index])
            if partner >= 0 and partner not in selected:
                selected.add(partner)
                frontier.append(partner)
        if context_hops > 1 and frontier:
            adjacency = _adjacency(graph.edge_index, graph.node_count)
            current = frontier
            for _ in range(context_hops - 1):
                nxt: list[int] = []
                for node in current:
                    for neighbour in adjacency[node]:
                        if neighbour not in selected:
                            selected.add(neighbour)
                            nxt.append(neighbour)
                current = nxt
                if not current:
                    break
    residue_index = np.fromiter(sorted(selected), dtype=np.int32, count=len(selected))
    node_roles = np.where(
        (residue_index >= start) & (residue_index < end),
        NODE_ROLE_CORE,
        NODE_ROLE_CONTEXT,
    ).astype(np.uint8, copy=False)
    return residue_index, node_roles


def _extract_slice(
    graph: Graph,
    start: int | None,
    end: int | None,
    *,
    keep_paired_neighbours: bool,
    context_hops: int,
) -> Graph:
    if start is None or end is None:
        raise GraphValidationError("sliced graph is missing start/end")
    residue_index, node_roles = _select_slice_nodes(
        graph,
        start,
        end,
        keep_paired_neighbours=keep_paired_neighbours,
        context_hops=context_hops,
    )
    remap = {int(old): new for new, old in enumerate(residue_index)}
    if graph.edge_index.size:
        keep = np.isin(graph.edge_index[0], residue_index) & np.isin(
            graph.edge_index[1], residue_index)
        sources = np.fromiter(
            (remap[int(value)] for value in graph.edge_index[0, keep]),
            dtype=np.int32,
            count=int(np.count_nonzero(keep)),
        )
        destinations = np.fromiter(
            (remap[int(value)] for value in graph.edge_index[1, keep]),
            dtype=np.int32,
            count=int(sources.shape[0]),
        )
        edge_index = np.vstack((sources, destinations))
        edge_types = np.ascontiguousarray(graph.edge_types[keep])
    else:
        edge_index = np.zeros((2, 0), dtype=np.int32)
        edge_types = np.zeros((0,), dtype=np.uint8)
    return Graph(
        identifier=graph.identifier,
        sequence=graph.sequence,
        structure=graph.structure,
        node_features=np.ascontiguousarray(graph.node_features[residue_index]),
        edge_index=np.ascontiguousarray(edge_index),
        edge_types=edge_types,
        spec=graph.spec,
        residue_index=np.ascontiguousarray(residue_index),
        node_roles=np.ascontiguousarray(node_roles),
    )


def _shard_has_nontrivial_node_metadata(shard: GraphShard) -> bool:
    """True when roles or residue indices cannot be inferred from sequences."""
    if bool(np.any(shard.node_roles != NODE_ROLE_CORE)):
        return True
    expected = np.asarray([len(sequence) for sequence in shard.sequences],
                          dtype=np.int64)
    if not np.array_equal(np.diff(shard.node_ptr), expected):
        return True
    offset = 0
    for sequence in shard.sequences:
        length = len(sequence)
        if not np.array_equal(
                shard.residue_index[offset:offset + length],
                np.arange(length, dtype=np.int32)):
            return True
        offset += length
    return False


def _legacy_node_metadata(
    sequences: Sequence[str],
    node_ptr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize full-molecule roles for shards written before slicing."""
    if node_ptr.dtype != np.int64 or node_ptr.shape != (len(sequences) + 1,):
        raise GraphValidationError("invalid graph shard offsets")
    expected = np.asarray([len(sequence) for sequence in sequences], dtype=np.int64)
    if not np.array_equal(np.diff(node_ptr), expected):
        raise GraphValidationError("sequence lengths do not match node offsets")
    node_count = int(node_ptr[-1])
    residue_parts = [
        np.arange(len(sequence), dtype=np.int32) for sequence in sequences]
    residue_index = (
        np.concatenate(residue_parts) if residue_parts
        else np.zeros((0,), dtype=np.int32))
    node_roles = np.full(node_count, NODE_ROLE_CORE, dtype=np.uint8)
    return residue_index, node_roles


def _pair_table(structure: str) -> np.ndarray:
    partners = np.full(len(structure), -1, dtype=np.int32)
    stack: list[int] = []
    for index, char in enumerate(structure):
        if char == "(":
            stack.append(index)
        elif char == ")":
            opening = stack.pop()
            partners[opening] = index
            partners[index] = opening
    return partners


def graph_metadata_path(tensor_path: str | Path) -> Path:
    """Return the conventional JSON sidecar path for a tensor shard."""
    path = Path(tensor_path)
    return path.with_suffix(".json")


def save_graph_shard(
    shard: GraphShard,
    tensor_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    checksum: bool = False,
) -> tuple[Path, Path]:
    """Atomically persist a graph shard; content hashing is opt-in."""
    tensor_path = Path(tensor_path)
    metadata_path = (Path(metadata_path) if metadata_path is not None
                     else graph_metadata_path(tensor_path))
    if tensor_path.resolve() == metadata_path.resolve():
        raise ValueError("tensor and metadata paths must be different")
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=tensor_path.parent, prefix=f".{tensor_path.name}.", suffix=".tmp")
    os.close(descriptor)
    temporary_tensor = Path(temporary_name)
    try:
        tensors = {
            "node_features": shard.node_features,
            "edge_index": shard.edge_index,
            "edge_types": shard.edge_types,
            "node_ptr": shard.node_ptr,
            "edge_ptr": shard.edge_ptr,
        }
        if _shard_has_nontrivial_node_metadata(shard):
            tensors["residue_index"] = shard.residue_index
            tensors["node_roles"] = shard.node_roles
        save_file(
            tensors,
            str(temporary_tensor),
            metadata={
                "format": GRAPH_SHARD_FORMAT,
                "format_version": str(GRAPH_SHARD_FORMAT_VERSION),
                "graph_spec_sha256": shard.spec.sha256,
            },
        )
        os.replace(temporary_tensor, tensor_path)
    finally:
        temporary_tensor.unlink(missing_ok=True)
    metadata = {
        "format": GRAPH_SHARD_FORMAT,
        "format_version": GRAPH_SHARD_FORMAT_VERSION,
        "graph_spec": shard.spec.to_dict(),
        "graph_spec_sha256": shard.spec.sha256,
        "tensor_file": tensor_path.name,
        "record_count": shard.record_count,
        "node_count": shard.node_count,
        "edge_count": shard.edge_count,
        "identifiers": list(shard.identifiers),
        "sequences": list(shard.sequences),
        "structures": list(shard.structures),
    }
    if checksum:
        metadata["tensor_sha256"] = _sha256(tensor_path)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=metadata_path.parent,
        prefix=f".{metadata_path.name}.", suffix=".tmp")
    os.close(descriptor)
    temporary_metadata = Path(temporary_name)
    try:
        temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
        os.replace(temporary_metadata, metadata_path)
    finally:
        temporary_metadata.unlink(missing_ok=True)
    return tensor_path, metadata_path


def load_graph_shard(
    tensor_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    expected_spec: GraphSpec | None = None,
    verify_checksum: bool = False,
    validation: Literal["metadata", "full"] = "metadata",
) -> GraphShard:
    """Load and validate a graph shard without deserializing Python objects."""
    tensor_path = Path(tensor_path)
    metadata_path = (Path(metadata_path) if metadata_path is not None
                     else graph_metadata_path(tensor_path))
    if validation not in {"metadata", "full"}:
        raise ValueError("validation must be 'metadata' or 'full'")
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise GraphValidationError(
            f"cannot read graph shard metadata: {error}") from error
    if (metadata.get("format") != GRAPH_SHARD_FORMAT
            or metadata.get("format_version") != GRAPH_SHARD_FORMAT_VERSION):
        raise GraphValidationError("unsupported graph shard format")
    try:
        spec = GraphSpec.from_dict(metadata["graph_spec"])
    except GraphValidationError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise GraphValidationError(
            f"invalid graph specification metadata: {error}") from error
    if metadata.get("graph_spec_sha256") != spec.sha256:
        raise GraphValidationError("graph specification fingerprint mismatch")
    if expected_spec is not None and spec.sha256 != expected_spec.sha256:
        raise GraphCompatibilityError(
            "graph shard specification is incompatible with the encoder")
    if verify_checksum:
        expected_checksum = metadata.get("tensor_sha256")
        if not expected_checksum:
            raise GraphValidationError("graph shard has no stored checksum")
        if _sha256(tensor_path) != expected_checksum:
            raise GraphValidationError("graph shard checksum mismatch")
    try:
        with safe_open(tensor_path, framework="np") as handle:
            tensor_metadata = handle.metadata() or {}
        if (tensor_metadata.get("format") != GRAPH_SHARD_FORMAT
                or tensor_metadata.get("format_version")
                != str(GRAPH_SHARD_FORMAT_VERSION)
                or tensor_metadata.get("graph_spec_sha256") != spec.sha256):
            raise GraphValidationError("tensor header metadata mismatch")
        arrays = load_file(tensor_path)
    except GraphValidationError:
        raise
    except Exception as error:
        raise GraphValidationError(
            f"cannot load graph shard tensors: {error}") from error
    names = set(arrays)
    if _SHARD_REQUIRED_TENSORS - names:
        raise GraphValidationError("graph shard tensor set mismatch")
    unexpected = names - _SHARD_REQUIRED_TENSORS - _SHARD_OPTIONAL_TENSORS
    if unexpected:
        raise GraphValidationError(
            "unexpected graph shard tensor(s): " + ", ".join(sorted(unexpected)))
    if ("residue_index" in arrays) != ("node_roles" in arrays):
        raise GraphValidationError(
            "residue_index and node_roles must be stored together")
    try:
        identifiers = tuple(metadata["identifiers"])
        sequences = tuple(metadata["sequences"])
        node_ptr = arrays["node_ptr"]
        if "residue_index" in arrays:
            residue_index = arrays["residue_index"]
            node_roles = arrays["node_roles"]
        else:
            residue_index, node_roles = _legacy_node_metadata(sequences, node_ptr)
        shard = GraphShard(
            identifiers=identifiers,
            sequences=sequences,
            structures=tuple(metadata["structures"]),
            node_features=arrays["node_features"],
            edge_index=arrays["edge_index"],
            edge_types=arrays["edge_types"],
            node_ptr=node_ptr,
            edge_ptr=arrays["edge_ptr"],
            spec=spec,
            residue_index=residue_index,
            node_roles=node_roles,
        )
    except GraphValidationError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise GraphValidationError(
            f"invalid graph shard metadata: {error}") from error
    if (metadata.get("record_count") != shard.record_count
            or metadata.get("node_count") != shard.node_count
            or metadata.get("edge_count") != shard.edge_count):
        raise GraphValidationError("graph shard count metadata mismatch")
    if validation == "full":
        shard.validate_values()
    return shard


__all__ = [
    "GRAPH_SHARD_FORMAT",
    "GRAPH_SHARD_FORMAT_VERSION",
    "NODE_ROLE_CONTEXT",
    "NODE_ROLE_CORE",
    "Graph",
    "GraphBuilder",
    "GraphCompatibilityError",
    "GraphShard",
    "GraphSpec",
    "GraphValidationError",
    "graph_metadata_path",
    "load_graph_shard",
    "partition_records",
    "save_graph_shard",
]
