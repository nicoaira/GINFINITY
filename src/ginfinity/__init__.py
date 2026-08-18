"""GINFINITY public API."""

from ._validation import InputValidationError, RNA
from .api import Ginfinity, ModelIntegrityError, default_alignment_parameters
from .graph import (GRAPH_SHARD_FORMAT, GRAPH_SHARD_FORMAT_VERSION,
                    NODE_ROLE_CONTEXT, NODE_ROLE_CORE, Graph, GraphBuilder,
                    GraphCompatibilityError, GraphShard, GraphSpec,
                    GraphValidationError, graph_metadata_path,
                    load_graph_shard, partition_records, save_graph_shard)
from .table import read_rna_table

__version__ = "1.2.1"

__all__ = [
    "Ginfinity",
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
    "InputValidationError",
    "ModelIntegrityError",
    "RNA",
    "default_alignment_parameters",
    "graph_metadata_path",
    "load_graph_shard",
    "partition_records",
    "read_rna_table",
    "save_graph_shard",
    "__version__",
]
