"""Self-contained GINE architecture and graph construction."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    hidden: int
    layers: int
    out_dim: int
    dropout: float
    struct_feature: str
    positional: bool
    residual: bool
    train_eps: bool
    edge_dim: int
    extra_edges: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict) -> "EncoderConfig":
        return cls(**{**value, "extra_edges": tuple(value.get("extra_edges", ()))})


class GINEConv(nn.Module):
    def __init__(self, hidden: int, edge_dim: int, dropout: float,
                 train_eps: bool) -> None:
        super().__init__()
        self.edge_lin = nn.Linear(edge_dim, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 2 * hidden), nn.BatchNorm1d(2 * hidden),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(2 * hidden, hidden))
        self.eps = nn.Parameter(torch.zeros(1), requires_grad=train_eps)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        source, destination = edge_index
        messages = F.relu(
            x.index_select(0, source) + self.edge_lin(edge_attr))
        aggregate = torch.zeros_like(x).index_add_(
            0, destination, messages)
        return self.mlp((1.0 + self.eps) * x + aggregate)


class GINEEncoder(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.cfg = config
        feature_dim = 4 + (1 if config.struct_feature == "A" else 3)
        feature_dim += 2 if config.positional else 0
        self.input = nn.Linear(feature_dim, config.hidden)
        self.convs = nn.ModuleList([
            GINEConv(config.hidden, config.edge_dim, config.dropout,
                     config.train_eps) for _ in range(config.layers)])
        self.norms = nn.ModuleList([
            nn.LayerNorm(config.hidden) for _ in range(config.layers)])
        self.head = nn.Sequential(
            nn.Linear(config.hidden, config.hidden), nn.ReLU(),
            nn.Linear(config.hidden, config.out_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        hidden = self.input(x)
        for convolution, normalization in zip(self.convs, self.norms):
            update = normalization(
                convolution(hidden, edge_index, edge_attr))
            hidden = hidden + update if self.cfg.residual else update
        return self.head(hidden)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
