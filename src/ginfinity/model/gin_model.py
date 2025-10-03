from typing import List

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, Set2Set, global_add_pool, global_mean_pool
from torch_geometric.nn.norm import (
    BatchNorm as PYG_BatchNorm,
    GraphNorm,
    InstanceNorm,
    LayerNorm as PYG_LayerNorm,
)

try:  # pragma: no cover - import-time guard for lightweight downstream scripts
    from ginfinity.utils import FORGI_NODE_TYPES
except Exception:  # pragma: no cover - fallback if utils has heavyweight deps unavailable
    FORGI_NODE_TYPES = [
        "five_prime",
        "stem",
        "hairpin",
        "internal",
        "multiloop",
        "three_prime",
        "other",
    ]


class JumpingKnowledgeAggregator(nn.Module):
    """Combine layer-wise node embeddings with flexible Jumping Knowledge modes."""

    def __init__(self, mode: str, hidden_dims: List[int]):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer dimension")

        normalized_mode = (mode or "last").lower()
        valid_modes = {"last", "concat", "attn"}
        if normalized_mode not in valid_modes:
            raise ValueError(f"jk_mode must be one of {sorted(valid_modes)}, got '{mode}'")

        self.mode = normalized_mode
        self.hidden_dims = list(hidden_dims)

        if self.mode == "concat":
            self.output_dim = int(sum(hidden_dims))
            self.projections = None
            self.attn = None
        else:  # 'last' or 'attn'
            self.output_dim = int(hidden_dims[-1])
            self.projections = nn.ModuleList()
            if self.mode == "attn":
                for dim in hidden_dims:
                    if dim == self.output_dim:
                        self.projections.append(nn.Identity())
                    else:
                        self.projections.append(nn.Linear(dim, self.output_dim, bias=False))
                self.attn = nn.Sequential(
                    nn.Linear(self.output_dim, self.output_dim),
                    nn.Tanh(),
                    nn.Linear(self.output_dim, 1, bias=False),
                )
            else:  # 'last'
                self.projections = None
                self.attn = None

    def forward(self, layer_outputs: List[torch.Tensor]) -> torch.Tensor:
        if not layer_outputs:
            raise ValueError("layer_outputs must contain at least one tensor")
        if len(layer_outputs) != len(self.hidden_dims):
            raise ValueError(
                f"Expected {len(self.hidden_dims)} layer outputs, got {len(layer_outputs)}"
            )

        if self.mode == "concat":
            return torch.cat(layer_outputs, dim=-1)

        if self.mode == "last":
            return layer_outputs[-1]

        # Attention over projected layer outputs (mode == 'attn')
        projected = [proj(h) for h, proj in zip(layer_outputs, self.projections)]
        stacked = torch.stack(projected, dim=1)  # [N, L, D]
        scores = self.attn(stacked)  # [N, L, 1]
        weights = torch.softmax(scores, dim=1)
        return torch.sum(weights * stacked, dim=1)

class GINModel(nn.Module):
    def __init__(
        self,
        hidden_dim,
        output_dim,
        graph_encoding="standard",
        gin_layers=1,
        dropout=0.05,
        jk_mode: str = "last",
        pooling_type="global_add_pool",
        node_embed_norm: str = "none",     # {"none","l2","zscore","zscore_l2"}
        eps: float = 1e-6,
        norm_type: str = "graph",          # {"none","batch","graph","layer","instance"}
        use_residual: bool = True,
        normalize_nodes_before_pool: bool = False,
        node_feature_dim: int = None,
        edge_feature_dim: int = 4,
        gin_eps: float = 0.0,              # GIN epsilon parameter
        train_eps: bool = True,            # Whether to make GIN epsilon learnable
    ):
        super().__init__()

        # Process hidden_dim to handle both int and list inputs
        if isinstance(hidden_dim, (int, float)):
            hidden_dims = [int(hidden_dim)] * gin_layers
        elif isinstance(hidden_dim, list):
            if len(hidden_dim) != gin_layers and len(hidden_dim) != 1:
                raise ValueError(
                    f"hidden_dim list must be of length 1 or {gin_layers}, got length {len(hidden_dim)}"
                )
            hidden_dims = hidden_dim if len(hidden_dim) == gin_layers else hidden_dim * gin_layers
        else:
            raise TypeError("hidden_dim must be an integer or a list of integers")

        # Feature dimensionalities (fallbacks depend on encoding)
        if node_feature_dim is None:
            if graph_encoding == "forgi":
                computed_node_dim = 1 + 2 + 4 + 1 + len(FORGI_NODE_TYPES)
            else:
                computed_node_dim = 3
        else:
            computed_node_dim = int(node_feature_dim)

        if edge_feature_dim is None:
            edge_dim = 7 if graph_encoding == "forgi" else 4
        else:
            edge_dim = int(edge_feature_dim)

        input_dim = computed_node_dim

        self.hidden_dims = list(hidden_dims)
        # Jumping Knowledge aggregator over node embeddings
        self.jk_aggregator = JumpingKnowledgeAggregator(jk_mode, self.hidden_dims)
        self.node_output_dim = self.jk_aggregator.output_dim

        # Store metadata for checkpointing
        self.metadata = {
            "hidden_dims": list(self.hidden_dims),
            "output_dim": output_dim,
            "graph_encoding": graph_encoding,
            "gin_layers": gin_layers,
            "dropout": dropout,
            "pooling_type": pooling_type,
            "node_embed_norm": node_embed_norm,
            "eps": eps,
            "norm_type": norm_type,
            "use_residual": use_residual,
            "normalize_nodes_before_pool": bool(normalize_nodes_before_pool),
            "node_feature_dim": input_dim,
            "edge_feature_dim": edge_dim,
            "gin_eps": gin_eps,
            "train_eps": train_eps,
            "jk_mode": self.jk_aggregator.mode,
            "node_output_dim": self.node_output_dim,
        }

        # Node encoder
        self.node_encoder = nn.Linear(input_dim, hidden_dims[0])

        # Stacks
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(gin_layers):
            in_dim = hidden_dims[i - 1] if i > 0 else hidden_dims[0]
            out_dim = hidden_dims[i]

            # MLP inside GINEConv (simple 2-layer MLP; keep norms outside for graph-aware ops)
            net_layers = [
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
            ]
            if dropout > 0:
                net_layers.append(nn.Dropout(p=dropout))
            net_layers.extend([
                nn.Linear(out_dim, out_dim),
                nn.ReLU(),
            ])
            net = nn.Sequential(*net_layers)
            self.convs.append(GINEConv(nn=net, eps=gin_eps, train_eps=train_eps, edge_dim=edge_dim))
            self.norms.append(self._make_norm(norm_type, out_dim))
            self.dropouts.append(nn.Dropout(p=dropout) if dropout > 0 else nn.Identity())

        # Pooling head
        if pooling_type == "set2set":
            self.pooling = Set2Set(self.node_output_dim, processing_steps=2)
            self.fc = nn.Linear(2 * self.node_output_dim, output_dim)
        elif pooling_type == "global_mean_pool":
            self.pooling = global_mean_pool
            self.fc = nn.Linear(self.node_output_dim, output_dim)
        else:
            self.pooling = global_add_pool
            self.fc = nn.Linear(self.node_output_dim, output_dim)

        # Output (post-hoc) node-embedding normalization config
        self.node_embed_norm = node_embed_norm.lower()
        self.eps = eps

        # μ/σ buffers for z-score of node embeddings (dimension = node_output_dim)
        self.register_buffer("node_mu", torch.zeros(self.node_output_dim), persistent=True)
        self.register_buffer("node_sigma", torch.ones(self.node_output_dim), persistent=True)

        self.use_residual = bool(use_residual)
        self.normalize_nodes_before_pool = bool(normalize_nodes_before_pool)

    # ---- internal helpers ----
    def _make_norm(self, norm_type: str, dim: int):
        norm_type = norm_type.lower()
        if norm_type == "none":
            return nn.Identity()
        elif norm_type == "batch":
            # Graph-aware BatchNorm from PyG; can accept (x, batch) but batch is optional
            return PYG_BatchNorm(dim)
        elif norm_type == "graph":
            return GraphNorm(dim)
        elif norm_type == "layer":
            # PyG LayerNorm is feature-wise and graph-aware
            return PYG_LayerNorm(dim)
        elif norm_type == "instance":
            return InstanceNorm(dim)
        else:
            raise ValueError("norm_type must be one of {'none','batch','graph','layer','instance'}")

    # ---------------- Checkpoint I/O ----------------
    @staticmethod
    def load_from_checkpoint(checkpoint_path, device='cpu'):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        metadata = checkpoint['metadata']
        node_feature_dim = metadata.get('node_feature_dim')
        edge_feature_dim = metadata.get('edge_feature_dim')
        if edge_feature_dim is None:
            edge_feature_dim = 4 if node_feature_dim is not None else 2
        model = GINModel(
            hidden_dim=metadata['hidden_dims'],
            output_dim=metadata['output_dim'],
            graph_encoding=metadata['graph_encoding'],
            gin_layers=metadata['gin_layers'],
            dropout=metadata['dropout'],
            pooling_type=metadata['pooling_type'],
            node_embed_norm=metadata.get('node_embed_norm', 'none'),
            eps=metadata.get('eps', 1e-6),
            norm_type=metadata.get('norm_type', 'none'),
            use_residual=metadata.get('use_residual', False),
            normalize_nodes_before_pool=metadata.get('normalize_nodes_before_pool', False),
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            gin_eps=metadata.get('gin_eps', 0.0),
            train_eps=metadata.get('train_eps', True),
            jk_mode=metadata.get('jk_mode', 'last'),
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def save_checkpoint(self, path, optimizer=None, epoch=None):
        checkpoint = {
            'metadata': self.metadata,
            'state_dict': self.state_dict(),
        }
        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        torch.save(checkpoint, path)

    # ------------- Node stats (for post-hoc normalization) -------------
    @torch.no_grad()
    def set_node_stats(self, mu: torch.Tensor, sigma: torch.Tensor):
        if mu.shape != self.node_mu.shape or sigma.shape != self.node_sigma.shape:
            raise ValueError(f"mu/sigma must have shape {self.node_mu.shape}, got {mu.shape}/{sigma.shape}")
        sigma = torch.clamp(sigma, min=self.eps)
        self.node_mu.copy_(mu)
        self.node_sigma.copy_(sigma)

    @torch.no_grad()
    def fit_node_stats_from_loader(self, loader, device='cpu', max_batches=None):
        self.eval()
        s = None
        ss = None
        n = 0
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            batch = batch.to(device)
            x = self._node_embeds_no_norm(batch)  # raw node embeddings
            if s is None:
                s = x.sum(dim=0)
                ss = (x * x).sum(dim=0)
            else:
                s += x.sum(dim=0)
                ss += (x * x).sum(dim=0)
            n += x.size(0)
        if n == 0:
            raise RuntimeError("No nodes seen while fitting node stats.")
        mu = s / n
        var = ss / n - mu * mu
        var = torch.clamp(var, min=0.0)
        sigma = torch.sqrt(var + self.eps)
        self.set_node_stats(mu, sigma)

    def set_node_embed_norm(self, mode: str):
        mode = mode.lower()
        if mode not in {"none","l2","zscore","zscore_l2"}:
            raise ValueError("node_embed_norm must be one of {'none','l2','zscore','zscore_l2'}")
        self.node_embed_norm = mode
        self.metadata["node_embed_norm"] = mode

    def set_normalize_nodes_before_pool(self, enabled: bool):
        self.normalize_nodes_before_pool = bool(enabled)
        self.metadata["normalize_nodes_before_pool"] = self.normalize_nodes_before_pool

    # ---------------- Core forward pieces ----------------
    def _encode_graph(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 1) initial projection
        x = self.node_encoder(x)

        # 2) conv → norm → (dropout) [+ residual]
        layer_outputs = []
        for i, conv in enumerate(self.convs):
            h_in = x
            x = conv(x, edge_index, edge_attr)
            # graph-aware norm layers (PyG norms accept (x, batch))
            norm = self.norms[i]
            # Not all norms require batch; PyG handles both signatures
            try:
                x = norm(x, batch)  # GraphNorm/BatchNorm/LN in PyG accept (x, batch)
            except TypeError:
                x = norm(x)         # Fallback if the norm doesn't take batch
            x = self.dropouts[i](x)
            if self.use_residual and h_in.shape == x.shape:
                x = x + h_in
            layer_outputs.append(x)
        return self.jk_aggregator(layer_outputs)

    def _apply_node_norm(self, x):
        mode = self.node_embed_norm
        if mode == "none":
            return x
        if mode.startswith("zscore"):
            x = (x - self.node_mu) / (self.node_sigma + self.eps)
        if mode.endswith("l2") or mode == "l2":
            norms = torch.linalg.norm(x, dim=1, keepdim=True)
            x = x / torch.clamp(norms, min=self.eps)
        return x

    def _node_embeds_no_norm(self, data):
        return self._encode_graph(data)

    def get_node_embeddings(self, data, apply_norm: bool = True):
        x = self._encode_graph(data)
        if apply_norm:
            x = self._apply_node_norm(x)
        return x

    def pool_and_project(self, x, batch):
        x_pooled = self.pooling(x, batch)
        return self.fc(x_pooled)

    def forward_once(self, data, normalize_nodes_before_pool: bool = None):
        if normalize_nodes_before_pool is None:
            normalize_nodes_before_pool = self.normalize_nodes_before_pool
        x = self.get_node_embeddings(data, apply_norm=normalize_nodes_before_pool)
        return self.pool_and_project(x, data.batch)

    def forward(self, anchor, positive, negative):
        a = self.forward_once(anchor)
        p = self.forward_once(positive)
        n = self.forward_once(negative)
        return a, p, n
