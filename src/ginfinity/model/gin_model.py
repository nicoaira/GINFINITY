import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_add_pool, Set2Set, global_mean_pool
from torch_geometric.nn.norm import BatchNorm as PYG_BatchNorm, GraphNorm, LayerNorm as PYG_LayerNorm, InstanceNorm

class GINModel(nn.Module):
    def __init__(
        self,
        hidden_dim,
        output_dim,
        graph_encoding="standard",
        gin_layers=1,
        dropout=0.05,
        pooling_type="global_add_pool",
        node_embed_norm: str = "none",     # {"none","l2","zscore","zscore_l2"}
        eps: float = 1e-6,
        norm_type: str = "graph",          # {"none","batch","graph","layer","instance"}
        use_residual: bool = True,
        normalize_nodes_before_pool: bool = False,
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

        # Store metadata for checkpointing
        self.metadata = {
            "hidden_dims": hidden_dims,
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
        }

        # Node feature dim
        input_dim = 1 if graph_encoding == "standard" else 7
        edge_dim = 2

        # Node encoder
        self.node_encoder = nn.Linear(input_dim, hidden_dims[0])

        # Stacks
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(gin_layers):
            # MLP inside GINEConv (simple 2-layer MLP; keep norms outside for graph-aware ops)
            net = nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i]),
                nn.ReLU(),
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dims[i], hidden_dims[i]),
                nn.ReLU(),
            )
            self.convs.append(GINEConv(nn=net, train_eps=True, edge_dim=edge_dim))
            self.norms.append(self._make_norm(norm_type, hidden_dims[i]))
            self.dropouts.append(nn.Dropout(p=dropout) if dropout > 0 else nn.Identity())

        # Pooling head
        if pooling_type == "set2set":
            self.pooling = Set2Set(hidden_dims[-1], processing_steps=2)
            self.fc = nn.Linear(2 * hidden_dims[-1], output_dim)
        elif pooling_type == "global_mean_pool":
            self.pooling = global_mean_pool
            self.fc = nn.Linear(hidden_dims[-1], output_dim)
        else:
            self.pooling = global_add_pool
            self.fc = nn.Linear(hidden_dims[-1], output_dim)

        # Output (post-hoc) node-embedding normalization config
        self.node_embed_norm = node_embed_norm.lower()
        self.eps = eps

        # μ/σ buffers for z-score of node embeddings (dimension = hidden_dims[-1])
        self.register_buffer("node_mu", torch.zeros(hidden_dims[-1]), persistent=True)
        self.register_buffer("node_sigma", torch.ones(hidden_dims[-1]), persistent=True)

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
        return x

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