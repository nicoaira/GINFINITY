import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_add_pool, Set2Set

class GINModel(nn.Module):
    def __init__(
        self,
        hidden_dim,
        output_dim,
        graph_encoding="standard",
        gin_layers=1,
        dropout=0.05,
        pooling_type="global_add_pool"
    ):
        super(GINModel, self).__init__()

        # Process hidden_dim to handle both int and list inputs
        if isinstance(hidden_dim, (int, float)):
            hidden_dims = [int(hidden_dim)] * gin_layers
        elif isinstance(hidden_dim, list):
            if len(hidden_dim) != gin_layers and len(hidden_dim) != 1:
                raise ValueError(
                    f"hidden_dim list must be of length 1 or {gin_layers}, "
                    f"got length {len(hidden_dim)}"
                )
            hidden_dims = hidden_dim if len(hidden_dim) == gin_layers else hidden_dim * gin_layers
        else:
            raise TypeError("hidden_dim must be an integer or a list of integers")

        # Store metadata as attributes (for checkpointing)
        self.metadata = {
            "hidden_dims": hidden_dims,
            "output_dim": output_dim,
            "graph_encoding": graph_encoding,
            "gin_layers": gin_layers,
            "dropout": dropout,
            "pooling_type": pooling_type
        }

        # 1) Node feature dimension:
        input_dim = 1 if graph_encoding == "standard" else 7

        # 2) Our edges are 2D ([1,0] or [0,1]) => pass edge_dim=2 to GINEConv
        edge_dim = 2  

        # We create one "node_encoder" from `input_dim` to `hidden_dims[0]`.
        self.node_encoder = nn.Linear(input_dim, hidden_dims[0])

        # Build each GINEConv layer
        self.convs = nn.ModuleList()
        for i in range(gin_layers):
            # The MLP that processes the sum x_j + edge_attr_embed
            # Make sure first layer is nn.Linear(...) so GINEConv can infer in_channels
            net = nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i]),
                nn.ReLU(),
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dims[i], hidden_dims[i]),
                nn.ReLU()
            )

            # GINEConv automatically does edge_attr -> [hidden_dims[i]] 
            # if you set edge_dim=2
            conv = GINEConv(nn=net, train_eps=True, edge_dim=2)
            self.convs.append(conv)

        # Pooling layer
        if pooling_type == "set2set":
            self.pooling = Set2Set(hidden_dims[-1], processing_steps=2)
            self.fc = nn.Linear(2 * hidden_dims[-1], output_dim)
        else:
            self.pooling = global_add_pool
            self.fc = nn.Linear(hidden_dims[-1], output_dim)

    @staticmethod
    def load_from_checkpoint(checkpoint_path, device='cpu'):
        """Load model from checkpoint with metadata"""
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        metadata = checkpoint['metadata']

        model = GINModel(
            hidden_dim=metadata['hidden_dims'],
            output_dim=metadata['output_dim'],
            graph_encoding=metadata['graph_encoding'],
            gin_layers=metadata['gin_layers'],
            dropout=metadata['dropout'],
            pooling_type=metadata['pooling_type']
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def save_checkpoint(self, path, optimizer=None, epoch=None):
        """Save model checkpoint with metadata"""
        checkpoint = {
            'metadata': self.metadata,
            'state_dict': self.state_dict(),
        }
        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        torch.save(checkpoint, path)

    def get_node_embeddings(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr

        # 1) Encode nodes from input_dim -> hidden_dims[0]
        x = self.node_encoder(x)

        # 2) Pass through each GINEConv
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)

        return x

    def pool_and_project(self, x, batch):
        x_pooled = self.pooling(x, batch)
        return self.fc(x_pooled)

    def forward_once(self, data):
        x = self.get_node_embeddings(data)
        return self.pool_and_project(x, data.batch)

    def forward(self, anchor, positive, negative):
        # Forward pass for each of the triplet inputs
        anchor_out = self.forward_once(anchor)
        positive_out = self.forward_once(positive)
        negative_out = self.forward_once(negative)
        return anchor_out, positive_out, negative_out
