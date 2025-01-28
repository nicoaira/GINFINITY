import torch.nn as nn
from torch_geometric.nn import global_add_pool, Set2Set
from torch_geometric.nn import GINConv
import torch

class GINModel(nn.Module):
    def __init__(self, hidden_dim, output_dim, graph_encoding="standard", gin_layers=1, dropout=0.05, pooling_type="global_add_pool"):
        super(GINModel, self).__init__()

        # Process hidden_dim to handle both int and list inputs
        if isinstance(hidden_dim, (int, float)):
            hidden_dims = [int(hidden_dim)] * gin_layers
        elif isinstance(hidden_dim, list):
            if len(hidden_dim) != gin_layers and len(hidden_dim) != 1:
                raise ValueError(f"hidden_dim list must be of length 1 or {gin_layers}, got length {len(hidden_dim)}")
            hidden_dims = hidden_dim if len(hidden_dim) == gin_layers else hidden_dim * gin_layers
        else:
            raise TypeError("hidden_dim must be an integer or a list of integers")

        # Store metadata as attributes
        self.metadata = {
            'hidden_dims': hidden_dims,
            'output_dim': output_dim,
            'graph_encoding': graph_encoding,
            'gin_layers': gin_layers,
            'dropout': dropout,
            'pooling_type': pooling_type
        }

        input_dim = 1 if graph_encoding == "standard" else 7

        # Define GIN MLP with variable hidden dimensions
        convs = []
        for i in range(gin_layers):
            if i == 0:
                net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dims[i]),
                    nn.ReLU(),
                    nn.Linear(hidden_dims[i], hidden_dims[i])
                )
            else:
                net = nn.Sequential(
                    nn.Linear(hidden_dims[i-1], hidden_dims[i]),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Linear(hidden_dims[i], hidden_dims[i])
                )
            convs.append(GINConv(net))
        
        self.convs = nn.ModuleList(convs)
       
        # Define pooling layer option
        if pooling_type == "set2set":
            self.pooling = Set2Set(hidden_dims[-1], processing_steps=2)
            self.fc = nn.Linear(2 * hidden_dims[-1], output_dim)
        else:
            self.pooling = global_add_pool
            self.fc = nn.Linear(hidden_dims[-1], output_dim)

    @staticmethod
    def load_from_checkpoint(checkpoint_path, device='cpu'):
        """Load model from checkpoint with metadata"""
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        
        # Create model instance using saved metadata
        model = GINModel(
            hidden_dim=checkpoint['metadata']['hidden_dims'],
            output_dim=checkpoint['metadata']['output_dim'],
            graph_encoding=checkpoint['metadata']['graph_encoding'],
            gin_layers=checkpoint['metadata']['gin_layers'],
            dropout=checkpoint['metadata']['dropout'],
            pooling_type=checkpoint['metadata']['pooling_type']
        )
        
        # Load state dict
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
        for conv in self.convs:
            x = conv(x, edge_index)
        # Return node-level embedding
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