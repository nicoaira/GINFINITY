import argparse
import dash
from dash import dcc, html, Input, Output, State
import networkx as nx
import torch
import plotly.graph_objects as go
import numpy as np
from flask import Flask

# Import the model and utils
from src.model.gin_model import GINModel
from src.utils import dotbracket_to_graph, graph_to_tensor

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="RNA Secondary Structure GIN Visualizer")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    return parser.parse_args()

args = parse_args()

# Load the trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GINModel.load_from_checkpoint(args.model_path, device=device)
model.to(device)
model.eval()

# Dash app setup
server = Flask(__name__)  # Required for deployment
app = dash.Dash(__name__, server=server)

app.layout = html.Div([
    html.H1("RNA Secondary Structure GIN Visualizer", style={"textAlign": "center"}),

    html.Div([
        html.Label("Paste Dot-Bracket Structure:"),
        dcc.Textarea(
            id="dotbracket-input",
            placeholder="Enter dot-bracket structure here...",
            style={"width": "100%", "height": "100px"}
        ),
        html.Button("Generate Graph", id="generate-button", n_clicks=0, style={"margin": "10px"}),
    ], style={"textAlign": "center"}),

    html.Div([
        html.Label("Select GIN Layer:"),
        dcc.Dropdown(id="layer-dropdown", options=[], value=None),
        
        html.Label("Select Embedding Dimension:"),
        dcc.Dropdown(id="dim-dropdown", options=[], value=None),
    ], style={"width": "40%", "margin": "auto"}),

    dcc.Graph(id="graph-plot", style={"width": "80%", "margin": "auto"}),
])


def extract_layerwise_embeddings(model, data):
    """
    Extract node embeddings after each GINEConv layer.
    This mirrors the logic in model.get_node_embeddings but returns
    intermediate outputs for visualization.
    """
    # Unpack Data object
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    
    # 1) Encode nodes from input_dim -> hidden_dims[0]
    x = model.node_encoder(x)
    
    # 2) Pass through each GINEConv, storing the embedding after each layer
    layerwise_embeddings = []
    for conv in model.convs:
        x = conv(x, edge_index, edge_attr)
        layerwise_embeddings.append(x.clone().detach())  # store the intermediate embedding
        
    return layerwise_embeddings


@app.callback(
    [Output("layer-dropdown", "options"),
     Output("dim-dropdown", "options"),
     Output("layer-dropdown", "value"),
     Output("dim-dropdown", "value"),
     Output("graph-plot", "figure")],
    [Input("generate-button", "n_clicks"),
     Input("layer-dropdown", "value"),
     Input("dim-dropdown", "value")],
    [State("dotbracket-input", "value")]
)
def update_visualization(n_clicks, layer, dim, dotbracket_structure):
    """Handles both graph generation and layer/dimension updates in one callback."""
    ctx = dash.callback_context

    if not ctx.triggered:
        return [], [], None, None, go.Figure()

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == "generate-button":
        # If button clicked, generate the graph and embeddings
        if not dotbracket_structure:
            return [], [], None, None, go.Figure()

        # Convert dot-bracket notation to graph
        graph = dotbracket_to_graph(dotbracket_structure)
        tensor_graph = graph_to_tensor(graph).to(device)

        # Compute embeddings
        layerwise_embeddings = extract_layerwise_embeddings(model, tensor_graph)
        embeddings_np = [emb.cpu().detach().numpy() for emb in layerwise_embeddings]

        num_layers = len(embeddings_np)
        embedding_dim = embeddings_np[0].shape[1]

        # Generate dropdown options
        layer_options = [{"label": f"Layer {i+1}", "value": i} for i in range(num_layers)]
        dim_options = [{"label": f"Dim {d}", "value": d} for d in range(embedding_dim)]

        pos = nx.planar_layout(graph)

        # Default visualization (first layer, first dimension)
        fig = generate_plot(graph, pos, embeddings_np, layer=0, dim=0)

        return layer_options, dim_options, 0, 0, fig

    elif trigger_id in ["layer-dropdown", "dim-dropdown"]:
        # If dropdown changed, update the plot
        if layer is None or dim is None or not dotbracket_structure:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, go.Figure()

        graph = dotbracket_to_graph(dotbracket_structure)
        tensor_graph = graph_to_tensor(graph).to(device)

        layerwise_embeddings = extract_layerwise_embeddings(model, tensor_graph)
        embeddings_np = [emb.cpu().detach().numpy() for emb in layerwise_embeddings]
        pos = nx.planar_layout(graph)

        fig = generate_plot(graph, pos, embeddings_np, layer, dim)

        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, fig

    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, go.Figure()


def generate_plot(graph, pos, embeddings_np, layer, dim):
    """Generate a Plotly figure for the graph."""
    selected_embedding = embeddings_np[layer][:, dim]  # Get embeddings for layer & dim
    min_val, max_val = np.min(selected_embedding), np.max(selected_embedding)

    # Normalize for color scale
    node_colors = (selected_embedding - min_val) / (max_val - min_val + 1e-6)

    # Prepare edges
    edge_x, edge_y = [], []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Prepare nodes
    node_x = [pos[node][0] for node in graph.nodes()]
    node_y = [pos[node][1] for node in graph.nodes()]

    fig = go.Figure()

    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='gray'),
        hoverinfo='none',
        mode='lines'
    ))

    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(size=10, color=node_colors, colorscale='Viridis', showscale=True),
        text=[f"Node {i}: {selected_embedding[i]:.3f}" for i in range(len(graph.nodes()))],
        hoverinfo="text"
    ))

    fig.update_layout(title=f"Layer {layer+1} - Dimension {dim} Embeddings",
                      xaxis=dict(showgrid=False, zeroline=False, visible=False),
                      yaxis=dict(showgrid=False, zeroline=False, visible=False),
                      showlegend=False)

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)