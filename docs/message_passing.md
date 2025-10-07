# Message Passing in GINFINITY

## Overview
GINFINITY encodes RNA structures as graphs and processes them with [PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io/) layers. Each graph consists of nodes that describe either nucleotide bases or FORGI structural elements, and directed edges that capture relationships such as backbone adjacency, base pairing, and FORGI parent/child membership. During training and inference, the model stacks modified Graph Isomorphism Network layers (`GINEConv`) to propagate information along these edges.

Routing a graph through the model happens inside `GINModel._encode_graph` (`src/ginfinity/model/gin_model.py`), which repeatedly applies:

1. A learnable linear projection of node features to the hidden dimension.
2. A `GINEConv` message-passing step that mixes neighbour information using edge attributes.
3. A graph-aware normalization layer (BatchNorm/GraphNorm/LayerNorm/InstanceNorm or identity), optional dropout, and—if dimensions allow—a residual skip connection.

The resulting node embeddings can be optionally normalized (z-score and/or L2) before the graph-level pooling head aggregates them into sequence-wide representations.

## Node Features
Node features are created in `src/ginfinity/utils.py` when NetworkX graphs are converted to PyG `Data` objects.

### Standard encoding
For the “standard” encoding, each base node carries four attributes (`_graph_to_tensor_standard`, lines 400–444):

- BackBone pairing status (paired/unpaired), scaled by the `seq_weight` hyperparameter.
- Loop size and loop position, normalized to `[0, 1]`.
- Optional one-hot nucleotide identity terms if the sequence weight is non-zero.

### FORGI encoding
For the FORGI encoding (`_graph_to_tensor_forgi`, lines 445–537), node features extend the above with:

- A binary flag indicating whether the node is a base or a FORGI meta-node.
- One-hot indicators over FORGI node types (hairpin, stem, internal, multiloop, five-prime, three-prime, other).

These richer node descriptors allow the same message-passing layers to reason jointly about sequence-level bases and higher-level structural motifs.

## Edge Attributes
Edges are duplicated in both directions so the message function can specialise by orientation. Every directed edge gets a feature vector appended to the PyG `edge_attr` tensor.

### Standard encoding
Standard graphs use four-dimensional edge features (`utils.py:420-435`):

1. `adjacent`: 1 if the edge connects sequential backbone neighbours.
2. `base_pair`: 1 if the edge represents a complementary base pair.
3. `is_forward`: 1 when `src_index < dst_index`.
4. `is_backward`: 1 when `src_index > dst_index`.

Because both orientations exist, the forward/backward slots swap when traversing the opposite direction, letting the network learn asymmetric transformations along 5'→3' versus 3'→5' flows.

### FORGI encoding
FORGI graphs use seven-dimensional edge features (`utils.py:500-531`):

1. Backbone adjacency.
2. Base-pair connectivity.
3. FORGI parent→child membership (FORGI node → base node).
4. FORGI child→parent membership (base node → FORGI node).
5. FORGI-to-FORGI connections.
6. `is_forward` indicator.
7. `is_backward` indicator.

This layout distinguishes not only the biological relationship but also the direction in which information travels. For example, messages from a base node up to its FORGI parent activate slot 4, whereas messages from the parent down to the base activate slot 3.

## GINEConv Message Passing
Each layer uses PyG’s `GINEConv`, which extends the GIN aggregator to incorporate edge features. For a destination node \(i\) and neighbour \(j\), the message before aggregation is:

\[
\mathbf{m}_{j\rightarrow i} = \operatorname{ReLU}\big(\mathbf{h}_j + W_e\, \mathbf{e}_{j\rightarrow i}\big),
\]

where:

- \(\mathbf{h}_j\) is the current hidden state of node \(j\).
- \(\mathbf{e}_{j\rightarrow i}\) is the edge feature vector (dim = 4 or 7).
- \(W_e\) is a learnable linear projection created automatically when `edge_dim` is passed to `GINEConv`.

After summing these messages, node \(i\)’s new state becomes:

\[
\mathbf{h}'_i = \text{MLP}\Big((1 + \epsilon) \cdot \mathbf{h}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{j\rightarrow i}\Big),
\]

with \(\epsilon\) either fixed or trained per layer. The MLP is the two-layer network built inside `GINModel.__init__`, consisting of Linear → ReLU → (Dropout) → Linear → ReLU.

Because `W_e` and the MLP weights are shared across all edges in a layer, the model learns different transformations for each relationship type by exploiting the sparsity pattern of the edge attributes. For instance:

- Base→FORGI messages emphasise the child→parent feature (slot 4) and can specialise via the corresponding column of \(W_e\).
- FORGI→Base messages emphasise slot 3, letting the layer push aggregated motif information back down to individual bases.
- Base→Base messages activate the adjacency and/or base-pair slots together with direction flags, allowing distinct handling of backbone context versus pairing context.

## Layer Stacking and Global Readout
Multiple such layers are stacked, optionally with residual connections (`GINModel.__init__`). After message passing, node embeddings can be normalised (`_apply_node_norm`) before pooling. Depending on configuration, the model applies one of:

- `global_add_pool`
- `global_mean_pool`
- `Set2Set`

The pooled representation is finally projected to the output dimension via a fully connected layer (`pool_and_project`).

## Practical Implications
- **Directional awareness**: Forward/backward bits make the network sensitive to traversal orientation, even though the underlying graph is undirected.
- **Cross-level communication**: Distinct FORGI membership slots encourage structured information flow between macro-level motifs and base-level nodes.
- **Edge-type specialisation**: The linear edge projection enables the model to learn different biases and mixing patterns for adjacency versus pairing relationships.
- **Scalability**: Using a single shared `GINEConv` per layer keeps the parameter count manageable while still allowing rich relational modelling through the edge attribute basis.

Together, these design choices let GINFINITY encode nuanced RNA structural dependencies while retaining the expressive power of GIN-style message passing.
