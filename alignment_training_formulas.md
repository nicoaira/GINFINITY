# Alignment Training Mathematical Formulas

## Overview

The alignment training mode in GINFINITY uses a contrastive learning approach to train Graph Isomorphism Networks (GINs) on RNA secondary structures. The goal is to learn node embeddings where structurally equivalent positions across different RNA structures are similar in embedding space, while non-equivalent positions are dissimilar.

## 1. Node Embedding Generation

For each RNA structure graph $G_i$, the GIN model generates node embeddings:

$$\mathbf{h}_i^{(l+1)} = \text{MLP}^{(l+1)}\left((1 + \epsilon^{(l)}) \cdot \mathbf{h}_i^{(l)} + \sum_{j \in \mathcal{N}(i)} \mathbf{h}_j^{(l)}\right)$$

Where:
- $\mathbf{h}_i^{(l)}$ is the embedding of node $i$ at layer $l$
- $\mathcal{N}(i)$ is the neighborhood of node $i$
- $\epsilon^{(l)}$ is a learnable parameter
- $\text{MLP}^{(l+1)}$ is a multi-layer perceptron

The final node embeddings are L2-normalized before computing similarities:

$$\mathbf{e}_i = \frac{\mathbf{h}_i^{(L)}}{\|\mathbf{h}_i^{(L)}\|_2}$$

## 2. Alignment Contrastive Loss

The core loss function is the **AlignmentContrastiveLoss**, which operates on batches containing nodes from multiple RNA structures.

### 2.1 Cosine Similarity Matrix

For a batch of embeddings $\mathbf{E} \in \mathbb{R}^{N \times D}$ (where $N$ is the total number of nodes and $D$ is the embedding dimension), we compute the pairwise cosine similarity matrix:

$$S_{ij} = \mathbf{e}_i^T \mathbf{e}_j$$

Since embeddings are L2-normalized, this directly gives cosine similarities in $[-1, 1]$.

### 2.2 Pair Classification

Each pair of nodes $(i,j)$ is classified based on three criteria:

**Same Graph Check:**
$$\text{same\_graph}_{ij} = \mathbb{I}[\text{graph\_id}_i = \text{graph\_id}_j]$$

**Same Label Check:**
$$\text{same\_label}_{ij} = \mathbb{I}[\text{label}_i = \text{label}_j]$$

**Different Graphs:**
$$\text{different\_graphs}_{ij} = \neg \text{same\_graph}_{ij}$$

### 2.3 Positive and Negative Masks

**Positive Pairs:** Nodes from different structures with the same alignment label (conserved positions)
$$\mathcal{P} = \{(i,j) : \text{same\_label}_{ij} \land \text{different\_graphs}_{ij}\}$$

**Negative Pairs:** Nodes from different structures with different alignment labels (**includes both aligned and unaligned nodes**)
$$\mathcal{N} = \{(i,j) : \neg\text{same\_label}_{ij} \land \text{different\_graphs}_{ij}\}$$

### 2.4 Loss Computation

The total loss combines two terms:

$$\mathcal{L} = \mathcal{L}_{\text{pos}} + \mathcal{L}_{\text{neg}}$$

**Positive Loss:** Encourages aligned nodes to be similar
$$\mathcal{L}_{\text{pos}} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} (1 - S_{ij})$$

**Negative Loss:** Penalizes non-aligned nodes that are too similar
$$\mathcal{L}_{\text{neg}} = \frac{1}{|\mathcal{N}|} \sum_{(i,j) \in \mathcal{N}} \max(0, S_{ij} - m)$$

Where $m$ is the margin parameter (typically $m = 0.2$).

## 3. Label Assignment Strategy

### 3.1 Aligned Nodes

Nodes that correspond to conserved positions across RNA structures receive the same positive integer label:

$$\text{label}_i = \begin{cases}
p & \text{if node } i \text{ corresponds to alignment position } p \\
\end{cases}$$

Where $p \in \{0, 1, 2, \ldots\}$ represents alignment positions from the JSON mapping.

### 3.2 Unaligned Nodes

Each unaligned node receives a unique negative label to ensure they only form negative pairs:

$$\text{label}_i = -(\text{graph\_id} \times 10^6 + \text{offset})$$

Where:
- $\text{graph\_id}$ identifies the RNA structure
- $\text{offset}$ is a unique identifier within that structure

## 4. Negative Pair Composition

**Important Clarification:** Negative pairs include **ALL** nodes with different labels from different structures:

### 4.1 Types of Negative Pairs

1. **Aligned vs. Aligned (Different Positions):**
   - Node from position $p_1$ in structure A vs. node from position $p_2$ in structure B
   - Where $p_1 \neq $p_2$ (both are conserved but at different alignment positions)

2. **Aligned vs. Unaligned:**
   - Node from conserved position $p$ in structure A vs. unaligned node in structure B
   - The aligned node has label $p$, unaligned node has unique negative label

3. **Unaligned vs. Unaligned:**
   - Unaligned node from structure A vs. unaligned node from structure B  
   - Both have different unique negative labels

### 4.2 Mathematical Representation

$$\mathcal{N} = \mathcal{N}_{\text{aligned-aligned}} \cup \mathcal{N}_{\text{aligned-unaligned}} \cup \mathcal{N}_{\text{unaligned-unaligned}}$$

Where:
- $\mathcal{N}_{\text{aligned-aligned}} = \{(i,j) : \text{label}_i, \text{label}_j \geq 0 \land \text{label}_i \neq \text{label}_j \land \text{different\_graphs}_{ij}\}$
- $\mathcal{N}_{\text{aligned-unaligned}} = \{(i,j) : (\text{label}_i \geq 0 \land \text{label}_j < 0) \lor (\text{label}_i < 0 \land \text{label}_j \geq 0) \land \text{different\_graphs}_{ij}\}$
- $\mathcal{N}_{\text{unaligned-unaligned}} = \{(i,j) : \text{label}_i, \text{label}_j < 0 \land \text{label}_i \neq \text{label}_j \land \text{different\_graphs}_{ij}\}$

## 5. Batch Construction

### 5.1 Node Selection

For each alignment containing structures $\{G_1, G_2, \ldots, G_k\}$:

**All Aligned Nodes:** Include all nodes with alignment mappings from the JSON
$$\mathcal{A} = \bigcup_{i=1}^k \{v \in V(G_i) : v \text{ has alignment mapping in JSON}\}$$

**Sampled Unaligned Nodes:** For each structure $G_i$, sample up to $M$ unaligned nodes
$$\mathcal{U}_i = \text{sample}(\{v \in V(G_i) : v \notin \mathcal{A}_i\}, M)$$

Where $M$ is controlled by `--alignment_unaligned_per_graph`.

### 5.2 Final Batch

The final batch for loss computation contains:
$$\mathcal{B} = \mathcal{A} \cup \bigcup_{i=1}^k \mathcal{U}_i$$

## 6. Training Dynamics

### 6.1 Positive Pair Objectives

For nodes $i$ and $j$ from different structures with the same alignment label:
- **Goal:** $S_{ij} \to 1$ (maximize cosine similarity)
- **Loss contribution:** $(1 - S_{ij})$

### 6.2 Negative Pair Objectives  

For **ANY** nodes $i$ and $j$ from different structures with different labels (whether aligned or unaligned):
- **Goal:** $S_{ij} \leq m$ (keep cosine similarity below margin)
- **Loss contribution:** $\max(0, S_{ij} - m)$

### 6.3 Gradient Flow

The gradients encourage:
1. **Conserved positions** (same alignment ID) to cluster together across structures
2. **Different conserved positions** to be separated by at least the margin distance
3. **Unaligned positions** to be separated from all other positions (aligned or unaligned) from different structures
4. **Structure-specific features** to be distinguished from conserved features

## 7. Implementation Details

### 7.1 All-vs-All Comparison

The loss function computes similarities between **all pairs** of nodes in the batch, not just random samples:

- Total comparisons: $O(|\mathcal{B}|^2)$
- Positive pairs: All aligned nodes with same position ID across different structures
- Negative pairs: **All other cross-structure node pairs** (regardless of whether they're aligned or unaligned)

### 7.2 Memory Efficiency

To manage memory usage:
- Batch size is typically 1 (one alignment per batch)
- Unaligned nodes are sampled rather than using all
- Embeddings are computed on-demand during forward pass

### 7.3 Hyperparameters

Key hyperparameters affecting the loss:

| Parameter | Symbol | Typical Value | Effect |
|-----------|--------|---------------|---------|
| Margin | $m$ | 0.2 | Controls negative pair penalty threshold |
| Max Unaligned | $M$ | 16 | Limits negative examples per structure |
| Sample Unaligned | - | True | Whether to randomly sample unaligned nodes |

## 8. Mathematical Properties

### 8.1 Loss Bounds

- **Positive loss:** $\mathcal{L}_{\text{pos}} \in [0, 2]$ (since $S_{ij} \in [-1, 1]$)
- **Negative loss:** $\mathcal{L}_{\text{neg}} \in [0, 1+m]$ (due to ReLU and similarity bounds)

### 8.2 Convergence Properties

As training progresses:
- Positive similarities approach 1: $S_{ij} \to 1$ for $(i,j) \in \mathcal{P}$
- Negative similarities stay below margin: $S_{ij} \leq m$ for $(i,j) \in \mathcal{N}$
- Embedding space develops cluster structure around conserved positions
- Unaligned positions form a "background" that's pushed away from conserved clusters

## 9. Key Insight: Unaligned Nodes as Universal Negatives

**Unaligned nodes serve as "universal negatives"** - they are pushed away from:
- All conserved positions across all structures
- All other unaligned positions from different structures

This creates a learned representation where:
- **Conserved regions** cluster together (positive pairs)
- **Non-conserved regions** are dispersed in embedding space (negative pairs)
- **Structure-specific elements** don't accidentally align with conserved features

This contrastive learning approach enables the model to learn meaningful representations of RNA structural conservation patterns across different sequences.