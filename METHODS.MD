**Methods**
GINFINITY encodes each RNA secondary structure as a heterogeneous graph whose vertices comprise nucleotide positions and higher-order structural elements. For each base vertex $v$, the feature vector
$$
\mathbf{x}_v = \big[p_v,~ \ell_v,~ \rho_v,~ b_v,~ f_v\big]
$$
collects the pairing indicator $p_v \in \{0,1\}$, loop-relative size $\ell_v$ and position $\rho_v$, an optional sequence one-hot vector $b_v \in \mathbb{R}^4$ scaled by the sequence weight (set to zero in alignment training), and an indicator $f_v$ denoting whether the node represents a nucleotide or a FORGI structural element. FORGI nodes expand the graph with additional vertices that summarize stems, loops, and termini; their feature block $f_v$ is a one-hot over the seven FORGI categories. Edges are attributed by type, capturing backbone adjacency, canonical base pairs, FORGI membership (parent→child and child→parent), and FORGI-to-FORGI contacts; each directed edge $e=(u,v)$ carries a 7-dimensional attribute vector that encodes these relations together with a direction flag, enabling edge-aware message passing.

Node features are linearly projected to the first hidden dimension (512 units), then propagated through $L=4$ edge-aware Graph Isomorphism (GINE) layers. For layer $l$,
$$
\mathbf{h}_v^{(l+1)} = \text{MLP}^{(l)}\!\left((1+\varepsilon^{(l)})\,\mathbf{h}_v^{(l)} + \sum_{u\in\mathcal{N}(v)} \phi^{(l)}\!\big(\mathbf{h}_u^{(l)}, \mathbf{e}_{uv}\big) \right),
$$
where $\varepsilon^{(l)}$ is trainable, $\phi^{(l)}$ is the GINE aggregation that conditions neighbor messages on edge attributes $\mathbf{e}_{uv}$, and $\text{MLP}^{(l)}$ is a two-layer perceptron with ReLU activations and dropout $p=0.05$. Each block applies GraphNorm for per-graph feature stabilization and uses residual connections when dimensions match. The final hidden layer expands to 1024 channels; these vectors constitute the raw node embeddings $\mathbf{h}_v^{(L)}$.

To promote comparability across structures, node embeddings undergo a two-step normalization. Running estimates of the mean $\boldsymbol{\mu}$ and standard deviation $\boldsymbol{\sigma}$ over all nodes seen in training define a z-score transform:
$$
\tilde{\mathbf{h}}_v = \frac{\mathbf{h}_v^{(L)} - \boldsymbol{\mu}}{\boldsymbol{\sigma} + \epsilon}.
$$
A subsequent unit-norm projection $\mathbf{z}_v = \tilde{\mathbf{h}}_v / \lVert\tilde{\mathbf{h}}_v\rVert_2$ (with the same $\epsilon$ safeguard) yields the embeddings used by the loss. This “zscore-then-$L_2$” normalization is also applied on-the-fly before any optional pooling, although the alignment objective operates directly at the node level.

Training batches are constructed around precomputed multiple-structure alignments. For each alignment identifier, all sequences with valid dot-bracket annotations are grouped, ensuring a minimum of two members. Conserved positions are supplied via a JSON mapping that specifies, for every structure, which nucleotide index corresponds to each alignment column and whether the nucleotide is the 5′ partner, the 3′ partner, or unpaired. Nodes lacking conservation labels are considered unaligned. During batching, every conserved node is retained, while up to $M=16$ unaligned nodes per structure are sampled to serve as negatives. Each node $v$ is tagged with a label $y_v$: conserved nodes share a nonnegative alignment index, whereas unaligned nodes receive unique negative labels that are disambiguated per structure, ensuring they never form spurious positives. Additional category metadata $c_v$ (5′-paired, 3′-paired, unpaired, and their unaligned counterparts) steers hard-negative selection.

The alignment objective is a contrastive loss built on cosine similarities. Let $\mathcal{B}$ denote the union of nodes in a mini-batch, and collect their normalized embeddings into $E = [\mathbf{z}_v]_{v\in\mathcal{B}}$. Positive pairs
$$
\mathcal{P} = \{(u,v) \mid y_u = y_v \ge 0,\ \text{graph}(u) \ne \text{graph}(v)\}
$$
are all cross-structure pairs that map to the same conserved alignment column. Negative candidates require distinct labels and different structures. A subset $\mathcal{N}$ is sampled with a bias toward “hard” pairs sharing the same category $c_u = c_v < 3$, controlled by a hard-negative fraction of 0.85; remaining capacity is filled with easier category combinations. The loss
$$
\mathcal{L} = \underbrace{\frac{1}{\lvert\mathcal{P}\rvert} \sum_{(u,v)\in\mathcal{P}} \bigl(1 - \mathbf{z}_u^\top \mathbf{z}_v\bigr)}_{\mathcal{L}_{\text{pos}}} + \underbrace{\frac{1}{\lvert\mathcal{N}\rvert} \sum_{(u,v)\in\mathcal{N}} \max\!\bigl(0,\ \mathbf{z}_u^\top \mathbf{z}_v - m\bigr)}_{\mathcal{L}_{\text{neg}}}
$$
pushes conserved nodes together and enforces a cosine margin $m = 0.2$ for mismatched pairs. By construction, negative sampling always involves at least one conserved node, preventing the objective from degenerating into comparisons among only unaligned positions.

Optimization uses Adam with a learnable $\varepsilon^{(l)}$ in every GINE layer (initialized at zero). Each training round employs a distinct dataset of alignments but reuses the parameters from the previous round (“keep weights”). Round 1 runs up to 200 epochs at an initial learning rate $5\cdot10^{-4}$, decayed multiplicatively by 0.98 after every epoch; patience 10 with minimum improvement $\Delta_{\min}=10^{-4}$ triggers early stopping based on validation loss. Rounds 2 and 3 repeat the 200-epoch budget with learning rates $10^{-4}$ and $10^{-5}$, decay factors 0.95 and 0.98, and patience 15. After each round, the model checkpoint with the best validation loss is retained. Training uses CUDA acceleration with mini-batches of 32 alignment groups, a 3\% validation split stratified by alignment identifier, and 16 dataloader workers that prefetch alignment graphs. An initial evaluation over 5\% of the training and validation batches establishes baseline losses before gradient updates.

The resulting pipeline yields node embeddings that are explicitly organized around alignment-defined correspondence, while still capturing higher-order RNA context via FORGI augmentations and edge-aware message passing.
