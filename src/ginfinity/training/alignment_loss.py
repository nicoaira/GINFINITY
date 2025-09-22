"""Loss utilities for alignment-based training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentContrastiveLoss(nn.Module):
    """Contrastive loss encouraging aligned nodes to be similar.

    The loss is composed of two terms:

    * ``1 - cos_sim`` for nodes that are annotated as equivalent across
      different structures (positives).
    * ``relu(cos_sim - margin)`` for nodes annotated with different
      alignment positions (negatives).

    Parameters
    ----------
    margin:
        Cosine similarity margin used for negative pairs. Similarities above
        ``margin`` are penalized while smaller similarities receive no
        penalty.
    """

    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = float(margin)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        graph_ids: torch.Tensor,
    ) -> torch.Tensor:
        if embeddings.numel() == 0 or embeddings.shape[0] < 2:
            # Return a zero scalar that participates in autograd.
            return embeddings.sum() * 0.0

        # Normalize before computing cosine similarity.
        embeddings = F.normalize(embeddings, p=2, dim=1)
        sim_matrix = embeddings @ embeddings.t()

        same_graph = graph_ids.unsqueeze(0) == graph_ids.unsqueeze(1)
        same_label = labels.unsqueeze(0) == labels.unsqueeze(1)

        different_graphs = ~same_graph
        positive_mask = same_label & different_graphs
        negative_mask = (~same_label) & different_graphs

        loss = torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

        if positive_mask.any():
            pos_sims = sim_matrix[positive_mask]
            loss = loss + (1.0 - pos_sims).mean()

        if negative_mask.any():
            neg_sims = sim_matrix[negative_mask]
            penalties = F.relu(neg_sims - self.margin)
            loss = loss + penalties.mean()

        return loss
