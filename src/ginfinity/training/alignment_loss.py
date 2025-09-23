"""Loss utilities for alignment-based training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentContrastiveLoss(nn.Module):
    """Optimized contrastive loss encouraging aligned nodes to be similar with hard negatives.

    The loss is composed of two terms:

    * ``1 - cos_sim`` for nodes that are annotated as equivalent across
      different structures (positives).
    * ``relu(cos_sim - margin)`` for nodes annotated with different
      alignment positions (negatives), with hard negatives prioritized.

    Parameters
    ----------
    margin:
        Cosine similarity margin used for negative pairs. Similarities above
        ``margin`` are penalized while smaller similarities receive no
        penalty.
    hard_negative_fraction:
        Fraction of negatives that should be hard negatives (same category).
        Default: 0.85
    max_negatives_per_positive:
        Maximum number of negatives to sample per positive pair for efficiency.
        Default: 50
    """

    def __init__(self, margin: float = 0.0, hard_negative_fraction: float = 0.85, max_negatives_per_positive: int = 50):
        super().__init__()
        self.margin = float(margin)
        self.hard_negative_fraction = float(hard_negative_fraction)
        self.max_negatives_per_positive = max_negatives_per_positive

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        graph_ids: torch.Tensor,
        categories: torch.Tensor,
    ) -> torch.Tensor:
        if embeddings.numel() == 0 or embeddings.shape[0] < 2:
            return embeddings.sum() * 0.0

        device = embeddings.device
        n = embeddings.shape[0]
        
        # Normalize embeddings once
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Vectorized positive pair identification 
        positive_pairs = self._find_positive_pairs(labels, graph_ids, categories, device)
        if len(positive_pairs) == 0:
            return embeddings.sum() * 0.0
            
        # Compute positive loss efficiently
        pos_emb1 = embeddings[positive_pairs[:, 0]]
        pos_emb2 = embeddings[positive_pairs[:, 1]]
        pos_sims = torch.sum(pos_emb1 * pos_emb2, dim=1)
        pos_loss = (1.0 - pos_sims).mean()
        
        # Sample and compute negative loss efficiently
        neg_loss = self._compute_negative_loss_efficient(
            embeddings, labels, graph_ids, categories, positive_pairs, device
        )
        
        return pos_loss + neg_loss

    def _find_positive_pairs(self, labels, graph_ids, categories, device):
        """Efficiently find positive pairs without creating N×N matrices."""
        # Group indices by label for conserved positions only
        conserved_mask = categories < 3
        if not conserved_mask.any():
            return torch.empty((0, 2), dtype=torch.long, device=device)
            
        conserved_indices = torch.nonzero(conserved_mask, as_tuple=False).squeeze(-1)
        conserved_labels = labels[conserved_indices]
        conserved_graphs = graph_ids[conserved_indices]
        
        positive_pairs = []
        
        # For each unique label, find cross-graph pairs
        unique_labels = torch.unique(conserved_labels)
        for label in unique_labels:
            label_mask = conserved_labels == label
            if label_mask.sum() < 2:
                continue
                
            label_indices = conserved_indices[label_mask]
            label_graphs = conserved_graphs[label_mask]
            
            # Find all pairs with different graphs
            for i in range(len(label_indices)):
                for j in range(i + 1, len(label_indices)):
                    if label_graphs[i] != label_graphs[j]:
                        positive_pairs.append([label_indices[i].item(), label_indices[j].item()])
        
        if not positive_pairs:
            return torch.empty((0, 2), dtype=torch.long, device=device)
            
        return torch.tensor(positive_pairs, dtype=torch.long, device=device)

    def _compute_negative_loss_efficient(self, embeddings, labels, graph_ids, categories, positive_pairs, device):
        """Compute negative loss with efficient sampling."""
        if len(positive_pairs) == 0:
            return torch.tensor(0.0, device=device)
            
        # For efficiency, sample a subset of negative pairs
        n_total = embeddings.shape[0]
        max_negatives = min(len(positive_pairs) * self.max_negatives_per_positive, n_total * 50)
        
        # Sample random pairs for negatives
        n_samples = min(max_negatives, 10000)  # Cap to avoid memory issues
        idx1 = torch.randint(0, n_total, (n_samples,), device=device)
        idx2 = torch.randint(0, n_total, (n_samples,), device=device)
        
        # Filter to get valid negative pairs
        different_graphs = graph_ids[idx1] != graph_ids[idx2]
        different_labels = labels[idx1] != labels[idx2]
        conserved_mask = (categories[idx1] < 3) | (categories[idx2] < 3)
        
        valid_negatives = different_graphs & different_labels & conserved_mask
        if not valid_negatives.any():
            return torch.tensor(0.0, device=device)
            
        idx1_neg = idx1[valid_negatives]
        idx2_neg = idx2[valid_negatives]
        
        # Separate into hard and easy negatives
        same_category = categories[idx1_neg] == categories[idx2_neg]
        n_hard = int(len(idx1_neg) * self.hard_negative_fraction)
        
        # Sample hard negatives
        hard_indices = torch.nonzero(same_category, as_tuple=False).squeeze(-1)
        if len(hard_indices) > n_hard:
            hard_perm = torch.randperm(len(hard_indices), device=device)[:n_hard]
            hard_indices = hard_indices[hard_perm]
        
        # Sample easy negatives
        easy_indices = torch.nonzero(~same_category, as_tuple=False).squeeze(-1)
        n_easy = min(len(easy_indices), max_negatives - len(hard_indices))
        if len(easy_indices) > n_easy:
            easy_perm = torch.randperm(len(easy_indices), device=device)[:n_easy]
            easy_indices = easy_indices[easy_perm]
        
        # Combine hard and easy negatives
        selected_indices = torch.cat([hard_indices, easy_indices]) if len(easy_indices) > 0 else hard_indices
        if len(selected_indices) == 0:
            return torch.tensor(0.0, device=device)
            
        neg_idx1 = idx1_neg[selected_indices]
        neg_idx2 = idx2_neg[selected_indices]
        
        # Compute negative similarities efficiently
        neg_emb1 = embeddings[neg_idx1]
        neg_emb2 = embeddings[neg_idx2]
        neg_sims = torch.sum(neg_emb1 * neg_emb2, dim=1)
        
        # Apply margin-based penalty
        penalties = F.relu(neg_sims - self.margin)
        return penalties.mean()
