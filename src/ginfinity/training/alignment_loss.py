"""Loss utilities for alignment-based training."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ginfinity.utils import log_information


class AlignmentContrastiveLoss(nn.Module):
    """Fully optimized contrastive loss with vectorized operations for maximum GPU performance.

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
    max_negatives:
        Maximum number of negatives to sample for efficiency. Default: 5000
    """

    def __init__(
        self,
        margin: float = 0.0,
        hard_negative_fraction: float = 0.85,
        max_negatives: int = 5000,
        debug: bool = False,
    ):
        super().__init__()
        self.margin = float(margin)
        self.hard_negative_fraction = float(hard_negative_fraction)
        self.max_negatives = max_negatives
        self.debug_enabled = bool(debug)
        self.debug_log_path: Optional[str] = None
        self._debug_batch_index = 0
        self._last_positive_pairs = 0

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

        # Normalize embeddings once
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Find positive pairs using efficient direct approach
        positive_pairs = self._create_positive_pairs_efficient(labels, graph_ids, categories)
        self._last_positive_pairs = int(positive_pairs.shape[0])
        if self.debug_enabled:
            self._debug_batch_index += 1
        
        zero = embeddings.sum() * 0.0

        pos_loss = zero
        if len(positive_pairs) > 0:
            # Compute positive similarities efficiently
            pos_emb1 = embeddings[positive_pairs[:, 0]]
            pos_emb2 = embeddings[positive_pairs[:, 1]]
            pos_sims = torch.sum(pos_emb1 * pos_emb2, dim=1)
            pos_loss = (1.0 - pos_sims).mean()

        # Compute negative loss using efficient sampling
        neg_loss = self._compute_negative_loss_vectorized(
            embeddings, labels, graph_ids, categories, device
        )

        return pos_loss + neg_loss

    def configure_debug(self, enabled: bool, log_path: Optional[str]) -> None:
        self.debug_enabled = bool(enabled)
        self.debug_log_path = log_path if self.debug_enabled else None

    def _serialize_debug_value(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.item())
            return value.detach().cpu().tolist()
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        return str(value)

    def _log_debug(self, event: str, payload: Dict[str, Any]) -> None:
        if not self.debug_enabled or not self.debug_log_path:
            return
        try:
            info = {"event": event, "batch_index": self._debug_batch_index}
            info.update({k: self._serialize_debug_value(v) for k, v in payload.items()})
            log_information(self.debug_log_path, info, "AlignmentLoss Debug")
        except Exception:
            pass

    def _create_positive_pairs_efficient(self, labels, graph_ids, categories):
        """Create positive pairs using pure tensor operations - no Python loops or .item() calls."""
        device = labels.device
        
        # Only consider conserved positions
        conserved_mask = categories < 3
        if not conserved_mask.any():
            return torch.empty((0, 2), dtype=torch.long, device=device)
            
        # Get conserved indices and their labels/graphs
        conserved_indices = torch.nonzero(conserved_mask, as_tuple=False).squeeze(-1)
        conserved_labels = labels[conserved_indices]
        conserved_graphs = graph_ids[conserved_indices]
        
        n_conserved = len(conserved_indices)
        if n_conserved < 2:
            return torch.empty((0, 2), dtype=torch.long, device=device)
        
        # Create all pairs of conserved indices
        i_indices = torch.arange(n_conserved, device=device).unsqueeze(1).expand(-1, n_conserved)
        j_indices = torch.arange(n_conserved, device=device).unsqueeze(0).expand(n_conserved, -1)
        
        # Keep only upper triangular pairs (i < j)
        upper_tri = i_indices < j_indices
        
        # Filter for same label and different graphs
        same_label = conserved_labels[i_indices] == conserved_labels[j_indices]
        diff_graphs = conserved_graphs[i_indices] != conserved_graphs[j_indices]
        
        # Combine all conditions
        valid_pairs = upper_tri & same_label & diff_graphs
        
        if not valid_pairs.any():
            return torch.empty((0, 2), dtype=torch.long, device=device)
        
        # Get the actual indices
        valid_i, valid_j = torch.nonzero(valid_pairs, as_tuple=True)
        actual_i = conserved_indices[valid_i]
        actual_j = conserved_indices[valid_j]
        
        return torch.stack([actual_i, actual_j], dim=1)

    def _compute_negative_loss_vectorized(self, embeddings, labels, graph_ids, categories, device):
        """Compute negative loss with fully vectorized sampling."""
        n = embeddings.shape[0]
        
        # Sample indices efficiently
        max_samples = min(self.max_negatives, n * n // 4)  # Quarter of all pairs
        
        zero = embeddings.sum() * 0.0

        if max_samples < 100:  # Not enough data
            self._log_debug(
                "negative_sampling_guard",
                {
                    "message": "max_samples < 100 guard triggered; negative loss set to zero",
                    "node_count": n,
                    "max_samples": max_samples,
                    "requested_max_negatives": self.max_negatives,
                    "positive_pairs": self._last_positive_pairs,
                },
            )
            return zero
            
        # Generate random pairs
        idx1 = torch.randint(0, n, (max_samples,), device=device)
        idx2 = torch.randint(0, n, (max_samples,), device=device)
        
        # Vectorized filtering for valid negatives
        different_graphs = graph_ids[idx1] != graph_ids[idx2]
        different_labels = labels[idx1] != labels[idx2]
        at_least_one_conserved = (categories[idx1] < 3) | (categories[idx2] < 3)
        
        valid_neg_mask = different_graphs & different_labels & at_least_one_conserved
        
        if not valid_neg_mask.any():
            self._log_debug(
                "no_valid_negatives",
                {
                    "message": "No valid negative pairs after masking; negative loss set to zero",
                    "node_count": n,
                    "max_samples": max_samples,
                    "positive_pairs": self._last_positive_pairs,
                },
            )
            return zero
            
        # Apply mask to get valid pairs
        valid_idx1 = idx1[valid_neg_mask]
        valid_idx2 = idx2[valid_neg_mask]
        
        # Separate hard and easy negatives vectorized
        same_category = categories[valid_idx1] == categories[valid_idx2]
        
        n_valid = len(valid_idx1)
        n_hard_target = int(n_valid * self.hard_negative_fraction)
        
        # Get hard negatives
        hard_mask = same_category
        hard_indices = torch.nonzero(hard_mask, as_tuple=False).squeeze(-1)
        
        # Get easy negatives  
        easy_mask = ~same_category
        easy_indices = torch.nonzero(easy_mask, as_tuple=False).squeeze(-1)
        
        # Sample from each category
        selected_indices = []
        
        if len(hard_indices) > 0:
            n_hard_sample = min(len(hard_indices), n_hard_target)
            if n_hard_sample < len(hard_indices):
                perm = torch.randperm(len(hard_indices), device=device)[:n_hard_sample]
                selected_indices.append(hard_indices[perm])
            else:
                selected_indices.append(hard_indices)
        
        if len(easy_indices) > 0:
            n_easy_target = max_samples - (len(selected_indices[0]) if selected_indices else 0)
            n_easy_sample = min(len(easy_indices), n_easy_target)
            if n_easy_sample > 0:
                if n_easy_sample < len(easy_indices):
                    perm = torch.randperm(len(easy_indices), device=device)[:n_easy_sample]
                    selected_indices.append(easy_indices[perm])
                else:
                    selected_indices.append(easy_indices)
        
        if not selected_indices:
            self._log_debug(
                "no_selected_negatives",
                {
                    "message": "Sampling produced no negative pairs; negative loss set to zero",
                    "node_count": n,
                    "available_pairs": len(valid_idx1),
                    "positive_pairs": self._last_positive_pairs,
                },
            )
            return zero
            
        # Combine all selected indices
        final_indices = torch.cat(selected_indices)
        
        # Get final pairs
        final_idx1 = valid_idx1[final_indices]
        final_idx2 = valid_idx2[final_indices]
        
        # Compute similarities vectorized
        emb1 = embeddings[final_idx1]
        emb2 = embeddings[final_idx2]
        neg_sims = torch.sum(emb1 * emb2, dim=1)
        
        # Apply margin penalty
        penalties = F.relu(neg_sims - self.margin)
        if len(penalties) == 0:
            self._log_debug(
                "empty_penalties",
                {
                    "message": "All negative similarities below margin; penalty is zero",
                    "node_count": n,
                    "selected_negatives": len(final_idx1),
                    "positive_pairs": self._last_positive_pairs,
                },
            )
            return zero
        return penalties.mean()
