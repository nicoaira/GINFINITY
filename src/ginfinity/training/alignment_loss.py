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
    * A supervised InfoNCE term that contrasts each positive pair against
      a temperature-scaled partition function built from sampled negatives.

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
        Maximum number of additional negative nodes to sample for the InfoNCE
        denominator. A value of ``None`` or ``0`` keeps only nodes that
        participate in positive pairs. Default: 5000
    temperature:
        Temperature applied to cosine similarities inside the InfoNCE term.
        Lower temperatures sharpen the distribution and penalize unrelated
        nodes with high similarity more aggressively. Default: 0.1
    """

    def __init__(
        self,
        margin: float = 0.0,
        hard_negative_fraction: float = 0.85,
        max_negatives: int = 5000,
        temperature: float = 0.1,
        debug: bool = False,
    ):
        super().__init__()
        self.margin = float(margin)
        self.hard_negative_fraction = float(hard_negative_fraction)
        self.max_negatives = max_negatives
        self.temperature = float(temperature)
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

        contrastive_loss = self._compute_contrastive_info_nce(
            embeddings, labels, graph_ids, categories
        )

        return pos_loss + contrastive_loss

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

    def _compute_contrastive_info_nce(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        graph_ids: torch.Tensor,
        categories: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a supervised InfoNCE loss over sampled nodes.

        Nodes that participate in positive pairs are always included while a
        limited number of additional nodes are sampled to act as negatives. The
        logits are built with cosine similarities (embeddings are already
        normalized) scaled by ``temperature``. Only pairs belonging to different
        graphs are considered valid, mirroring the alignment setup.
        """

        device = embeddings.device
        n = embeddings.shape[0]
        zero = embeddings.sum() * 0.0

        if n < 2:
            return zero

        same_graph = graph_ids.unsqueeze(0) == graph_ids.unsqueeze(1)
        same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
        conserved_i = categories.unsqueeze(0) < 3
        conserved_j = categories.unsqueeze(1) < 3

        positive_mask = same_label & (~same_graph) & conserved_i & conserved_j

        if not positive_mask.any():
            self._log_debug(
                "no_positive_pairs",
                {
                    "message": "No positive pairs available for InfoNCE; returning zero loss",
                    "node_count": n,
                },
            )
            return zero

        # Determine which nodes need to be kept in the contrastive set
        participating_mask = positive_mask.any(dim=0) | positive_mask.any(dim=1)
        participating_indices = torch.nonzero(participating_mask, as_tuple=False).squeeze(-1)

        if participating_indices.numel() == 0:
            return zero

        if self.max_negatives is not None and self.max_negatives > 0 and participating_indices.numel() < n:
            selection_mask = torch.zeros(n, dtype=torch.bool, device=device)
            selection_mask[participating_indices] = True

            negative_candidates = torch.nonzero(~selection_mask, as_tuple=False).squeeze(-1)
            if negative_candidates.numel() > 0:
                sample_size = min(self.max_negatives, int(negative_candidates.numel()))
                if sample_size > 0:
                    hard_candidates = negative_candidates[categories[negative_candidates] < 3]
                    easy_candidates = negative_candidates[categories[negative_candidates] >= 3]

                    n_hard = int(round(sample_size * self.hard_negative_fraction))
                    n_hard = min(n_hard, int(hard_candidates.numel()))
                    n_easy = sample_size - n_hard
                    n_easy = min(n_easy, int(easy_candidates.numel()))

                    selected_parts = []
                    if n_hard > 0:
                        perm = torch.randperm(hard_candidates.numel(), device=device)[:n_hard]
                        selected_parts.append(hard_candidates[perm])
                    if n_easy > 0:
                        perm = torch.randperm(easy_candidates.numel(), device=device)[:n_easy]
                        selected_parts.append(easy_candidates[perm])

                    if selected_parts:
                        sampled_negatives = torch.cat(selected_parts)
                        selection_mask[sampled_negatives] = True

            subset_indices = torch.nonzero(selection_mask, as_tuple=False).squeeze(-1)
        else:
            subset_indices = participating_indices

        subset_embeddings = embeddings[subset_indices]
        subset_labels = labels[subset_indices]
        subset_graphs = graph_ids[subset_indices]
        subset_categories = categories[subset_indices]

        sim_matrix = torch.matmul(subset_embeddings, subset_embeddings.T) / max(self.temperature, 1e-8)

        same_graph_subset = subset_graphs.unsqueeze(0) == subset_graphs.unsqueeze(1)
        same_label_subset = subset_labels.unsqueeze(0) == subset_labels.unsqueeze(1)
        conserved_i_subset = subset_categories.unsqueeze(0) < 3
        conserved_j_subset = subset_categories.unsqueeze(1) < 3

        positive_mask_subset = same_label_subset & (~same_graph_subset) & conserved_i_subset & conserved_j_subset
        # Treat every pair drawn from different graphs with different labels
        # as a valid negative.  The previous implementation restricted
        # negatives to cases where at least one node was annotated as
        # conserved.  As training progressed this allowed groups of
        # unannotated nodes to collapse towards the same representation since
        # they never received any explicit repulsive signal.  Including those
        # pairs in the negative mask prevents this behaviour and preserves the
        # contrast between unrelated structures.
        negative_mask_subset = (~same_graph_subset) & (~same_label_subset)

        valid_mask = positive_mask_subset | negative_mask_subset

        if not positive_mask_subset.any():
            self._log_debug(
                "no_positive_pairs_subset",
                {
                    "message": "Positive pairs disappeared after sampling; returning zero loss",
                    "node_count": int(subset_indices.numel()),
                    "original_nodes": n,
                    "positive_pairs": self._last_positive_pairs,
                },
            )
            return zero

        diag_mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=device)
        valid_mask = valid_mask & (~diag_mask)

        masked_logits = sim_matrix.masked_fill(~valid_mask, float("-inf"))

        logsumexp = torch.logsumexp(masked_logits, dim=1, keepdim=True)
        logsumexp = torch.where(torch.isfinite(logsumexp), logsumexp, torch.zeros_like(logsumexp))

        log_probs = masked_logits - logsumexp
        log_probs = torch.where(torch.isfinite(log_probs), log_probs, torch.zeros_like(log_probs))

        positive_log_probs = log_probs[positive_mask_subset]
        if positive_log_probs.numel() == 0:
            return zero

        contrastive_loss = -positive_log_probs.mean()

        # Add a soft margin regularizer to keep unrelated nodes apart even when
        # they slip through the sampling mask.
        if self.margin > 0.0 and negative_mask_subset.any():
            negative_sims = sim_matrix[negative_mask_subset]
            margin_penalty = F.relu(negative_sims - self.margin).mean()
            contrastive_loss = contrastive_loss + margin_penalty

        return contrastive_loss
