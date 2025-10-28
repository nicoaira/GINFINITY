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
    lambda_contrastive:
        Weight for the contrastive loss term relative to the positive loss.
        Higher values increase the emphasis on separating negatives.
        Default: 1.0
    intra_structure_weight:
        Additional weight for intra-structure negative pairs (same RNA, different positions).
        Higher values penalize false positives within the same structure more strongly.
        Default: 2.0
    diversity_weight:
        Weight for diversity regularization that penalizes low variance within structures.
        Encourages embeddings of different nodes in same structure to remain distinct.
        Default: 0.1
    """

    def __init__(
        self,
        margin: float = 0.0,
        hard_negative_fraction: float = 0.85,
        max_negatives: int = 5000,
        temperature: float = 0.1,
        lambda_contrastive: float = 1.0,
        intra_structure_weight: float = 2.0,
        diversity_weight: float = 0.1,
        debug: bool = False,
    ):
        super().__init__()
        self.margin = float(margin)
        self.hard_negative_fraction = float(hard_negative_fraction)
        self.max_negatives = max_negatives
        self.temperature = float(temperature)
        self.lambda_contrastive = float(lambda_contrastive)
        self.intra_structure_weight = float(intra_structure_weight)
        self.diversity_weight = float(diversity_weight)
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

        # Add diversity regularization to prevent collapse where all nodes in same structure become identical
        diversity_loss = zero
        if self.diversity_weight > 0.0:
            diversity_loss = self._compute_diversity_loss(embeddings, graph_ids)

        return pos_loss + self.lambda_contrastive * contrastive_loss + self.diversity_weight * diversity_loss

    def configure_debug(self, enabled: bool, log_path: Optional[str]) -> None:
        self.debug_enabled = bool(enabled)
        self.debug_log_path = log_path if self.debug_enabled else None

    def set_temperature(self, temperature: float) -> None:
        """Update the temperature parameter (useful for annealing schedules)."""
        self.temperature = float(temperature)

    def _compute_diversity_loss(self, embeddings: torch.Tensor, graph_ids: torch.Tensor) -> torch.Tensor:
        """Compute diversity regularization loss.

        Penalizes low variance of embeddings within each structure.
        This prevents all nodes in the same structure from collapsing to identical embeddings.

        Returns:
            Loss value (higher when variance is low, i.e., nodes are too similar)
        """
        device = embeddings.device
        unique_graphs = torch.unique(graph_ids)

        if len(unique_graphs) == 0:
            return embeddings.sum() * 0.0

        per_graph_diversity_losses = []

        for graph_id in unique_graphs:
            # Get all nodes from this graph
            graph_mask = graph_ids == graph_id
            graph_embeddings = embeddings[graph_mask]

            if graph_embeddings.shape[0] < 2:
                # Need at least 2 nodes to compute variance
                continue

            # Compute variance along feature dimension, then mean across features
            # High variance = high diversity (good), so loss = -log(variance + eps)
            variance = torch.var(graph_embeddings, dim=0, unbiased=False)  # variance per feature
            mean_variance = variance.mean()  # average variance across features

            # Loss is inversely related to variance (we want to maximize variance)
            # Using negative log to make it numerically stable
            diversity_loss_for_graph = -torch.log(mean_variance + 1e-8)
            per_graph_diversity_losses.append(diversity_loss_for_graph)

        if len(per_graph_diversity_losses) == 0:
            return embeddings.sum() * 0.0

        # Average diversity loss across all graphs
        return torch.stack(per_graph_diversity_losses).mean()

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
        normalized) scaled by ``temperature``.  All pairs that do not share the
        same alignment label are treated as negatives, even when the nodes come
        from the same graph.  This prevents the model from mapping unrelated
        positions inside a graph to similar representations, which would in turn
        make cross-graph comparisons degenerate.
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
        # Treat every pair of nodes that do not share an alignment label as a
        # valid negative, regardless of whether the nodes originate from the
        # same structure.  Earlier implementations ignored intra-graph pairs
        # which meant that unrelated positions inside the same RNA could drift
        # towards identical embeddings without receiving any penalty.  By
        # expanding the negative mask to cover all mismatched labels we provide
        # a consistent repulsive signal that keeps unrelated nodes separated
        # both within a graph and across graphs, stabilising the similarity
        # matrix during training.
        negative_mask_subset = ~same_label_subset

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

        # Add explicit penalty for intra-structure negatives (same RNA, different positions)
        # This directly addresses the problem of different loops within the same structure
        # having spuriously high similarity
        if self.intra_structure_weight > 0.0:
            intra_structure_negative_mask = same_graph_subset & negative_mask_subset & (~diag_mask)
            if intra_structure_negative_mask.any():
                # For each node, compute the mean log probability of intra-structure negatives
                # We want to maximize the probability of assigning low similarity to these pairs
                intra_negative_logits = masked_logits.clone()
                intra_negative_logits[~intra_structure_negative_mask] = float("-inf")

                # Count how many intra-structure negatives each node has
                intra_neg_count = intra_structure_negative_mask.sum(dim=1, keepdim=True).float()
                has_intra_negs = intra_neg_count > 0

                if has_intra_negs.any():
                    # Compute log-sum-exp over intra-structure negatives only
                    intra_logsumexp = torch.logsumexp(intra_negative_logits, dim=1, keepdim=True)
                    intra_logsumexp = torch.where(torch.isfinite(intra_logsumexp), intra_logsumexp, torch.zeros_like(intra_logsumexp))

                    # Compute mean similarity to intra-structure negatives (we want this low)
                    # Average over the temperature-scaled logits
                    intra_mean_sim = intra_logsumexp.squeeze() - torch.log(intra_neg_count.squeeze() + 1e-8)
                    intra_mean_sim = intra_mean_sim[has_intra_negs.squeeze()]

                    # Penalize high similarity to intra-structure negatives
                    # The higher the similarity, the higher the penalty
                    intra_structure_penalty = intra_mean_sim.mean()
                    contrastive_loss = contrastive_loss + self.intra_structure_weight * intra_structure_penalty

        # Add a soft margin regularizer to keep unrelated nodes apart even when
        # they slip through the sampling mask.
        if self.margin > 0.0 and negative_mask_subset.any():
            negative_sims = sim_matrix[negative_mask_subset]
            margin_penalty = F.relu(negative_sims - self.margin).mean()
            contrastive_loss = contrastive_loss + margin_penalty

        return contrastive_loss
