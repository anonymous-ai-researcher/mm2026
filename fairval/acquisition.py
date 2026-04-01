"""Acquisition functions for FairVAL: α_acc, α_fair, and composite scoring."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class AccuracyScorer:
    """Gradient-embedding uncertainty scorer (BADGE-style).

    For each unlabeled x, computes:
        g_x = ∇_w ℓ(ŷ_x, σ(w^T φ(x) + b))
        α_acc(x) = ‖g_x‖₂

    Args:
        head: Linear classification head with .weight and .bias.
    """

    def __init__(self, head: nn.Linear):
        self.head = head

    @torch.no_grad()
    def score(
        self, embeddings: torch.Tensor, normalize: bool = True
    ) -> torch.Tensor:
        """Compute accuracy scores for a batch of embeddings.

        Args:
            embeddings: (B, p) tensor.
            normalize: If True, scale scores to [0, 1].

        Returns:
            (B,) tensor of accuracy scores.
        """
        # Pseudo-labels and predicted probabilities
        logits = embeddings @ self.head.weight.T + self.head.bias  # (B, 1)
        probs = torch.sigmoid(logits.squeeze(-1))  # (B,)
        pseudo_labels = (probs > 0.5).float()

        # Gradient embedding: g_x = (p̂ - ŷ) · φ(x)
        residuals = probs - pseudo_labels  # (B,)
        grad_embeddings = residuals.unsqueeze(-1) * embeddings  # (B, p)

        # Gradient norms
        scores = torch.norm(grad_embeddings, dim=-1)  # (B,)

        if normalize and scores.max() > 0:
            scores = scores / scores.max()

        return scores


class FairnessScorer:
    """TPR-gap-driven group urgency scorer.

    For each unlabeled x with estimated group â = D(φ(x)):
        α_fair(x) = w_â(t) · p̂(Y=1 | x)

    where w_a(t) = |TPR̂_a - TPR̄| + c / √max(n_{a,+}, 1)

    Args:
        num_groups: Number of protected groups (k).
        c: Scaling constant for the uncertainty term (default: 1.0).
    """

    def __init__(self, num_groups: int, c: float = 1.0):
        self.num_groups = num_groups
        self.c = c
        # Per-group statistics
        self.tpr_estimates = torch.zeros(num_groups)
        self.positive_counts = torch.zeros(num_groups)

    def update_stats(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        groups: torch.Tensor,
    ):
        """Update per-group TPR estimates from labeled data.

        Args:
            predictions: (N,) binary predictions h(x).
            labels: (N,) true labels Y.
            groups: (N,) group assignments Â.
        """
        for a in range(self.num_groups):
            mask_group_pos = (groups == a) & (labels == 1)
            n_a_plus = mask_group_pos.sum().item()
            self.positive_counts[a] = n_a_plus

            if n_a_plus > 0:
                self.tpr_estimates[a] = (
                    predictions[mask_group_pos].float().mean().item()
                )
            else:
                self.tpr_estimates[a] = 0.0

    def _compute_weights(self) -> torch.Tensor:
        """Compute per-group fairness urgency weights w_a(t).

        w_a(t) = |TPR̂_a - TPR̄| + c / √max(n_{a,+}, 1)
        """
        mean_tpr = self.tpr_estimates.mean()
        tpr_deviation = (self.tpr_estimates - mean_tpr).abs()
        uncertainty = self.c / torch.sqrt(
            torch.clamp(self.positive_counts, min=1.0)
        )
        return tpr_deviation + uncertainty

    @torch.no_grad()
    def score(
        self,
        embeddings: torch.Tensor,
        estimated_groups: torch.Tensor,
        positive_probs: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Compute fairness scores for a batch.

        Args:
            embeddings: (B, p) tensor (unused, for API consistency).
            estimated_groups: (B,) tensor of estimated group labels.
            positive_probs: (B,) tensor of p̂(Y=1|x).
            normalize: If True, scale to [0, 1].

        Returns:
            (B,) tensor of fairness scores.
        """
        weights = self._compute_weights().to(positive_probs.device)  # (k,)
        group_weights = weights[estimated_groups]  # (B,)
        scores = group_weights * positive_probs  # (B,)

        if normalize and scores.max() > 0:
            scores = scores / scores.max()

        return scores


class CompositeScorer:
    """Joint acquisition function combining accuracy and fairness.

    α(x; t) = λ(t) · α_acc(x) + (1 - λ(t)) · α_fair(x)

    Args:
        accuracy_scorer: AccuracyScorer instance.
        fairness_scorer: FairnessScorer instance.
    """

    def __init__(
        self,
        accuracy_scorer: AccuracyScorer,
        fairness_scorer: FairnessScorer,
    ):
        self.accuracy = accuracy_scorer
        self.fairness = fairness_scorer

    def score(
        self,
        embeddings: torch.Tensor,
        estimated_groups: torch.Tensor,
        positive_probs: torch.Tensor,
        lambda_t: float,
    ) -> torch.Tensor:
        """Compute composite acquisition scores.

        Args:
            embeddings: (B, p) tensor.
            estimated_groups: (B,) tensor.
            positive_probs: (B,) tensor.
            lambda_t: Trade-off parameter λ(t) ∈ [0, 1].

        Returns:
            (B,) tensor of composite scores.
        """
        alpha_acc = self.accuracy.score(embeddings, normalize=True)
        alpha_fair = self.fairness.score(
            embeddings, estimated_groups, positive_probs, normalize=True
        )
        return lambda_t * alpha_acc + (1 - lambda_t) * alpha_fair

    def select_batch(
        self,
        embeddings: torch.Tensor,
        estimated_groups: torch.Tensor,
        positive_probs: torch.Tensor,
        lambda_t: float,
        batch_size: int,
    ) -> torch.Tensor:
        """Select top-m samples by composite score.

        Returns:
            (m,) tensor of selected indices.
        """
        scores = self.score(
            embeddings, estimated_groups, positive_probs, lambda_t
        )
        _, indices = scores.topk(min(batch_size, len(scores)))
        return indices
