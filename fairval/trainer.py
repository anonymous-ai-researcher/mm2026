"""Fairness-constrained empirical risk minimization (ERM) trainer."""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional


class LinearHead(nn.Module):
    """Linear classification head: g(φ(x)) = σ(w^T φ(x) + b).

    Args:
        embed_dim: Input dimension (p).
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """(B, p) -> (B,) logits."""
        return self.linear(embeddings).squeeze(-1)

    def predict_proba(self, embeddings: torch.Tensor) -> torch.Tensor:
        """(B, p) -> (B,) probabilities."""
        return torch.sigmoid(self.forward(embeddings))

    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        """(B, p) -> (B,) binary predictions."""
        return (self.predict_proba(embeddings) > 0.5).long()


class FairConstrainedERM:
    """Fairness-constrained ERM via exponentiated-gradient reduction.

    Implements the reduction approach of Agarwal et al. (2018):
    min_h L(h) subject to fairness constraints.

    The Lagrangian is:
        L(h) + μ · violation(h)

    where violation measures the EO/EOD gap.

    Args:
        embed_dim: Backbone embedding dimension.
        num_groups: Number of protected groups.
        gamma: Fairness tolerance.
        lr: Learning rate for the head.
        mu_lr: Learning rate for Lagrange multipliers.
        epochs: Number of training epochs per AL round.
    """

    def __init__(
        self,
        embed_dim: int,
        num_groups: int,
        gamma: float = 0.05,
        lr: float = 1e-2,
        mu_lr: float = 1e-1,
        epochs: int = 100,
    ):
        self.embed_dim = embed_dim
        self.num_groups = num_groups
        self.gamma = gamma
        self.lr = lr
        self.mu_lr = mu_lr
        self.epochs = epochs
        self.head = LinearHead(embed_dim)
        # Lagrange multipliers for per-group EO constraints
        self.mu = torch.zeros(num_groups, requires_grad=False)

    def to(self, device: torch.device) -> "FairConstrainedERM":
        self.head = self.head.to(device)
        self.mu = self.mu.to(device)
        return self

    def _compute_eo_violations(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        groups: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-group EO violation: |TPR_a - TPR̄|.

        Returns:
            (k,) tensor of per-group violations.
        """
        tprs = torch.zeros(self.num_groups, device=predictions.device)
        for a in range(self.num_groups):
            mask = (groups == a) & (labels == 1)
            if mask.sum() > 0:
                tprs[a] = predictions[mask].float().mean()
            else:
                tprs[a] = 0.5  # Default when no positives in group

        mean_tpr = tprs.mean()
        violations = (tprs - mean_tpr).abs()
        return violations

    def train_round(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        groups: torch.Tensor,
    ) -> dict:
        """Train the head on labeled data with fairness constraints.

        Args:
            embeddings: (N_labeled, p) tensor.
            labels: (N_labeled,) binary labels.
            groups: (N_labeled,) group labels.

        Returns:
            Dictionary with training metrics.
        """
        self.head.train()
        optimizer = optim.Adam(self.head.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        best_loss = float("inf")
        for epoch in range(self.epochs):
            logits = self.head(embeddings)
            # Primary loss
            base_loss = criterion(logits, labels.float())

            # Fairness penalty
            preds = (torch.sigmoid(logits) > 0.5).long()
            violations = self._compute_eo_violations(preds, labels, groups)
            # Hinge-style: penalize only when violation > γ
            fair_penalty = torch.clamp(violations - self.gamma, min=0.0)
            lagrangian = base_loss + (self.mu * fair_penalty).sum()

            optimizer.zero_grad()
            lagrangian.backward()
            optimizer.step()

            # Update Lagrange multipliers (exponentiated gradient)
            with torch.no_grad():
                self.mu = torch.clamp(
                    self.mu + self.mu_lr * fair_penalty.detach(), min=0.0
                )

            if base_loss.item() < best_loss:
                best_loss = base_loss.item()

        self.head.eval()
        return {
            "loss": best_loss,
            "violations": violations.detach().cpu().tolist(),
            "mu": self.mu.detach().cpu().tolist(),
        }
