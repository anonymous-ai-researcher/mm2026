"""Demographic estimator D: predicts group attribute from backbone embeddings."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class DemographicEstimator(nn.Module):
    """Two-layer MLP that predicts group membership from embeddings.

    Maps φ(x) ∈ ℝ^p → [k], implementing D(φ(x)) = Â.
    Can optionally inject synthetic noise at rate η for evaluation.

    Args:
        embed_dim: Dimension of backbone embeddings (p).
        num_groups: Number of protected groups (k).
        hidden_dim: Hidden layer dimension.
        noise_rate: Synthetic noise rate η ∈ [0, 0.5) for evaluation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_groups: int,
        hidden_dim: int = 128,
        noise_rate: float = 0.0,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.noise_rate = noise_rate

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_groups),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Predict group logits: (B, p) -> (B, k)."""
        return self.mlp(embeddings)

    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Predict group labels: (B, p) -> (B,) with optional noise injection."""
        logits = self.forward(embeddings)
        preds = logits.argmax(dim=-1)

        if self.noise_rate > 0 and self.training is False:
            mask = torch.rand(preds.shape, device=preds.device) < self.noise_rate
            random_groups = torch.randint(
                0, self.num_groups, preds.shape, device=preds.device
            )
            preds = torch.where(mask, random_groups, preds)

        return preds

    def fit(
        self,
        embeddings: torch.Tensor,
        group_labels: torch.Tensor,
        epochs: int = 50,
        lr: float = 1e-3,
    ) -> float:
        """Train the estimator on a pilot set with known group labels.

        Args:
            embeddings: (N_pilot, p) tensor of backbone embeddings.
            group_labels: (N_pilot,) tensor of true group labels in [0, k).
            epochs: Training epochs.
            lr: Learning rate.

        Returns:
            Estimated error rate η̂ on the training set.
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for _ in range(epochs):
            logits = self.forward(embeddings)
            loss = criterion(logits, group_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Estimate η
        self.eval()
        with torch.no_grad():
            preds = self.forward(embeddings).argmax(dim=-1)
            eta_hat = (preds != group_labels).float().mean().item()

        return eta_hat

    def estimate_noise_rate(
        self,
        embeddings: torch.Tensor,
        group_labels: torch.Tensor,
    ) -> float:
        """Estimate η on a held-out set."""
        self.eval()
        with torch.no_grad():
            preds = self.forward(embeddings).argmax(dim=-1)
            return (preds != group_labels).float().mean().item()
