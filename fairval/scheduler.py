"""Adaptive trade-off scheduling for FairVAL.

Implements the sigmoid schedule:
    λ(t) = σ(-β · (Δ̂(t) - γ))
         = 1 / (1 + exp(β · (Δ̂(t) - γ)))

where Δ̂(t) = max_{a,a'} |TPR̂_a - TPR̂_{a'}| is the empirical fairness gap.
"""

import math
import torch
from typing import Optional


class SigmoidScheduler:
    """Adaptive accuracy-fairness trade-off via sigmoid schedule.

    When Δ̂(t) >> γ (severe fairness violation): λ → 0 (fairness-driven).
    When Δ̂(t) << γ (fairness satisfied):       λ → 1 (accuracy-driven).
    When Δ̂(t) ≈ γ (boundary):                  λ ≈ 0.5 (balanced).

    Args:
        gamma: Fairness tolerance γ > 0.
        beta: Temperature controlling transition sharpness (default: 10).
    """

    def __init__(self, gamma: float = 0.05, beta: float = 10.0):
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        self.gamma = gamma
        self.beta = beta
        self._history: list[dict] = []

    def compute_fairness_gap(self, tpr_estimates: torch.Tensor) -> float:
        """Compute Δ̂(t) = max_{a,a'} |TPR̂_a - TPR̂_{a'}|.

        Args:
            tpr_estimates: (k,) tensor of per-group TPR estimates.

        Returns:
            Scalar fairness gap.
        """
        if len(tpr_estimates) < 2:
            return 0.0
        return (tpr_estimates.max() - tpr_estimates.min()).item()

    def step(self, tpr_estimates: torch.Tensor) -> float:
        """Compute λ(t) for the current round.

        Args:
            tpr_estimates: (k,) tensor of current per-group TPR estimates.

        Returns:
            λ(t) ∈ [0, 1].
        """
        gap = self.compute_fairness_gap(tpr_estimates)
        lambda_t = 1.0 / (1.0 + math.exp(self.beta * (gap - self.gamma)))

        self._history.append({
            "gap": gap,
            "lambda": lambda_t,
            "gamma": self.gamma,
        })

        return lambda_t

    @property
    def history(self) -> list[dict]:
        """Returns the full scheduling history."""
        return self._history

    def reset(self):
        """Clear scheduling history."""
        self._history = []


class FixedScheduler:
    """Baseline: fixed λ for ablation studies.

    Args:
        lambda_fixed: Constant trade-off value.
    """

    def __init__(self, lambda_fixed: float = 0.5):
        self.lambda_fixed = lambda_fixed

    def step(self, tpr_estimates: torch.Tensor) -> float:
        return self.lambda_fixed

    def compute_fairness_gap(self, tpr_estimates: torch.Tensor) -> float:
        return (tpr_estimates.max() - tpr_estimates.min()).item()
