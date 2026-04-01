"""FairVAL: main active learning loop (Algorithm 1 in the paper)."""

import torch
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional

from .backbone import FrozenBackbone
from .estimator import DemographicEstimator
from .acquisition import AccuracyScorer, FairnessScorer, CompositeScorer
from .scheduler import SigmoidScheduler, FixedScheduler
from .trainer import FairConstrainedERM, LinearHead
from .metrics import full_evaluation, labels_to_target

logger = logging.getLogger(__name__)


@dataclass
class FairVALConfig:
    """Configuration for FairVAL."""
    # Active learning
    budget_fraction: float = 0.10
    rounds: int = 10
    seed_fraction: float = 0.01
    # Fairness
    gamma: float = 0.05
    beta: float = 10.0
    c: float = 1.0
    # Demographic estimator
    noise_rate: float = 0.0
    estimator_hidden: int = 128
    estimator_epochs: int = 50
    # Trainer
    train_lr: float = 1e-2
    train_epochs: int = 100
    # Scheduler
    use_adaptive_lambda: bool = True
    fixed_lambda: float = 0.5
    # Ablation flags
    use_alpha_acc: bool = True
    use_alpha_fair: bool = True
    use_tpr_deviation: bool = True
    use_uncertainty: bool = True


@dataclass
class RoundResult:
    """Result from a single AL round."""
    round_idx: int
    n_labeled: int
    lambda_t: float
    fairness_gap: float
    metrics: dict
    selected_indices: list


class FairVAL:
    """FairVAL active learning algorithm.

    Implements Algorithm 1 from the paper:
        Input: pool U, backbone φ, estimator D, budget B, rounds T, γ, β
        For each round t:
            1. Compute α_acc and α_fair for all unlabeled samples
            2. Adapt λ(t) via sigmoid schedule
            3. Select top-m by composite α
            4. Query oracle, retrain under fairness constraints
        Output: fair classifier h

    Args:
        backbone: Frozen visual backbone.
        num_groups: Number of protected groups (k).
        config: FairVALConfig instance.
        device: Target device.
    """

    def __init__(
        self,
        backbone: FrozenBackbone,
        num_groups: int,
        config: Optional[FairVALConfig] = None,
        device: str = "cuda",
    ):
        self.backbone = backbone
        self.num_groups = num_groups
        self.config = config or FairVALConfig()
        self.device = torch.device(device)

        # Initialize components
        self.trainer = FairConstrainedERM(
            embed_dim=backbone.embed_dim,
            num_groups=num_groups,
            gamma=self.config.gamma,
            lr=self.config.train_lr,
            epochs=self.config.train_epochs,
        ).to(self.device)

        self.estimator = DemographicEstimator(
            embed_dim=backbone.embed_dim,
            num_groups=num_groups,
            hidden_dim=self.config.estimator_hidden,
            noise_rate=self.config.noise_rate,
        ).to(self.device)

        if self.config.use_adaptive_lambda:
            self.scheduler = SigmoidScheduler(
                gamma=self.config.gamma,
                beta=self.config.beta,
            )
        else:
            self.scheduler = FixedScheduler(self.config.fixed_lambda)

        self.acc_scorer = AccuracyScorer(self.trainer.head.linear)
        self.fair_scorer = FairnessScorer(
            num_groups=num_groups,
            c=self.config.c if self.config.use_uncertainty else 0.0,
        )
        if not self.config.use_tpr_deviation:
            # Override to remove TPR deviation term
            original_weights = self.fair_scorer._compute_weights
            def weights_no_deviation():
                w = original_weights()
                # Zero out the TPR deviation, keep only uncertainty
                tprs = self.fair_scorer.tpr_estimates
                mean_tpr = tprs.mean()
                deviation = (tprs - mean_tpr).abs()
                return w - deviation
            self.fair_scorer._compute_weights = weights_no_deviation

        self.composite = CompositeScorer(self.acc_scorer, self.fair_scorer)
        self.history: list[RoundResult] = []

    @torch.no_grad()
    def _embed_pool(self, images: torch.Tensor) -> torch.Tensor:
        """Embed all images through the frozen backbone."""
        embeddings = []
        batch_size = 256
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(self.device)
            emb = self.backbone(batch)
            embeddings.append(emb.cpu())
        return torch.cat(embeddings, dim=0).to(self.device)

    def run(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        groups: torch.Tensor,
        pilot_mask: Optional[torch.Tensor] = None,
    ) -> list[RoundResult]:
        """Run the full FairVAL active learning loop.

        Args:
            images: (N, C, H, W) tensor of all pool images.
            labels: (N,) tensor of true labels (revealed only when queried).
            groups: (N,) tensor of true group labels (for evaluation only).
            pilot_mask: Optional (N,) bool mask for pilot set with known groups.

        Returns:
            List of RoundResult for each AL round.
        """
        N = len(images)
        budget = int(self.config.budget_fraction * N)
        m = budget // self.config.rounds  # batch size per round
        seed_size = max(int(self.config.seed_fraction * N), self.num_groups * 2)

        logger.info(
            f"FairVAL: N={N}, budget={budget}, rounds={self.config.rounds}, "
            f"m={m}, k={self.num_groups}"
        )

        # Embed all images
        logger.info("Embedding pool...")
        embeddings = self._embed_pool(images)

        # Train demographic estimator on pilot set
        if pilot_mask is not None:
            pilot_idx = pilot_mask.nonzero(as_tuple=True)[0]
        else:
            pilot_idx = torch.arange(min(seed_size * 2, N))
        eta_hat = self.estimator.fit(
            embeddings[pilot_idx],
            groups[pilot_idx].to(self.device),
            epochs=self.config.estimator_epochs,
        )
        logger.info(f"Demographic estimator trained, η̂ = {eta_hat:.3f}")

        # Initialize labeled set with stratified seed
        labeled_mask = torch.zeros(N, dtype=torch.bool)
        # Stratified: ensure at least 1 sample per group in seed
        for a in range(self.num_groups):
            group_idx = (groups == a).nonzero(as_tuple=True)[0]
            n_seed_group = max(1, seed_size // self.num_groups)
            perm = torch.randperm(len(group_idx))[:n_seed_group]
            labeled_mask[group_idx[perm]] = True

        labels_device = labels.to(self.device)
        groups_device = groups.to(self.device)
        self.history = []

        for t in range(self.config.rounds):
            # Current sets
            labeled_idx = labeled_mask.nonzero(as_tuple=True)[0]
            unlabeled_idx = (~labeled_mask).nonzero(as_tuple=True)[0]

            if len(unlabeled_idx) == 0 or len(unlabeled_idx) < m:
                break

            # Retrain classifier on labeled data
            train_result = self.trainer.train_round(
                embeddings[labeled_idx],
                labels_device[labeled_idx],
                groups_device[labeled_idx],
            )

            # Update scorer references
            self.acc_scorer.head = self.trainer.head.linear

            # Estimate groups for unlabeled samples
            est_groups = self.estimator.predict(embeddings[unlabeled_idx])

            # Update fairness statistics
            with torch.no_grad():
                labeled_preds = self.trainer.head.predict(embeddings[labeled_idx])
            self.fair_scorer.update_stats(
                labeled_preds,
                labels_device[labeled_idx],
                groups_device[labeled_idx],  # Use true groups for stats
            )

            # Compute λ(t)
            lambda_t = self.scheduler.step(self.fair_scorer.tpr_estimates)

            # Compute positive probabilities
            pos_probs = self.trainer.head.predict_proba(embeddings[unlabeled_idx])

            # Compute composite scores and select batch
            if self.config.use_alpha_acc and self.config.use_alpha_fair:
                selected_local = self.composite.select_batch(
                    embeddings[unlabeled_idx],
                    est_groups,
                    pos_probs,
                    lambda_t,
                    m,
                )
            elif self.config.use_alpha_acc:
                scores = self.acc_scorer.score(embeddings[unlabeled_idx])
                _, selected_local = scores.topk(min(m, len(scores)))
            elif self.config.use_alpha_fair:
                scores = self.fair_scorer.score(
                    embeddings[unlabeled_idx], est_groups, pos_probs
                )
                _, selected_local = scores.topk(min(m, len(scores)))
            else:
                selected_local = torch.randperm(len(unlabeled_idx))[:m]

            # Map local indices back to global
            selected_global = unlabeled_idx[selected_local.cpu()]
            labeled_mask[selected_global] = True

            # Evaluate on full test set
            with torch.no_grad():
                all_preds = self.trainer.head.predict(embeddings)
                all_probs = self.trainer.head.predict_proba(embeddings)

            metrics = full_evaluation(
                all_preds, all_probs, labels_device, groups_device, self.num_groups
            )

            gap = self.scheduler.compute_fairness_gap(
                self.fair_scorer.tpr_estimates
            )

            result = RoundResult(
                round_idx=t,
                n_labeled=labeled_mask.sum().item(),
                lambda_t=lambda_t,
                fairness_gap=gap,
                metrics=metrics,
                selected_indices=selected_global.tolist(),
            )
            self.history.append(result)

            logger.info(
                f"Round {t+1}/{self.config.rounds}: "
                f"n_labeled={result.n_labeled}, λ={lambda_t:.3f}, "
                f"Δ={gap:.3f}, EOD={metrics['eod']:.3f}, "
                f"F1={metrics['f1']:.3f}, WGR={metrics['wgr']:.3f}"
            )

        return self.history
