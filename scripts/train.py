"""Training entry point for FairVAL experiments."""

import argparse
import logging
import json
import os
import yaml
import torch
import numpy as np
from pathlib import Path

from fairval import FairVAL
from fairval.algorithm import FairVALConfig
from fairval.backbone import load_backbone
from fairval.datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="FairVAL Training")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--noise_rate", type=float, default=None)
    parser.add_argument("--method", type=str, default="fairval",
                        choices=["fairval", "random", "entropy", "badge",
                                 "coreset", "typiclust", "fare"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--subsample_groups", type=int, default=None)
    parser.add_argument("--subsample_prevalence", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_config(config_path: str, args) -> dict:
    """Load YAML config and override with CLI arguments."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.backbone:
        cfg["backbone"] = args.backbone
    if args.budget:
        cfg["budget_fraction"] = args.budget
    if args.gamma:
        cfg["gamma"] = args.gamma
    if args.noise_rate is not None:
        cfg["noise_rate"] = args.noise_rate

    return cfg


def run_single_seed(cfg: dict, seed: int, args) -> dict:
    """Run one experiment with a given seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load dataset
    dataset = load_dataset(
        name=cfg["dataset"],
        root=os.path.join(cfg.get("data_root", "data"), cfg["dataset"]),
        split="train",
    )

    # Load backbone
    backbone = load_backbone(cfg["backbone"], device=args.device)

    # Build FairVAL config
    fv_config = FairVALConfig(
        budget_fraction=cfg.get("budget_fraction", 0.10),
        rounds=cfg.get("rounds", 10),
        seed_fraction=cfg.get("seed_fraction", 0.01),
        gamma=cfg.get("gamma", 0.05),
        beta=cfg.get("beta", 10.0),
        c=cfg.get("c", 1.0),
        noise_rate=cfg.get("noise_rate", 0.0),
        train_lr=cfg.get("train_lr", 1e-2),
        train_epochs=cfg.get("train_epochs", 100),
    )

    # Method-specific ablations
    if args.method == "random":
        fv_config.use_alpha_acc = False
        fv_config.use_alpha_fair = False
    elif args.method == "badge":
        fv_config.use_alpha_fair = False
    elif args.method == "fare":
        fv_config.use_alpha_acc = False

    # Collate dataset
    images = torch.stack([dataset[i][0] for i in range(len(dataset))])
    labels = dataset.labels
    groups = dataset.groups

    # Run FairVAL
    fairval = FairVAL(
        backbone=backbone,
        num_groups=dataset.num_groups,
        config=fv_config,
        device=args.device,
    )

    history = fairval.run(images, labels, groups)

    # Final metrics
    final = history[-1].metrics if history else {}
    return {
        "seed": seed,
        "method": args.method,
        "dataset": cfg["dataset"],
        "final_metrics": final,
        "history": [
            {
                "round": r.round_idx,
                "n_labeled": r.n_labeled,
                "lambda": r.lambda_t,
                "gap": r.fairness_gap,
                **r.metrics,
            }
            for r in history
        ],
    }


def main():
    args = parse_args()
    cfg = load_config(args.config, args)

    seeds = args.seeds or [args.seed]
    all_results = []

    for seed in seeds:
        logger.info(f"Running seed {seed}...")
        result = run_single_seed(cfg, seed, args)
        all_results.append(result)

    # Save results
    output_dir = Path(args.output_dir) / cfg["dataset"] / args.method
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_path}")

    # Print summary
    if all_results:
        final = all_results[-1]["final_metrics"]
        logger.info(
            f"Final: F1={final.get('f1', 0):.3f}, "
            f"EOD={final.get('eod', 0):.3f}, "
            f"WGR={final.get('wgr', 0):.3f}"
        )


if __name__ == "__main__":
    main()
