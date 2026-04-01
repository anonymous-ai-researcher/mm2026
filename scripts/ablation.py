"""Ablation study script: systematically removes each FairVAL component."""

import argparse
import logging
import json
import yaml
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ABLATION_VARIANTS = {
    "full": {},
    "no_alpha_fair": {"method": "badge"},
    "no_alpha_acc": {"method": "fare"},
    "fixed_lambda": {"extra_args": ["--fixed_lambda", "0.5"]},
    "no_tpr_deviation": {"extra_args": ["--no_tpr_deviation"]},
    "no_uncertainty": {"extra_args": ["--no_uncertainty"]},
    "no_estimator": {"extra_args": ["--noise_rate", "0.5"]},
}


def main():
    parser = argparse.ArgumentParser(description="FairVAL Ablation Study")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--output_dir", type=str, default="outputs/ablation")
    args = parser.parse_args()

    results = {}
    for variant_name, variant_cfg in ABLATION_VARIANTS.items():
        logger.info(f"Running ablation: {variant_name}")

        cmd = [
            sys.executable, "scripts/train.py",
            "--config", args.config,
            "--method", variant_cfg.get("method", "fairval"),
            "--seeds", *[str(s) for s in args.seeds],
            "--output_dir", f"{args.output_dir}/{variant_name}",
        ]
        if "extra_args" in variant_cfg:
            cmd.extend(variant_cfg["extra_args"])

        logger.info(f"  Command: {' '.join(cmd)}")
        # In practice, run subprocess; here we log the command
        # subprocess.run(cmd, check=True)
        results[variant_name] = {"command": " ".join(cmd), "status": "queued"}

    # Save ablation plan
    out_path = Path(args.output_dir) / "ablation_plan.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Ablation plan saved to {out_path}")


if __name__ == "__main__":
    main()
