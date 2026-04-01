"""Evaluation script for trained FairVAL models."""

import argparse
import json
import logging
import torch
from pathlib import Path

from fairval.backbone import load_backbone
from fairval.datasets import load_dataset
from fairval.trainer import LinearHead
from fairval.metrics import full_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate FairVAL checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--backbone", type=str, default="clip-vit-b16")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset(args.dataset, f"{args.data_root}/{args.dataset}", args.split)
    images = torch.stack([dataset[i][0] for i in range(len(dataset))])
    labels = dataset.labels
    groups = dataset.groups

    # Load backbone and head
    device = torch.device(args.device)
    backbone = load_backbone(args.backbone, device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    head = LinearHead(backbone.embed_dim).to(device)
    head.load_state_dict(checkpoint["head_state_dict"])
    head.eval()

    # Embed and evaluate
    logger.info("Embedding test set...")
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(images), 256):
            batch = images[i:i+256].to(device)
            embeddings.append(backbone(batch).cpu())
    embeddings = torch.cat(embeddings).to(device)

    with torch.no_grad():
        preds = head.predict(embeddings)
        probs = head.predict_proba(embeddings)

    results = full_evaluation(preds, probs, labels.to(device), groups.to(device), dataset.num_groups)

    logger.info(f"Results on {args.dataset} ({args.split}):")
    for k, v in results.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")

    # Save
    out_path = Path(args.checkpoint).parent / f"eval_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
