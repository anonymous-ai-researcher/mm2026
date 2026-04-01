"""Frozen visual backbone wrappers for CLIP, DINOv2, and ResNet-50."""

import torch
import torch.nn as nn
from typing import Literal

BackboneName = Literal["clip-vit-b16", "dinov2-vit-s14", "resnet50"]

_EMBED_DIMS = {
    "clip-vit-b16": 512,
    "dinov2-vit-s14": 384,
    "resnet50": 2048,
}


class FrozenBackbone(nn.Module):
    """Wrapper that extracts embeddings from a frozen pretrained backbone.

    All parameters are frozen; no gradients flow through the backbone.
    """

    def __init__(self, model: nn.Module, embed_dim: int, name: str):
        super().__init__()
        self.model = model
        self.embed_dim = embed_dim
        self.name = name
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings: (B, C, H, W) -> (B, embed_dim)."""
        return self.model(x)


def _load_clip(device: torch.device) -> FrozenBackbone:
    """Load CLIP ViT-B/16 visual encoder."""
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai"
        )
        visual = model.visual
        visual.eval()
        visual = visual.to(device)
        return FrozenBackbone(visual, 512, "clip-vit-b16")
    except ImportError:
        raise ImportError(
            "open_clip is required for CLIP backbone. "
            "Install with: pip install open-clip-torch"
        )


def _load_dinov2(device: torch.device) -> FrozenBackbone:
    """Load DINOv2 ViT-S/14."""
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval()
    model = model.to(device)
    return FrozenBackbone(model, 384, "dinov2-vit-s14")


def _load_resnet50(device: torch.device) -> FrozenBackbone:
    """Load ResNet-50 with ImageNet-pretrained weights, headless."""
    from torchvision.models import resnet50, ResNet50_Weights

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # Remove final FC layer, keep avgpool output
    modules = list(model.children())[:-1]
    modules.append(nn.Flatten())
    backbone = nn.Sequential(*modules)
    backbone.eval()
    backbone = backbone.to(device)
    return FrozenBackbone(backbone, 2048, "resnet50")


def load_backbone(
    name: BackboneName, device: torch.device | str = "cuda"
) -> FrozenBackbone:
    """Load a frozen visual backbone by name.

    Args:
        name: One of 'clip-vit-b16', 'dinov2-vit-s14', 'resnet50'.
        device: Target device.

    Returns:
        FrozenBackbone instance with .embed_dim attribute.
    """
    if isinstance(device, str):
        device = torch.device(device)

    loaders = {
        "clip-vit-b16": _load_clip,
        "dinov2-vit-s14": _load_dinov2,
        "resnet50": _load_resnet50,
    }
    if name not in loaders:
        raise ValueError(f"Unknown backbone: {name}. Choose from {list(loaders)}")
    return loaders[name](device)
