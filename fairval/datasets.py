"""Dataset loaders and preprocessing for FairVAL experiments."""

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import Optional, Callable
import pandas as pd
import numpy as np


# Standard ImageNet normalization
IMAGENET_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    IMAGENET_NORMALIZE,
])


class FairVisionDataset(Dataset):
    """Base class for fair vision datasets.

    Each sample returns (image, label, group):
        image: (3, 224, 224) tensor
        label: binary int (Y ∈ {0, 1})
        group: int in [0, k) (A ∈ [k])
    """

    def __init__(
        self,
        image_paths: list[str],
        labels: np.ndarray,
        groups: np.ndarray,
        transform: Optional[Callable] = None,
    ):
        self.image_paths = image_paths
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.groups = torch.tensor(groups, dtype=torch.long)
        self.transform = transform or DEFAULT_TRANSFORM
        self.num_groups = int(groups.max()) + 1

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.labels[idx], self.groups[idx]


class Fitzpatrick17k(FairVisionDataset):
    """Fitzpatrick-17k: skin disease classification with 6 skin type groups.

    Y=1: malignant lesion. A ∈ {0,...,5}: Fitzpatrick skin types I-VI.
    """

    def __init__(self, root: str, split: str = "train", transform=None):
        csv_path = os.path.join(root, "fitzpatrick17k.csv")
        df = pd.read_csv(csv_path)

        if split != "all":
            np.random.seed(42)
            indices = np.random.permutation(len(df))
            n_train = int(0.8 * len(df))
            if split == "train":
                df = df.iloc[indices[:n_train]]
            else:
                df = df.iloc[indices[n_train:]]

        image_paths = [
            os.path.join(root, "images", p) for p in df["image_path"]
        ]
        labels = (df["malignant"] == 1).astype(int).values
        groups = (df["fitzpatrick_skin_type"] - 1).astype(int).values  # 0-indexed

        super().__init__(image_paths, labels, groups, transform)


class ISIC2019(FairVisionDataset):
    """ISIC 2019: melanoma classification with ITA-derived skin types.

    Y=1: melanoma. A ∈ {0,...,5}: ITA-derived skin types.
    """

    def __init__(self, root: str, split: str = "train", transform=None):
        csv_path = os.path.join(root, "isic2019_metadata.csv")
        df = pd.read_csv(csv_path)

        image_paths = [
            os.path.join(root, "images", f"{uid}.jpg")
            for uid in df["image_id"]
        ]
        labels = (df["diagnosis"] == "MEL").astype(int).values
        groups = df["ita_skin_type"].astype(int).values

        super().__init__(image_paths, labels, groups, transform)


class CelebA(FairVisionDataset):
    """CelebA: attribute prediction with gender groups.

    Y=1: attractive. A ∈ {0, 1}: male/female.
    """

    def __init__(self, root: str, split: str = "train", transform=None):
        attr_path = os.path.join(root, "list_attr_celeba.txt")
        partition_path = os.path.join(root, "list_eval_partition.txt")

        # Load attributes and partitions
        attrs = pd.read_csv(attr_path, sep=r"\s+", skiprows=1)
        partitions = pd.read_csv(
            partition_path, sep=r"\s+", header=None, names=["image", "split"]
        )

        split_map = {"train": 0, "val": 1, "test": 2}
        mask = partitions["split"] == split_map.get(split, 0)
        selected = partitions[mask]["image"].values

        attrs_sel = attrs.loc[attrs.index.isin(selected)]
        image_paths = [os.path.join(root, "img_align_celeba", p) for p in selected]
        labels = ((attrs_sel["Attractive"] + 1) // 2).values
        groups = ((attrs_sel["Male"] + 1) // 2).values

        super().__init__(image_paths, labels, groups, transform)


class FairFace(FairVisionDataset):
    """FairFace: gender prediction with 7 racial groups.

    Y=1: female. A ∈ {0,...,6}: race categories.
    """

    RACE_MAP = {
        "White": 0, "Black": 1, "Latino_Hispanic": 2,
        "East Asian": 3, "Southeast Asian": 4,
        "Indian": 5, "Middle Eastern": 6,
    }

    def __init__(self, root: str, split: str = "train", transform=None):
        csv_path = os.path.join(root, f"fairface_label_{split}.csv")
        df = pd.read_csv(csv_path)

        image_paths = [os.path.join(root, p) for p in df["file"]]
        labels = (df["gender"] == "Female").astype(int).values
        groups = df["race"].map(self.RACE_MAP).astype(int).values

        super().__init__(image_paths, labels, groups, transform)


DATASET_REGISTRY = {
    "fitzpatrick": Fitzpatrick17k,
    "isic": ISIC2019,
    "celeba": CelebA,
    "fairface": FairFace,
}


def load_dataset(
    name: str,
    root: str,
    split: str = "train",
    transform: Optional[Callable] = None,
) -> FairVisionDataset:
    """Load a dataset by name.

    Args:
        name: One of 'fitzpatrick', 'isic', 'celeba', 'fairface'.
        root: Path to dataset root directory.
        split: 'train', 'val', or 'test'.
        transform: Optional image transform.

    Returns:
        FairVisionDataset instance.
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name}. Choose from {list(DATASET_REGISTRY)}"
        )
    return DATASET_REGISTRY[name](root, split, transform)
