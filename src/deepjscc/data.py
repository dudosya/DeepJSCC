from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import DeepJSCCConfig


def build_cifar10_loaders(cfg: DeepJSCCConfig) -> tuple[DataLoader, DataLoader]:
    root = Path(cfg.dataset.data_dir)

    train_tf = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_ds = datasets.CIFAR10(root=root, train=True, transform=train_tf, download=cfg.dataset.download)
    test_ds = datasets.CIFAR10(root=root, train=False, transform=test_tf, download=cfg.dataset.download)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
    )
    return train_loader, test_loader
