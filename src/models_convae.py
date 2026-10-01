"""Phase-1 lightweight 2-D Convolutional Autoencoder (reference-only).

Input : 1 x H x W grayscale SEM image in [0, 1]   (H = W = 256 by default)
Encoder: 4 x [Conv2d(k=4, s=2, p=1) + ReLU]        256 -> 16 spatial
Decoder: 4 x [ConvTranspose2d(k=4, s=2, p=1)]      16 -> 256 spatial, Sigmoid output
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 32, latent_channels: int = 64):
        super().__init__()
        b = base_channels
        self.config = {"in_channels": in_channels, "base_channels": b, "latent_channels": latent_channels}
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, b, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(b, 2 * b, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(2 * b, 4 * b, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(4 * b, latent_channels, 4, 2, 1), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 4 * b, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(4 * b, 2 * b, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2 * b, b, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(b, in_channels, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def save_checkpoint(model: ConvAutoencoder, path: Path, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.config, **(extra or {})}, path)


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> tuple[ConvAutoencoder, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ConvAutoencoder(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)  # frozen backbone
    return model, ckpt


def checkpoint_is_valid(path: Path, image_size: int = 256) -> bool:
    """A checkpoint is reusable if it loads and maps (1,1,S,S) -> (1,1,S,S)."""
    if not path.exists():
        return False
    try:
        model, _ = load_checkpoint(path)
        with torch.no_grad():
            y = model(torch.zeros(1, 1, image_size, image_size))
        return tuple(y.shape) == (1, 1, image_size, image_size)
    except Exception:
        return False
