"""Phase-1 reference-only training with early stopping on reference-validation loss."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .models_convae import ConvAutoencoder, count_parameters, save_checkpoint
from .preprocessing import load_all_images, to_tensor
from .seed import set_seed
from .utils import get_device, log


def _augment(batch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = []
    for img in batch:
        if rng.random() < 0.5:
            img = img[:, ::-1]
        if rng.random() < 0.5:
            img = img[::-1, :]
        img = np.rot90(img, k=int(rng.integers(0, 4)))
        out.append(np.ascontiguousarray(img))
    return np.stack(out)


@torch.no_grad()
def reconstruction_loss(model: nn.Module, images: np.ndarray, device, batch_size: int = 16) -> float:
    model.eval()
    total, n = 0.0, 0
    for s in range(0, len(images), batch_size):
        x = to_tensor(images[s:s + batch_size]).to(device)
        total += float(((model(x) - x) ** 2).mean(dim=(1, 2, 3)).sum())
        n += len(x)
    return total / max(n, 1)


def train_phase1(train_refs: pd.DataFrame, val_refs: pd.DataFrame, cfg: dict, ckpt_path: Path) -> dict:
    """Train the ConvAE on TRAIN reference images only; early-stop on VALIDATION reference loss."""
    seed, tcfg, mcfg = cfg["seed"], cfg["training"], cfg["model"]
    size = cfg["data"]["image_size"]
    set_seed(seed)
    device = get_device(tcfg.get("device", "auto"))
    assert (train_refs.is_anomaly == 0).all(), "Phase-1 TRAIN must contain reference images only"
    assert (val_refs.is_anomaly == 0).all(), "Early stopping uses reference-validation images only"

    log(f"Loading {len(train_refs)} train / {len(val_refs)} validation reference images ({size}x{size})")
    x_train = load_all_images(train_refs, size)
    x_val = load_all_images(val_refs, size)

    model = ConvAutoencoder(1, mcfg["base_channels"], mcfg["latent_channels"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"])
    rng = np.random.default_rng(seed)
    bs, patience = tcfg["batch_size"], tcfg["patience"]
    best, best_epoch, wait, history = float("inf"), -1, 0, []
    t0 = time.time()
    log(f"Training ConvAE ({count_parameters(model):,} params) on {device} for <= {tcfg['epochs']} epochs")
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        order = rng.permutation(len(x_train))
        run, n = 0.0, 0
        for s in range(0, len(order), bs):
            xb = x_train[order[s:s + bs]]
            if tcfg.get("augment", True):
                xb = _augment(xb, rng)
            x = to_tensor(xb).to(device)
            loss = ((model(x) - x) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item() * len(x)
            n += len(x)
        val_loss = reconstruction_loss(model, x_val, device)
        history.append({"epoch": epoch, "train_loss": run / n, "val_loss": val_loss})
        improved = val_loss < best - 1e-7
        if improved:
            best, best_epoch, wait = val_loss, epoch, 0
            save_checkpoint(model, ckpt_path, {"epoch": epoch, "val_loss": val_loss, "seed": seed,
                                               "image_size": size})
        else:
            wait += 1
        if epoch == 1 or epoch % 5 == 0 or improved:
            log(f"epoch {epoch:3d} train={run / n:.6f} val={val_loss:.6f}{' *' if improved else ''}")
        if wait >= patience:
            log(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
            break
    return {"best_epoch": best_epoch, "best_val_loss": best, "epochs_run": len(history),
            "train_seconds": round(time.time() - t0, 1), "device": str(device),
            "n_parameters": count_parameters(model), "history": history}
