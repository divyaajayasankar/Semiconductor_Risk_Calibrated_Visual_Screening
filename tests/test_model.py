import torch

from src.models_convae import ConvAutoencoder, checkpoint_is_valid, count_parameters, save_checkpoint


def test_forward_shape_and_range():
    m = ConvAutoencoder()
    x = torch.rand(2, 1, 256, 256)
    y = m(x)
    assert y.shape == x.shape
    assert float(y.min()) >= 0 and float(y.max()) <= 1
    assert count_parameters(m) > 0


def test_checkpoint_roundtrip(tmp_path):
    m = ConvAutoencoder(base_channels=8, latent_channels=16)
    p = tmp_path / "ck.pt"
    save_checkpoint(m, p, {"epoch": 1})
    assert checkpoint_is_valid(p, 64)
    assert not checkpoint_is_valid(tmp_path / "missing.pt")
