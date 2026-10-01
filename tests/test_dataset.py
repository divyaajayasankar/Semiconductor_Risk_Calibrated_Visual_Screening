import numpy as np
import pandas as pd

from src import data as D
from src.config import get_paths
from src.preprocessing import load_image, load_mask


def test_dataset_loading_and_pairing(cfg):
    p = get_paths(cfg)
    df = D.load_index(p.data_dir, p.csv)
    assert list(df.columns[:4]) == D.REQUIRED_COLUMNS
    for r in df.head(10).itertuples():
        assert r.image_abs.endswith(".jpg") and r.mask_abs.endswith(".png")
        assert r.filename in r.image_abs and r.filename in r.mask_abs


def test_empty_mask_ground_truth(cfg):
    p = get_paths(cfg)
    report, meta = D.verify_dataset(p.data_dir, p.csv)
    assert report["status"] == "PASS"
    assert report["empty_masks_reference"] == 50 and report["non_empty_masks_anomaly"] == 70
    # label comes from the mask, never from the class id
    assert (meta.loc[meta.is_anomaly == 0, "mask_fg_pixels"] == 0).all()
    assert (meta.loc[meta.is_anomaly == 1, "mask_fg_pixels"] > 0).all()


def test_preprocessing_shapes(cfg):
    p = get_paths(cfg)
    df = D.load_index(p.data_dir, p.csv)
    x = load_image(df.image_abs[0], 64)
    m = load_mask(df.mask_abs[len(df) - 1], 64)
    assert x.shape == (64, 64) and x.dtype == np.float32 and 0 <= x.min() and x.max() <= 1
    assert m.shape == (64, 64) and m.dtype == bool and m.any()


def test_split_roles(cfg):
    p = get_paths(cfg)
    _, meta = D.verify_dataset(p.data_dir, p.csv)
    split = D.make_split(meta, cfg["split"], 42)
    ok, problems = D.validate_split(split, meta)
    assert ok, problems
    again = D.make_split(meta, cfg["split"], 42)
    pd.testing.assert_frame_equal(split, again)   # deterministic
