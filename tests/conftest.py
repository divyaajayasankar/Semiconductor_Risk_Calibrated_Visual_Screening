"""Shared fixtures. A tiny SYNTHETIC SEM-like dataset is generated in a temp dir so every test
runs without the real Carinthia-S data (synthetic results are never reported)."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from synthetic_fixture import make_config, make_fixture  # noqa: E402


@pytest.fixture(scope="session")
def synth_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("semivision")
    make_fixture(root, n_ref=50, n_anom=70, size=64)
    cfg_path = make_config(root, ROOT / "experiment_config.yaml", image_size=64, epochs=2)
    old = {k: os.environ.get(k) for k in ("SEMIVISION_ROOT", "SEMIVISION_CONFIG")}
    os.environ["SEMIVISION_ROOT"] = str(root)
    os.environ["SEMIVISION_CONFIG"] = str(cfg_path)
    yield root
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(scope="session")
def cfg(synth_root):
    from src.config import load_config
    return load_config()


@pytest.fixture(scope="session")
def pipeline(synth_root, cfg):
    """Run the whole pipeline (steps 01-09) once on the synthetic fixture."""
    spec = importlib.util.spec_from_file_location("run_all", ROOT / "run_all.py")
    run_all = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_all)

    class A:
        retrain = False
    for sid, _, fn in run_all.STEPS:
        if sid != "10":
            fn(cfg, A())
    from src.config import get_paths
    return get_paths(cfg)
