"""Check Python version, packages, device and dataset location."""
from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PKGS = ["torch", "numpy", "pandas", "scikit-learn", "scipy", "Pillow", "matplotlib", "PyYAML", "streamlit",
        "plotly", "pytest"]


def main() -> int:
    ok = True
    print(f"Python      : {sys.version.split()[0]} {'OK' if sys.version_info >= (3, 10) else 'NEED >= 3.10'}")
    ok &= sys.version_info >= (3, 10)
    for p in PKGS:
        try:
            print(f"{p:<12}: {metadata.version(p)}")
        except metadata.PackageNotFoundError:
            print(f"{p:<12}: MISSING")
            ok = False
    try:
        import torch
        print(f"Device      : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    except Exception:
        pass
    from src.config import get_paths, load_config
    paths = get_paths(load_config())
    print(f"Dataset CSV     : {'FOUND' if paths.csv.exists() else 'MISSING'}  ({paths.csv})")
    for name in ["images", "masks"]:
        d = paths.data_dir / name
        n = sum(1 for f in d.glob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}) \
            if d.exists() else 0
        print(f"Dataset {name + '/':<8}: {n} files  ({d})")
    print(f"Checkpoint  : {'FOUND' if paths.checkpoint.exists() else 'not yet trained'}")
    print("ENVIRONMENT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
