"""Step 01 - Verify Carinthia-S and create/preserve the leakage-free split.

Outputs:
  results/dataset/verification.json
  results/dataset/image_metadata.csv
  results/dataset/split_manifest.csv
  results/dataset/leakage_report.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data as D  # noqa: E402
from src.config import get_paths, load_config  # noqa: E402
from src.utils import log, save_json  # noqa: E402


def main(cfg=None) -> dict:
    cfg = cfg or load_config()
    paths = get_paths(cfg)
    out = paths.results / "dataset"
    log(f"Verifying dataset at {paths.data_dir}")
    report, meta = D.verify_dataset(paths.data_dir, paths.csv, cfg["data"]["mask_threshold"])
    meta.to_csv(out / "image_metadata.csv", index=False)

    split, how = D.get_or_create_split(meta, paths.split_manifest, cfg["split"], cfg["seed"])
    ok, problems = D.validate_split(split, meta)
    if not ok:
        raise RuntimeError(f"Split validation failed: {problems}")
    D.save_split(split, paths.split_manifest)
    leak = D.leakage_report(split)
    leak["split_manifest"] = how
    leak["manifest_hash"] = D.manifest_hash(split)
    leak["seed"] = cfg["seed"]
    save_json(leak, out / "leakage_report.json")
    report["split"] = {"status": how, "manifest_hash": leak["manifest_hash"], "role_counts": leak["role_counts"]}
    save_json(report, out / "verification.json")

    log(f"Images={report['images_found']}  Masks={report['masks_found']}  CSV rows={report['csv_rows']}")
    log(f"Sizes={report['image_sizes']}  Modes={report['image_modes']}")
    log(f"Empty masks (REFERENCE)={report['empty_masks_reference']}  "
        f"Non-empty masks (ANOMALY)={report['non_empty_masks_anomaly']}")
    log(f"Class distribution={report['class_distribution']}")
    for r, c in leak["role_counts"].items():
        log(f"  {r:<18} total={c['total']:5d}  reference={c['reference']:4d}  anomaly={c['anomaly']:5d}")
    log(f"Dataset verification: {report['status']} | Leakage check: {leak['status']}")
    if report["status"] != "PASS" or leak["status"] != "PASS":
        raise RuntimeError("Dataset verification or leakage check FAILED - see results/dataset/*.json")
    return report


if __name__ == "__main__":
    main()
