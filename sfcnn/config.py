"""YAML config loading with defaults (nested dicts, attribute-free on purpose)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "data": {"manifest": "data/synthetic/manifest.yaml", "output": "data/processed/dataset.npz"},
    "beam_fwhm_arcsec": 14.1,
    "detection": {"snr_threshold": 5.0, "min_separation_arcsec": 15.0, "smooth_fwhm_arcsec": 7.0},
    "labels": {"sf_classes": ["0", "I", "0/I", "FLAT"], "match_radius_arcsec": 15.0,
               "exclusion_radius_arcsec": 30.0, "ambiguous_if_other_yso": True,
               "coverage_radius_arcmin": 10.0},
    "cutout": {"size_arcsec": 96.0, "npix": 32, "max_nan_fraction": 0.1},
    "split": {"mode": "spatial", "val_fraction": 0.15, "test_fraction": 0.15,
              "block_arcsec": 300.0, "buffer": True, "seed": 42,
              "val_regions": [], "test_regions": []},
    "model": {"widths": [32, 64, 128], "dropout": 0.3},
    "train": {"epochs": 60, "batch_size": 64, "lr": 1.0e-3, "weight_decay": 1.0e-4,
              "min_epochs": 0, "patience": 12, "seed": 42, "augment": True, "run_dir": "runs/sfnet"},
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path | None) -> dict:
    if path is None:
        return copy.deepcopy(DEFAULTS)
    with open(path, encoding="utf-8") as f:
        return _merge(DEFAULTS, yaml.safe_load(f) or {})


def load_manifest(path: str | Path) -> list[dict]:
    """Manifest lists one entry per observed region; relative paths resolve against the manifest."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        regions = yaml.safe_load(f)["regions"]
    for r in regions:
        for key in ("map", "map_450", "yso_catalog", "source_catalog"):
            if r.get(key) and not Path(r[key]).is_absolute():
                r[key] = str((path.parent / r[key]).resolve())
    return regions
