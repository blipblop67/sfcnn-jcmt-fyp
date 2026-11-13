"""Building the labelled cutout dataset and leakage-free train/val/test splits.

Leakage matters here: neighbouring cores share overlapping cutouts and the same
cloud environment, so a random split over-estimates accuracy. Supported modes:
  * ``region``  - hold out whole regions (best test of generalisation)
  * ``spatial`` - split each map into blocks; blocks go to one split only, and
                  training cores whose cutouts overlap a val/test core are dropped
  * ``random``  - for comparison only
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import load_manifest
from .cutouts import extract_cutouts, extract_snr_channel
from .fits_io import load_map
from .labels import label_sources, load_yso_catalog
from .sources import find_peaks, sources_from_catalog


def build_dataset(cfg: dict, manifest_path: str | None = None, output: str | None = None,
                  verbose: bool = True) -> Path:
    manifest_path = manifest_path or cfg["data"]["manifest"]
    output = Path(output or cfg["data"]["output"])
    det, lab, cut = cfg["detection"], cfg["labels"], cfg["cutout"]
    Xs, metas = [], []
    for reg in load_manifest(manifest_path):
        sm = load_map(reg["map"], variance_hdu=reg.get("variance_hdu", "auto"), name=reg["name"])
        if reg.get("source_catalog"):
            cols = reg.get("source_columns", {})
            src = sources_from_catalog(sm, pd.read_csv(reg["source_catalog"]),
                                       cols.get("ra", "ra"), cols.get("dec", "dec"))
        else:
            src = find_peaks(sm, det["snr_threshold"], det["min_separation_arcsec"], det["smooth_fwhm_arcsec"])
        ycols = reg.get("yso_columns", {})
        ysos = load_yso_catalog(reg["yso_catalog"], ycols.get("ra", "ra"), ycols.get("dec", "dec"),
                                ycols.get("class", "class"))
        src = label_sources(src, ysos, lab["sf_classes"], lab["match_radius_arcsec"],
                            lab["exclusion_radius_arcsec"], lab["ambiguous_if_other_yso"],
                            lab.get("coverage_radius_arcmin"))
        n_amb = int((src["label"] == -1).sum())
        n_cov = int((src["label"] == -2).sum())
        src = src[src["label"] >= 0].reset_index(drop=True)
        size_arcsec = cutout_size_arcsec(cut, reg)
        X, keep = extract_cutouts(sm, src["x"], src["y"], size_arcsec, cut["npix"], cut["max_nan_fraction"])
        src = src[keep].reset_index(drop=True)
        if cut.get("use_450"):
            if not reg.get("map_450"):
                raise ValueError(f"cutout.use_450 is on but region {reg['name']} has no map_450")
            sm450 = load_map(reg["map_450"], name=reg["name"] + "_450")
            x4, y4 = sm450.wcs.world_to_pixel(sm.wcs.pixel_to_world(src["x"].to_numpy(), src["y"].to_numpy()))
            X = np.concatenate([X, extract_snr_channel(sm450, x4, y4, size_arcsec, cut["npix"])], axis=1)
        src["region"] = reg["name"]
        src["cloud"] = reg.get("cloud", reg["name"])
        src["distance_pc"] = reg.get("distance_pc", np.nan)
        src["cutout_arcsec"] = size_arcsec
        src["pixel_scale_arcsec"] = sm.pixel_scale_arcsec
        Xs.append(X)
        metas.append(src)
        if verbose:
            print(f"[{reg['name']}] {len(src)} cores: {int(src['label'].sum())} star-forming, "
                  f"{int((src['label'] == 0).sum())} non-SF ({n_amb} ambiguous, "
                  f"{n_cov} outside YSO coverage, {int((~keep).sum())} off-edge; all dropped)")
    if not Xs or sum(len(m) for m in metas) == 0:
        raise RuntimeError("no labelled cores found in any region")
    X = np.concatenate(Xs)
    meta = pd.concat(metas, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, X=X, y=meta["label"].to_numpy(np.int64))
    meta.to_csv(output.with_suffix(".csv"), index=False)
    if verbose:
        print(f"Saved {len(meta)} cutouts {X.shape[1:]} -> {output}  (+ metadata {output.with_suffix('.csv')})")
    return output


def cutout_size_arcsec(cut: dict, reg: dict) -> float:
    """Fixed angular size, or fixed physical size (cutout.size_pc) if the region has distance_pc."""
    if cut.get("size_pc") and reg.get("distance_pc"):
        return float(cut["size_pc"]) / float(reg["distance_pc"]) * 206264.806
    return float(cut["size_arcsec"])


def load_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    path = Path(path)
    d = np.load(path)
    return d["X"], d["y"], pd.read_csv(path.with_suffix(".csv"))


def make_splits(meta: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
    sp = cfg["split"]
    rng = np.random.default_rng(sp["seed"])
    n = len(meta)
    assign = np.full(n, "train", dtype=object)
    mode = sp["mode"]

    if mode == "region":
        # group by parent cloud when the manifest gives one, so maps of one cloud stay together
        gcol = "cloud" if "cloud" in meta.columns else "region"
        regions = sorted(meta[gcol].unique())
        val_r, test_r = list(sp["val_regions"]), list(sp["test_regions"])
        if not val_r and not test_r:
            perm = list(rng.permutation(regions))
            n_test = max(1, round(sp["test_fraction"] * len(regions)))
            n_val = max(1, round(sp["val_fraction"] * len(regions)))
            test_r, val_r = perm[:n_test], perm[n_test:n_test + n_val]
        in_val = meta[gcol].isin(val_r) | meta["region"].isin(val_r)
        in_test = meta[gcol].isin(test_r) | meta["region"].isin(test_r)
        assign[in_val.to_numpy()] = "val"
        assign[in_test.to_numpy()] = "test"
    elif mode == "spatial":
        block_pix = sp["block_arcsec"] / meta["pixel_scale_arcsec"].to_numpy()
        groups = (meta["region"].astype(str) + "_" + (meta["x"] // block_pix).astype(int).astype(str)
                  + "_" + (meta["y"] // block_pix).astype(int).astype(str)).to_numpy()
        ug = rng.permutation(np.unique(groups))
        sizes = pd.Series(groups).value_counts()
        target_test, target_val = sp["test_fraction"] * n, sp["val_fraction"] * n
        c_test = c_val = 0
        for g in ug:
            if c_test < target_test:
                assign[groups == g] = "test"; c_test += sizes[g]
            elif c_val < target_val:
                assign[groups == g] = "val"; c_val += sizes[g]
    elif mode == "random":
        perm = rng.permutation(n)
        n_test, n_val = int(sp["test_fraction"] * n), int(sp["val_fraction"] * n)
        assign[perm[:n_test]] = "test"
        assign[perm[n_test:n_test + n_val]] = "val"
    else:
        raise ValueError(f"unknown split mode {mode!r}")

    if sp.get("buffer", True) and mode != "random":
        # drop training cores whose cutout footprint overlaps a held-out core's
        footprint = cfg["cutout"]["size_arcsec"]  # cutouts overlap if centres are closer than this
        drop = np.zeros(n, dtype=bool)
        for reg, sub in meta.groupby("region"):
            idx = sub.index.to_numpy()
            held = idx[assign[idx] != "train"]
            tr = idx[assign[idx] == "train"]
            if len(held) == 0 or len(tr) == 0:
                continue
            scale = sub["pixel_scale_arcsec"].iloc[0]
            pt = meta.loc[tr, ["x", "y"]].to_numpy() * scale
            ph = meta.loc[held, ["x", "y"]].to_numpy() * scale
            d = np.sqrt(((pt[:, None, :] - ph[None, :, :]) ** 2).sum(-1)).min(1)
            drop[tr[d < footprint]] = True
        assign[drop] = "dropped"

    return {k: np.nonzero(assign == k)[0] for k in ("train", "val", "test")}


def describe_splits(y: np.ndarray, splits: dict) -> str:
    return "  ".join(f"{k}: n={len(v)} (SF={int(y[v].sum())})" for k, v in splits.items())


class CutoutDataset(Dataset):
    """On-the-fly augmentation: the sky has no preferred orientation, so the 8
    dihedral transforms (rotations by 90 deg + flips) are label-preserving.
    A +-1 pixel shift adds robustness to peak-position errors."""

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False, max_shift: int = 1):
        self.X = torch.from_numpy(np.ascontiguousarray(X))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.augment = augment
        self.max_shift = max_shift

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment:
            x = torch.rot90(x, int(torch.randint(4, ())), dims=(1, 2))
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=(2,))
            if self.max_shift:
                s = torch.randint(-self.max_shift, self.max_shift + 1, (2,))
                x = torch.roll(x, shifts=(int(s[0]), int(s[1])), dims=(1, 2))
        return x, self.y[i]


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
