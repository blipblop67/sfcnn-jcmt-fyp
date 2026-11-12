"""Applying a trained model to a new SCUBA-2 map.

* ``classify_sources``: detect cores (or take a catalogue), output P(star-forming) per core.
* ``probability_map``:  sliding-window P(star-forming) over every significant pixel,
                        written as a FITS image with a valid WCS (open in DS9 / CARTA).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .cutouts import extract_cutouts
from .fits_io import load_map
from .model import load_checkpoint, predict_proba
from .sources import find_peaks, sources_from_catalog


def classify_sources(checkpoint: str, map_path: str, out_dir: str = "predictions",
                     source_catalog: str | None = None, threshold: float | None = None) -> pd.DataFrame:
    model, ck = load_checkpoint(checkpoint)
    cfg = ck["config"]
    sm = load_map(map_path)
    if source_catalog:
        src = sources_from_catalog(sm, pd.read_csv(source_catalog))
    else:
        d = cfg["detection"]
        src = find_peaks(sm, d["snr_threshold"], d["min_separation_arcsec"], d["smooth_fwhm_arcsec"])
    X, keep = extract_cutouts(sm, src["x"], src["y"], cfg["cutout"]["size_arcsec"], cfg["cutout"]["npix"],
                              cfg["cutout"]["max_nan_fraction"])
    src = src[keep].reset_index(drop=True)
    src["p_star_forming"] = predict_proba(model, X)
    thr = threshold if threshold is not None else _saved_threshold(checkpoint)
    src["predicted_star_forming"] = (src["p_star_forming"] >= thr).astype(int)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    stem = Path(map_path).stem
    src.to_csv(out / f"{stem}_cores.csv", index=False)
    _plot_sources(sm, src, thr, out / f"{stem}_cores.png")
    print(f"{len(src)} cores, {int(src['predicted_star_forming'].sum())} predicted star-forming "
          f"(threshold {thr:.2f}) -> {out}")
    return src


def _saved_threshold(checkpoint: str) -> float:
    import json
    mpath = Path(checkpoint).parent / "eval" / "metrics.json"
    if mpath.exists():
        return float(json.loads(mpath.read_text())["CNN (SFNet)"]["val_tuned"]["threshold"])
    return 0.5


def probability_map(checkpoint: str, map_path: str, out_path: str | None = None,
                    stride_pix: int = 2, min_snr: float = 3.0, batch: int = 512) -> Path:
    from astropy.io import fits
    model, ck = load_checkpoint(checkpoint)
    cfg = ck["config"]
    sm = load_map(map_path)
    snr = sm.snr_map()
    ny, nx = sm.shape
    gy, gx = np.mgrid[0:ny:stride_pix, 0:nx:stride_pix]
    prob = np.full(gy.shape, np.nan, np.float32)
    sel = np.isfinite(snr[gy, gx]) & (snr[gy, gx] >= min_snr)
    ys, xs = gy[sel], gx[sel]
    vals = np.full(len(ys), np.nan, np.float32)
    for i in range(0, len(ys), batch):
        X, keep = extract_cutouts(sm, xs[i:i + batch], ys[i:i + batch], cfg["cutout"]["size_arcsec"],
                                  cfg["cutout"]["npix"], cfg["cutout"]["max_nan_fraction"])
        if len(X):
            v = np.full(keep.shape, np.nan, np.float32)
            v[keep] = predict_proba(model, X, tta=False)
            vals[i:i + batch] = v
    prob[sel] = vals

    w = sm.wcs.deepcopy()
    w.wcs.crpix = (w.wcs.crpix - 1.0) / stride_pix + 1.0
    if w.wcs.has_cd():
        w.wcs.cd = w.wcs.cd * stride_pix
    else:
        w.wcs.cdelt = w.wcs.cdelt * stride_pix
    hdr = w.to_header()
    hdr["BUNIT"] = ""
    hdr["COMMENT"] = f"P(star-forming) from sfcnn, stride {stride_pix} px, pixels with S/N<{min_snr} blank"
    out_path = Path(out_path or Path("predictions") / f"{Path(map_path).stem}_psf.fits")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(prob, header=hdr).writeto(out_path, overwrite=True)
    _plot_probmap(sm, prob, w, out_path.with_suffix(".png"))
    print(f"Probability map -> {out_path}")
    return out_path


def _plot_sources(sm, src, thr, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection=sm.wcs)
    ax.imshow(np.arcsinh(sm.snr_map() / 3), origin="lower", cmap="gray_r", vmin=-0.5, vmax=4)
    sc = ax.scatter(src["x"], src["y"], c=src["p_star_forming"], cmap="coolwarm", vmin=0, vmax=1,
                    s=np.where(src["p_star_forming"] >= thr, 60, 25), edgecolor="k", lw=0.5)
    fig.colorbar(sc, ax=ax, label="P(star-forming)")
    ax.set(xlabel="RA", ylabel="Dec", title=f"{sm.name}: core classification")
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def _plot_probmap(sm, prob, w, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13, 6))
    a0 = fig.add_subplot(1, 2, 1, projection=sm.wcs)
    a0.imshow(np.arcsinh(sm.snr_map() / 3), origin="lower", cmap="inferno", vmin=-0.5, vmax=4)
    a0.set(title=f"{sm.name}: asinh(S/N/3)", xlabel="RA", ylabel="Dec")
    a1 = fig.add_subplot(1, 2, 2, projection=w)
    im = a1.imshow(prob, origin="lower", cmap="coolwarm", vmin=0, vmax=1)
    a1.set(title="P(star-forming), sliding window", xlabel="RA", ylabel=" ")
    fig.colorbar(im, ax=a1, fraction=0.046)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
