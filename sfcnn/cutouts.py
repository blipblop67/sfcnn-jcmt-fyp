"""Cutout extraction and input normalisation.

Each cutout is resampled to a fixed angular grid (``size_arcsec`` across,
``npix`` pixels) so maps with different pixel scales give identical inputs.

Two input channels:
  0. ``asinh(S/N / 3)``  - significance, compresses the huge dynamic range of
     dust emission while staying linear near the noise floor. Unit-free, so
     maps calibrated differently (pW vs mJy/beam) are comparable.
  1. ``flux / central flux`` clipped to [-1, 2] - pure morphology (how
     centrally concentrated the core is), independent of brightness.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .fits_io import SkyMap

N_CHANNELS = 2


def _sample(img: np.ndarray, yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
    return ndimage.map_coordinates(img, [yy, xx], order=1, mode="constant", cval=np.nan)


def cutout_grid(x: float, y: float, npix: int, step_pix: float) -> tuple[np.ndarray, np.ndarray]:
    offs = (np.arange(npix) - (npix - 1) / 2.0) * step_pix
    oy, ox = np.meshgrid(offs, offs, indexing="ij")
    return y + oy, x + ox


def make_channels(flux: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """flux, noise: (..., H, W) arrays with NaN for missing data -> (..., 2, H, W)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = flux / noise
    ch0 = np.arcsinh(snr / 3.0)
    h, w = flux.shape[-2:]
    cy, cx = h // 2, w // 2
    centre = np.nanmax(flux[..., cy - 1:cy + 1, cx - 1:cx + 1].reshape(*flux.shape[:-2], -1), axis=-1)
    centre_noise = np.nanmedian(noise[..., cy - 1:cy + 1, cx - 1:cx + 1].reshape(*flux.shape[:-2], -1), axis=-1)
    scale = np.fmax(centre, 3.0 * centre_noise)[..., None, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        ch1 = np.clip(flux / scale, -1.0, 2.0)
    out = np.stack([ch0, ch1], axis=-3)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def extract_cutouts(skymap: SkyMap, xs, ys, size_arcsec: float = 96.0, npix: int = 32,
                    max_nan_fraction: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X, keep) where X has shape (n_kept, 2, npix, npix) and ``keep`` is
    a boolean mask over the input positions (False = too much blank/off-map area)."""
    step = (size_arcsec / npix) / skymap.pixel_scale_arcsec
    data = skymap.data
    noise = skymap.noise_map()
    fluxes, noises, keep = [], [], np.zeros(len(xs), dtype=bool)
    for i, (x, y) in enumerate(zip(xs, ys)):
        yy, xx = cutout_grid(float(x), float(y), npix, step)
        f = _sample(data, yy, xx)
        n = _sample(noise, yy, xx)
        bad = ~np.isfinite(f) | ~np.isfinite(n)
        if bad.mean() > max_nan_fraction:
            continue
        keep[i] = True
        fluxes.append(f)
        noises.append(n)
    if not fluxes:
        return np.zeros((0, N_CHANNELS, npix, npix), np.float32), keep
    return make_channels(np.stack(fluxes), np.stack(noises)), keep


def extract_snr_channel(skymap: SkyMap, xs, ys, size_arcsec: float = 96.0, npix: int = 32) -> np.ndarray:
    """Single asinh(S/N/3) channel from another map (e.g. 450 um) at the given pixel positions.
    Missing data become 0. Returns (N, 1, npix, npix)."""
    step = (size_arcsec / npix) / skymap.pixel_scale_arcsec
    data, noise = skymap.data, skymap.noise_map()
    out = np.zeros((len(xs), 1, npix, npix), np.float32)
    for i, (x, y) in enumerate(zip(xs, ys)):
        yy, xx = cutout_grid(float(x), float(y), npix, step)
        with np.errstate(invalid="ignore", divide="ignore"):
            ch = np.arcsinh(_sample(data, yy, xx) / _sample(noise, yy, xx) / 3.0)
        out[i, 0] = np.nan_to_num(ch, nan=0.0, posinf=0.0, neginf=0.0)
    return out
