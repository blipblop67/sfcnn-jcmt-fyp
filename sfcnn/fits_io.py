"""Loading JCMT SCUBA-2 maps (FITS) into a simple 2-D container.

SCUBA-2 maps reduced with ORAC-DR / SMURF and converted to FITS usually have:
  * a primary HDU whose data is 3-D with a degenerate spectral axis, shape (1, ny, nx)
  * a ``VARIANCE`` extension with the same shape
  * units of mJy/beam (or pW before FCF calibration)
This module squeezes those down to 2-D arrays plus a celestial WCS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

_NOISE_HDU_NAMES = ("VARIANCE", "VAR", "ERR", "ERROR", "NOISE", "RMS", "SIGMA")


@dataclass
class SkyMap:
    data: np.ndarray
    wcs: WCS
    variance: np.ndarray | None = None
    unit: str = ""
    name: str = ""
    header: fits.Header = field(default_factory=fits.Header, repr=False)

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def pixel_scale_arcsec(self) -> float:
        scales = proj_plane_pixel_scales(self.wcs)
        unit = u.Unit(self.wcs.wcs.cunit[0] or "deg")
        return float((np.mean(scales) * unit).to_value(u.arcsec))

    def noise_map(self) -> np.ndarray:
        """1-sigma noise per pixel. Uses the variance array if present, else a
        robust global estimate (sigma-clipped std), which is a poor substitute
        near map edges where SCUBA-2 noise rises."""
        if self.variance is not None:
            with np.errstate(invalid="ignore"):
                noise = np.sqrt(self.variance)
            noise[~np.isfinite(noise) | (noise <= 0)] = np.nan
            return noise
        _, _, std = sigma_clipped_stats(self.data[np.isfinite(self.data)], sigma=3.0, maxiters=10)
        noise = np.full(self.data.shape, std, dtype=float)
        noise[~np.isfinite(self.data)] = np.nan
        return noise

    def snr_map(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.data / self.noise_map()


def _squeeze(arr: np.ndarray) -> np.ndarray:
    arr = np.squeeze(np.asarray(arr, dtype=np.float64))
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D image after squeezing, got shape {arr.shape}")
    return arr


def load_map(path: str | Path, hdu: int | str | None = None,
             variance_hdu: int | str | None = "auto", name: str | None = None) -> SkyMap:
    """Load a SCUBA-2 style FITS map.

    hdu:          HDU holding the image; default = first HDU with >=2-D data.
    variance_hdu: "auto" searches for a VARIANCE/ERR-style extension; None disables.
                  Extensions named ERR/ERROR/NOISE/RMS/SIGMA are squared into variance.
    """
    path = Path(path)
    with fits.open(path, memmap=False) as hdul:
        if hdu is None:
            idx = next(i for i, h in enumerate(hdul) if h.data is not None and h.data.ndim >= 2)
            img_hdu = hdul[idx]
        else:
            img_hdu = hdul[hdu]
        data = _squeeze(img_hdu.data)
        header = img_hdu.header.copy()
        wcs = WCS(header).celestial
        if wcs.naxis != 2:
            raise ValueError(f"{path}: could not find a 2-D celestial WCS")

        variance = None
        var_h = None
        if variance_hdu == "auto":
            for h in hdul:
                if h is not img_hdu and h.name.upper() in _NOISE_HDU_NAMES and h.data is not None:
                    var_h = h
                    break
        elif variance_hdu is not None:
            var_h = hdul[variance_hdu]
        if var_h is not None:
            variance = _squeeze(var_h.data)
            if var_h.name.upper() not in ("VARIANCE", "VAR"):
                variance = variance ** 2
            if variance.shape != data.shape:
                raise ValueError(f"{path}: variance shape {variance.shape} != data shape {data.shape}")

    unit = str(header.get("BUNIT", "")).strip()
    return SkyMap(data=data, wcs=wcs, variance=variance, unit=unit,
                  name=name or path.stem, header=header)


def save_map(path: str | Path, data: np.ndarray, wcs: WCS, variance: np.ndarray | None = None,
             unit: str = "mJy/beam", extra_header: dict | None = None) -> None:
    hdr = wcs.to_header()
    hdr["BUNIT"] = unit
    for k, v in (extra_header or {}).items():
        hdr[k] = v
    hdus = [fits.PrimaryHDU(data.astype(np.float32), header=hdr)]
    if variance is not None:
        hdus.append(fits.ImageHDU(variance.astype(np.float32), header=wcs.to_header(), name="VARIANCE"))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(path, overwrite=True)
