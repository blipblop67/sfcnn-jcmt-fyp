"""Finding candidate dense cores / clumps in a SCUBA-2 map.

The classifier works on cutouts centred on emission peaks. Peaks come either
from this simple local-maximum finder or from a published clump catalogue
(e.g. a JCMT Gould Belt Survey core catalogue, FellWalker/getsources output).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy import ndimage

from .fits_io import SkyMap

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def find_peaks(skymap: SkyMap, snr_threshold: float = 5.0, min_separation_arcsec: float = 15.0,
               smooth_fwhm_arcsec: float = 7.0, edge_buffer_pix: int = 0) -> pd.DataFrame:
    """Local maxima in a lightly smoothed S/N map above ``snr_threshold``.

    Returns columns: x, y (0-based pixel), ra, dec (deg), peak_snr, peak_flux.
    """
    snr = skymap.snr_map()
    valid = np.isfinite(snr)
    work = np.where(valid, snr, 0.0)
    pix = skymap.pixel_scale_arcsec
    if smooth_fwhm_arcsec > 0:
        work = ndimage.gaussian_filter(work, smooth_fwhm_arcsec / pix * FWHM_TO_SIGMA)
    size = max(3, int(2 * round(min_separation_arcsec / pix) + 1))
    local_max = ndimage.maximum_filter(work, size=size, mode="nearest") == work
    cand = local_max & valid & (work >= snr_threshold)
    if edge_buffer_pix > 0:
        cand[:edge_buffer_pix, :] = cand[-edge_buffer_pix:, :] = False
        cand[:, :edge_buffer_pix] = cand[:, -edge_buffer_pix:] = False
    ys, xs = np.nonzero(cand)
    order = np.argsort(-work[ys, xs])
    ys, xs = ys[order], xs[order]
    sky = skymap.wcs.pixel_to_world(xs, ys)
    return pd.DataFrame({
        "x": xs.astype(float), "y": ys.astype(float),
        "ra": sky.icrs.ra.deg if len(xs) else [], "dec": sky.icrs.dec.deg if len(xs) else [],
        "peak_snr": work[ys, xs], "peak_flux": skymap.data[ys, xs],
    })


def sources_from_catalog(skymap: SkyMap, catalog: pd.DataFrame, ra_col: str = "ra",
                         dec_col: str = "dec") -> pd.DataFrame:
    """Convert an external clump catalogue (RA/Dec in degrees) to the same table format."""
    coords = SkyCoord(catalog[ra_col].to_numpy() * u.deg, catalog[dec_col].to_numpy() * u.deg)
    x, y = skymap.wcs.world_to_pixel(coords)
    ny, nx = skymap.shape
    xi, yi = np.round(x).astype(int), np.round(y).astype(int)
    inside = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
    snr = skymap.snr_map()
    peak_snr = np.full(len(x), np.nan)
    peak_flux = np.full(len(x), np.nan)
    peak_snr[inside] = snr[yi[inside], xi[inside]]
    peak_flux[inside] = skymap.data[yi[inside], xi[inside]]
    out = pd.DataFrame({"x": x, "y": y, "ra": coords.ra.deg, "dec": coords.dec.deg,
                        "peak_snr": peak_snr, "peak_flux": peak_flux})
    return out[inside].reset_index(drop=True)
