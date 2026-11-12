"""Synthetic SCUBA-2 850 um maps + YSO catalogues for developing/testing the pipeline.

NOT a substitute for real JCMT data: use it to debug the code end-to-end, to run
controlled experiments (e.g. accuracy vs noise), and as a sanity check that the
CNN learns what you think it learns. Report results on real data in the thesis.

What is modelled:
  * lognormal cloud with a power-law (P(k) ~ k^-beta) spectrum + filaments
  * starless cores: flat-topped Plummer-like profiles (Bonnor-Ebert-ish), r_c ~ 10-25"
  * protostellar cores: centrally peaked power-law envelopes + an unresolved
    warm inner component; optional faint bipolar lobes (CO(3-2) contamination
    of the 850 um band by outflows)
  * overlapping brightness distributions + "hard" cases (compact prestellar
    cores, faint VeLLO-like protostars) so the task is not trivial
  * SCUBA-2 two-component beam (13.0" main + 48" error beam; Dempsey et al. 2013)
  * loss of large-scale emission from the map-maker (high-pass filtering)
  * noise rising towards the map edge (PONG-like), with a variance map
  * YSO catalogue with positional scatter and unrelated Class II/III stars
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage

PIX_ARCSEC = 3.0
F2S = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def power_law_field(n: int, beta: float, rng) -> np.ndarray:
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij"))
    k[0, 0] = np.inf
    f = np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) * k ** (-beta / 2)).real
    return (f - f.mean()) / f.std()


def _filaments(n: int, rng, count: int) -> np.ndarray:
    img = np.zeros((n, n))
    for _ in range(count):
        pos = rng.uniform(0.1 * n, 0.9 * n, 2)
        ang = rng.uniform(0, 2 * np.pi)
        length = rng.uniform(0.2, 0.6) * n
        canvas = np.zeros((n, n))
        for _ in range(int(length * 2)):
            ang += rng.normal(0, 0.03)
            pos = pos + 0.5 * np.array([np.sin(ang), np.cos(ang)])
            iy, ix = int(pos[0]), int(pos[1])
            if 0 <= iy < n and 0 <= ix < n:
                canvas[iy, ix] += 1.0
        canvas = ndimage.gaussian_filter(canvas, rng.uniform(1.5, 3.0))
        if canvas.max() > 0:
            img += rng.uniform(10, 45) * canvas / canvas.max()
    return img


def _profile(n, y0, x0, rc, p, q, theta, half: int = 100):
    """Elliptical Plummer-like profile, evaluated within +-half pixels (wings beyond are negligible)."""
    out = np.zeros((n, n))
    y1, y2 = max(0, int(y0) - half), min(n, int(y0) + half + 1)
    x1, x2 = max(0, int(x0) - half), min(n, int(x0) + half + 1)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    dy, dx = yy - y0, xx - x0
    u = dx * np.cos(theta) + dy * np.sin(theta)
    v = (-dx * np.sin(theta) + dy * np.cos(theta)) / q
    out[y1:y2, x1:x2] = (1.0 + (u ** 2 + v ** 2) / rc ** 2) ** (-p)
    return out


def _beam_convolve(img: np.ndarray) -> np.ndarray:
    main = ndimage.gaussian_filter(img, 13.0 / PIX_ARCSEC * F2S)
    err = ndimage.gaussian_filter(img, 48.0 / PIX_ARCSEC * F2S)
    return 0.98 * main + 0.02 * err


def _place(n, weight, rng, k, min_sep, existing):
    pts = list(existing)
    w = weight.ravel() / weight.sum()
    out, tries = [], 0
    while len(out) < k and tries < 50 * k:
        tries += 1
        idx = rng.choice(w.size, p=w)
        y, x = divmod(idx, n)
        y, x = y + rng.uniform(-0.5, 0.5), x + rng.uniform(-0.5, 0.5)
        r = np.hypot(y - n / 2, x - n / 2)
        if r > 0.42 * n:
            continue
        if all(np.hypot(y - a, x - b) >= min_sep for a, b in pts):
            pts.append((y, x)); out.append((y, x))
    return out


def make_region(name: str, out_dir: str | Path, seed: int, n: int = 800,
                n_starless: int = 45, n_proto: int = 30, noise_mjy: float = 4.0) -> dict:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    beam_area_pix = 1.1331 * (14.1 / PIX_ARCSEC) ** 2

    cloud = np.exp(0.8 * power_law_field(n, rng.uniform(2.6, 3.2), rng))
    sky = rng.uniform(4, 10) * cloud + _filaments(n, rng, rng.integers(3, 7))

    weight = ndimage.gaussian_filter(sky, 6) ** 2
    min_sep = 40.0 / PIX_ARCSEC
    proto_pos = _place(n, weight, rng, n_proto, min_sep, [])
    starless_pos = _place(n, weight, rng, n_starless, min_sep, proto_pos)

    for (y, x) in starless_pos:
        hard = rng.random() < 0.2
        rc = rng.uniform(1.5, 3.0) if hard else rng.uniform(3.5, 8.0)
        sky += np.exp(rng.normal(np.log(45), 0.7)) * _profile(
            n, y, x, rc, rng.uniform(0.8, 1.5), rng.uniform(0.5, 1.0), rng.uniform(0, np.pi))

    ysos = []
    for (y, x) in proto_pos:
        faint = rng.random() < 0.15
        amp = np.exp(rng.normal(np.log(22 if faint else 60), 0.6))
        sky += amp * _profile(n, y, x, rng.uniform(1.0, 2.5), rng.uniform(0.55, 0.95),
                              rng.uniform(0.7, 1.0), rng.uniform(0, np.pi))
        point = np.zeros((n, n)); point[int(round(y)), int(round(x))] = 1.0
        sky += rng.uniform(0.2, 0.7) * amp * beam_area_pix * point * 0.9  # unresolved warm component
        if rng.random() < 0.5:
            th, d = rng.uniform(0, np.pi), rng.uniform(5, 12)
            for sgn in (-1, 1):
                sky += rng.uniform(0.05, 0.15) * amp * _profile(
                    n, y + sgn * d * np.sin(th), x + sgn * d * np.cos(th), 2.0, 1.5, 0.35, th)
        cls = rng.choice(["0", "I", "Flat"], p=[0.35, 0.4, 0.25])
        ysos.append((y + rng.normal(0, 2.0 / PIX_ARCSEC), x + rng.normal(0, 2.0 / PIX_ARCSEC), cls))
    for _ in range(5):  # protostars with no detectable envelope
        ysos.append((*rng.uniform(0.1 * n, 0.9 * n, 2), rng.choice(["I", "Flat"])))
    for _ in range(rng.integers(40, 90)):  # disc-bearing / diskless stars, unrelated to cores
        ysos.append((*rng.uniform(0.05 * n, 0.95 * n, 2), rng.choice(["II", "III"], p=[0.8, 0.2])))

    m = _beam_convolve(sky)
    m -= ndimage.gaussian_filter(m, 300.0 / PIX_ARCSEC * F2S, mode="reflect")  # map-maker filtering
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(yy - n / 2, xx - n / 2)
    sigma = noise_mjy * np.sqrt(1 + (r / (0.33 * n)) ** 6)
    white = ndimage.gaussian_filter(rng.standard_normal((n, n)), 0.8)
    m += sigma * white / white.std()
    m[r > 0.48 * n] = np.nan
    var = sigma ** 2
    var[r > 0.48 * n] = np.nan

    ra0, dec0 = rng.uniform(40, 90), rng.uniform(-10, 35)
    hdr = fits.Header()
    hdr["CTYPE1"], hdr["CTYPE2"], hdr["CTYPE3"] = "RA---TAN", "DEC--TAN", "FREQ"
    hdr["CRPIX1"], hdr["CRPIX2"], hdr["CRPIX3"] = n / 2 + 0.5, n / 2 + 0.5, 1.0
    hdr["CRVAL1"], hdr["CRVAL2"], hdr["CRVAL3"] = ra0, dec0, 3.5269e11
    hdr["CDELT1"], hdr["CDELT2"], hdr["CDELT3"] = -PIX_ARCSEC / 3600, PIX_ARCSEC / 3600, 1.0e9
    hdr["CUNIT1"], hdr["CUNIT2"], hdr["CUNIT3"] = "deg", "deg", "Hz"
    hdr["RADESYS"] = "ICRS"
    hdr["BUNIT"] = "mJy/beam"
    hdr["TELESCOP"], hdr["INSTRUME"] = "JCMT", "SCUBA-2"
    hdr["OBJECT"] = name
    hdr["COMMENT"] = "SYNTHETIC map generated by sfcnn.synthetic - not real data"
    map_path = out_dir / f"{name}_850um.fits"
    fits.HDUList([fits.PrimaryHDU(m[None].astype(np.float32), header=hdr),
                  fits.ImageHDU(var[None].astype(np.float32), header=hdr, name="VARIANCE")]
                 ).writeto(map_path, overwrite=True)

    w = WCS(hdr).celestial
    ys, xs, cls = zip(*ysos)
    sky_c = w.pixel_to_world(np.array(xs), np.array(ys))
    cat_path = out_dir / f"{name}_ysos.csv"
    pd.DataFrame({"ra": sky_c.ra.deg, "dec": sky_c.dec.deg, "class": cls}).to_csv(cat_path, index=False)
    return {"name": name, "map": map_path.name, "yso_catalog": cat_path.name}


def make_synthetic_survey(out_dir: str | Path = "data/synthetic", n_regions: int = 12, seed: int = 0,
                          n: int = 800, verbose: bool = True) -> Path:
    out_dir = Path(out_dir)
    regions = []
    for i in range(n_regions):
        name = f"Synth{chr(65 + i)}"
        regions.append(make_region(name, out_dir, seed * 1000 + i, n=n))
        if verbose:
            print(f"  wrote {name}")
    manifest = out_dir / "manifest.yaml"
    with open(manifest, "w", encoding="utf-8") as f:
        yaml.safe_dump({"regions": regions}, f, sort_keys=False)
    if verbose:
        print(f"Manifest: {manifest}")
    return manifest
