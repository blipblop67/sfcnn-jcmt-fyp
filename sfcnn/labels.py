"""Ground-truth labels from cross-matching SCUBA-2 peaks with YSO catalogues.

A submillimetre core is labelled **star-forming (1)** if an embedded protostar
(Class 0, I or flat-spectrum by default) lies within ``match_radius`` of the peak,
i.e. roughly within one 850 um beam (14.1"). It is labelled **non-star-forming
(0)** (starless / prestellar / diffuse) only if no protostar lies within the
larger ``exclusion_radius``. Cores in between are *ambiguous* and dropped (-1),
so that positional uncertainty does not inject label noise.

Typical catalogues: Spitzer c2d + Gould Belt YSOs (Dunham et al. 2015),
Spitzer/Herschel Orion protostars (Megeath et al. 2012; Furlan et al. 2016).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord

DEFAULT_SF_CLASSES = ("0", "I", "0/I", "0+I", "FLAT", "F", "FS")


def normalise_class(c) -> str:
    s = str(c).strip().upper().replace("CLASS", "").strip()
    return {"FLAT SPECTRUM": "FLAT", "FLAT-SPECTRUM": "FLAT", "1": "I", "2": "II", "3": "III"}.get(s, s)


def load_yso_catalog(path, ra_col: str = "ra", dec_col: str = "dec", class_col: str = "class") -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {ra_col, dec_col, class_col} - set(df.columns)
    if missing:
        raise KeyError(f"{path}: missing columns {missing}; have {list(df.columns)}")
    return pd.DataFrame({"ra": df[ra_col].astype(float), "dec": df[dec_col].astype(float),
                         "cls": df[class_col].map(normalise_class)})


def _nearest_sep_arcsec(src: SkyCoord, cat: pd.DataFrame) -> np.ndarray:
    if len(cat) == 0 or len(src) == 0:
        return np.full(len(src), np.inf)
    cc = SkyCoord(cat["ra"].to_numpy() * u.deg, cat["dec"].to_numpy() * u.deg)
    _, sep, _ = src.match_to_catalog_sky(cc)
    return sep.to_value(u.arcsec)


def label_sources(sources: pd.DataFrame, ysos: pd.DataFrame,
                  sf_classes=DEFAULT_SF_CLASSES, match_radius_arcsec: float = 15.0,
                  exclusion_radius_arcsec: float = 30.0,
                  ambiguous_if_other_yso: bool = True) -> pd.DataFrame:
    """Adds columns ``sep_protostar``, ``sep_other_yso`` (arcsec) and ``label`` (1/0/-1)."""
    sf_set = {normalise_class(c) for c in sf_classes}
    is_proto = ysos["cls"].isin(sf_set)
    src = SkyCoord(sources["ra"].to_numpy() * u.deg, sources["dec"].to_numpy() * u.deg)
    sep_p = _nearest_sep_arcsec(src, ysos[is_proto])
    sep_o = _nearest_sep_arcsec(src, ysos[~is_proto])

    label = np.full(len(sources), -1, dtype=int)
    label[sep_p <= match_radius_arcsec] = 1
    neg = sep_p > exclusion_radius_arcsec
    if ambiguous_if_other_yso:
        # A Class II/III star sitting on the core peak: could be an evolved
        # protostar whose classification is wrong, so don't call it starless.
        neg &= sep_o > match_radius_arcsec
    label[neg] = 0
    out = sources.copy()
    out["sep_protostar"] = sep_p
    out["sep_other_yso"] = sep_o
    out["label"] = label
    return out
