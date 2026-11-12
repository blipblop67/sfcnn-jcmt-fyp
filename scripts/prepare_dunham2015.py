"""Convert a VizieR TSV export of Dunham et al. (2015, ApJS 220, 11; J/ApJS/220/11/fluxobs)
into the YSO CSV format sfcnn expects (ra, dec, class).

Classes follow Dunham et al. (2015), using the extinction-corrected spectral index alpha0
(alpha' in the paper) and bolometric temperature:
    Class 0      alpha0 >= 0.3 and Tbol < 70 K
    Class I      alpha0 >= 0.3 (Tbol >= 70 K or unknown)
    Flat         -0.3 <= alpha0 < 0.3
    Class II     -1.6 <= alpha0 < -0.3
    Class III    alpha0 < -1.6
Rows with no alpha0 fall back to the observed alpha.

Query used to make the TSV (Perseus example):
  https://vizier.cfa.harvard.edu/viz-bin/asu-tsv?-source=J/ApJS/220/11/fluxobs&Cloud=Perseus
    &-out=_RAJ2000,_DEJ2000,Cloud,ID,alpha,alpha0,Tbol&-oc.form=d&-out.max=unlimited

Usage:  python scripts/prepare_dunham2015.py data/real/dunham2015_perseus.tsv data/real/perseus_ysos.csv
"""
import sys

import numpy as np
import pandas as pd


def read_vizier_tsv(path: str) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip() and not ln.startswith("#")]
    header = lines[0].split("\t")
    rows = [ln.split("\t") for ln in lines[3:]]  # skip units + dashes lines
    df = pd.DataFrame(rows, columns=header).apply(lambda c: c.str.strip())
    for col in ("_RAJ2000", "_DEJ2000", "alpha", "alpha0", "Tbol"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def classify(alpha0: float, tbol: float) -> str:
    if np.isnan(alpha0):
        return "UNKNOWN"
    if alpha0 >= 0.3:
        return "0" if (not np.isnan(tbol) and tbol < 70) else "I"
    if alpha0 >= -0.3:
        return "FLAT"
    if alpha0 >= -1.6:
        return "II"
    return "III"


def main(src: str, dst: str) -> None:
    df = read_vizier_tsv(src)
    a = df["alpha0"].fillna(df["alpha"])
    out = pd.DataFrame({"ra": df["_RAJ2000"], "dec": df["_DEJ2000"], "id": df["ID"],
                        "class": [classify(x, t) for x, t in zip(a, df["Tbol"])],
                        "alpha0": df["alpha0"], "tbol": df["Tbol"]}).dropna(subset=["ra", "dec"])
    out.to_csv(dst, index=False)
    print(f"{len(out)} YSOs -> {dst}")
    print(out["class"].value_counts().to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
