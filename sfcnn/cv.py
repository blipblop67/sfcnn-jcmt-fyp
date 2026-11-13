"""Cross-validation: compare model variants over many train/test splits.

A single 30-core test set cannot tell two models apart, so every variant is
trained and scored on the same set of folds and we report mean +- std over
folds, plus a pooled ROC-AUC over all out-of-fold predictions.

Fold schemes
  cloud    leave-one-cloud-out: each cloud with >= min_test_pos star-forming cores
           is the test set once; the other clouds are used for training. This is the
           strictest test: does the model work on a cloud it has never seen?
  spatial  repeated spatial-block splits with different seeds (for one or two clouds).

Validation data (for early stopping, threshold and calibration) always come
from spatial blocks of the *training* clouds, never from the test cloud.

Variants
  peak, lr, rf      classical baselines (peak S/N; logistic regression and
                    random forest on the six hand-crafted features)
  cnn               SFNet, one seed
  cnn_ens           mean of `ensemble` SFNets trained with different seeds
  hybrid            SFNet + hand-crafted features in the dense head
  hybrid_ens        ensemble of hybrid SFNets
  mixup             SFNet trained with mixup (alpha 0.2)
  small             SFNet with half the width (16/32/64 filters)
  pretrained        SFNet initialised from a checkpoint trained on synthetic maps
  cnn_2ch           SFNet on the 850 um channels only (for a 3-channel 450 um dataset)
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import load_dataset, make_splits, save_json
from .evaluate import best_f1_threshold, metrics
from .features import cutout_features
from .model import predict_ensemble, predict_proba
from .train import fit_model

CNN_OVERRIDES = {
    "cnn": {}, "cnn_ens": {}, "cnn_2ch": {},
    "hybrid": {"model": {"hybrid": True}}, "hybrid_ens": {"model": {"hybrid": True}},
    "mixup": {"train": {"mixup_alpha": 0.2}},
    "small": {"model": {"widths": [16, 32, 64]}},
    "pretrained": {"train": {"lr": 3.0e-4}},  # init_checkpoint is set from the cv options
}
LABELS = {"peak": "Peak S/N only", "lr": "Logistic regression (features)", "rf": "Random forest (features)",
          "cnn": "CNN", "cnn_ens": "CNN ensemble", "hybrid": "Hybrid CNN + features",
          "hybrid_ens": "Hybrid ensemble", "mixup": "CNN + mixup", "small": "CNN (half width)",
          "pretrained": "CNN pretrained on synthetic", "cnn_2ch": "CNN, 850 um only"}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out.get(k, {}), v) if isinstance(v, dict) else v
    return out


def _val_from_train(meta: pd.DataFrame, idx: np.ndarray, cfg: dict, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split training indices into train/val by spatial blocks (with the overlap buffer)."""
    sub = meta.iloc[idx].reset_index(drop=True)
    c = _merge(cfg, {"split": {"mode": "spatial", "test_fraction": 0.0, "seed": seed}})
    c["split"]["val_fraction"] = cfg["split"].get("val_fraction", 0.15)
    s = make_splits(sub, c)
    return idx[s["train"]], idx[s["val"]]


def make_folds(meta: pd.DataFrame, y: np.ndarray, cfg: dict, scheme: str, n_repeats: int,
               min_test_pos: int) -> list[dict]:
    folds = []
    if scheme == "cloud":
        gcol = "cloud" if "cloud" in meta.columns else "region"
        for c in sorted(meta[gcol].unique()):
            te = np.nonzero((meta[gcol] == c).to_numpy())[0]
            if y[te].sum() < min_test_pos or (1 - y[te]).sum() < min_test_pos:
                continue
            rest = np.nonzero((meta[gcol] != c).to_numpy())[0]
            tr, va = _val_from_train(meta, rest, cfg, seed=0)
            folds.append({"name": c, "train": tr, "val": va, "test": te})
    elif scheme == "spatial":
        for r in range(n_repeats):
            c = _merge(cfg, {"split": {"mode": "spatial", "seed": 1000 + r}})
            s = make_splits(meta, c)
            if len(np.unique(y[s["val"]])) < 2 or len(np.unique(y[s["test"]])) < 2:
                continue
            folds.append({"name": f"split{r}", **s})
    else:
        raise ValueError(scheme)
    return folds


def cross_validate(cfg: dict, dataset_path: str | None, out_dir: str, variants: list[str],
                   scheme: str = "cloud", n_repeats: int = 5, ensemble: int = 3,
                   min_test_pos: int = 5, pretrained: str | None = None, verbose: bool = True) -> pd.DataFrame:
    X, y, meta = load_dataset(dataset_path or cfg["data"]["output"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    folds = make_folds(meta, y, cfg, scheme, n_repeats, min_test_pos)
    if verbose:
        print(f"{len(folds)} folds ({scheme}); {len(X)} cores, {int(y.sum())} star-forming; variants: {variants}")
    F = cutout_features(X)
    rows, preds = [], []
    for fold in folds:
        tr, va, te = fold["train"], fold["val"], fold["test"]
        if verbose:
            print(f"\n== fold {fold['name']}: train {len(tr)} ({int(y[tr].sum())} SF), val {len(va)} "
                  f"({int(y[va].sum())} SF), test {len(te)} ({int(y[te].sum())} SF)")
        cache: dict[str, list] = {}
        for v in variants:
            t0 = time.time()
            if v == "peak":
                pv, pt = F[va, 0], F[te, 0]
            elif v in ("lr", "rf"):
                clf = (make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
                       if v == "lr" else
                       RandomForestClassifier(400, min_samples_leaf=3, class_weight="balanced", random_state=0))
                clf.fit(F[tr], y[tr])
                pv, pt = clf.predict_proba(F[va])[:, 1], clf.predict_proba(F[te])[:, 1]
            else:
                over = copy.deepcopy(CNN_OVERRIDES[v])
                if v == "pretrained":
                    if not pretrained:
                        raise ValueError("variant 'pretrained' needs --pretrained CHECKPOINT")
                    over["train"]["init_checkpoint"] = pretrained
                c = _merge(cfg, over)
                Xv = X[:, :2] if v == "cnn_2ch" else X
                base = v.replace("_ens", "")
                k = ensemble if v.endswith("_ens") else 1
                models = cache.setdefault(base, [])
                while len(models) < k:  # cnn_ens reuses the cnn model as its first member
                    m, _ = fit_model(c, Xv, y, tr, va, seed=cfg["train"]["seed"] + len(models), verbose=False)
                    models.append(m)
                ms = models[:k]
                pv, pt = predict_ensemble(ms, Xv[va]), predict_ensemble(ms, Xv[te])
            thr = best_f1_threshold(y[va], pv)
            r = metrics(y[te], pt, thr)
            rows.append({"fold": fold["name"], "variant": v, "n_test": len(te), "n_test_sf": int(y[te].sum()), **r})
            preds.append(pd.DataFrame({"fold": fold["name"], "variant": v, "idx": te, "y": y[te], "p": pt}))
            if verbose:
                print(f"  {v:11s} AUC {r['roc_auc']:.3f}  PR-AUC {r['pr_auc']:.3f}  bal.acc {r['balanced_accuracy']:.3f}"
                      f"  F1 {r['f1']:.3f}  ({time.time() - t0:.0f}s)")
        pd.DataFrame(rows).to_csv(out / "folds.csv", index=False)  # save progress after every fold

    df = pd.DataFrame(rows)
    P = pd.concat(preds, ignore_index=True)
    P.to_csv(out / "oof_predictions.csv", index=False)
    summary = summarise(df, P, variants)
    summary.to_csv(out / "summary.csv")
    save_json({"scheme": scheme, "folds": [f["name"] for f in folds], "variants": variants,
               "ensemble": ensemble, "n_cores": len(X), "n_sf": int(y.sum())}, out / "cv_info.json")
    _plot(df, variants, out / "cv_auc.png", scheme)
    if verbose:
        print("\n" + summary.round(3).to_string())
        print(f"\nOutputs in {out}")
    return summary


def summarise(df: pd.DataFrame, P: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    out = {}
    for v in variants:
        d = df[df.variant == v]
        p = P[P.variant == v]
        out[LABELS.get(v, v)] = {
            "auc_mean": d.roc_auc.mean(), "auc_std": d.roc_auc.std(ddof=1) if len(d) > 1 else 0.0,
            "pr_auc_mean": d.pr_auc.mean(), "bal_acc_mean": d.balanced_accuracy.mean(),
            "f1_mean": d.f1.mean(), "pooled_auc": roc_auc_score(p.y, p.p), "n_folds": len(d)}
    return pd.DataFrame(out).T


def _plot(df: pd.DataFrame, variants: list[str], path: Path, scheme: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(variants)), 4))
    data = [df[df.variant == v].roc_auc.to_numpy() for v in variants]
    ax.boxplot(data, showmeans=True)
    for i, d in enumerate(data, 1):
        ax.scatter(np.full(len(d), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(d)), d, s=12, c="k", alpha=0.6)
    ax.set_xticks(range(1, len(variants) + 1), [LABELS.get(v, v) for v in variants], rotation=30, ha="right", fontsize=8)
    ax.set(ylabel="test ROC-AUC per fold", title=f"Cross-validation ({scheme} folds)")
    ax.axhline(0.5, color="grey", ls=":", lw=0.8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
