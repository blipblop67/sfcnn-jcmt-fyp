"""Test-set evaluation: CNN vs classical baselines, plots for the report."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import load_dataset, save_json
from .explain import grad_cam
from .features import FEATURE_NAMES, cutout_features
from .model import load_checkpoint, predict_proba


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return float(thr[np.argmax(f1[:-1])]) if len(thr) else 0.5


def metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {"threshold": thr, "accuracy": accuracy_score(y, yhat),
            "balanced_accuracy": balanced_accuracy_score(y, yhat),
            "precision": precision_score(y, yhat, zero_division=0), "recall": recall_score(y, yhat),
            "f1": f1_score(y, yhat), "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def evaluate(checkpoint: str, dataset_path: str | None = None, out_dir: str | None = None,
             tta: bool = True, verbose: bool = True) -> dict:
    model, ck = load_checkpoint(checkpoint)
    cfg = ck["config"]
    X, y, meta = load_dataset(dataset_path or cfg["data"]["output"])
    splits = {k: np.asarray(v, dtype=int) for k, v in ck["splits"].items()}
    tr, va, te = splits["train"], splits["val"], splits["test"]
    out = Path(out_dir or Path(checkpoint).parent / "eval")
    out.mkdir(parents=True, exist_ok=True)

    probs = {"CNN (SFNet)": (predict_proba(model, X[va], tta=tta), predict_proba(model, X[te], tta=tta))}

    F = cutout_features(X)
    for name, clf in [("Logistic regression (features)",
                       make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))),
                      ("Random forest (features)",
                       RandomForestClassifier(400, min_samples_leaf=3, class_weight="balanced", random_state=0))]:
        clf.fit(F[tr], y[tr])
        probs[name] = (clf.predict_proba(F[va])[:, 1], clf.predict_proba(F[te])[:, 1])
    probs["Peak S/N only"] = (F[va, 0], F[te, 0])

    results = {}
    for name, (pv, pt) in probs.items():
        thr = best_f1_threshold(y[va], pv)
        results[name] = {"val_tuned": metrics(y[te], pt, thr)}
        if name.startswith("CNN"):
            results[name]["at_0.5"] = metrics(y[te], pt, 0.5)

    p_cnn = probs["CNN (SFNet)"][1]
    thr_cnn = results["CNN (SFNet)"]["val_tuned"]["threshold"]
    test_meta = meta.iloc[te].copy()
    test_meta["p_sf"] = p_cnn
    test_meta["pred"] = (p_cnn >= thr_cnn).astype(int)
    test_meta.to_csv(out / "test_predictions.csv", index=False)
    per_region = {}
    for reg, sub in test_meta.groupby("region"):
        if sub["label"].nunique() == 2:
            per_region[reg] = metrics(sub["label"].to_numpy(), sub["p_sf"].to_numpy(), thr_cnn)
    results["CNN (SFNet)"]["per_region"] = per_region
    results["n_test"] = int(len(te)); results["n_test_sf"] = int(y[te].sum())
    results["checkpoint_epoch"] = ck["epoch"]
    save_json(results, out / "metrics.json")

    _plots(y[te], {k: v[1] for k, v in probs.items()}, thr_cnn, out)
    _gallery(model, X[te], y[te], p_cnn, thr_cnn, out / "gallery_gradcam.png")

    table = pd.DataFrame({k: v["val_tuned"] for k, v in results.items() if isinstance(v, dict) and "val_tuned" in v}).T
    table = table[["roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "precision", "recall", "f1"]]
    table.to_csv(out / "comparison.csv")
    if verbose:
        print(f"Test set: {len(te)} cores ({int(y[te].sum())} star-forming). Thresholds tuned on validation.")
        print(table.round(3).to_string())
        print(f"\nOutputs in {out}")
    return results


def _plots(y, probs: dict, thr: float, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})", lw=2 if "CNN" in name else 1.2)
        pr, rc, _ = precision_recall_curve(y, p)
        ax[1].plot(rc, pr, label=f"{name} (AP {average_precision_score(y, p):.3f})", lw=2 if "CNN" in name else 1.2)
    ax[0].plot([0, 1], [0, 1], "k:", lw=0.8)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (test)")
    ax[1].axhline(y.mean(), color="k", ls=":", lw=0.8)
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall (test)")
    for a in ax:
        a.legend(fontsize=8, loc="lower right" if a is ax[0] else "lower left")
    fig.tight_layout(); fig.savefig(out / "roc_pr.png", dpi=140); plt.close(fig)

    p = probs["CNN (SFNet)"]
    cm = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
    ax[0].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax[0].text(j, i, str(v), ha="center", va="center", fontsize=14,
                   color="white" if v > cm.max() / 2 else "black")
    ax[0].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["non-SF", "SF"], yticklabels=["non-SF", "SF"],
              xlabel="Predicted", ylabel="True", title=f"CNN confusion matrix (thr={thr:.2f})")
    frac, mean = calibration_curve(y, p, n_bins=8, strategy="quantile")
    ax[1].plot(mean, frac, "o-", label="CNN"); ax[1].plot([0, 1], [0, 1], "k:")
    ax[1].set(xlabel="Predicted P(star-forming)", ylabel="Observed fraction", title="Reliability diagram")
    fig.tight_layout(); fig.savefig(out / "confusion_calibration.png", dpi=140); plt.close(fig)


def _gallery(model, X, y, p, thr, path: Path, per_cell: int = 4) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    yhat = p >= thr
    cats = [("True SF", (y == 1) & yhat), ("True non-SF", (y == 0) & ~yhat),
            ("False SF (FP)", (y == 0) & yhat), ("Missed SF (FN)", (y == 1) & ~yhat)]
    fig, axes = plt.subplots(len(cats), per_cell * 2, figsize=(per_cell * 3.2, len(cats) * 1.75))
    rows = []
    for _, mask in cats:
        idx = np.nonzero(mask)[0]
        rows.append(idx[np.argsort(-np.abs(p[idx] - thr))][:per_cell])
    all_idx = np.concatenate(rows)
    all_cams = grad_cam(model, X[all_idx]) if len(all_idx) else np.zeros((0,) + X.shape[-2:])
    offsets = np.cumsum([0] + [len(r) for r in rows])
    for r, (title, _) in enumerate(cats):
        idx = rows[r]
        cams = all_cams[offsets[r]:offsets[r + 1]]
        for c in range(per_cell):
            a_img, a_cam = axes[r, 2 * c], axes[r, 2 * c + 1]
            for a in (a_img, a_cam):
                a.set_xticks([]); a.set_yticks([])
            if c < len(idx):
                img = X[idx[c], 0]
                a_img.imshow(img, origin="lower", cmap="inferno")
                a_img.set_title(f"p={p[idx[c]]:.2f}", fontsize=7)
                a_cam.imshow(img, origin="lower", cmap="gray")
                a_cam.imshow(cams[c], origin="lower", cmap="jet", alpha=0.45, vmin=0, vmax=1)
                a_cam.set_title("Grad-CAM", fontsize=7)
            else:
                a_img.axis("off"); a_cam.axis("off")
        axes[r, 0].set_ylabel(title, fontsize=8)
    fig.suptitle("Test cutouts: asinh(S/N) and Grad-CAM saliency", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
