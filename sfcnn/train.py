"""Training: class-imbalance weighting, cosine LR, early stopping on val ROC-AUC,
optional mixup, initialisation from a pretrained checkpoint, Platt calibration,
and seed ensembles."""
from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .dataset import CutoutDataset, describe_splits, load_dataset, make_splits, save_json
from .features import model_features
from .model import build_model, load_checkpoint, predict_proba, raw_logits


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class _WithFeatures(torch.utils.data.Dataset):
    """Wraps CutoutDataset so each item is (x, features, y)."""

    def __init__(self, base: CutoutDataset, F: np.ndarray):
        self.base, self.F = base, torch.as_tensor(F, dtype=torch.float32)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x, self.F[i], y


def _load_init(model: nn.Module, path: str, verbose: bool) -> None:
    """Copy every pretrained tensor whose name and shape match (e.g. skips a first conv
    layer with a different number of input channels)."""
    src = load_checkpoint(path)[0].state_dict()
    dst = model.state_dict()
    copied = [k for k, v in src.items() if k in dst and dst[k].shape == v.shape
              and k not in ("feat_mean", "feat_std", "cal_a", "cal_b")]
    dst.update({k: src[k] for k in copied})
    model.load_state_dict(dst)
    if verbose:
        print(f"Initialised {len(copied)}/{len(dst)} tensors from {path}")


def calibrate(model, X_val: np.ndarray, y_val: np.ndarray) -> tuple[float, float]:
    """Platt scaling on the validation set: p = sigmoid(a * logit + b). Kept only if a > 0."""
    z = raw_logits(model, X_val).reshape(-1, 1)
    if len(np.unique(y_val)) < 2:
        return 1.0, 0.0
    lr = LogisticRegression(C=1e4, max_iter=1000).fit(z, y_val)
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    if a <= 0:
        return 1.0, 0.0
    model.cal_a.fill_(a)
    model.cal_b.fill_(b)
    return a, b


def fit_model(cfg: dict, X: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray, seed: int,
              device: str = "cpu", verbose: bool = True):
    """Train one network on X[tr], select on X[va]. Returns (model, info)."""
    tc = cfg["train"]
    seed_everything(seed)
    model = build_model(cfg, X.shape[1]).to(device)
    if tc.get("init_checkpoint"):
        _load_init(model, tc["init_checkpoint"], verbose)
    F = None
    if model.n_features:
        F = model_features(X)
        model.feat_mean.copy_(torch.as_tensor(F[tr].mean(0)))
        model.feat_std.copy_(torch.as_tensor(F[tr].std(0) + 1e-6))

    base = CutoutDataset(X[tr], y[tr], augment=tc["augment"])
    ds = _WithFeatures(base, F[tr]) if F is not None else base
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=tc["batch_size"], shuffle=True, generator=g,
                        drop_last=len(tr) > tc["batch_size"])
    n_pos, n_neg = y[tr].sum(), (1 - y[tr]).sum()
    pw = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"])
    alpha = float(tc.get("mixup_alpha", 0.0))

    history, best_auc, best_epoch, bad, best_state = [], -1.0, -1, 0, None
    for epoch in range(1, tc["epochs"] + 1):
        t0 = time.time()
        model.train()
        tot, nb = 0.0, 0
        for batch in loader:
            xb, fb, yb = batch if F is not None else (batch[0], None, batch[1])
            xb, yb = xb.to(device), yb.to(device)
            fb = fb.to(device) if fb is not None else None
            if alpha > 0:  # mixup: convex combinations of pairs of examples and their labels
                lam = float(np.random.beta(alpha, alpha))
                perm = torch.randperm(len(yb), device=device)
                xb = lam * xb + (1 - lam) * xb[perm]
                yb = lam * yb + (1 - lam) * yb[perm]
                if fb is not None:
                    fb = lam * fb + (1 - lam) * fb[perm]
            opt.zero_grad()
            loss = loss_fn(model(xb, fb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb); nb += len(yb)
        sched.step()

        p_va = predict_proba(model, X[va], tta=False, device=device)
        val_auc = roc_auc_score(y[va], p_va)
        val_acc = float(((p_va > 0.5) == y[va]).mean())
        history.append({"epoch": epoch, "train_loss": tot / nb, "val_auc": val_auc, "val_acc": val_acc,
                        "lr": sched.get_last_lr()[0]})
        # With tiny validation sets the AUC is noisy, so ignore checkpoints before min_epochs.
        warm = epoch >= tc.get("min_epochs", 0)
        improved = warm and val_auc > best_auc + 1e-4
        if improved:
            best_auc, best_epoch, bad = val_auc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        elif warm:
            bad += 1
        if verbose:
            print(f"ep {epoch:3d}  train_loss {tot / nb:.4f}  val_auc {val_auc:.4f}  val_acc {val_acc:.3f}  "
                  f"{time.time() - t0:.1f}s{'  *' if improved else ''}")
        if bad >= tc["patience"]:
            if verbose:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch} (val AUC {best_auc:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    cal = calibrate(model, X[va], y[va]) if tc.get("calibrate", True) else (1.0, 0.0)
    return model.eval(), {"best_epoch": best_epoch, "val_auc": best_auc, "history": history,
                          "calibration": cal}


def train(cfg: dict, dataset_path: str | None = None, run_dir: str | None = None,
          device: str | None = None, verbose: bool = True) -> Path:
    tc = cfg["train"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(run_dir or tc["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    X, y, meta = load_dataset(dataset_path or cfg["data"]["output"])
    splits = make_splits(meta, cfg)
    if verbose:
        print("Splits:", describe_splits(y, splits))
    tr, va = splits["train"], splits["val"]
    if len(np.unique(y[va])) < 2:
        raise RuntimeError("Validation split has only one class; change split seed/fractions.")

    k = int(tc.get("ensemble", 1))
    paths = []
    for i in range(k):
        seed = tc["seed"] + i
        model, info = fit_model(cfg, X, y, tr, va, seed, device, verbose)
        name = "best.pt" if i == 0 else f"best_seed{seed}.pt"
        torch.save({"state_dict": model.state_dict(), "config": cfg, "in_channels": X.shape[1],
                    "splits": {s: v.tolist() for s, v in splits.items()}, "epoch": info["best_epoch"],
                    "val_auc": info["val_auc"], "seed": seed}, run_dir / name)
        paths.append(run_dir / name)
        if i == 0:
            save_json(info["history"], run_dir / "history.json")
            _plot_history(info["history"], run_dir / "training_curves.png")
        if verbose:
            print(f"Saved {run_dir / name} (epoch {info['best_epoch']}, val AUC {info['val_auc']:.4f}, "
                  f"calibration a={info['calibration'][0]:.2f} b={info['calibration'][1]:.2f})")
    if verbose and k > 1:
        print("Ensemble members:", " ".join(str(p) for p in paths))
    return run_dir


def _plot_history(history: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train (weighted BCE)")
    ax[0].set(xlabel="epoch", ylabel="loss", title="Training loss")
    ax[0].legend()
    ax[1].plot(ep, [h["val_auc"] for h in history], label="val ROC-AUC")
    ax[1].plot(ep, [h["val_acc"] for h in history], label="val accuracy")
    ax[1].set(xlabel="epoch", title="Validation metrics", ylim=(0.4, 1.0))
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
