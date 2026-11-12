"""Training loop with class-imbalance weighting, cosine LR and early stopping on val ROC-AUC."""
from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .dataset import CutoutDataset, describe_splits, load_dataset, make_splits, save_json
from .model import build_model, predict_proba


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(cfg: dict, dataset_path: str | None = None, run_dir: str | None = None,
          device: str | None = None, verbose: bool = True) -> Path:
    tc = cfg["train"]
    seed_everything(tc["seed"])
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

    loader = DataLoader(CutoutDataset(X[tr], y[tr], augment=tc["augment"]),
                        batch_size=tc["batch_size"], shuffle=True, drop_last=len(tr) > tc["batch_size"])
    model = build_model(cfg, X.shape[1]).to(device)
    n_pos, n_neg = y[tr].sum(), (1 - y[tr]).sum()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"])

    history, best_auc, best_epoch, bad = [], -1.0, -1, 0
    if verbose:
        print(f"Model: {sum(p.numel() for p in model.parameters()):,} params, device={device}, "
              f"pos_weight={n_neg / max(n_pos, 1):.2f}")
    for epoch in range(1, tc["epochs"] + 1):
        t0 = time.time()
        model.train()
        tot, nb = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb); nb += len(yb)
        sched.step()

        p_va = predict_proba(model, X[va], tta=False, device=device)
        with torch.no_grad():
            val_loss = loss_fn(torch.logit(torch.as_tensor(p_va).clamp(1e-6, 1 - 1e-6)).to(device),
                               torch.as_tensor(y[va], dtype=torch.float32, device=device)).item()
        val_auc = roc_auc_score(y[va], p_va)
        val_acc = float(((p_va > 0.5) == y[va]).mean())
        history.append({"epoch": epoch, "train_loss": tot / nb, "val_loss": val_loss,
                        "val_auc": val_auc, "val_acc": val_acc, "lr": sched.get_last_lr()[0]})
        improved = val_auc > best_auc + 1e-4
        if improved:
            best_auc, best_epoch, bad = val_auc, epoch, 0
            torch.save({"state_dict": model.state_dict(), "config": cfg, "in_channels": X.shape[1],
                        "splits": {k: v.tolist() for k, v in splits.items()}, "epoch": epoch,
                        "val_auc": val_auc}, run_dir / "best.pt")
        else:
            bad += 1
        if verbose:
            print(f"ep {epoch:3d}  train_loss {tot / nb:.4f}  val_loss {val_loss:.4f}  "
                  f"val_auc {val_auc:.4f}  val_acc {val_acc:.3f}  {time.time() - t0:.1f}s{'  *' if improved else ''}")
        if bad >= tc["patience"]:
            if verbose:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch} (val AUC {best_auc:.4f})")
            break

    save_json(history, run_dir / "history.json")
    _plot_history(history, run_dir / "training_curves.png")
    if verbose:
        print(f"Best checkpoint: {run_dir / 'best.pt'} (epoch {best_epoch}, val AUC {best_auc:.4f})")
    return run_dir


def _plot_history(history: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="validation")
    ax[0].set(xlabel="epoch", ylabel="weighted BCE loss", title="Loss")
    ax[0].legend()
    ax[1].plot(ep, [h["val_auc"] for h in history], label="val ROC-AUC")
    ax[1].plot(ep, [h["val_acc"] for h in history], label="val accuracy")
    ax[1].set(xlabel="epoch", title="Validation metrics", ylim=(0.4, 1.0))
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
