"""SFNet: a small VGG-style CNN for 32x32 SCUBA-2 cutouts, with an optional hybrid head.

Deliberately compact (~0.3M parameters): labelled cores number in the hundreds
to low thousands, so a large network would just memorise them.

Hybrid mode (``n_features > 0``): the six hand-crafted morphology features
(sfcnn.features) are standardised and concatenated with the CNN embedding
before the dense layers, so the network gets explicit shape information that
it struggles to learn from few examples.

Calibration: ``cal_a``/``cal_b`` buffers apply Platt scaling (a*logit + b) fitted
on the validation set after training. It is monotonic, so ROC-AUC is unchanged,
but the outputs become usable probabilities.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .features import FEATURE_NAMES, model_features


def conv_block(cin: int, cout: int, pool: bool) -> nn.Sequential:
    layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
              nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class SFNet(nn.Module):
    def __init__(self, in_channels: int = 2, widths=(32, 64, 128), dropout: float = 0.3, n_features: int = 0):
        super().__init__()
        widths = list(widths)
        chans = [in_channels] + widths
        # pool after every block except the last, whose feature map is used by Grad-CAM
        self.blocks = nn.ModuleList(conv_block(chans[i], chans[i + 1], pool=i < len(widths) - 1)
                                    for i in range(len(widths)))
        self.n_features = n_features
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(widths[-1] + n_features, 64), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout), nn.Linear(64, 1))
        self.register_buffer("feat_mean", torch.zeros(n_features))
        self.register_buffer("feat_std", torch.ones(n_features))
        self.register_buffer("cal_a", torch.ones(()))
        self.register_buffer("cal_b", torch.zeros(()))

    @property
    def last_conv(self) -> nn.Module:
        return self.blocks[-1]

    def raw_logits(self, x: torch.Tensor, f: torch.Tensor | None = None) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        z = self.head[1](self.head[0](x))
        if self.n_features:
            if f is None:
                raise ValueError("hybrid SFNet needs the feature tensor f")
            z = torch.cat([z, (f - self.feat_mean) / self.feat_std], dim=1)
        return self.head[2:](z).squeeze(1)

    def forward(self, x: torch.Tensor, f: torch.Tensor | None = None) -> torch.Tensor:
        return self.cal_a * self.raw_logits(x, f) + self.cal_b  # calibrated logits


def build_model(cfg: dict, in_channels: int = 2, n_features: int | None = None) -> SFNet:
    m = cfg["model"]
    if n_features is None:
        n_features = len(FEATURE_NAMES) if m.get("hybrid") else 0
    return SFNet(in_channels, m["widths"], m["dropout"], n_features)


def dihedral_tta(model: SFNet, x: torch.Tensor, f: torch.Tensor | None = None) -> torch.Tensor:
    """Average sigmoid probability over the 8 rotations/flips (test-time augmentation).
    The hand-crafted features are rotation-invariant, so f is reused unchanged."""
    probs = []
    for flip in (False, True):
        xf = torch.flip(x, dims=(3,)) if flip else x
        for k in range(4):
            probs.append(torch.sigmoid(model(torch.rot90(xf, k, dims=(2, 3)), f)))
    return torch.stack(probs).mean(0)


@torch.no_grad()
def predict_proba(model: SFNet, X, batch_size: int = 256, tta: bool = True, device: str = "cpu") -> np.ndarray:
    model.eval().to(device)
    F = model_features(X) if model.n_features else None
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[i:i + batch_size]).to(device)
        fb = torch.as_tensor(F[i:i + batch_size], dtype=torch.float32).to(device) if F is not None else None
        out.append((dihedral_tta(model, xb, fb) if tta else torch.sigmoid(model(xb, fb))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


@torch.no_grad()
def raw_logits(model: SFNet, X, batch_size: int = 256, device: str = "cpu") -> np.ndarray:
    model.eval().to(device)
    F = model_features(X) if model.n_features else None
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[i:i + batch_size]).to(device)
        fb = torch.as_tensor(F[i:i + batch_size], dtype=torch.float32).to(device) if F is not None else None
        out.append(model.raw_logits(xb, fb).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def predict_ensemble(models: list[SFNet], X, tta: bool = True) -> np.ndarray:
    """Mean probability of several independently trained networks."""
    return np.mean([predict_proba(m, X, tta=tta) for m in models], axis=0)


def load_checkpoint(path, device: str = "cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck["state_dict"]
    n_feat = sd["feat_mean"].numel() if "feat_mean" in sd else 0
    model = build_model(ck["config"], ck.get("in_channels", 2), n_feat)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # checkpoints from before calibration/hybrid support lack only these buffers
    bad = [k for k in missing if k not in ("feat_mean", "feat_std", "cal_a", "cal_b")]
    if bad or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing {bad}, unexpected {unexpected}")
    model.eval()
    return model, ck
