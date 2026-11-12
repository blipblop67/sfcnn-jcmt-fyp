"""SFNet: a small VGG-style CNN for 32x32 two-channel SCUBA-2 cutouts.

Deliberately compact (~0.3M parameters): labelled cores number in the hundreds
to low thousands, so a large network would just memorise them.
"""
from __future__ import annotations

import torch
from torch import nn


def conv_block(cin: int, cout: int, pool: bool) -> nn.Sequential:
    layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
              nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class SFNet(nn.Module):
    def __init__(self, in_channels: int = 2, widths=(32, 64, 128), dropout: float = 0.3):
        super().__init__()
        widths = list(widths)
        chans = [in_channels] + widths
        # pool after every block except the last, whose feature map is used by Grad-CAM
        self.blocks = nn.ModuleList(conv_block(chans[i], chans[i + 1], pool=i < len(widths) - 1)
                                    for i in range(len(widths)))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(widths[-1], 64), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout), nn.Linear(64, 1))

    @property
    def last_conv(self) -> nn.Module:
        return self.blocks[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return self.head(x).squeeze(1)  # logits


def build_model(cfg: dict, in_channels: int = 2) -> SFNet:
    m = cfg["model"]
    return SFNet(in_channels, m["widths"], m["dropout"])


def dihedral_tta(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Average sigmoid probability over the 8 rotations/flips (test-time augmentation)."""
    probs = []
    for flip in (False, True):
        xf = torch.flip(x, dims=(3,)) if flip else x
        for k in range(4):
            probs.append(torch.sigmoid(model(torch.rot90(xf, k, dims=(2, 3)))))
    return torch.stack(probs).mean(0)


@torch.no_grad()
def predict_proba(model: nn.Module, X, batch_size: int = 256, tta: bool = True, device: str = "cpu"):
    import numpy as np
    model.eval().to(device)
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[i:i + batch_size]).to(device)
        out.append((dihedral_tta(model, xb) if tta else torch.sigmoid(model(xb))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def load_checkpoint(path, device: str = "cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["config"], ck.get("in_channels", 2))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
