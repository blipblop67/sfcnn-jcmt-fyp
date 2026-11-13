"""Grad-CAM saliency for SFNet: which part of the cutout drove the decision?

For a physically sensible model the heat should sit on the compact central
peak for star-forming cores, not on map edges or unrelated background.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .features import model_features
from .model import SFNet


def grad_cam(model: SFNet, X: np.ndarray, device: str = "cpu", normalize: str = "batch") -> np.ndarray:
    """Returns (N, H, W) maps in [0, 1] for the star-forming logit.

    normalize="batch" scales by the maximum over all inputs, so confident
    non-star-forming cores (little positive evidence) correctly show ~no heat;
    "each" rescales every map to its own maximum (shape only)."""
    model.eval().to(device)
    acts, grads = {}, {}
    h1 = model.last_conv.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = model.last_conv.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    try:
        x = torch.as_tensor(X, device=device).requires_grad_(True)
        f = (torch.as_tensor(model_features(X), device=device) if getattr(model, "n_features", 0) else None)
        model.zero_grad()
        model(x, f).sum().backward()
        a, g = acts["a"], grads["g"]
        cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=X.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        if normalize == "each":
            cam = cam / cam.flatten(1).max(1).values.clamp_min(1e-8)[:, None, None]
        else:
            cam = cam / cam.max().clamp_min(1e-8)
        return cam.detach().cpu().numpy()
    finally:
        h1.remove(); h2.remove()
