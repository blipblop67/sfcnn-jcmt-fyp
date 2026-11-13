import numpy as np
import pandas as pd
import pytest
import torch

from sfcnn.config import load_config
from sfcnn.cutouts import extract_cutouts, make_channels
from sfcnn.dataset import build_dataset, load_dataset, make_splits
from sfcnn.explain import grad_cam
from sfcnn.features import FEATURE_NAMES, cutout_features
from sfcnn.fits_io import load_map
from sfcnn.labels import label_sources, normalise_class
from sfcnn.model import SFNet, dihedral_tta
from sfcnn.synthetic import make_synthetic_survey


@pytest.fixture(scope="module")
def survey(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    manifest = make_synthetic_survey(d, n_regions=3, seed=1, n=320, verbose=False)
    return d, manifest


def test_load_scuba2_style_map(survey):
    d, _ = survey
    sm = load_map(d / "SynthA_850um.fits")
    assert sm.data.ndim == 2 and sm.variance is not None  # 3-D cube squeezed, VARIANCE found
    assert sm.pixel_scale_arcsec == pytest.approx(3.0, rel=1e-6)
    assert sm.unit == "mJy/beam"


def test_class_normalisation():
    assert normalise_class("Class I") == "I"
    assert normalise_class("flat spectrum") == "FLAT"
    assert normalise_class(" 0 ") == "0"


def test_labels_match_and_ambiguity():
    src = pd.DataFrame({"ra": [10.0, 10.0, 10.0, 10.0], "dec": [0.0, 0.01, 0.025, 0.03]})
    a = 1 / 3600
    ysos = pd.DataFrame({"ra": [10.0, 10.0, 10.0], "dec": [5 * a, 0.01 + 20 * a, 0.03], "cls": ["I", "0", "II"]})
    out = label_sources(src, ysos, ["0", "I"], 15, 30)
    assert list(out["label"]) == [1, -1, 0, -1]  # match / between radii / clear / Class II on peak


def test_cutout_shape_and_channels(survey):
    d, _ = survey
    sm = load_map(d / "SynthA_850um.fits")
    X, keep = extract_cutouts(sm, [160, 2], [160, 2], 96, 32)
    assert X.shape == (1, 2, 32, 32) and list(keep) == [True, False]
    assert np.isfinite(X).all()


def test_make_channels_unit_free():
    rng = np.random.default_rng(0)
    f, n = rng.normal(5, 1, (32, 32)), np.full((32, 32), 1.0)
    np.testing.assert_allclose(make_channels(f, n), make_channels(f * 1e3, n * 1e3), atol=1e-5)


def test_build_and_spatial_split_no_overlap(survey, tmp_path):
    _, manifest = survey
    cfg = load_config(None)
    out = build_dataset(cfg, str(manifest), str(tmp_path / "ds.npz"), verbose=False)
    X, y, meta = load_dataset(out)
    assert len(X) == len(y) == len(meta) > 10 and set(np.unique(y)) <= {0, 1}
    cfg["split"].update(mode="spatial", block_arcsec=200.0)
    s = make_splits(meta, cfg)
    tr, held = set(s["train"]), set(s["val"]) | set(s["test"])
    assert not tr & held
    for i in tr:  # buffer: no training cutout overlaps a held-out one in the same region
        for j in held:
            if meta.region[i] == meta.region[j]:
                assert np.hypot(meta.x[i] - meta.x[j], meta.y[i] - meta.y[j]) * 3.0 >= 96.0
    cfg["split"]["mode"] = "region"
    s = make_splits(meta, cfg)
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not set(meta.region[s[a]]) & set(meta.region[s[b]])


def test_model_forward_tta_and_gradcam():
    m = SFNet().eval()
    x = torch.randn(4, 2, 32, 32)
    assert m(x).shape == (4,)
    p = dihedral_tta(m, x)
    assert ((p >= 0) & (p <= 1)).all()
    # TTA output is invariant to rotating the input
    torch.testing.assert_close(p, dihedral_tta(m, torch.rot90(x, 1, dims=(2, 3))), atol=1e-5, rtol=1e-4)
    cam = grad_cam(m, x.numpy())
    assert cam.shape == (4, 32, 32) and cam.max() <= 1 + 1e-6


def test_features_finite():
    X = np.random.default_rng(0).normal(0, 1, (5, 2, 32, 32)).astype(np.float32)
    F = cutout_features(X)
    assert F.shape == (5, len(FEATURE_NAMES)) and np.isfinite(F).all()


def test_coverage_cut_drops_cores_far_from_any_yso():
    src = pd.DataFrame({"ra": [10.0, 10.0], "dec": [0.0, 1.0]})
    ysos = pd.DataFrame({"ra": [10.0], "dec": [0.05], "cls": ["II"]})  # 3' from core 0, 57' from core 1
    out = label_sources(src, ysos, ["0", "I"], 15, 30, coverage_radius_arcmin=10)
    assert list(out["label"]) == [0, -2]


def test_hybrid_model_and_calibration():
    from sfcnn.config import load_config
    from sfcnn.model import build_model, predict_proba
    from sfcnn.train import calibrate
    cfg = load_config(None)
    cfg["model"]["hybrid"] = True
    m = build_model(cfg).eval()
    assert m.n_features == len(FEATURE_NAMES)
    X = np.random.default_rng(1).normal(0, 1, (12, 2, 32, 32)).astype(np.float32)
    y = np.array([0, 1] * 6)
    p0 = predict_proba(m, X, tta=False)
    calibrate(m, X, y)
    p1 = predict_proba(m, X, tta=False)
    assert ((p1 >= 0) & (p1 <= 1)).all()
    # calibration is monotonic (or skipped): the ranking, hence AUC, is preserved
    assert np.array_equal(np.argsort(p0, kind="stable"), np.argsort(p1, kind="stable"))


def test_fit_model_short_run_and_ensemble_shapes():
    from sfcnn.config import load_config
    from sfcnn.model import predict_ensemble
    from sfcnn.train import fit_model
    cfg = load_config(None)
    cfg["train"].update(epochs=2, patience=5, batch_size=8, mixup_alpha=0.2)
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (40, 2, 32, 32)).astype(np.float32)
    y = np.array([0, 1] * 20)
    tr, va = np.arange(30), np.arange(30, 40)
    ms = [fit_model(cfg, X, y, tr, va, seed=s, verbose=False)[0] for s in (0, 1)]
    p = predict_ensemble(ms, X[va], tta=False)
    assert p.shape == (10,) and np.isfinite(p).all()
