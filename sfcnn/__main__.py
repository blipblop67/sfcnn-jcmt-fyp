"""Command line: python -m sfcnn <command> [options]

  synth      generate a synthetic SCUBA-2 survey (for testing)
  build      detect + label cores, extract cutouts -> dataset.npz/.csv
  train      train SFNet
  evaluate   test-set metrics, baselines, plots, Grad-CAM
  classify   classify cores in a new map
  probmap    sliding-window P(star-forming) FITS map
  demo       synth -> build -> train -> evaluate -> classify -> probmap
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="sfcnn", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default="configs/default.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth"); s.add_argument("--out", default="data/synthetic")
    s.add_argument("--regions", type=int, default=12); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--size", type=int, default=800)

    b = sub.add_parser("build"); b.add_argument("--manifest"); b.add_argument("--output")

    t = sub.add_parser("train"); t.add_argument("--dataset"); t.add_argument("--run-dir")
    t.add_argument("--epochs", type=int); t.add_argument("--split-mode", choices=["region", "spatial", "random"])

    e = sub.add_parser("evaluate"); e.add_argument("--checkpoint", required=True)
    e.add_argument("--dataset"); e.add_argument("--out"); e.add_argument("--no-tta", action="store_true")

    k = sub.add_parser("classify"); k.add_argument("--checkpoint", required=True); k.add_argument("--map", required=True)
    k.add_argument("--sources", help="optional CSV of core positions (ra, dec in deg)")
    k.add_argument("--out", default="predictions"); k.add_argument("--threshold", type=float)

    p = sub.add_parser("probmap"); p.add_argument("--checkpoint", required=True); p.add_argument("--map", required=True)
    p.add_argument("--out"); p.add_argument("--stride", type=int, default=2); p.add_argument("--min-snr", type=float, default=3.0)

    d = sub.add_parser("demo"); d.add_argument("--epochs", type=int)

    a = ap.parse_args(argv)
    cfg = load_config(a.config if Path(a.config).exists() else None)

    if a.cmd == "synth":
        from .synthetic import make_synthetic_survey
        make_synthetic_survey(a.out, a.regions, a.seed, a.size)
    elif a.cmd == "build":
        from .dataset import build_dataset
        build_dataset(cfg, a.manifest, a.output)
    elif a.cmd == "train":
        from .train import train
        if a.epochs: cfg["train"]["epochs"] = a.epochs
        if a.split_mode: cfg["split"]["mode"] = a.split_mode
        train(cfg, a.dataset, a.run_dir)
    elif a.cmd == "evaluate":
        from .evaluate import evaluate
        evaluate(a.checkpoint, a.dataset, a.out, tta=not a.no_tta)
    elif a.cmd == "classify":
        from .predict import classify_sources
        classify_sources(a.checkpoint, a.map, a.out, a.sources, a.threshold)
    elif a.cmd == "probmap":
        from .predict import probability_map
        probability_map(a.checkpoint, a.map, a.out, a.stride, a.min_snr)
    elif a.cmd == "demo":
        from .synthetic import make_synthetic_survey
        from .dataset import build_dataset
        from .train import train
        from .evaluate import evaluate
        from .predict import classify_sources, probability_map
        if a.epochs: cfg["train"]["epochs"] = a.epochs
        manifest = make_synthetic_survey("data/synthetic", 12, 0)
        build_dataset(cfg, str(manifest))
        run = train(cfg)
        ckpt = str(run / "best.pt")
        evaluate(ckpt)
        new = make_synthetic_survey("data/synthetic_unseen", 1, 99)
        m = str(Path("data/synthetic_unseen") / "SynthA_850um.fits")
        classify_sources(ckpt, m)
        probability_map(ckpt, m)


if __name__ == "__main__":
    main()
