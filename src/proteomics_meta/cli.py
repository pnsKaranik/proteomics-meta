"""Command-line runner for the proteomics meta-analysis pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .capabilities import get_logger

logger = get_logger(__name__)


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    separator = "," if suffix == ".csv" else "\t"
    return pd.read_csv(path, sep=separator, engine="python")


def _dummy_frame(n_proteins: int = 600, n_samples: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        rng.random((n_proteins, n_samples)),
        index=[f"P{i:04d}_HUMAN" for i in range(n_proteins)],
        columns=[f"S{i}" for i in range(n_samples)],
    )
    frame["Protein.Group"] = frame.index
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="proteomics-meta", description="Sample-aware proteomics meta-analysis")
    parser.add_argument("--file", type=Path, help="input matrix (.csv/.tsv/.parquet); omit for a synthetic demo run")
    parser.add_argument("--outdir", default="Meta_Analysis_Results")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--latent-dim", type=int, default=10)
    parser.add_argument("--beta-vae", type=float, default=1.0)
    parser.add_argument("--pcorr-threshold", type=float, default=0.10)
    parser.add_argument("--pips-alpha", type=float, default=0.5)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--no-jackknife", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from .engine import run_pipeline

    if args.file is None:
        logger.info("No input file provided; running on synthetic data (600x12).")
        frame = _dummy_frame()
    else:
        if not args.file.exists():
            logger.error("File not found: %s", args.file)
            sys.exit(1)
        try:
            frame = _read_table(args.file)
        except Exception as exc:
            logger.error("Cannot read %s: %s", args.file, exc)
            sys.exit(1)

    run_pipeline(
        frame,
        {
            "work_dir": args.outdir,
            "iterations": args.iterations,
            "epochs": args.epochs,
            "latent_dim": args.latent_dim,
            "beta_vae": args.beta_vae,
            "pcorr_threshold": args.pcorr_threshold,
            "pips_alpha": args.pips_alpha,
            "n_bootstrap": args.n_bootstrap,
            "jackknife": not args.no_jackknife,
        },
    )


if __name__ == "__main__":
    main()
