"""Launcher that starts the Streamlit dashboard for the packaged app."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def app_path() -> Path:
    return Path(__file__).resolve().parent / "app.py"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = [sys.executable, "-m", "streamlit", "run", str(app_path()), *argv]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
