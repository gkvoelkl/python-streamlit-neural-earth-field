"""The Earth as a neural field: f(lon, lat) → elevation, weights as compressed data."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

__all__ = ["main"]


def main() -> None:
    """Launches the Streamlit app."""
    app = Path(__file__).resolve().parents[2] / "app.py"
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)]))
