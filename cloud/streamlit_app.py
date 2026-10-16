"""Entry point for the online version on Streamlit Community Cloud.

Community Cloud installs the packages from ``cloud/requirements.txt`` (PyTorch for
the CPU) instead of ``uv.lock``. The averaged elevation grids are fetched from the
GitHub release ``data-v1`` instead of the 456 MB ETOPO file.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("NEURAL_EARTH_CLOUD", "1")

import torch  # noqa: E402

from python_streamlit_neural_earth_field.app import main  # noqa: E402

# One core per training run, so a single visitor does not occupy the whole shared machine
torch.set_num_threads(1)

main()
