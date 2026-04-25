"""
paths.py — points to experiment/data and experiment/results
"""

import os
from pathlib import Path

_ROOT = Path(__file__).parent          # experiment/ directory itself

DATA_DIR    = Path(os.environ.get("PRISM_EXP_DATA_DIR",    _ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("PRISM_EXP_RESULTS_DIR", _ROOT / "results"))

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
