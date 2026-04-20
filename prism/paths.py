"""
paths.py
========
Centralised path configuration for the PRISM project.

All modules should import DATA_DIR / RESULTS_DIR from here rather than
constructing paths independently.

Environment variable overrides (set these in Colab before importing any
prism module):

    import os
    os.environ["PRISM_DATA_DIR"]    = "/content/drive/MyDrive/PRISM/data"
    os.environ["PRISM_RESULTS_DIR"] = "/content/drive/MyDrive/PRISM/results"
"""

import os
from pathlib import Path

_ROOT = Path(__file__).parent.parent

DATA_DIR    = Path(os.environ.get("PRISM_DATA_DIR",    _ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("PRISM_RESULTS_DIR", _ROOT / "results"))

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
