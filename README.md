# PRiSM
**Put-spread Regime-informed Simulation & Management**

MGT 6081 — Derivative Securities | Georgia Institute of Technology | Spring 2026

---

## Local Development

### First-time setup

```bash
git clone https://github.com/RubiscoYHY/PRiSM.git
cd PRiSM
conda create -n prism python=3.13
conda activate prism
pip install -e .
```

No additional configuration needed — `data/` and `results/` resolve automatically relative to the project root.

### Launch the GUI

After the environment is set up, activate the conda environment and run:

```bash
conda activate prism
prism
```

---

## Google Colab (Simulation Notebooks Only)

Colab sessions are ephemeral and install packages into a non-writable system path, so output directories must be redirected to Google Drive before importing any `prism` module.

Run the following setup cell at the top of every simulation notebook:

```python
# ── Colab Setup ───────────────────────────────────────────────
import os, sys
from pathlib import Path
from google.colab import drive

# 1. Mount Google Drive
drive.mount("/content/drive")
PRISM_ROOT = Path("/content/drive/MyDrive/PRiSM")

# 2. Point to data/ — update DATA_SRC if the shared folder is at a different path
#    (e.g. after adding a Drive shortcut to your My Drive)
DATA_SRC = Path("/content/drive/MyDrive/PRiSM_data")   # ← shared folder shortcut
os.environ["PRISM_DATA_DIR"]    = str(DATA_SRC if DATA_SRC.exists() else PRISM_ROOT / "data")
os.environ["PRISM_RESULTS_DIR"] = str(PRISM_ROOT / "results")

# 3. Clone repo and install (skip if already done)
if not (PRISM_ROOT / "prism").exists():
    os.system(f"git clone https://github.com/RubiscoYHY/PRiSM.git {PRISM_ROOT}")
os.system(f"pip install -e {PRISM_ROOT} -q")

sys.path.insert(0, str(PRISM_ROOT))
print(f"DATA_DIR    → {os.environ['PRISM_DATA_DIR']}")
print(f"RESULTS_DIR → {os.environ['PRISM_RESULTS_DIR']}")
# ─────────────────────────────────────────────────────────────
```

> **Important:** `os.environ` must be set before any `from prism import ...` statement.
> Once set, all modules resolve paths automatically — no further configuration required.

---

## Project Structure

```
PRiSM/
├── prism/
│   ├── paths.py          # centralised DATA_DIR / RESULTS_DIR (import from here)
│   ├── data_collection.py
│   └── ...               # ML, pricing, simulation, backtesting modules
├── notebooks/            # simulation notebooks (Colab)
├── notes/                # reference materials
├── data/                 # cached data (not tracked by git — shared via Google Drive)
└── results/              # charts and output (not tracked by git)
```
