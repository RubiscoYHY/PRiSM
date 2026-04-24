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

### Download data from Google Drive

Download the shared `data/`, `results/`, and `notes/` folders from Google Drive and place them in the project root. **Important:** macOS may rename extracted folders to `data 2`, `results 2`, etc. if a folder with the same name already exists. Make sure the folders are named exactly `data/`, `results/`, and `notes/` — otherwise the project will not find them.

### Pull updates

After the initial setup, pull the latest code with:

```bash
git pull
```

Because the package is installed in editable mode (`-e`), all code changes — including the `prism` command — take effect immediately. No need to re-run `pip install -e .` unless you are told that new dependencies have been added.

### Launch the GUI

After the environment is set up, activate the conda environment and run:

```bash
conda activate prism
prism
```

---

## Google Colab (Simulation Notebooks Only)

Colab sessions are ephemeral, so `data/` and `results/` must live on Google Drive. The repo itself is cloned into the Colab runtime (faster than Drive I/O).

### Setup steps

1. **Copy the shared `data/` and `results/` folders** into a location on your own Google Drive (any path works).
2. **Find that path.** Open the folder in Google Drive, look at the URL, or browse the left sidebar in Colab after mounting. For example, if you placed the folders under `My Drive > Courses > MGT6081 > PRiSM`, the path would be:
   ```
   /content/drive/MyDrive/Courses/MGT6081/PRiSM
   ```
3. **Paste your path into the setup cell below** — only the line marked `# ← CHANGE THIS` needs editing.

### Setup cell

Run this at the top of every simulation notebook:

```python
# ── Colab Setup ───────────────────────────────────────────────
import os, sys
from pathlib import Path
from google.colab import drive

# 1. Mount Google Drive
drive.mount("/content/drive")

# 2. ★ CHANGE THIS to the Drive folder that contains your data/ and results/ ★
DRIVE_ROOT = Path("/content/drive/MyDrive/Courses/MGT6081/PRiSM")   # ← CHANGE THIS

# 3. Point prism.paths to Drive folders (MUST set before importing prism)
os.environ["PRISM_DATA_DIR"]    = str(DRIVE_ROOT / "data")
os.environ["PRISM_RESULTS_DIR"] = str(DRIVE_ROOT / "results")

# 4. Clone repo into Colab runtime and install (no changes needed below)
REPO_DIR = Path("/content/PRiSM")
if not (REPO_DIR / "prism").exists():
    !git clone https://github.com/RubiscoYHY/PRiSM.git {REPO_DIR}
!pip install -e {REPO_DIR} -q

sys.path.insert(0, str(REPO_DIR))

print(f"DATA_DIR    → {os.environ['PRISM_DATA_DIR']}")
print(f"RESULTS_DIR → {os.environ['PRISM_RESULTS_DIR']}")
# ─────────────────────────────────────────────────────────────
```

> **Important:** `os.environ` must be set **before** any `from prism import ...` statement.
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
