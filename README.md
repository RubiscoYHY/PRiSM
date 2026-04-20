# PRISM
**Put-spread Regime-Informed Simulation & Management**

MGT 6081 — Quantitative Finance | Georgia Institute of Technology | Spring 2026

---

## Local Development

```bash
git clone https://github.com/RubiscoYHY/PRISM.git
cd PRISM
conda create -n prism python=3.13
conda activate prism
pip install -e .
```

---

## Google Colab

Run at the top of each notebook:

```python
!git clone https://github.com/RubiscoYHY/PRISM.git
%cd PRISM
!pip install -e . -q

from google.colab import drive
drive.mount('/content/drive')
```

---

## Project Structure

```
PRISM/
├── prism/       # core library (data, ML, pricing, simulation, backtesting)
├── notebooks/   # analysis notebooks
├── notes/       # reference materials
└── data/        # cached data (not tracked by git)
```
