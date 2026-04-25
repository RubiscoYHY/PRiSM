"""
compare_models.py
=================
Compare backtest performance between:
  - Local repo model (no SMOTE, Optuna-tuned scale_pos_weight)
  - Notebook-style model (SMOTE + max_delta_step=1 + scale_pos_weight=1)

Both models are evaluated through the same backtest engine (threshold_grid.py)
on the validation (2020-2022) and test (2022-2024) periods.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

from prism.paths import DATA_DIR
from prism.xgboost_train import build_features, FEATURE_COLS
from prism.threshold_grid import (
    load_backtest_data, load_skew_fn, attach_p_safe, run_backtest,
)

OUT_DIR = DATA_DIR / "analysis"
OUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════
# STEP 1: Load data and prepare features
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("Model Comparison: Local vs Notebook-style XGBoost")
print("=" * 60)

print("\n[1/5] Loading data...")
feat = load_backtest_data()
skew_fn = load_skew_fn()

# Split for XGBoost training (same as xgboost_train.py)
train_mask = feat.index < "2020-01-01"
val_mask   = (feat.index >= "2020-01-01") & (feat.index < "2022-01-01")
test_mask  = feat.index >= "2022-01-01"

X_train = feat.loc[train_mask, FEATURE_COLS].values
y_train = feat.loc[train_mask, "label"].values
X_val   = feat.loc[val_mask, FEATURE_COLS].values
X_test  = feat.loc[test_mask, FEATURE_COLS].values


# ══════════════════════════════════════════════════════════════
# STEP 2: Local model — load existing
# ══════════════════════════════════════════════════════════════
print("\n[2/5] Loading local model (no SMOTE)...")
model_local = xgb.XGBClassifier()
model_local.load_model(str(DATA_DIR / "XGBoost" / "xgb_model.json"))

p_safe_local = model_local.predict_proba(feat[FEATURE_COLS].values)[:, 0]
feat_local = feat.copy()
feat_local["p_safe"] = p_safe_local

print(f"  P(Safe) local — mean: {p_safe_local.mean():.4f}, "
      f"median: {np.median(p_safe_local):.4f}, "
      f">0.95: {(p_safe_local > 0.95).mean():.1%}")


# ══════════════════════════════════════════════════════════════
# STEP 3: Notebook model — train with SMOTE
# ══════════════════════════════════════════════════════════════
print("\n[3/5] Training notebook-style model (SMOTE)...")

# Note: notebook uses label 1=Safe, 0=Crash
# Local repo uses label 1=Crash, 0=Safe
# We need to flip labels for notebook-style training
y_train_nb = 1 - y_train  # flip: 0=Crash→1=Safe, 1=Crash→0=Crash...
# Actually, let me re-check. In the local repo:
#   label = (future_ret < -0.05).astype(int)  →  1 = Crash, 0 = Safe
# In the notebook:
#   labels[date] = 0 if drop <= -threshold else 1  →  0 = Crash, 1 = Safe
# So yes, we flip.
y_train_nb = 1 - y_train

# SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train_nb)

print(f"  Before SMOTE — Safe: {(y_train_nb == 1).sum()}, Crash: {(y_train_nb == 0).sum()}")
print(f"  After SMOTE  — Safe: {(y_train_bal == 1).sum()}, Crash: {(y_train_bal == 0).sum()}")

# Optuna search (notebook style: narrower range, 50 trials for speed)
print("  Running Optuna (50 trials)...")


def objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 50, 300),
        "max_depth":        trial.suggest_int("max_depth", 2, 6),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": 1.0,       # notebook forces 1.0 after SMOTE
        "max_delta_step":   1,          # notebook adds this
        "objective":        "binary:logistic",
        "eval_metric":      "logloss",
        "verbosity":        0,
        "random_state":     42,
    }
    tscv = TimeSeriesSplit(n_splits=5)
    f1_scores = []
    for tr_idx, vl_idx in tscv.split(X_train_bal):
        X_t, X_v = X_train_bal[tr_idx], X_train_bal[vl_idx]
        y_t, y_v = y_train_bal[tr_idx], y_train_bal[vl_idx]
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_t, y_t, verbose=False)
        preds = clf.predict(X_v)
        f1_scores.append(f1_score(y_v, preds, zero_division=0))
    return np.mean(f1_scores)


study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
)
study.optimize(objective, n_trials=50, show_progress_bar=True)

best_params_nb = study.best_params
best_params_nb.update({
    "scale_pos_weight": 1.0,
    "max_delta_step":   1,
    "objective":        "binary:logistic",
    "eval_metric":      "logloss",
    "verbosity":        0,
    "random_state":     42,
})

print(f"  Best CV F1: {study.best_value:.4f}")

# Retrain on full SMOTE-balanced training set
model_nb = xgb.XGBClassifier(**best_params_nb)
model_nb.fit(X_train_bal, y_train_bal, verbose=False)

# P(Safe) = P(class=1) for notebook convention
p_safe_nb = model_nb.predict_proba(feat[FEATURE_COLS].values)[:, 1]

feat_nb = feat.copy()
feat_nb["p_safe"] = p_safe_nb

print(f"  P(Safe) notebook — mean: {p_safe_nb.mean():.4f}, "
      f"median: {np.median(p_safe_nb):.4f}, "
      f"max: {p_safe_nb.max():.4f}, "
      f">0.70: {(p_safe_nb > 0.70).mean():.1%}, "
      f">0.95: {(p_safe_nb > 0.95).mean():.1%}")


# ══════════════════════════════════════════════════════════════
# STEP 4: Run backtests with both models
# ══════════════════════════════════════════════════════════════
print("\n[4/5] Running backtests...")

# Thresholds
TC = 0.71
TS_LOCAL = 0.95   # optimized for local model
TS_NB    = 0.60   # notebook had to use this

# Determine appropriate notebook threshold from distribution
pcts = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
print(f"\n  Notebook P(Safe) threshold scan (val period):")
for ts in pcts:
    sub = feat_nb[val_mask]
    eligible = ((sub["p_calm"] > TC) & (sub["p_safe"] > ts)).mean()
    print(f"    ts={ts:.2f} → {eligible:.1%} of days eligible")

# Also scan test period
print(f"\n  Notebook P(Safe) threshold scan (test period):")
for ts in pcts:
    sub = feat_nb[test_mask]
    eligible = ((sub["p_calm"] > TC) & (sub["p_safe"] > ts)).mean()
    print(f"    ts={ts:.2f} → {eligible:.1%} of days eligible")

# Run backtests on val + test combined for clearer comparison
# We'll also try multiple notebook thresholds

configs = {
    "Local (tc=0.71, ts=0.95)":     (feat_local, TC, TS_LOCAL),
    "Notebook (tc=0.71, ts=0.60)":  (feat_nb,    TC, 0.60),
    "Notebook (tc=0.71, ts=0.50)":  (feat_nb,    TC, 0.50),
    "Notebook (tc=0.71, ts=0.65)":  (feat_nb,    TC, 0.65),
    "Baseline (tc=0, ts=0)":        (feat_local, 0.0, 0.0),
}

results = {}

for period_name, period_mask in [("Validation (2020-2022)", val_mask),
                                  ("Test (2022-2024)", test_mask)]:
    print(f"\n  ── {period_name} ──")
    for config_name, (data, tc, ts) in configs.items():
        data_period = data[period_mask]
        nav_df, metrics = run_backtest(data_period, tc, ts, skew_fn)
        results[(period_name, config_name)] = (nav_df, metrics)
        print(f"    {config_name:<35s}  "
              f"Sharpe={metrics['sharpe']:+.4f}  "
              f"Return={metrics['total_return']:+.2%}  "
              f"MaxDD={metrics['max_dd']:.2%}  "
              f"Vol={metrics['vol']:.2%}")


# ══════════════════════════════════════════════════════════════
# STEP 5: Plot comparisons
# ══════════════════════════════════════════════════════════════
print("\n[5/5] Generating comparison plots...")

fig, axes = plt.subplots(1, 2, figsize=(20, 7))

plot_configs = [
    ("Local (tc=0.71, ts=0.95)",    "#1f77b4", "-",  2.0),
    ("Notebook (tc=0.71, ts=0.60)", "#d62728", "-",  2.0),
    ("Notebook (tc=0.71, ts=0.50)", "#ff7f0e", "--", 1.5),
    ("Notebook (tc=0.71, ts=0.65)", "#2ca02c", "--", 1.5),
    ("Baseline (tc=0, ts=0)",       "gray",    ":",  1.2),
]

for ax_idx, (period_name, _) in enumerate([("Validation (2020-2022)", val_mask),
                                            ("Test (2022-2024)", test_mask)]):
    ax = axes[ax_idx]

    for config_name, color, ls, lw in plot_configs:
        nav_df, metrics = results[(period_name, config_name)]
        # Normalize to 100
        nav_norm = nav_df["nav"] / nav_df["nav"].iloc[0] * 100
        label = f"{config_name}\n  Sharpe={metrics['sharpe']:+.3f}  Ret={metrics['total_return']:+.1%}  MaxDD={metrics['max_dd']:.1%}"
        ax.plot(nav_norm.index, nav_norm.values, color=color, linestyle=ls,
                linewidth=lw, label=label)

    ax.axhline(100, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.set_ylabel("NAV (normalized to 100)", fontsize=11)
    ax.set_title(period_name, fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=30)

fig.suptitle("Model Comparison: Local (no SMOTE) vs Notebook (SMOTE)\nBacktest NAV Curves",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "model_comparison_nav.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {OUT_DIR / 'model_comparison_nav.png'}")


# ── P(Safe) distribution comparison ──
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for ax, (period_name, mask) in zip(axes,
        [("Validation (2020-2022)", val_mask), ("Test (2022-2024)", test_mask)]):
    p_local = p_safe_local[mask]
    p_nb    = p_safe_nb[mask]

    ax.hist(p_local, bins=50, alpha=0.6, color="#1f77b4", label="Local (no SMOTE)", density=True)
    ax.hist(p_nb,    bins=50, alpha=0.6, color="#d62728", label="Notebook (SMOTE)", density=True)

    ax.axvline(0.95, color="#1f77b4", linestyle="--", linewidth=1.5, label="ts=0.95 (local)")
    ax.axvline(0.60, color="#d62728", linestyle="--", linewidth=1.5, label="ts=0.60 (notebook)")

    ax.set_xlabel("P(Safe)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"P(Safe) Distribution — {period_name}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "psafe_distribution_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {OUT_DIR / 'psafe_distribution_comparison.png'}")


# ── Summary table ──
print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
print(f"\n{'Period':<25s} {'Config':<35s} {'Sharpe':>8s} {'Return':>10s} {'MaxDD':>10s} {'Vol':>8s}")
print("-" * 100)
for (period_name, config_name), (_, metrics) in sorted(results.items()):
    print(f"{period_name:<25s} {config_name:<35s} "
          f"{metrics['sharpe']:>+8.4f} {metrics['total_return']:>+10.2%} "
          f"{metrics['max_dd']:>10.2%} {metrics['vol']:>8.2%}")

print("\n" + "=" * 60)
print(f"All outputs saved to: {OUT_DIR}")
print("=" * 60)
