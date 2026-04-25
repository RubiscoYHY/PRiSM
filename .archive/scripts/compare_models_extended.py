"""
compare_models_extended.py
==========================
Compare Local vs Notebook-style XGBoost on the extended test period
(2022-01-01 – 2026-04-24) using data/extended/ price + HMM data.
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
from scipy.interpolate import interp1d

from prism.paths import DATA_DIR
from prism.xgboost_train import FEATURE_COLS
from prism.data_collection import get_option_price, next_nth_friday
from prism.threshold_grid import (
    load_skew_fn, run_backtest, INITIAL_CASH, MAX_POSITIONS, CAR_FRACTION, R,
)

EXT_DIR = DATA_DIR / "extended"
XGB_DIR = DATA_DIR / "XGBoost"
OUT_DIR = DATA_DIR / "analysis"
OUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════
# Feature engineering (same as xgboost_train.build_features but
# without the label column — we don't need labels for inference)
# ══════════════════════════════════════════════════════════════

def build_features_nolabel(df: pd.DataFrame) -> pd.DataFrame:
    """Build the 9 XGBoost features without the look-ahead label."""
    r = df["log_return"]
    S = df["close"]
    feat = pd.DataFrame(index=df.index)
    feat["RV_5d"]    = r.rolling(5).std()  * np.sqrt(252)
    feat["RV_20d"]   = r.rolling(20).std() * np.sqrt(252)
    feat["RV_60d"]   = r.rolling(60).std() * np.sqrt(252)
    feat["RV_ratio"] = feat["RV_20d"] / feat["RV_60d"].replace(0, np.nan)
    feat["Mom_5d"]   = r.rolling(5).sum()
    feat["Mom_20d"]  = r.rolling(20).sum()
    rolling_max_60   = S.rolling(60).max()
    feat["DD_60d"]   = (S - rolling_max_60) / rolling_max_60
    feat["RSkew_20d"] = r.rolling(20).skew()
    feat["p_calm"]   = df["p_calm"]
    feat = feat.dropna()
    return feat


def build_features_with_label(df: pd.DataFrame) -> pd.DataFrame:
    """Build features WITH label — needed for training."""
    r = df["log_return"]
    S = df["close"]
    feat = build_features_nolabel(df)
    # Label: did SPY drop >5% in next 30 cal days (~21 trading days)?
    future_min = S.shift(-1).rolling(21, min_periods=1).min().shift(-(21 - 1))
    future_ret = (future_min - S) / S
    feat["label"] = (future_ret < -0.05).astype(int)
    feat = feat.dropna()
    return feat


# ══════════════════════════════════════════════════════════════
# STEP 1: Load extended data
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("Extended Test Comparison: Local vs Notebook XGBoost")
print("  Test period: 2022-01-01 – 2026-04-24")
print("=" * 60)

print("\n[1/5] Loading extended data...")
spy = pd.read_csv(EXT_DIR / "spy_vix_daily.csv", index_col="date", parse_dates=True)
hmm = pd.read_csv(EXT_DIR / "hmm_pcalm_daily.csv", index_col="date", parse_dates=True)

close_col = next(c for c in spy.columns if "spy" in c and "close" in c)
spy = spy.rename(columns={close_col: "close"})
df = spy.join(hmm[["p_calm"]], how="inner").sort_index()

# Build features without label for the full dataset (inference only)
feat_full = build_features_nolabel(df)
feat_full = feat_full.join(df[["close", "vix_close", "vix9d_close"]], how="left")

print(f"  Full dataset: {len(feat_full)} rows "
      f"({feat_full.index[0].date()} – {feat_full.index[-1].date()})")

# Build features WITH label for training subset (only need train period)
feat_train_labeled = build_features_with_label(df)

skew_fn = load_skew_fn()


# ══════════════════════════════════════════════════════════════
# STEP 2: Local model — load existing
# ══════════════════════════════════════════════════════════════
print("\n[2/5] Loading local model...")
model_local = xgb.XGBClassifier()
model_local.load_model(str(XGB_DIR / "xgb_model.json"))

p_safe_local = model_local.predict_proba(feat_full[FEATURE_COLS].values)[:, 0]
feat_local = feat_full.copy()
feat_local["p_safe"] = p_safe_local

print(f"  P(Safe) local — mean: {p_safe_local.mean():.4f}, "
      f">0.95: {(p_safe_local > 0.95).mean():.1%}")


# ══════════════════════════════════════════════════════════════
# STEP 3: Notebook model — train with SMOTE (on 2015-2020 data)
# ══════════════════════════════════════════════════════════════
print("\n[3/5] Training notebook-style model (SMOTE)...")

train_data = feat_train_labeled[feat_train_labeled.index < "2020-01-01"]
X_train = train_data[FEATURE_COLS].values
y_train = train_data["label"].values

# Flip labels: local 1=Crash,0=Safe → notebook 1=Safe,0=Crash
y_train_nb = 1 - y_train

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train_nb)
print(f"  Before SMOTE — Safe: {(y_train_nb == 1).sum()}, Crash: {(y_train_nb == 0).sum()}")
print(f"  After SMOTE  — Safe: {(y_train_bal == 1).sum()}, Crash: {(y_train_bal == 0).sum()}")

print("  Running Optuna (50 trials)...")

def objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 50, 300),
        "max_depth":        trial.suggest_int("max_depth", 2, 6),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": 1.0,
        "max_delta_step":   1,
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

model_nb = xgb.XGBClassifier(**best_params_nb)
model_nb.fit(X_train_bal, y_train_bal, verbose=False)

# P(Safe) = P(class=1) for notebook convention
p_safe_nb = model_nb.predict_proba(feat_full[FEATURE_COLS].values)[:, 1]
feat_nb = feat_full.copy()
feat_nb["p_safe"] = p_safe_nb

print(f"  P(Safe) notebook — mean: {p_safe_nb.mean():.4f}, "
      f"max: {p_safe_nb.max():.4f}, "
      f">0.60: {(p_safe_nb > 0.60).mean():.1%}, "
      f">0.95: {(p_safe_nb > 0.95).mean():.1%}")


# ══════════════════════════════════════════════════════════════
# STEP 4: Run backtests — test period only (2022 – 2026-04)
# ══════════════════════════════════════════════════════════════
print("\n[4/5] Running backtests on extended test period (2022 – 2026-04)...")

test_mask = feat_full.index >= "2022-01-01"

TC = 0.71
configs = {
    "Local (ts=0.95)":    (feat_local, TC, 0.95),
    "Notebook (ts=0.60)": (feat_nb,    TC, 0.60),
    "Notebook (ts=0.50)": (feat_nb,    TC, 0.50),
    "Notebook (ts=0.65)": (feat_nb,    TC, 0.65),
    "Notebook (ts=0.70)": (feat_nb,    TC, 0.70),
    "Baseline (no filter)": (feat_local, 0.0, 0.0),
}

results = {}
for config_name, (data, tc, ts) in configs.items():
    data_test = data[test_mask]
    nav_df, metrics = run_backtest(data_test, tc, ts, skew_fn)
    results[config_name] = (nav_df, metrics)

# Print metrics
print(f"\n{'Config':<30s} {'Sharpe':>8s} {'AnnRet':>10s} {'TotRet':>10s} "
      f"{'MaxDD':>10s} {'Vol':>8s}")
print("-" * 80)
for name, (_, m) in results.items():
    print(f"{name:<30s} {m['sharpe']:>+8.4f} {m['annual_return']:>+10.2%} "
          f"{m['total_return']:>+10.2%} {m['max_dd']:>10.2%} {m['vol']:>8.2%}")


# ══════════════════════════════════════════════════════════════
# STEP 5: Plot
# ══════════════════════════════════════════════════════════════
print("\n[5/5] Generating plots...")

# ── NAV curve comparison ──
fig, ax = plt.subplots(figsize=(16, 8))

plot_styles = [
    ("Local (ts=0.95)",      "#1f77b4", "-",  2.5),
    ("Notebook (ts=0.60)",   "#d62728", "-",  2.5),
    ("Notebook (ts=0.50)",   "#ff7f0e", "--", 1.5),
    ("Notebook (ts=0.65)",   "#2ca02c", "--", 1.5),
    ("Notebook (ts=0.70)",   "#9467bd", "--", 1.5),
    ("Baseline (no filter)", "gray",    ":",  1.2),
]

for config_name, color, ls, lw in plot_styles:
    nav_df, metrics = results[config_name]
    nav_norm = nav_df["nav"] / nav_df["nav"].iloc[0] * 100
    label = (f"{config_name}  |  "
             f"Sharpe={metrics['sharpe']:+.3f}  "
             f"Ann={metrics['annual_return']:+.1%}  "
             f"Tot={metrics['total_return']:+.1%}  "
             f"MaxDD={metrics['max_dd']:.1%}")
    ax.plot(nav_norm.index, nav_norm.values, color=color,
            linestyle=ls, linewidth=lw, label=label)

ax.axhline(100, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
ax.set_ylabel("NAV (normalized to 100)", fontsize=12)
ax.set_xlabel("")
ax.set_title(
    "Extended Test Period: Local (no SMOTE) vs Notebook (SMOTE)\n"
    f"Test: 2022-01-01 – {feat_full[test_mask].index[-1].date()}  |  "
    f"tc = {TC}",
    fontsize=13, fontweight="bold",
)
ax.legend(fontsize=9, loc="upper left",
          framealpha=0.9, edgecolor="gray")
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

plt.tight_layout()
plt.savefig(OUT_DIR / "model_comparison_extended_test.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {OUT_DIR / 'model_comparison_extended_test.png'}")


# ── P(Safe) time series comparison (test period) ──
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True,
                                gridspec_kw={"hspace": 0.08})

test_dates = feat_full[test_mask].index

ax1.plot(test_dates, p_safe_local[test_mask], color="#1f77b4",
         linewidth=0.8, alpha=0.8, label="Local model P(Safe)")
ax1.axhline(0.95, color="#1f77b4", linestyle="--", linewidth=1,
            alpha=0.6, label="ts = 0.95")
ax1.set_ylabel("P(Safe) — Local", fontsize=11)
ax1.set_ylim(-0.05, 1.05)
ax1.legend(fontsize=9, loc="lower left")
ax1.grid(True, alpha=0.3)
ax1.set_title("P(Safe) Time Series — Extended Test Period", fontsize=13,
              fontweight="bold")

ax2.plot(test_dates, p_safe_nb[test_mask], color="#d62728",
         linewidth=0.8, alpha=0.8, label="Notebook model P(Safe)")
ax2.axhline(0.60, color="#d62728", linestyle="--", linewidth=1,
            alpha=0.6, label="ts = 0.60")
ax2.set_ylabel("P(Safe) — Notebook", fontsize=11)
ax2.set_ylim(-0.05, 1.05)
ax2.legend(fontsize=9, loc="lower left")
ax2.grid(True, alpha=0.3)

ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

plt.tight_layout()
plt.savefig(OUT_DIR / "psafe_timeseries_comparison_extended.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {OUT_DIR / 'psafe_timeseries_comparison_extended.png'}")

# ── Recent 30 trading days detail ──
recent = feat_full.iloc[-30:]
print(f"\n  Recent 30 days ({recent.index[0].date()} – {recent.index[-1].date()}):")
print(f"  {'Date':<12s} {'SPY':>8s} {'Local':>8s} {'NB':>8s} {'Calm':>7s}  Status")
print(f"  {'-'*60}")
for d, row in recent.iterrows():
    i = feat_full.index.get_loc(d)
    ps_l = p_safe_local[i]
    ps_n = p_safe_nb[i]
    pc   = row["p_calm"]
    flag_l = "L" if (pc > TC and ps_l > 0.95) else " "
    flag_n = "N" if (pc > TC and ps_n > 0.60) else " "
    print(f"  {d.date()}  {row['close']:>8.2f}  {ps_l:>8.4f}  {ps_n:>8.4f}  {pc:>7.4f}  {flag_l}{flag_n}")

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)
