"""
compare_timing.py
=================
Pure market-timing comparison: hold SPY when both signals pass,
hold cash otherwise. No options, no spreads — just long SPY vs cash.

Compares:
  - Local model  (tc=0.71, ts=0.95)
  - Notebook model (tc=0.71, ts=0.60/0.70)
  - Buy-and-hold SPY

Test period: 2022-01-01 – 2026-04-24 (extended data)
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
from prism.xgboost_train import FEATURE_COLS

EXT_DIR = DATA_DIR / "extended"
XGB_DIR = DATA_DIR / "XGBoost"
OUT_DIR = DATA_DIR / "analysis"
OUT_DIR.mkdir(exist_ok=True)


# ── Feature engineering (no label) ──

def build_features_nolabel(df):
    r, S = df["log_return"], df["close"]
    feat = pd.DataFrame(index=df.index)
    feat["RV_5d"]     = r.rolling(5).std()  * np.sqrt(252)
    feat["RV_20d"]    = r.rolling(20).std() * np.sqrt(252)
    feat["RV_60d"]    = r.rolling(60).std() * np.sqrt(252)
    feat["RV_ratio"]  = feat["RV_20d"] / feat["RV_60d"].replace(0, np.nan)
    feat["Mom_5d"]    = r.rolling(5).sum()
    feat["Mom_20d"]   = r.rolling(20).sum()
    feat["DD_60d"]    = (S - S.rolling(60).max()) / S.rolling(60).max()
    feat["RSkew_20d"] = r.rolling(20).skew()
    feat["p_calm"]    = df["p_calm"]
    return feat.dropna()


def build_features_with_label(df):
    feat = build_features_nolabel(df)
    S = df["close"]
    future_min = S.shift(-1).rolling(21, min_periods=1).min().shift(-20)
    feat["label"] = ((future_min - S.reindex(feat.index)) / S.reindex(feat.index) < -0.05).astype(int)
    return feat.dropna()


# ── Timing backtest ──

def run_timing(prices, signal, name=""):
    """
    prices: Series of daily SPY close (aligned index with signal)
    signal: boolean Series — True = hold SPY, False = hold cash
    Returns nav Series and metrics dict.
    """
    daily_ret = prices.pct_change().fillna(0.0)
    # When signal is True on day t, we hold SPY and earn day t+1 return
    # Use signal shifted by 1: decision at close of day t → exposed on day t+1
    position = signal.shift(1).fillna(False).astype(float)
    strat_ret = daily_ret * position
    nav = (1 + strat_ret).cumprod() * 100

    n_days     = len(nav)
    total_ret  = nav.iloc[-1] / 100 - 1
    years      = n_days / 252
    annual_ret = (1 + total_ret) ** (1 / years) - 1
    ann_vol    = strat_ret.std() * np.sqrt(252)
    sharpe     = annual_ret / ann_vol if ann_vol > 1e-8 else 0.0
    max_dd     = ((nav / nav.cummax()) - 1).min()

    # Sortino
    downside   = strat_ret[strat_ret < 0]
    down_vol   = downside.std() * np.sqrt(252) if len(downside) > 1 else 1e-8
    sortino    = annual_ret / down_vol

    # CVaR 95%
    var_95     = np.percentile(strat_ret, 5)
    cvar_95    = strat_ret[strat_ret <= var_95].mean()

    # Exposure
    exposure   = position.mean()
    n_trades   = (position.diff().abs() > 0.5).sum()

    metrics = {
        "sharpe":     round(sharpe, 4),
        "sortino":    round(sortino, 4),
        "annual_ret": round(annual_ret, 4),
        "total_ret":  round(total_ret, 4),
        "ann_vol":    round(ann_vol, 4),
        "max_dd":     round(max_dd, 4),
        "cvar_95":    round(cvar_95, 6),
        "exposure":   round(exposure, 4),
        "n_trades":   int(n_trades),
    }
    return nav, strat_ret, metrics


# ══════════════════════════════════════════════════════════════
# STEP 1: Load data
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("Market Timing Comparison: Long SPY vs Cash")
print("=" * 60)

print("\n[1/4] Loading extended data...")
spy = pd.read_csv(EXT_DIR / "spy_vix_daily.csv", index_col="date", parse_dates=True)
hmm = pd.read_csv(EXT_DIR / "hmm_pcalm_daily.csv", index_col="date", parse_dates=True)

close_col = next(c for c in spy.columns if "spy" in c and "close" in c)
spy = spy.rename(columns={close_col: "close"})
df = spy.join(hmm[["p_calm"]], how="inner").sort_index()

feat = build_features_nolabel(df)
feat = feat.join(df[["close", "vix_close", "vix9d_close"]], how="left")

feat_labeled = build_features_with_label(df)
print(f"  Dataset: {len(feat)} rows ({feat.index[0].date()} – {feat.index[-1].date()})")


# ══════════════════════════════════════════════════════════════
# STEP 2: Load / train models
# ══════════════════════════════════════════════════════════════
print("\n[2/4] Preparing models...")

# Local model
model_local = xgb.XGBClassifier()
model_local.load_model(str(XGB_DIR / "xgb_model.json"))
feat["p_safe_local"] = model_local.predict_proba(feat[FEATURE_COLS].values)[:, 0]

# Notebook model (SMOTE)
train_data = feat_labeled[feat_labeled.index < "2020-01-01"]
X_train = train_data[FEATURE_COLS].values
y_train_nb = 1 - train_data["label"].values  # flip labels

smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_train, y_train_nb)
print(f"  SMOTE: {(y_train_nb==1).sum()} Safe + {(y_train_nb==0).sum()} Crash → "
      f"{(y_bal==1).sum()} + {(y_bal==0).sum()}")

print("  Optuna (50 trials)...")

def objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 50, 300),
        "max_depth":        trial.suggest_int("max_depth", 2, 6),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": 1.0, "max_delta_step": 1,
        "objective": "binary:logistic", "eval_metric": "logloss",
        "verbosity": 0, "random_state": 42,
    }
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    for tr_idx, vl_idx in tscv.split(X_bal):
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_bal[tr_idx], y_bal[tr_idx], verbose=False)
        scores.append(f1_score(y_bal[vl_idx], clf.predict(X_bal[vl_idx]), zero_division=0))
    return np.mean(scores)

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50, show_progress_bar=True)

bp = {**study.best_params, "scale_pos_weight": 1.0, "max_delta_step": 1,
      "objective": "binary:logistic", "eval_metric": "logloss",
      "verbosity": 0, "random_state": 42}
model_nb = xgb.XGBClassifier(**bp)
model_nb.fit(X_bal, y_bal, verbose=False)
feat["p_safe_nb"] = model_nb.predict_proba(feat[FEATURE_COLS].values)[:, 1]

print(f"  Best CV F1: {study.best_value:.4f}")


# ══════════════════════════════════════════════════════════════
# STEP 3: Run timing backtests
# ══════════════════════════════════════════════════════════════
print("\n[3/4] Running timing backtests (2022 – 2026-04)...")

test = feat[feat.index >= "2022-01-01"].copy()
prices = test["close"]

TC = 0.71

configs = {
    "Buy & Hold SPY": pd.Series(True, index=test.index),
    "Local (tc=0.71, ts=0.95)":
        (test["p_calm"] > TC) & (test["p_safe_local"] > 0.95),
    "Notebook (tc=0.71, ts=0.60)":
        (test["p_calm"] > TC) & (test["p_safe_nb"] > 0.60),
    "Notebook (tc=0.71, ts=0.70)":
        (test["p_calm"] > TC) & (test["p_safe_nb"] > 0.70),
    "HMM only (tc=0.71)":
        (test["p_calm"] > TC),
    "Local XGB only (ts=0.95)":
        (test["p_safe_local"] > 0.95),
    "Notebook XGB only (ts=0.60)":
        (test["p_safe_nb"] > 0.60),
}

results = {}
for name, signal in configs.items():
    nav, strat_ret, metrics = run_timing(prices, signal, name)
    results[name] = (nav, strat_ret, metrics)

# Print table
print(f"\n{'Config':<32s} {'Sharpe':>7s} {'Sortino':>8s} {'AnnRet':>8s} "
      f"{'TotRet':>8s} {'MaxDD':>8s} {'Vol':>7s} {'CVaR95':>8s} "
      f"{'Expos':>6s} {'Trades':>6s}")
print("-" * 110)
for name, (_, _, m) in results.items():
    print(f"{name:<32s} {m['sharpe']:>+7.3f} {m['sortino']:>+8.3f} "
          f"{m['annual_ret']:>+8.1%} {m['total_ret']:>+8.1%} "
          f"{m['max_dd']:>8.1%} {m['ann_vol']:>7.1%} {m['cvar_95']:>+8.4f} "
          f"{m['exposure']:>6.1%} {m['n_trades']:>6d}")


# ══════════════════════════════════════════════════════════════
# STEP 4: Plots
# ══════════════════════════════════════════════════════════════
print("\n[4/4] Generating plots...")

# ── Plot 1: NAV curves ──
fig, ax = plt.subplots(figsize=(16, 8))

styles = [
    ("Buy & Hold SPY",              "black",   "-",  2.0),
    ("Local (tc=0.71, ts=0.95)",    "#1f77b4", "-",  2.5),
    ("Notebook (tc=0.71, ts=0.60)", "#d62728", "-",  2.5),
    ("Notebook (tc=0.71, ts=0.70)", "#9467bd", "--", 1.8),
    ("HMM only (tc=0.71)",         "#2ca02c", "--", 1.5),
    ("Local XGB only (ts=0.95)",    "#17becf", ":", 1.5),
    ("Notebook XGB only (ts=0.60)", "#ff7f0e", ":",  1.5),
]

for name, color, ls, lw in styles:
    nav, _, m = results[name]
    label = (f"{name}  |  Sharpe={m['sharpe']:+.3f}  "
             f"Ann={m['annual_ret']:+.1%}  MaxDD={m['max_dd']:.1%}  "
             f"Exp={m['exposure']:.0%}")
    ax.plot(nav.index, nav.values, color=color, ls=ls, lw=lw, label=label)

ax.axhline(100, color="gray", ls=":", lw=0.8, alpha=0.4)
ax.set_ylabel("NAV (normalized to 100)", fontsize=12)
ax.set_title(
    "Market Timing: Hold SPY when signals pass, Cash otherwise\n"
    f"Test: 2022-01-01 – {test.index[-1].date()}",
    fontsize=14, fontweight="bold")
ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "timing_comparison_nav.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {OUT_DIR / 'timing_comparison_nav.png'}")


# ── Plot 2: Exposure & drawdown ──
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
                          gridspec_kw={"height_ratios": [2, 1, 1], "hspace": 0.08})

# Panel 1: NAV (top 3 only)
ax = axes[0]
for name, color, ls, lw in styles[:4]:
    nav, _, m = results[name]
    ax.plot(nav.index, nav.values, color=color, ls=ls, lw=lw, label=name)
ax.axhline(100, color="gray", ls=":", lw=0.8, alpha=0.4)
ax.set_ylabel("NAV", fontsize=11)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)
ax.set_title("Market Timing: NAV, Position & Drawdown", fontsize=13, fontweight="bold")

# Panel 2: Position (in/out)
ax2 = axes[1]
sig_local = configs["Local (tc=0.71, ts=0.95)"].astype(float)
sig_nb    = configs["Notebook (tc=0.71, ts=0.60)"].astype(float)
ax2.fill_between(test.index, 0, sig_local, alpha=0.4, color="#1f77b4",
                 step="post", label="Local in-market")
ax2.fill_between(test.index, 0, -sig_nb, alpha=0.4, color="#d62728",
                 step="post", label="Notebook in-market (inverted)")
ax2.set_ylabel("Position", fontsize=11)
ax2.set_ylim(-1.3, 1.3)
ax2.set_yticks([-1, 0, 1])
ax2.set_yticklabels(["NB in", "Cash", "Local in"])
ax2.legend(fontsize=9, loc="lower left")
ax2.grid(True, alpha=0.3)

# Panel 3: Drawdown
ax3 = axes[2]
for name, color in [("Buy & Hold SPY", "black"),
                     ("Local (tc=0.71, ts=0.95)", "#1f77b4"),
                     ("Notebook (tc=0.71, ts=0.60)", "#d62728")]:
    nav, _, _ = results[name]
    dd = (nav / nav.cummax() - 1) * 100
    ax3.fill_between(dd.index, dd.values, 0, alpha=0.3, color=color, label=name)
    ax3.plot(dd.index, dd.values, color=color, lw=0.8)
ax3.set_ylabel("Drawdown (%)", fontsize=11)
ax3.legend(fontsize=9, loc="lower left")
ax3.grid(True, alpha=0.3)

ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "timing_comparison_detail.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {OUT_DIR / 'timing_comparison_detail.png'}")


# ── Plot 3: Monthly returns heatmap ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, cname) in zip(axes, [
    ("Buy & Hold SPY", "SPY B&H"),
    ("Local (tc=0.71, ts=0.95)", "Local Timing"),
    ("Notebook (tc=0.71, ts=0.60)", "Notebook Timing"),
]):
    _, strat_ret, _ = results[name]
    monthly = strat_ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    pivot = pd.DataFrame({
        "year":  monthly.index.year,
        "month": monthly.index.month,
        "ret":   monthly.values,
    }).pivot(index="year", columns="month", values="ret")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]

    im = ax.imshow(pivot.values * 100, cmap="RdYlGn", aspect="auto",
                   vmin=-10, vmax=10)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(cname, fontsize=11, fontweight="bold")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v*100:+.1f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(v) < 0.06 else "white")

fig.colorbar(im, ax=axes, label="Monthly Return (%)", shrink=0.8)
fig.suptitle("Monthly Returns Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "timing_monthly_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {OUT_DIR / 'timing_monthly_heatmap.png'}")

print("\n" + "=" * 60)
print("Done. All outputs in:", OUT_DIR)
print("=" * 60)
