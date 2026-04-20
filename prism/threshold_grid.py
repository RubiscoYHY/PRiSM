"""
threshold_grid.py
=================
Backtest engine and validation-set threshold grid search for PRISM.

Daily execution loop (per proposal Section 3.1):
  A. Mark-to-market all open positions
  B. Close: DTE <= 5 | profit >= 80% of max | loss >= 50% of max
  C. Open: p_calm > tc AND p_safe > ts AND n_open < 4
  D. Record daily NAV

Delta close rule omission:
  For a short put spread, max loss is capped at (K1-K2-premium) * contracts * 100.
  CaR=20%*NAV controls contracts, bounding max loss per trade to 20% of NAV.
  Spread delta is bounded in [0,1] by construction (unlike naked shorts).
  The stop-loss (loss >= 50% max) closes before deep-ITM delta accumulation.
  A separate delta threshold is therefore redundant for capped-risk spreads.

Outputs (data/threshold_grid/):
  psafe_distribution.png         -- p_safe histogram (validation set)
  threshold_sharpe_contour.png   -- Sharpe ratio over (tc, ts) grid
  threshold_return_contour.png   -- Total return over (tc, ts) grid
  threshold_grid_results.csv     -- Full grid metrics
  threshold_best.txt             -- Best pair and metrics
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import xgboost as xgb
from tqdm import tqdm

from prism.paths import DATA_DIR
from prism.data_collection import get_option_price, next_nth_friday
from prism.xgboost_train import build_features, FEATURE_COLS

XGB_DIR  = DATA_DIR / "XGBoost"        # model lives here (read-only)
GRID_DIR = DATA_DIR / "threshold_grid" # all outputs go here
GRID_DIR.mkdir(exist_ok=True)

INITIAL_CASH  = 100_000.0
MAX_POSITIONS = 4
CAR_FRACTION  = 0.20
R             = 0.04


# ─────────────────────────────────────────────────────────────
# SECTION 1: Data loading
# ─────────────────────────────────────────────────────────────

def load_backtest_data() -> pd.DataFrame:
    """
    Load and merge all data needed for the backtest:
      - SPY / VIX prices (for repricing)
      - HMM p_calm (Layer 1)
      - Engineered features (for XGBoost inference)
    Returns a single DataFrame indexed by date.
    """
    spy = pd.read_csv(DATA_DIR / "spy_vix_daily.csv", index_col="date", parse_dates=True)
    hmm = pd.read_csv(DATA_DIR / "HMM" / "hmm_pcalm_daily.csv", index_col="date", parse_dates=True)

    close_col = next(c for c in spy.columns if "spy" in c and "close" in c)
    spy = spy.rename(columns={close_col: "close"})

    df = spy.join(hmm[["p_calm"]], how="inner").sort_index()
    feat = build_features(df)
    feat = feat.join(df[["close", "vix_close", "vix9d_close"]], how="left")

    print(f"  Full dataset: {len(feat)} rows "
          f"({feat.index[0].date()} – {feat.index[-1].date()})")
    return feat


def load_skew_fn() -> interp1d:
    mults = pd.read_csv(DATA_DIR / "skew_multipliers.csv")
    return interp1d(
        mults["moneyness"].values,
        mults["skew_multiplier"].values,
        kind="linear", fill_value="extrapolate", bounds_error=False,
    )


def attach_p_safe(feat: pd.DataFrame) -> pd.DataFrame:
    """
    Load the saved XGBoost model and compute p_safe = P(no crash) for all dates.
    No look-ahead: inference uses only past features, model trained on 2015-2020 only.
    """
    model = xgb.XGBClassifier()
    model.load_model(str(XGB_DIR / "xgb_model.json"))
    proba = model.predict_proba(feat[FEATURE_COLS].values)
    feat = feat.copy()
    feat["p_safe"] = proba[:, 0]   # P(class=0) = P(safe, no crash in next 30d)
    return feat


# ─────────────────────────────────────────────────────────────
# SECTION 2: p_safe distribution diagnostic
# ─────────────────────────────────────────────────────────────

def plot_psafe_distribution(feat_val: pd.DataFrame) -> None:
    """
    Plot p_safe histogram for the validation set to diagnose classifier
    calibration and explain the contour map structure.

    A bimodal distribution (mass near 0 and near 1, gap in 0.80-0.94)
    would confirm that XGBoost acts as a hard switch rather than a
    continuous probability filter, explaining the 3-region pattern.
    """
    p = feat_val["p_safe"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: full histogram
    ax = axes[0]
    ax.hist(p, bins=60, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.axvline(0.95, color="crimson",   linestyle="--", linewidth=1.5, label="ts = 0.95 (optimal)")
    ax.axvline(0.80, color="orange",    linestyle="--", linewidth=1.5, label="ts = 0.80")
    ax.axvline(0.50, color="seagreen",  linestyle="--", linewidth=1.5, label="ts = 0.50")
    ax.set_xlabel("p_safe  (XGBoost P(no crash))", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("p_safe distribution — Validation set (2020–2022)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: cumulative distribution (shows how many days pass each threshold)
    ax2 = axes[1]
    sorted_p = np.sort(p)
    cdf = np.arange(1, len(sorted_p) + 1) / len(sorted_p)
    ax2.plot(sorted_p, 1 - cdf, color="steelblue", linewidth=2)
    ax2.axvline(0.95, color="crimson",  linestyle="--", linewidth=1.5,
                label=f"ts=0.95 → {(p > 0.95).mean():.1%} of days")
    ax2.axvline(0.80, color="orange",   linestyle="--", linewidth=1.5,
                label=f"ts=0.80 → {(p > 0.80).mean():.1%} of days")
    ax2.axvline(0.50, color="seagreen", linestyle="--", linewidth=1.5,
                label=f"ts=0.50 → {(p > 0.50).mean():.1%} of days")
    ax2.set_xlabel("p_safe threshold", fontsize=11)
    ax2.set_ylabel("Fraction of days passing threshold", fontsize=11)
    ax2.set_title("Complementary CDF of p_safe", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(GRID_DIR / "psafe_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved → {GRID_DIR / 'psafe_distribution.png'}")

    # Print key statistics
    print(f"  p_safe stats: mean={p.mean():.3f}  median={p.median():.3f}  "
          f"std={p.std():.3f}")
    print(f"  Days with p_safe > 0.95 : {(p > 0.95).sum():4d}  ({(p > 0.95).mean():.1%})")
    print(f"  Days with p_safe > 0.80 : {(p > 0.80).sum():4d}  ({(p > 0.80).mean():.1%})")
    print(f"  Days with p_safe > 0.50 : {(p > 0.50).sum():4d}  ({(p > 0.50).mean():.1%})")
    print(f"  Days with p_safe in [0.80, 0.95): "
          f"{((p >= 0.80) & (p < 0.95)).sum():3d}  "
          f"({((p >= 0.80) & (p < 0.95)).mean():.1%})  ← 'no man's land'")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Single-run backtest engine
# ─────────────────────────────────────────────────────────────

def _reprice(pos: dict, S: float, vix: float, vix9d,
             date: pd.Timestamp, skew_fn: interp1d) -> float:
    T_rem = (pos["expiry"] - date).days
    if T_rem <= 0:
        return max(pos["K1"] - S, 0.0) - max(pos["K2"] - S, 0.0)
    short_val = get_option_price(S, pos["K1"], T_rem, vix, skew_fn, R, vix9d)
    long_val  = get_option_price(S, pos["K2"], T_rem, vix, skew_fn, R, vix9d)
    return short_val - long_val


def run_backtest(
    data: pd.DataFrame,
    threshold_calm: float,
    threshold_safe: float,
    skew_fn: interp1d,
    initial_cash: float = INITIAL_CASH,
) -> tuple[pd.DataFrame, dict]:
    """
    Run one full backtest pass on `data` with given thresholds.

    NAV accounting:
      NAV_t = initial_cash + closed_pnl_cumulative + sum(unrealized_pnl_open_positions)
      On open : NAV unchanged (unrealized_pnl of new position = 0)
      On close: unrealized P&L becomes realized, NAV unchanged at moment of close
    """
    positions   = []
    closed_pnl  = 0.0
    nav_records = []

    for date, row in data.iterrows():
        S      = float(row["close"])
        vix    = float(row["vix_close"])
        vix9d  = float(row["vix9d_close"]) if not np.isnan(row.get("vix9d_close", np.nan)) else None
        p_calm = float(row["p_calm"])
        p_safe = float(row["p_safe"])

        if np.isnan(S) or np.isnan(vix) or S <= 0:
            continue

        # ── A. Mark-to-market ──
        for pos in positions:
            pos["current_value"] = _reprice(pos, S, vix, vix9d, date, skew_fn)

        # ── B. Close check ──
        remaining = []
        for pos in positions:
            dte      = (pos["expiry"] - date).days
            val      = pos["current_value"]
            credit   = pos["open_credit"]
            max_loss = pos["max_loss_per_share"]

            should_close = (
                dte <= 5
                or val <= 0.20 * credit
                or val >= credit + 0.50 * max_loss
            )

            if should_close:
                closed_pnl += (credit - val) * pos["contracts"] * 100
            else:
                remaining.append(pos)

        positions = remaining

        # ── Current NAV ──
        unrealized = sum((p["open_credit"] - p["current_value"]) * p["contracts"] * 100
                         for p in positions)
        nav = initial_cash + closed_pnl + unrealized

        # ── C. Open check ──
        if (p_calm > threshold_calm
                and p_safe > threshold_safe
                and len(positions) < MAX_POSITIONS):

            expiry = next_nth_friday(pd.Timestamp(date), n=4)
            T_days = (expiry - pd.Timestamp(date)).days
            K1     = S * 0.95
            K2     = S * 0.91

            sp = get_option_price(S, K1, T_days, vix, skew_fn, R, vix9d)
            lp = get_option_price(S, K2, T_days, vix, skew_fn, R, vix9d)
            open_credit        = sp - lp
            max_loss_per_share = (K1 - K2) - open_credit

            if max_loss_per_share > 1e-3 and open_credit > 0:
                contracts = (CAR_FRACTION * nav) / (max_loss_per_share * 100)
                positions.append({
                    "expiry":             expiry,
                    "K1":                 K1,
                    "K2":                 K2,
                    "open_credit":        open_credit,
                    "max_loss_per_share": max_loss_per_share,
                    "contracts":          contracts,
                    "current_value":      open_credit,
                })

        nav_records.append({"date": date, "nav": nav})

    nav_df = pd.DataFrame(nav_records).set_index("date")
    nav_df["daily_return"] = nav_df["nav"].pct_change().fillna(0.0)

    n_days     = len(nav_df)
    total_ret  = nav_df["nav"].iloc[-1] / initial_cash - 1
    annual_ret = (1 + total_ret) ** (252 / n_days) - 1
    ann_vol    = nav_df["daily_return"].std() * np.sqrt(252)
    sharpe     = annual_ret / ann_vol if ann_vol > 1e-8 else 0.0
    max_dd     = ((nav_df["nav"] / nav_df["nav"].cummax()) - 1).min()

    return nav_df, {
        "sharpe":        round(sharpe, 4),
        "total_return":  round(total_ret, 4),
        "annual_return": round(annual_ret, 4),
        "vol":           round(ann_vol, 4),
        "max_dd":        round(max_dd, 4),
    }


# ─────────────────────────────────────────────────────────────
# SECTION 4: Threshold grid search
# ─────────────────────────────────────────────────────────────

def run_threshold_grid(
    data_val: pd.DataFrame,
    skew_fn: interp1d,
    tc_grid: np.ndarray,
    ts_grid: np.ndarray,
) -> pd.DataFrame:
    records = []
    with tqdm(total=len(tc_grid) * len(ts_grid), desc="  Grid search") as pbar:
        for tc in tc_grid:
            for ts in ts_grid:
                _, metrics = run_backtest(data_val, tc, ts, skew_fn)
                records.append({"tc": tc, "ts": ts, **metrics})
                pbar.update(1)
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# SECTION 5: Contour plots
# ─────────────────────────────────────────────────────────────

def _contour_plot(
    results: pd.DataFrame,
    tc_grid: np.ndarray,
    ts_grid: np.ndarray,
    metric: str,
    title: str,
    cmap: str,
    fname: str,
    best_tc: float,
    best_ts: float,
) -> None:
    Z = results.pivot(index="ts", columns="tc", values=metric).values

    fig, ax = plt.subplots(figsize=(7, 6))
    cf = ax.contourf(tc_grid, ts_grid, Z, levels=40, cmap=cmap)
    cs = ax.contour(tc_grid, ts_grid, Z, levels=15,
                    colors="white", linewidths=0.5, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
    plt.colorbar(cf, ax=ax, label=metric)

    ax.scatter([best_tc], [best_ts], color="red", s=80, zorder=5,
               label=f"Best ({best_tc:.2f}, {best_ts:.2f})")
    ax.legend(fontsize=9)

    ax.set_xlabel("threshold_calm  (HMM P(Calm) > tc)", fontsize=11)
    ax.set_ylabel("threshold_safe  (XGBoost P(Safe) > ts)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(tc_grid[::5])
    ax.set_yticks(ts_grid[::5])
    ax.tick_params(axis="both", labelsize=8)
    plt.tight_layout()
    plt.savefig(GRID_DIR / fname, dpi=150)
    plt.close()
    print(f"  Saved → {GRID_DIR / fname}")


def plot_contours(results: pd.DataFrame, tc_grid: np.ndarray, ts_grid: np.ndarray) -> None:
    best_row = results.loc[results["sharpe"].idxmax()]
    best_tc, best_ts = best_row["tc"], best_row["ts"]

    _contour_plot(results, tc_grid, ts_grid,
                  metric="sharpe",
                  title="Sharpe Ratio — Threshold Grid (Validation 2020–2022)",
                  cmap="RdYlGn", fname="threshold_sharpe_contour.png",
                  best_tc=best_tc, best_ts=best_ts)
    _contour_plot(results, tc_grid, ts_grid,
                  metric="total_return",
                  title="Total Return — Threshold Grid (Validation 2020–2022)",
                  cmap="Blues", fname="threshold_return_contour.png",
                  best_tc=best_tc, best_ts=best_ts)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("PRISM — Threshold Grid Search")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    feat    = load_backtest_data()
    skew_fn = load_skew_fn()
    feat    = attach_p_safe(feat)

    feat_val = feat[(feat.index >= "2020-01-01") & (feat.index < "2022-01-01")]
    print(f"  Validation rows: {len(feat_val)} "
          f"({feat_val.index[0].date()} – {feat_val.index[-1].date()})")

    print("\n[2/5] p_safe distribution diagnostic...")
    plot_psafe_distribution(feat_val)

    print("\n[3/5] Running threshold grid search (46×46 = 2,116 combinations)...")
    tc_grid = np.round(np.arange(0.50, 0.96, 0.01), 2)
    ts_grid = np.round(np.arange(0.50, 0.96, 0.01), 2)
    results = run_threshold_grid(feat_val, skew_fn, tc_grid, ts_grid)
    results.to_csv(GRID_DIR / "threshold_grid_results.csv", index=False)
    print(f"  Saved → {GRID_DIR / 'threshold_grid_results.csv'}")

    print("\n[4/5] Plotting contour maps...")
    plot_contours(results, tc_grid, ts_grid)

    print("\n[5/5] Best threshold pair:")
    best = results.loc[results["sharpe"].idxmax()]
    summary = (
        f"Best threshold pair (Sharpe-optimal on validation set)\n"
        f"{'='*45}\n"
        f"  threshold_calm  : {best['tc']:.2f}\n"
        f"  threshold_safe  : {best['ts']:.2f}\n"
        f"  Sharpe          : {best['sharpe']:.4f}\n"
        f"  Annual return   : {best['annual_return']:.2%}\n"
        f"  Total return    : {best['total_return']:.2%}\n"
        f"  Ann. volatility : {best['vol']:.2%}\n"
        f"  Max drawdown    : {best['max_dd']:.2%}\n"
    )
    print(summary)
    (GRID_DIR / "threshold_best.txt").write_text(summary)

    print("  Running baseline (no ML filter, tc=0, ts=0)...")
    _, base = run_backtest(feat_val, 0.0, 0.0, skew_fn)
    print(f"  Baseline  Sharpe={base['sharpe']:.4f}  "
          f"Return={base['total_return']:.2%}  MaxDD={base['max_dd']:.2%}")

    print("\n" + "=" * 60)
    print("Done. All outputs in:", GRID_DIR)
    print("=" * 60)
