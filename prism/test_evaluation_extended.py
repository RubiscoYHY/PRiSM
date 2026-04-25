"""
test_evaluation_extended.py
============================
Extended test-set evaluation (2022 – 2026-04) for PRISM.

Compares three investment approaches using the SMOTE-trained XGBoost model:
  1. Short put spread strategy  (options, signal-gated)
  2. SPY buy-and-hold           (passive benchmark)
  3. Long-only market timing    (hold SPY when both signals pass, cash otherwise)

Models:
  - HMM     : 2-state GaussianHMM trained on 2015-2020
  - XGBoost  : SMOTE-balanced, loaded from data/XGBoost/xgb_model.json
  - Thresholds: tc=0.71, ts=0.62 (Sharpe-optimal on validation set)

Run data_collection_extended.py first to populate data/extended/.

Outputs (results/test_evaluation_extended/):
  test_nav_comparison_ext.png  -- NAV curves for all strategies
  test_metrics_ext.csv         -- full metrics table
  test_metrics_ext.txt         -- human-readable summary
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import xgboost as xgb
from scipy.interpolate import interp1d

from prism.paths import DATA_DIR, RESULTS_DIR
from prism.xgboost_train import build_features, FEATURE_COLS
from prism.threshold_grid import run_backtest, load_skew_fn, INITIAL_CASH

EXT_DIR      = DATA_DIR / "extended"
XGB_DIR      = DATA_DIR / "XGBoost"
EXT_TEST_DIR = RESULTS_DIR / "test_evaluation_extended"
EXT_TEST_DIR.mkdir(exist_ok=True)

TC_OPT = 0.71
TS_OPT = 0.62

TEST_START = "2022-01-01"
TEST_END   = "2026-05-01"

# Option spread strategies (run through the backtest engine)
SPREAD_STRATEGIES = [
    {"name": "Spread: Baseline",    "tc": 0.00,   "ts": 0.00,   "color": "#888888", "ls": ":"},
    {"name": "Spread: HMM-only",    "tc": TC_OPT, "ts": 0.00,   "color": "#F5A623", "ls": "-."},
    {"name": "Spread: HMM+XGBoost", "tc": TC_OPT, "ts": TS_OPT, "color": "#2A9D8F", "ls": "-"},
]


# ─────────────────────────────────────────────────────────────
# SECTION 1: Data loading
# ─────────────────────────────────────────────────────────────

def load_extended_data() -> pd.DataFrame:
    """
    Load extended SPY/VIX + HMM p_calm from data/extended/.
    Build XGBoost feature matrix and compute p_safe using the
    original trained model (no retraining).
    """
    spy = pd.read_csv(
        EXT_DIR / "spy_vix_daily.csv", index_col="date", parse_dates=True
    )
    hmm = pd.read_csv(
        EXT_DIR / "hmm_pcalm_daily.csv", index_col="date", parse_dates=True
    )

    close_col = next(c for c in spy.columns if "spy" in c and "close" in c)
    spy = spy.rename(columns={close_col: "close"})

    df = spy.join(hmm[["p_calm"]], how="inner").sort_index()
    feat = build_features(df)
    feat = feat.join(df[["close", "vix_close", "vix9d_close"]], how="left")

    # XGBoost inference — load original trained model (no retraining)
    model = xgb.XGBClassifier()
    model.load_model(str(XGB_DIR / "xgb_model.json"))
    proba = model.predict_proba(feat[FEATURE_COLS].values)
    feat = feat.copy()
    feat["p_safe"] = proba[:, 0]   # P(class=0) = P(safe, no crash in next 30d)

    print(f"  Extended dataset: {len(feat)} rows "
          f"({feat.index[0].date()} – {feat.index[-1].date()})")
    return feat


# ─────────────────────────────────────────────────────────────
# SECTION 2: Metrics
# ─────────────────────────────────────────────────────────────

def compute_extended_metrics(nav_df: pd.DataFrame, base_metrics: dict) -> dict:
    r = nav_df["daily_return"]

    var_95  = r.quantile(0.05)
    cvar_95 = r[r <= var_95].mean()

    neg_r        = r[r < 0]
    downside_vol = np.sqrt((neg_r ** 2).mean()) * np.sqrt(252) if len(neg_r) > 0 else 1e-8
    annual_ret   = base_metrics["annual_return"]
    sortino      = annual_ret / downside_vol if downside_vol > 1e-8 else 0.0

    return {
        **base_metrics,
        "cvar_95": round(float(cvar_95), 4),
        "sortino": round(float(sortino), 4),
    }


def compute_spy_metrics(spy_series: pd.Series) -> dict:
    r = spy_series.pct_change().fillna(0.0)

    n_days     = len(spy_series)
    total_ret  = spy_series.iloc[-1] / spy_series.iloc[0] - 1
    annual_ret = (1 + total_ret) ** (252 / n_days) - 1
    ann_vol    = r.std() * np.sqrt(252)
    sharpe     = annual_ret / ann_vol if ann_vol > 1e-8 else 0.0
    max_dd     = ((spy_series / spy_series.cummax()) - 1).min()

    var_95  = r.quantile(0.05)
    cvar_95 = r[r <= var_95].mean()

    neg_r        = r[r < 0]
    downside_vol = np.sqrt((neg_r ** 2).mean()) * np.sqrt(252) if len(neg_r) > 0 else 1e-8
    sortino      = annual_ret / downside_vol if downside_vol > 1e-8 else 0.0

    return {
        "sharpe":        round(float(sharpe),     4),
        "total_return":  round(float(total_ret),  4),
        "annual_return": round(float(annual_ret), 4),
        "vol":           round(float(ann_vol),    4),
        "max_dd":        round(float(max_dd),     4),
        "cvar_95":       round(float(cvar_95),    4),
        "sortino":       round(float(sortino),    4),
    }


# ─────────────────────────────────────────────────────────────
# SECTION 3: Plot
# ─────────────────────────────────────────────────────────────

def run_timing_strategy(
    prices: pd.Series,
    signal: pd.Series,
) -> tuple[pd.Series, dict]:
    """
    Long-only market timing: hold SPY when signal is True, hold cash otherwise.
    Signal on day t → exposed to day t+1 return.
    Returns (nav_series, metrics_dict).
    """
    daily_ret = prices.pct_change().fillna(0.0)
    position  = signal.shift(1).fillna(False).astype(float)
    strat_ret = daily_ret * position
    nav       = (1 + strat_ret).cumprod() * INITIAL_CASH

    n_days     = len(nav)
    total_ret  = nav.iloc[-1] / INITIAL_CASH - 1
    years      = n_days / 252
    annual_ret = (1 + total_ret) ** (1 / years) - 1
    ann_vol    = strat_ret.std() * np.sqrt(252)
    sharpe     = annual_ret / ann_vol if ann_vol > 1e-8 else 0.0
    max_dd     = ((nav / nav.cummax()) - 1).min()

    var_95  = strat_ret.quantile(0.05)
    cvar_95 = strat_ret[strat_ret <= var_95].mean()

    neg_r        = strat_ret[strat_ret < 0]
    downside_vol = np.sqrt((neg_r ** 2).mean()) * np.sqrt(252) if len(neg_r) > 0 else 1e-8
    sortino      = annual_ret / downside_vol

    exposure = position.mean()
    n_trades = int((position.diff().abs() > 0.5).sum())

    return nav, {
        "sharpe":        round(float(sharpe), 4),
        "total_return":  round(float(total_ret), 4),
        "annual_return": round(float(annual_ret), 4),
        "vol":           round(float(ann_vol), 4),
        "max_dd":        round(float(max_dd), 4),
        "cvar_95":       round(float(cvar_95), 4),
        "sortino":       round(float(sortino), 4),
        "exposure":      round(float(exposure), 4),
        "n_trades":      n_trades,
    }


def plot_nav_comparison(
    spy_series: pd.Series,
    all_navs: dict[str, pd.Series],
    all_metrics: list[dict],
    test_end: str,
) -> None:
    """
    Plot NAV curves for all strategies (spread, timing, SPY B&H).
    """
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(2, 1, figsize=(16, 12),
                              gridspec_kw={"height_ratios": [2, 1], "hspace": 0.15})

    # ── Top panel: NAV curves ──
    ax = axes[0]
    colors = {
        "SPY (Buy & Hold)":        "#264653",
        "Timing: HMM+XGBoost":    "#E76F51",
        "Spread: Baseline":       "#888888",
        "Spread: HMM-only":       "#F5A623",
        "Spread: HMM+XGBoost":    "#2A9D8F",
    }
    lstyles = {
        "SPY (Buy & Hold)":        "-",
        "Timing: HMM+XGBoost":    "-",
        "Spread: Baseline":       ":",
        "Spread: HMM-only":       "-.",
        "Spread: HMM+XGBoost":    "-",
    }
    lwidths = {
        "SPY (Buy & Hold)":        2.0,
        "Timing: HMM+XGBoost":    2.5,
        "Spread: Baseline":       1.2,
        "Spread: HMM-only":       1.5,
        "Spread: HMM+XGBoost":    2.5,
    }

    metrics_lookup = {m["strategy"]: m for m in all_metrics}

    for name, nav in all_navs.items():
        norm = nav / nav.iloc[0] * 100
        m = metrics_lookup.get(name, {})
        sharpe = m.get("sharpe", 0)
        ann_ret = m.get("annual_return", 0)
        max_dd = m.get("max_dd", 0)
        label = f"{name}  |  Sharpe={sharpe:+.3f}  Ann={ann_ret:+.1%}  MaxDD={max_dd:.1%}"
        ax.plot(norm.index, norm.values,
                color=colors.get(name, "black"),
                linestyle=lstyles.get(name, "-"),
                linewidth=lwidths.get(name, 1.5),
                label=label, zorder=4)

    ax.axhline(100, color="black", linewidth=0.6, linestyle="--", alpha=0.4)

    orig_end = pd.Timestamp("2024-12-31")
    ax.axvline(orig_end, color="navy", linewidth=1.2, linestyle="--", alpha=0.5)
    ax.text(orig_end, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 200,
            "  Original test end", fontsize=8, color="navy", va="top")

    ax.set_title(
        f"PRiSM — Strategy Comparison: Short Put Spread vs Market Timing vs Buy & Hold\n"
        f"Test: 2022-01-01 – {test_end}  |  tc={TC_OPT}  ts={TS_OPT}",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylabel("NAV (normalized to 100)", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.25)

    # ── Bottom panel: Drawdown ──
    ax2 = axes[1]
    highlight = ["SPY (Buy & Hold)", "Timing: HMM+XGBoost", "Spread: HMM+XGBoost"]
    for name in highlight:
        if name not in all_navs:
            continue
        nav = all_navs[name]
        dd = (nav / nav.cummax() - 1) * 100
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.3,
                         color=colors.get(name, "gray"))
        ax2.plot(dd.index, dd.values, color=colors.get(name, "gray"),
                 linewidth=0.8, label=name)

    ax2.set_ylabel("Drawdown (%)", fontsize=11)
    ax2.legend(fontsize=9, loc="lower left")
    ax2.grid(True, alpha=0.25)

    for a in axes:
        a.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(a.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    out = EXT_TEST_DIR / "test_nav_comparison_ext.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Metrics table
# ─────────────────────────────────────────────────────────────

def save_metrics(all_metrics: list[dict], n_days: int, date_range: str) -> None:
    df = pd.DataFrame(all_metrics).set_index("strategy")
    df.to_csv(EXT_TEST_DIR / "test_metrics_ext.csv")
    print(f"  Saved → {EXT_TEST_DIR / 'test_metrics_ext.csv'}")

    header = (
        f"{'Strategy':<16}  {'Sharpe':>7}  {'AnnRet':>8}  "
        f"{'Vol':>7}  {'MaxDD':>8}  {'CVaR95':>8}  {'Sortino':>8}"
    )
    sep_full  = "=" * len(header)
    sep_inner = "-" * len(header)
    lines = [
        f"PRiSM — Extended Test Set Metrics  ({date_range}, {n_days} trading days)",
        sep_full,
        header,
        sep_inner,
    ]

    spy_row = None
    for rec in all_metrics:
        row = df.loc[rec["strategy"]]
        line = (
            f"{rec['strategy']:<16}  "
            f"{row['sharpe']:>7.3f}  "
            f"{row['annual_return']:>8.2%}  "
            f"{row['vol']:>7.2%}  "
            f"{row['max_dd']:>8.2%}  "
            f"{row['cvar_95']:>8.4f}  "
            f"{row['sortino']:>8.3f}"
        )
        if rec["strategy"] == "SPY (B&H)":
            spy_row = line
        else:
            lines.append(line)

    lines.append(sep_inner)
    if spy_row:
        lines.append(spy_row)
    lines.append(sep_full)
    lines.append("")
    lines.append("CVaR95  : mean of worst 5% daily returns (raw, not annualised).")
    lines.append("Sortino : annualised return / annualised downside deviation (negative days only).")
    lines.append("Sharpe  : annual_return / ann_vol (no explicit risk-free deduction).")
    lines.append("")
    lines.append("Note: 2025-01-01 – present is entirely out-of-sample for all model components.")
    txt = "\n".join(lines)

    (EXT_TEST_DIR / "test_metrics_ext.txt").write_text(txt)
    print(f"  Saved → {EXT_TEST_DIR / 'test_metrics_ext.txt'}")
    print()
    print(txt)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("PRiSM — Extended Test Evaluation: 3-Strategy Comparison")
    print("=" * 60)
    print(f"  Thresholds : tc={TC_OPT}, ts={TS_OPT}")
    print(f"  XGBoost    : SMOTE-balanced model")

    print("\n[1/4] Loading extended data...")
    feat = load_extended_data()
    skew_fn = load_skew_fn()

    feat_test = feat[
        (feat.index >= TEST_START) & (feat.index < TEST_END)
    ]
    print(f"  Test rows: {len(feat_test)} "
          f"({feat_test.index[0].date()} – {feat_test.index[-1].date()})")

    spy_test = feat_test["close"].copy()

    # ── Collect all NAV series and metrics ──
    all_navs    = {}
    all_metrics = []

    # ── A. Short put spread strategies ──
    print("\n[2/4] Running spread backtests...")
    for strat in SPREAD_STRATEGIES:
        name = strat["name"]
        print(f"  [{name}]  tc={strat['tc']:.2f}  ts={strat['ts']:.2f}")
        nav_df, base_m = run_backtest(feat_test, strat["tc"], strat["ts"], skew_fn)
        ext_m = compute_extended_metrics(nav_df, base_m)

        all_navs[name] = nav_df["nav"]
        all_metrics.append({"strategy": name, **ext_m})

        print(f"    Sharpe={ext_m['sharpe']:.3f}  "
              f"AnnRet={ext_m['annual_return']:.2%}  "
              f"MaxDD={ext_m['max_dd']:.2%}")

    # ── B. Market timing (long-only) ──
    print("\n[3/4] Running market timing backtest...")
    timing_signal = (feat_test["p_calm"] > TC_OPT) & (feat_test["p_safe"] > TS_OPT)
    timing_nav, timing_m = run_timing_strategy(spy_test, timing_signal)

    all_navs["Timing: HMM+XGBoost"] = timing_nav
    all_metrics.append({"strategy": "Timing: HMM+XGBoost", **timing_m})
    print(f"  [Timing: HMM+XGBoost]  tc={TC_OPT}  ts={TS_OPT}")
    print(f"    Sharpe={timing_m['sharpe']:.3f}  "
          f"AnnRet={timing_m['annual_return']:.2%}  "
          f"MaxDD={timing_m['max_dd']:.2%}  "
          f"Exposure={timing_m['exposure']:.1%}  "
          f"Trades={timing_m['n_trades']}")

    # ── C. SPY buy-and-hold ──
    spy_m = compute_spy_metrics(spy_test)
    all_navs["SPY (Buy & Hold)"] = spy_test
    all_metrics.append({"strategy": "SPY (Buy & Hold)", **spy_m})
    print(f"  [SPY (Buy & Hold)]")
    print(f"    Sharpe={spy_m['sharpe']:.3f}  "
          f"AnnRet={spy_m['annual_return']:.2%}  "
          f"MaxDD={spy_m['max_dd']:.2%}")

    # ── D. Save and plot ──
    print("\n[4/4] Plotting and saving results...")
    date_range = (f"{feat_test.index[0].strftime('%Y-%m-%d')} – "
                  f"{feat_test.index[-1].strftime('%Y-%m-%d')}")
    test_end_str = feat_test.index[-1].strftime("%Y-%m-%d")
    plot_nav_comparison(spy_test, all_navs, all_metrics, test_end_str)
    save_metrics(all_metrics, len(feat_test), date_range)

    print("\n" + "=" * 60)
    print("Done. All outputs in:", EXT_TEST_DIR)
    print("=" * 60)
