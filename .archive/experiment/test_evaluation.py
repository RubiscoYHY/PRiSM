"""
test_evaluation.py
==================
Test-set evaluation (2022–2024) for PRISM.

Three strategy variants run exactly once with FIXED, pre-registered
parameters (thresholds were locked on the validation set — test set
is untouched until this script).

  Baseline     : tc=0.00, ts=0.00  (always open, no ML filter)
  HMM-only     : tc=0.71, ts=0.00  (HMM regime filter only)
  HMM+XGBoost  : tc=0.71, ts=0.95  (full dual-layer filter, optimal pair)

SPY buy-and-hold is included as a passive benchmark (no leverage).

Outputs (results/test_evaluation/):
  test_nav_comparison.png  -- 4-curve normalised NAV (SPY + 3 strategies)
  test_metrics.csv         -- Sharpe, MaxDD, CVaR95, Sortino for all 4 rows
  test_metrics.txt         -- Human-readable summary table
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from experiment.paths import DATA_DIR, RESULTS_DIR
from experiment.threshold_grid import (
    load_backtest_data,
    load_skew_fn,
    attach_p_safe,
    run_backtest,
    INITIAL_CASH,
)

TEST_DIR = RESULTS_DIR / "test_evaluation"
TEST_DIR.mkdir(exist_ok=True)

TC_OPT = 0.71
TS_OPT = 0.95

STRATEGIES = [
    {"name": "Baseline",    "tc": 0.00, "ts": 0.00, "color": "#888888", "ls": "--"},
    {"name": "HMM-only",    "tc": TC_OPT, "ts": 0.00, "color": "#F5A623", "ls": "-."},
    {"name": "HMM+XGBoost", "tc": TC_OPT, "ts": TS_OPT, "color": "#2A9D8F", "ls": "-"},
]


# ─────────────────────────────────────────────────────────────
# SECTION 1: Extended metrics
# ─────────────────────────────────────────────────────────────

def compute_extended_metrics(nav_df: pd.DataFrame, base_metrics: dict) -> dict:
    """
    Append CVaR(95%) and Sortino ratio to the dict returned by run_backtest.

    CVaR(95%): mean of the worst 5% of daily returns (Expected Shortfall).
    Sortino  : annualised excess return / annualised downside deviation.
               Downside deviation uses only days where daily_return < 0.
    """
    r = nav_df["daily_return"]

    # CVaR 95%
    var_95  = r.quantile(0.05)
    cvar_95 = r[r <= var_95].mean()

    # Sortino
    neg_r         = r[r < 0]
    downside_vol  = np.sqrt((neg_r ** 2).mean()) * np.sqrt(252) if len(neg_r) > 0 else 1e-8
    annual_ret    = base_metrics["annual_return"]
    sortino       = annual_ret / downside_vol if downside_vol > 1e-8 else 0.0

    return {
        **base_metrics,
        "cvar_95": round(float(cvar_95), 4),
        "sortino": round(float(sortino), 4),
    }


# ─────────────────────────────────────────────────────────────
# SECTION 2: SPY buy-and-hold metrics
# ─────────────────────────────────────────────────────────────

def compute_spy_metrics(spy_series: pd.Series) -> dict:
    """
    Compute the same metric set for SPY buy-and-hold over the test period.
    Sharpe is computed identically to run_backtest (annual_return / ann_vol,
    no explicit risk-free deduction) for a like-for-like comparison.
    """
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
# SECTION 3: Comparison plot
# ─────────────────────────────────────────────────────────────

def plot_nav_comparison(
    spy_series: pd.Series,
    nav_results: dict[str, pd.DataFrame],
) -> None:
    """
    Plot four normalised NAV curves on the same axes:
      SPY index, Baseline, HMM-only, HMM+XGBoost.
    All series are normalised to 100 at the first test date.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # SPY (buy-and-hold)
    spy_norm = spy_series / spy_series.iloc[0] * 100
    ax.plot(spy_norm.index, spy_norm.values,
            color="#264653", linewidth=1.8, linestyle=":",
            label="SPY (buy-and-hold)", zorder=3)

    # Option strategy curves
    for strat in STRATEGIES:
        name  = strat["name"]
        nav_df = nav_results[name]
        norm  = nav_df["nav"] / INITIAL_CASH * 100
        ax.plot(norm.index, norm.values,
                color=strat["color"], linewidth=1.8,
                linestyle=strat["ls"], label=name, zorder=4)

    # Equity timing curve (if present)
    if "Equity Timing" in nav_results:
        eq_norm = nav_results["Equity Timing"]["nav"] / INITIAL_CASH * 100
        ax.plot(eq_norm.index, eq_norm.values,
                color="#9B59B6", linewidth=1.8,
                linestyle="-", label="Equity Timing", zorder=4)

    # Reference line at 100
    ax.axhline(100, color="black", linewidth=0.6, linestyle="--", alpha=0.4)

    ax.set_title("PRiSM — Test Set Portfolio Performance (2022–2024)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Normalised NAV  (start = 100)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    out = TEST_DIR / "test_nav_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Metrics table
# ─────────────────────────────────────────────────────────────

def save_metrics(all_metrics: list[dict]) -> None:
    df = pd.DataFrame(all_metrics).set_index("strategy")
    df.to_csv(TEST_DIR / "test_metrics.csv")
    print(f"  Saved → {TEST_DIR / 'test_metrics.csv'}")

    header = (
        f"{'Strategy':<16}  {'Sharpe':>7}  {'AnnRet':>8}  "
        f"{'Vol':>7}  {'MaxDD':>8}  {'CVaR95':>8}  {'Sortino':>8}"
    )
    sep_full  = "=" * len(header)
    sep_inner = "-" * len(header)
    lines = [
        "PRiSM — Test Set Metrics (2022–2024)",
        sep_full,
        header,
        sep_inner,
    ]

    spy_row  = None
    strat_rows = []
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
            strat_rows.append(line)

    lines.extend(strat_rows)
    lines.append(sep_inner)
    if spy_row:
        lines.append(spy_row)          # SPY benchmark after divider
    lines.append(sep_full)
    lines.append("")
    lines.append("CVaR95  : mean of worst 5% daily returns (raw, not annualised).")
    lines.append("Sortino : annualised return / annualised downside deviation (negative days only).")
    lines.append("Sharpe  : annual_return / ann_vol (no explicit risk-free deduction, consistent across all rows).")
    txt = "\n".join(lines)

    (TEST_DIR / "test_metrics.txt").write_text(txt)
    print(f"  Saved → {TEST_DIR / 'test_metrics.txt'}")
    print()
    print(txt)


# ─────────────────────────────────────────────────────────────
# SECTION 5: Equity timing strategy (buy SPY when calm & safe)
# ─────────────────────────────────────────────────────────────

def run_equity_timing(
    data: pd.DataFrame,
    threshold_calm: float,
    threshold_safe: float,
    initial_cash: float = INITIAL_CASH,
) -> tuple[pd.DataFrame, dict]:
    """
    Simple equity timing: hold SPY when calm & safe, otherwise hold cash.
    """
    cash      = initial_cash
    shares    = 0.0
    invested  = False
    nav_records = []

    for date, row in data.iterrows():
        S      = float(row["close"])
        p_calm = float(row["p_calm"])
        p_safe = float(row["p_safe"])

        if np.isnan(S) or S <= 0:
            continue

        should_hold = (p_calm > threshold_calm) and (p_safe > threshold_safe)

        if should_hold and not invested:
            shares = cash / S
            cash = 0.0
            invested = True
        elif not should_hold and invested:
            cash = shares * S
            shares = 0.0
            invested = False

        nav = cash + shares * S
        nav_records.append({"date": date, "nav": nav})

    nav_df = pd.DataFrame(nav_records).set_index("date")
    nav_df["daily_return"] = nav_df["nav"].pct_change().fillna(0.0)

    n_days     = len(nav_df)
    total_ret  = nav_df["nav"].iloc[-1] / initial_cash - 1
    annual_ret = (1 + total_ret) ** (252 / max(n_days, 1)) - 1
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
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("PRiSM — Test Set Evaluation (2022–2024, 200 trials)")
    print("=" * 60)

    print("\n[1/4] Loading data...")
    feat    = load_backtest_data()
    skew_fn = load_skew_fn()
    feat    = attach_p_safe(feat)

    feat_test = feat[(feat.index >= "2022-01-01") & (feat.index < "2025-01-01")]
    print(f"  Test rows : {len(feat_test)} "
          f"({feat_test.index[0].date()} – {feat_test.index[-1].date()})")

    spy_test = feat_test["close"].copy()

    print("\n[2/4] Running option strategy backtests...")
    nav_results  = {}
    all_metrics  = []

    for strat in STRATEGIES:
        name = strat["name"]
        print(f"  [{name}]  tc={strat['tc']:.2f}  ts={strat['ts']:.2f}")
        nav_df, base_m = run_backtest(feat_test, strat["tc"], strat["ts"], skew_fn)
        ext_m = compute_extended_metrics(nav_df, base_m)

        nav_results[name] = nav_df
        all_metrics.append({"strategy": name, **ext_m})

        print(f"    Sharpe={ext_m['sharpe']:.3f}  "
              f"AnnRet={ext_m['annual_return']:.2%}  "
              f"MaxDD={ext_m['max_dd']:.2%}  "
              f"CVaR95={ext_m['cvar_95']:.4f}  "
              f"Sortino={ext_m['sortino']:.3f}")

    print("\n[3/4] Running equity timing strategy...")
    # Equity timing: buy SPY when calm & safe, sell to cash otherwise
    eq_nav, eq_base = run_equity_timing(feat_test, TC_OPT, TS_OPT)
    eq_m = compute_extended_metrics(eq_nav, eq_base)
    nav_results["Equity Timing"] = eq_nav
    all_metrics.append({"strategy": "Equity Timing", **eq_m})
    print(f"  [Equity Timing]  tc={TC_OPT:.2f}  ts={TS_OPT:.2f}")
    print(f"    Sharpe={eq_m['sharpe']:.3f}  "
          f"AnnRet={eq_m['annual_return']:.2%}  "
          f"MaxDD={eq_m['max_dd']:.2%}  "
          f"CVaR95={eq_m['cvar_95']:.4f}  "
          f"Sortino={eq_m['sortino']:.3f}")

    # SPY buy-and-hold benchmark
    spy_m = compute_spy_metrics(spy_test)
    all_metrics.append({"strategy": "SPY (B&H)", **spy_m})
    print(f"  [SPY (B&H)]")
    print(f"    Sharpe={spy_m['sharpe']:.3f}  "
          f"AnnRet={spy_m['annual_return']:.2%}  "
          f"MaxDD={spy_m['max_dd']:.2%}  "
          f"CVaR95={spy_m['cvar_95']:.4f}  "
          f"Sortino={spy_m['sortino']:.3f}")

    print("\n[4/4] Plotting and saving results...")
    plot_nav_comparison(spy_test, nav_results)
    save_metrics(all_metrics)

    print("\n" + "=" * 60)
    print("Done. All outputs in:", TEST_DIR)
    print("=" * 60)
