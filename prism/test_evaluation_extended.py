"""
test_evaluation_extended.py
============================
Extended test-set evaluation (2022 – 2026-04-01) for PRISM.

Uses the SAME trained models as the locked test evaluation:
  - HMM : same 2-state GaussianHMM trained on 2015-2020 (refit in
          data_collection_extended.py, converges to identical parameters)
  - XGBoost : loaded directly from data/XGBoost/xgb_model.json (no retraining)
  - Thresholds : tc=0.71, ts=0.95 (locked on validation set, unchanged)

This is a pure out-of-sample extension experiment. The 2025-2026 period
has never been seen by any model component.

Run data_collection_extended.py first to populate data/extended/.

Outputs (results/test_evaluation_extended/):
  test_nav_comparison_ext.png  -- 4-curve normalised NAV (SPY + 3 strategies)
  test_metrics_ext.csv         -- Sharpe, MaxDD, CVaR95, Sortino for all 4 rows
  test_metrics_ext.txt         -- Human-readable summary table
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
TS_OPT = 0.95

TEST_START = "2022-01-01"
TEST_END   = "2026-04-01"

STRATEGIES = [
    {"name": "Baseline",    "tc": 0.00,   "ts": 0.00,   "color": "#888888", "ls": "--"},
    {"name": "HMM-only",    "tc": TC_OPT, "ts": 0.00,   "color": "#F5A623", "ls": "-."},
    {"name": "HMM+XGBoost", "tc": TC_OPT, "ts": TS_OPT, "color": "#2A9D8F", "ls": "-"},
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

def plot_nav_comparison(
    spy_series: pd.Series,
    nav_results: dict[str, pd.DataFrame],
    test_end: str,
) -> None:
    """
    4-curve normalised NAV: SPY + 3 strategies.
    Vertical dashed line marks the original test-set end (2024-12-31)
    to visually separate the locked test period from the live extension.
    """
    fig, ax = plt.subplots(figsize=(13, 6))

    spy_norm = spy_series / spy_series.iloc[0] * 100
    ax.plot(spy_norm.index, spy_norm.values,
            color="#264653", linewidth=1.8, linestyle=":",
            label="SPY (buy-and-hold)", zorder=3)

    for strat in STRATEGIES:
        name   = strat["name"]
        nav_df = nav_results[name]
        norm   = nav_df["nav"] / INITIAL_CASH * 100
        ax.plot(norm.index, norm.values,
                color=strat["color"], linewidth=1.8,
                linestyle=strat["ls"], label=name, zorder=4)

    ax.axhline(100, color="black", linewidth=0.6, linestyle="--", alpha=0.4)

    # Mark original test boundary (2024-12-31)
    orig_end = pd.Timestamp("2024-12-31")
    ax.axvline(orig_end, color="navy", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.text(orig_end, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 200,
            "  Original\n  test end\n  (2024-12-31)",
            fontsize=8, color="navy", va="top", ha="left")

    ax.set_title(
        f"PRiSM — Extended Test Set Portfolio Performance (2022 – {test_end})",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Normalised NAV  (start = 100)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    out = EXT_TEST_DIR / "test_nav_comparison_ext.png"
    plt.savefig(out, dpi=150)
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
    print("PRiSM — Extended Test Set Evaluation (2022 – 2026-04-01)")
    print("=" * 60)
    print("  Models used:")
    print("    HMM      : refit on 2015-2020 (same window, data/extended/)")
    print("    XGBoost  : data/XGBoost/xgb_model.json  (NO retraining)")
    print("    Thresholds: tc=0.71, ts=0.95  (locked on validation set)")

    print("\n[1/3] Loading extended data...")
    feat = load_extended_data()
    skew_fn = load_skew_fn()

    feat_test = feat[
        (feat.index >= TEST_START) & (feat.index < TEST_END)
    ]
    print(f"  Extended test rows: {len(feat_test)} "
          f"({feat_test.index[0].date()} – {feat_test.index[-1].date()})")

    spy_test = feat_test["close"].copy()

    print("\n[2/3] Running backtests...")
    nav_results = {}
    all_metrics = []

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

    # SPY benchmark
    spy_m = compute_spy_metrics(spy_test)
    all_metrics.append({"strategy": "SPY (B&H)", **spy_m})
    print(f"  [SPY (B&H)]")
    print(f"    Sharpe={spy_m['sharpe']:.3f}  "
          f"AnnRet={spy_m['annual_return']:.2%}  "
          f"MaxDD={spy_m['max_dd']:.2%}  "
          f"CVaR95={spy_m['cvar_95']:.4f}  "
          f"Sortino={spy_m['sortino']:.3f}")

    print("\n[3/3] Plotting and saving results...")
    date_range = (f"{feat_test.index[0].strftime('%Y-%m-%d')} – "
                  f"{feat_test.index[-1].strftime('%Y-%m-%d')}")
    plot_nav_comparison(spy_test, nav_results, feat_test.index[-1].strftime("%Y-%m-%d"))
    save_metrics(all_metrics, len(feat_test), date_range)

    print("\n" + "=" * 60)
    print("Done. All outputs in:", EXT_TEST_DIR)
    print("=" * 60)
