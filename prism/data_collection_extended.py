"""
data_collection_extended.py
============================
Extended data collection experiment — SPY + VIX through 2026-04-01.

All outputs go to data/extended/ (completely separate from the locked
original data/ directory — no original files are overwritten).

Pipeline:
  1. Download SPY OHLCV + VIX + VIX9D  (2015-01-01 – 2026-04-01)
  2. Refit 2-state Gaussian HMM on the SAME training window (2015-2020)
     and apply to the extended dataset → new p_calm series through 2026.
     The XGBoost model (data/XGBoost/xgb_model.json) is NOT retrained.

Outputs (data/extended/):
  spy_vix_daily.csv      — SPY OHLCV + VIX + log_return, 2015 – 2026-04
  hmm_pcalm_daily.csv    — daily P(Calm) on extended dataset
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from prism.paths import DATA_DIR
from prism.hmm_regime import train_hmm, apply_model

EXT_DIR = DATA_DIR / "extended"
EXT_DIR.mkdir(exist_ok=True)

DATA_START = "2015-01-01"
DATA_END   = "2026-04-26"


# ─────────────────────────────────────────────────────────────
# SECTION 1: Download extended price history
# ─────────────────────────────────────────────────────────────

def download_extended(start: str = DATA_START, end: str = DATA_END) -> pd.DataFrame:
    """
    Download SPY + VIX + VIX9D to the extended end date.
    Saves to data/extended/spy_vix_daily.csv (never touches data/).
    """
    def _flatten(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(c).strip().lower() for c in df.columns]
        else:
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        return df

    print(f"Downloading SPY  ({start} – {end})...")
    spy_raw   = _flatten(yf.download("SPY",   start=start, end=end,
                                     auto_adjust=True, progress=False))
    print(f"Downloading ^VIX ({start} – {end})...")
    vix_raw   = _flatten(yf.download("^VIX",  start=start, end=end,
                                     auto_adjust=True, progress=False))
    print(f"Downloading ^VIX9D ({start} – {end})...")
    vix9d_raw = _flatten(yf.download("^VIX9D", start=start, end=end,
                                     auto_adjust=True, progress=False))

    spy_close_col   = next(c for c in spy_raw.columns   if "close" in c)
    vix_close_col   = next(c for c in vix_raw.columns   if "close" in c)
    vix9d_close_col = next(c for c in vix9d_raw.columns if "close" in c)

    df = spy_raw.copy()
    df.columns = [f"spy_{c}" for c in df.columns]
    df["vix_close"]   = vix_raw[vix_close_col]
    df["vix9d_close"] = vix9d_raw[vix9d_close_col]
    df["vix9d_close"] = df["vix9d_close"].fillna(df["vix_close"])

    df = df.dropna(subset=[f"spy_{spy_close_col}", "vix_close"])

    df["log_return"] = np.log(
        df[f"spy_{spy_close_col}"] / df[f"spy_{spy_close_col}"].shift(1)
    )
    df = df.dropna(subset=["log_return"])
    df.index.name = "date"

    out = EXT_DIR / "spy_vix_daily.csv"
    df.to_csv(out)
    print(f"  Saved {len(df)} rows → {out}")
    print(f"  Date range: {df.index[0].date()} – {df.index[-1].date()}")
    return df


# ─────────────────────────────────────────────────────────────
# SECTION 2: Refit HMM and extend p_calm
# ─────────────────────────────────────────────────────────────

def build_extended_pcalm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Refit 2-state HMM on 2015-2020 (identical training window to original).
    Apply to the full extended dataset → p_calm through 2026.

    Because the training data (2015-2020) is unchanged, this model will
    converge to parameters indistinguishable from the original HMM.
    """
    print("\nFitting HMM (train window 2015-2020)...")
    best_model, runs_df = train_hmm(df)
    best_ll = runs_df["log_likelihood"].iloc[0]
    print(f"  Best LL = {best_ll:.4f}  (original was 5044.26 — should match)")

    pcalm_df = apply_model(best_model, df)
    out = EXT_DIR / "hmm_pcalm_daily.csv"
    pcalm_df.to_csv(out)
    print(f"  Saved {len(pcalm_df)} rows → {out}")
    return pcalm_df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("PRISM — Extended Data Collection (2015 – 2026-04-01)")
    print("=" * 60)
    print(f"  Output directory: {EXT_DIR}")
    print(f"  Original data/   : UNTOUCHED")

    print("\n[1/2] Downloading price data...")
    df = download_extended()

    print("\n[2/2] Building extended HMM p_calm series...")
    pcalm_df = build_extended_pcalm(df)

    new_rows = pcalm_df[pcalm_df.index >= "2025-01-01"]
    print(f"\n  New rows beyond original data end: {len(new_rows)}")
    print(f"  ({new_rows.index[0].date() if len(new_rows) > 0 else 'n/a'} "
          f"– {new_rows.index[-1].date() if len(new_rows) > 0 else 'n/a'})")

    print("\n" + "=" * 60)
    print("Done.")
    print(f"  {EXT_DIR / 'spy_vix_daily.csv'}")
    print(f"  {EXT_DIR / 'hmm_pcalm_daily.csv'}")
    print("=" * 60)
