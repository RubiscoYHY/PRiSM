"""
data_collection.py
==================
Fallback data collection for MGT 6081 Final Project.
Uses Yahoo Finance (yfinance) as the primary source when Bloomberg is unavailable.

Pipeline:
  1. Download SPY daily OHLCV + VIX daily close  (2015–2024)
  2. Calibrate per-moneyness skew multipliers from a current yfinance option-chain snapshot
  3. Build a pricing function  get_option_price(date, K, T)  that applies:
       IV(K, T) = VIX(date) × skew_multiplier(K/S) × term_structure_adj(T)
     and returns a Black-Scholes mid price
  4. For each backtest day, compute short-leg and long-leg mid prices and save to CSV
     Output: data/option_prices_fallback.csv
             data/spy_vix_daily.csv
             data/skew_multipliers.csv

Dependencies:
    pip install yfinance pandas numpy scipy
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
from experiment.paths import DATA_DIR

# ─────────────────────────────────────────────
# SECTION 1: Download SPY and VIX daily data
# ─────────────────────────────────────────────

def download_spy_vix(start: str = "2015-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """
    Download SPY OHLCV, VIX close, and VIX9D close from Yahoo Finance.

    Returns
    -------
    pd.DataFrame
        Daily index with columns:
        spy_open, spy_high, spy_low, spy_close, spy_adj_close, spy_volume,
        vix_close, vix9d_close, log_return
    """
    print("Downloading SPY daily data...")
    spy_raw = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)

    print("Downloading VIX daily data...")
    vix_raw = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)

    print("Downloading VIX9D daily data...")
    vix9d_raw = yf.download("^VIX9D", start=start, end=end, auto_adjust=True, progress=False)

    def _flatten(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(c).strip().lower() for c in df.columns]
        else:
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        return df

    spy_raw   = _flatten(spy_raw)
    vix_raw   = _flatten(vix_raw)
    vix9d_raw = _flatten(vix9d_raw)

    spy_close_col  = next(c for c in spy_raw.columns  if "close" in c)
    vix_close_col  = next(c for c in vix_raw.columns  if "close" in c)
    vix9d_close_col = next(c for c in vix9d_raw.columns if "close" in c)

    df = spy_raw.copy()
    df.columns = [f"spy_{c}" for c in df.columns]
    df["vix_close"]   = vix_raw[vix_close_col]
    df["vix9d_close"] = vix9d_raw[vix9d_close_col]

    # VIX9D starts in 2011; before that, fall back to VIX as proxy
    df["vix9d_close"] = df["vix9d_close"].fillna(df["vix_close"])

    df = df.dropna(subset=[f"spy_{spy_close_col}", "vix_close"])

    # Log returns for HMM / XGBoost feature pipeline
    df["log_return"] = np.log(df[f"spy_{spy_close_col}"] / df[f"spy_{spy_close_col}"].shift(1))
    df = df.dropna(subset=["log_return"])

    df.index.name = "date"
    df.to_csv(DATA_DIR / "spy_vix_daily.csv")
    print(f"  Saved {len(df)} rows to {DATA_DIR / 'spy_vix_daily.csv'}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Calibrate skew multipliers from current yfinance option-chain snapshot
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_skew_multipliers(
    moneyness_grid: list = None,
    min_volume: int = 10,
    min_open_interest: int = 50,
) -> interp1d:
    """
    Use the current yfinance SPY option chain (live snapshot) to estimate
    how much OTM put IV exceeds VIX as a function of moneyness (K/S).

    This is the KEY correction for error source #1 (skew).
    The multipliers are assumed stable across time (slow-moving structural parameter).

    Parameters
    ----------
    moneyness_grid : list of floats, optional
        Moneyness nodes at which to evaluate and save multipliers.
        Defaults to [0.88, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 1.00]
    min_volume : int
        Minimum option volume to include in calibration (filters illiquid strikes).
    min_open_interest : int
        Minimum open interest to include.

    Returns
    -------
    scipy.interpolate.interp1d
        Interpolator: moneyness (float) → skew_multiplier (float).
        Extrapolates flat beyond the observed range.
    """
    if moneyness_grid is None:
        moneyness_grid = [0.88, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95,
                          0.96, 0.97, 0.98, 0.99, 1.00, 1.01]

    print("Calibrating skew multipliers from current yfinance snapshot...")

    ticker = yf.Ticker("SPY")
    spot = ticker.info.get("regularMarketPrice") or ticker.info.get("previousClose")
    if spot is None:
        # Fallback: use last close from recent history
        spot = ticker.history(period="1d")["Close"].iloc[-1]
    print(f"  SPY spot: {spot:.2f}")

    # Current VIX
    vix_ticker = yf.Ticker("^VIX")
    vix_now = vix_ticker.history(period="1d")["Close"].iloc[-1] / 100.0
    print(f"  VIX now: {vix_now*100:.2f}")

    # Find expiry closest to 30 DTE
    expiries = ticker.options  # tuple of date strings
    today = datetime.today()
    target_dte = 30

    def dte(exp_str):
        return (datetime.strptime(exp_str, "%Y-%m-%d") - today).days

    expiry = min(expiries, key=lambda e: abs(dte(e) - target_dte))
    actual_dte = dte(expiry)
    print(f"  Using expiry {expiry} ({actual_dte} DTE)")

    chain = ticker.option_chain(expiry)
    puts = chain.puts.copy()

    # Filter for liquid strikes only
    puts = puts[
        (puts["volume"] >= min_volume) &
        (puts["openInterest"] >= min_open_interest) &
        (puts["impliedVolatility"] > 0.01) &
        (puts["impliedVolatility"] < 5.0)      # remove clearly bad IV quotes
    ].copy()

    puts["moneyness"] = puts["strike"] / spot
    puts["iv_ratio"] = puts["impliedVolatility"] / vix_now  # raw skew multiplier per strike

    # Keep only puts in sensible moneyness range
    puts = puts[(puts["moneyness"] >= 0.80) & (puts["moneyness"] <= 1.05)]
    puts = puts.sort_values("moneyness")

    print(f"  {len(puts)} liquid put strikes available for calibration")

    if len(puts) < 4:
        print("  WARNING: Too few liquid strikes — using fallback hardcoded multipliers.")
        # Empirical fallback for SPY (typical calm-market values)
        fallback_m = [0.88, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95,
                      0.96, 0.97, 0.98, 1.00, 1.01]
        fallback_v = [1.55, 1.45, 1.40, 1.35, 1.28, 1.22, 1.15,
                      1.10, 1.06, 1.03, 1.00, 0.98]
        multiplier_fn = interp1d(fallback_m, fallback_v,
                                 kind="linear", fill_value="extrapolate")
        df_mults = pd.DataFrame({"moneyness": fallback_m, "skew_multiplier": fallback_v,
                                 "source": "hardcoded_fallback"})
        df_mults.to_csv(DATA_DIR / "skew_multipliers.csv", index=False)
        return multiplier_fn

    # Smooth the raw iv_ratio by fitting a rolling median
    puts["iv_ratio_smooth"] = (
        puts.set_index("moneyness")["iv_ratio"]
        .rolling(window=3, center=True, min_periods=1)
        .median()
        .values
    )

    # Evaluate at our target moneyness grid by interpolation
    interp_raw = interp1d(
        puts["moneyness"].values,
        puts["iv_ratio_smooth"].values,
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )
    grid_values = np.clip(interp_raw(moneyness_grid), 0.80, 3.0)

    # Force ATM (moneyness≈1.0) multiplier = 1.0 exactly (VIX IS ATM vol by definition)
    atm_idx = np.argmin(np.abs(np.array(moneyness_grid) - 1.0))
    atm_observed = grid_values[atm_idx]
    grid_values = grid_values / atm_observed   # normalize so ATM = 1.0

    multiplier_fn = interp1d(
        moneyness_grid, grid_values,
        kind="linear", fill_value="extrapolate", bounds_error=False
    )

    df_mults = pd.DataFrame({
        "moneyness": moneyness_grid,
        "skew_multiplier": grid_values,
        "source": f"yfinance_snapshot_{expiry}",
    })
    df_mults.to_csv(DATA_DIR / "skew_multipliers.csv", index=False)
    print(f"  Saved skew multipliers to {DATA_DIR / 'skew_multipliers.csv'}")
    print(df_mults[["moneyness", "skew_multiplier"]].to_string(index=False))

    return multiplier_fn


# ──────────────────────────────────────────────────────────────
# SECTION 3: Black-Scholes pricing with IV(K,T) correction
# ──────────────────────────────────────────────────────────────

def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Standard Black-Scholes European put price.

    Parameters
    ----------
    S : float  Current underlying price
    K : float  Strike price
    T : float  Time to expiry in years
    r : float  Risk-free rate (annualised, e.g. 0.05 for 5%)
    sigma : float  Implied volatility (annualised, e.g. 0.20 for 20%)

    Returns
    -------
    float  Put price
    """
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def get_option_price(
    S: float,
    K: float,
    T_days: float,
    vix: float,
    skew_fn: interp1d,
    r: float = 0.04,
    vix9d: float = None,
) -> float:
    """
    Core pricing function used by the backtest engine.

    IV(K, T) = IV_atm(T) × skew_multiplier(K/S)

    Term structure (ATM IV at horizon T):
        Uses linear variance interpolation between VIX9D (9-day) and VIX (30-day).
        Total variance is proportional to time, so interpolation is done in
        variance space (not vol space) to avoid Jensen's inequality bias.

        TotalVar(9)  = (VIX9D/100)² × 9
        TotalVar(30) = (VIX/100)²   × 30

        For any T:
            TotalVar(T) = TotalVar(9) + [TotalVar(30) - TotalVar(9)]
                          × (T - 9) / (30 - 9)
            IV_atm(T)   = sqrt(TotalVar(T) / T)

        When VIX9D is unavailable, falls back to VIX as the single anchor
        (equivalent to flat variance term structure assumption).

    Parameters
    ----------
    S       : float   SPY spot price
    K       : float   Option strike
    T_days  : float   Days to expiry
    vix     : float   VIX index value (e.g. 18.5 for 18.5%)
    skew_fn : interp1d  Calibrated skew multiplier function
    r       : float   Risk-free rate (annualised), default 4%
    vix9d   : float   VIX9D index value; if None, falls back to flat term structure

    Returns
    -------
    float  Black-Scholes put mid price
    """
    T_years   = T_days / 365.0
    moneyness = K / S

    # --- Term structure: variance interpolation ---
    if T_days <= 0:
        # Expired or same day: intrinsic only
        return max(K - S, 0.0)

    total_var_30 = (vix / 100.0) ** 2 * 30.0

    if vix9d is not None and vix9d > 0:
        total_var_9 = (vix9d / 100.0) ** 2 * 9.0
        # Linear interpolation / extrapolation in variance space
        slope = (total_var_30 - total_var_9) / (30.0 - 9.0)
        total_var_T = total_var_9 + slope * (T_days - 9.0)
    else:
        # Flat variance term structure fallback (old behaviour)
        total_var_T = total_var_30 * (T_days / 30.0)

    # Guard against negative variance from aggressive extrapolation
    total_var_T = max(total_var_T, 1e-8)
    iv_atm = np.sqrt(total_var_T / T_days)      # annualised ATM vol at horizon T

    # --- Skew correction ---
    skew_mult = float(np.clip(skew_fn(moneyness), 0.80, 3.0))

    sigma = iv_atm * skew_mult

    return black_scholes_put(S, K, T_years, r, sigma)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4: Build the full historical option price dataset for the backtest
# ──────────────────────────────────────────────────────────────────────────────

def next_nth_friday(date: pd.Timestamp, n: int = 4) -> pd.Timestamp:
    """
    Return the Nth Friday strictly after `date`, using pure timedelta arithmetic.

    How it works:
      1. Find the next Friday on or after (date + 1 day).
      2. Jump forward (n-1) more weeks of 7 days each.

    This is the correct primitive for the backtest engine:
      - Open-day expiry selection  → next_nth_friday(today, n=4)  ≈ 28-35 DTE
      - Close-check DTE countdown  → (expiry - today).days <= 5   → close
      - Any "N weeks out" query    → next_nth_friday(today, n=N)

    Parameters
    ----------
    date : pd.Timestamp   Reference date (open date / today)
    n    : int            Which Friday to target (1=next, 2=2nd, 4=4th, etc.)

    Returns
    -------
    pd.Timestamp  The target Friday. Always a valid date, no month-boundary edge cases.

    Examples
    --------
    >>> next_nth_friday(pd.Timestamp("2024-02-01"), n=4)   # short month, no issue
    Timestamp('2024-03-01 00:00:00')
    >>> next_nth_friday(pd.Timestamp("2024-12-30"), n=4)   # year boundary
    Timestamp('2025-01-24 00:00:00')
    """
    days_ahead = (4 - date.day_of_week) % 7   # Friday = weekday 4
    if days_ahead == 0:
        days_ahead = 7                          # already Friday → go to NEXT one
    first_friday = date + pd.Timedelta(days=days_ahead)
    return first_friday + pd.Timedelta(weeks=n - 1)


def build_option_price_history(
    spy_vix_df: pd.DataFrame,
    skew_fn: interp1d,
    short_leg_moneyness: float = 0.95,
    long_leg_moneyness: float = 0.91,
    r: float = 0.04,
) -> pd.DataFrame:
    """
    For each trading day, compute:
      - short leg price  (K1 = 0.95 × S)
      - long leg price   (K2 = 0.91 × S)
      - spread mid price (short_leg - long_leg)
      - spread max loss  (K1 - K2 - spread_mid) per share, × 100 per contract
      - implied IV used for each leg

    Parameters
    ----------
    spy_vix_df          : output of download_spy_vix()
    skew_fn             : output of calibrate_skew_multipliers()
    short_leg_moneyness : float, default 0.95
    long_leg_moneyness  : float, default 0.91
    r                   : float, annualised risk-free rate

    Returns
    -------
    pd.DataFrame saved to data/option_prices_fallback.csv
    """
    print("\nBuilding historical option price series...")

    # Identify close column
    close_col = next(c for c in spy_vix_df.columns if "spy" in c and "close" in c)

    records = []

    for date, row in spy_vix_df.iterrows():
        S    = float(row[close_col])
        vix  = float(row["vix_close"])
        vix9d = float(row["vix9d_close"]) if "vix9d_close" in row.index else None

        if np.isnan(S) or np.isnan(vix) or S <= 0 or vix <= 0:
            continue
        if vix9d is not None and (np.isnan(vix9d) or vix9d <= 0):
            vix9d = None

        # Strikes (moneyness-based, model-free — avoids circular IV dependency)
        K1 = round(S * short_leg_moneyness, 2)  # short leg
        K2 = round(S * long_leg_moneyness, 2)   # long leg

        # Expiry ~ next monthly 4th Friday
        expiry = next_nth_friday(pd.Timestamp(date), n=4)
        T_days = (expiry - pd.Timestamp(date)).days

        # Option prices
        price_short = get_option_price(S, K1, T_days, vix, skew_fn, r, vix9d)
        price_long  = get_option_price(S, K2, T_days, vix, skew_fn, r, vix9d)

        # Spread economics
        spread_mid      = price_short - price_long          # net credit received
        spread_max_loss = (K1 - K2) - spread_mid            # per share
        spread_max_loss_contract = spread_max_loss * 100    # per contract (100 shares)

        # IV used (for diagnostics) — mirrors get_option_price variance interpolation
        total_var_30 = (vix / 100.0) ** 2 * 30.0
        if vix9d is not None:
            total_var_9 = (vix9d / 100.0) ** 2 * 9.0
            slope = (total_var_30 - total_var_9) / (30.0 - 9.0)
            total_var_T = max(total_var_9 + slope * (T_days - 9.0), 1e-8)
        else:
            total_var_T = max(total_var_30 * (T_days / 30.0), 1e-8)
        iv_atm   = np.sqrt(total_var_T / T_days)
        iv_short = iv_atm * float(skew_fn(K1 / S))
        iv_long  = iv_atm * float(skew_fn(K2 / S))

        records.append({
            "date":                    date,
            "spy_close":               round(S, 2),
            "vix_close":               round(vix, 2),
            "vix9d_close":             round(vix9d, 2) if vix9d is not None else np.nan,
            "K1_short":                K1,
            "K2_long":                 K2,
            "T_days":                  T_days,
            "expiry":                  expiry.strftime("%Y-%m-%d"),
            "price_short_leg":         round(price_short, 4),
            "price_long_leg":          round(price_long, 4),
            "spread_mid":              round(spread_mid, 4),
            "spread_max_loss":         round(spread_max_loss, 4),
            "spread_max_loss_contract":round(spread_max_loss_contract, 2),
            "iv_short_used":           round(iv_short, 4),
            "iv_long_used":            round(iv_long, 4),
        })

    df_out = pd.DataFrame(records).set_index("date")
    df_out.to_csv(DATA_DIR / "option_prices_fallback.csv")
    print(f"  Saved {len(df_out)} rows to {DATA_DIR / 'option_prices_fallback.csv'}")
    print("\nSample output (last 5 rows):")
    print(df_out.tail(5)[["spy_close","vix_close","K1_short","K2_long",
                           "spread_mid","spread_max_loss_contract","iv_short_used"]].to_string())
    return df_out


# ──────────────────────────────────────────────────────────────────────────
# SECTION 5: Diagnostic — quantify skew correction impact
# ──────────────────────────────────────────────────────────────────────────

def skew_correction_diagnostic(spy_vix_df: pd.DataFrame, skew_fn: interp1d) -> None:
    """
    Print a table comparing naive (no skew) vs corrected spread prices
    for a representative sample of VIX regimes.
    Answers: 'how much does the skew correction matter?'
    """
    print("\n=== Skew Correction Diagnostic ===")
    close_col = next(c for c in spy_vix_df.columns if "spy" in c and "close" in c)

    # No-skew function (flat multiplier = 1.0 everywhere)
    flat_fn = interp1d([0.0, 2.0], [1.0, 1.0], fill_value=1.0, bounds_error=False)

    sample = spy_vix_df.dropna().sample(n=min(300, len(spy_vix_df)), random_state=42)

    results = []
    for date, row in sample.iterrows():
        S     = float(row[close_col])
        vix   = float(row["vix_close"])
        vix9d = float(row["vix9d_close"]) if "vix9d_close" in row.index else None
        if vix9d is not None and (np.isnan(vix9d) or vix9d <= 0):
            vix9d = None
        K1  = S * 0.95
        K2  = S * 0.91
        T   = 30

        spread_naive    = (get_option_price(S, K1, T, vix, flat_fn, vix9d=vix9d) -
                           get_option_price(S, K2, T, vix, flat_fn, vix9d=vix9d))
        spread_corrected = (get_option_price(S, K1, T, vix, skew_fn, vix9d=vix9d) -
                            get_option_price(S, K2, T, vix, skew_fn, vix9d=vix9d))
        results.append({
            "vix_regime": pd.cut([vix], bins=[0,15,20,30,100],
                                  labels=["low(<15)","mid(15-20)","high(20-30)","spike(>30)"])[0],
            "spread_naive":     spread_naive,
            "spread_corrected": spread_corrected,
            "pct_diff":         (spread_corrected - spread_naive) / max(spread_naive, 1e-6) * 100,
        })

    diag_df = (pd.DataFrame(results)
               .groupby("vix_regime")[["spread_naive","spread_corrected","pct_diff"]]
               .mean()
               .round(4))
    print(diag_df.to_string())
    print("\n  pct_diff = how much larger corrected spread is vs naive (positive = naive underestimates)")
    diag_df.to_csv(DATA_DIR / "skew_diagnostic.csv")
    print(f"  Saved to {DATA_DIR / 'skew_diagnostic.csv'}")


# ──────────────────────────────────────────────────────────────────
# MAIN: Run the full pipeline
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("MGT 6081 — Fallback Data Collection Pipeline")
    print("=" * 60)

    # Step 1: Download price history
    spy_vix = download_spy_vix(start="2015-01-01", end="2024-12-31")

    # Step 2: Calibrate skew multipliers from live snapshot
    skew_fn = calibrate_skew_multipliers()

    # Step 3: Build full option price history
    option_prices = build_option_price_history(spy_vix, skew_fn)

    # Step 4: Show how much the skew correction matters
    skew_correction_diagnostic(spy_vix, skew_fn)

    print("\n=== Pipeline complete ===")
    print(f"Output directory: {DATA_DIR}")
    print("Output files:")
    print("  spy_vix_daily.csv           — SPY OHLCV + VIX + log_returns")
    print("  skew_multipliers.csv        — per-moneyness IV multipliers")
    print("  option_prices_fallback.csv  — daily spread prices (K1=0.95S, K2=0.91S)")
    print("  skew_diagnostic.csv         — correction impact by VIX regime")
