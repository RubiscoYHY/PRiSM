"""
live_calculator.py
==================
PRISM Live Trading Calculator — Flask backend.

Answers the question: "Am I allowed to open a new short put spread today?"

Signal pipeline (mirrors backtest exactly):
  1. Load historical SPY/VIX data + extend with fresh yfinance data to today
  2. Reconstruct trained HMM from saved parameters → compute p_calm for full sequence
  3. Build the 9 XGBoost features from the extended series
  4. Load trained XGBoost model → compute p_safe for today
  5. Compare against optimal thresholds (tc=0.71, ts=0.95)
  6. Fetch live SPY / VIX prices → compute spread parameters at current market prices

Usage:
    pip install flask
    python live_calculator.py
    Then open http://localhost:5719 in your browser.

Data sources (no broker API needed):
  - Historical: data/spy_vix_daily.csv (already on disk)
  - Live:       yfinance (~15-min delayed during market hours, last close otherwise)
"""

import warnings
warnings.filterwarnings("ignore")

import re
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from datetime import datetime, timezone
from pathlib import Path
from hmmlearn.hmm import GaussianHMM
from scipy.interpolate import interp1d
from flask import Flask, jsonify, render_template, request
import prism.position_manager as pm

# ── Project paths ──────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data"
HMM_DIR  = DATA_DIR / "HMM"
XGB_DIR  = DATA_DIR / "XGBoost"
GRID_DIR = DATA_DIR / "threshold_grid"

# ── Strategy constants (mirrors backtest) ──────────────────────────────────────
R             = 0.04      # risk-free rate
CAR_FRACTION  = 0.20      # capital at risk per position
MAX_POSITIONS = 4         # max concurrent positions
SHORT_LEG_M   = 0.95      # K1 = S * 0.95
LONG_LEG_M    = 0.91      # K2 = S * 0.91
N_WEEKS_OUT   = 4         # target expiry: 4th Friday from today

FEATURE_COLS = [
    "RV_5d", "RV_20d", "RV_60d", "RV_ratio",
    "Mom_5d", "Mom_20d", "DD_60d", "RSkew_20d", "p_calm"
]

app = Flask(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Load / reconstruct trained models
# ══════════════════════════════════════════════════════════════════════════════

def reconstruct_hmm() -> GaussianHMM:
    """
    Rebuild the trained GaussianHMM from the saved parameter CSV.
    The best run is the first row (sorted by log-likelihood descending).
    State 0 = Calm (lower sigma), State 1 = Turbulent.
    """
    runs = pd.read_csv(HMM_DIR / "hmm_all_runs.csv")
    best = runs.iloc[0]

    model = GaussianHMM(n_components=2, covariance_type="diag")
    model.means_   = np.array([[best["calm_mu"]], [best["turb_mu"]]])
    model.covars_  = np.array([[best["calm_sigma"] ** 2], [best["turb_sigma"] ** 2]])
    model.transmat_ = np.array([
        [best["calm_stay_prob"], 1.0 - best["calm_stay_prob"]],
        [1.0 - best["turb_stay_prob"], best["turb_stay_prob"]],
    ])
    pi = np.array([best["calm_pi"], best["turb_pi"]], dtype=float)
    # Guard: ensure valid probability vector
    pi = np.clip(pi, 1e-6, None)
    model.startprob_ = pi / pi.sum()
    return model


def load_xgb_model() -> xgb.XGBClassifier:
    model = xgb.XGBClassifier()
    model.load_model(str(XGB_DIR / "xgb_model.json"))
    return model


def load_thresholds() -> tuple[float, float]:
    """Parse threshold_best.txt → (tc, ts)."""
    text = (GRID_DIR / "threshold_best.txt").read_text()
    tc = float(re.search(r"threshold_calm\s*:\s*([\d.]+)", text).group(1))
    ts = float(re.search(r"threshold_safe\s*:\s*([\d.]+)", text).group(1))
    return tc, ts


def load_skew_fn() -> interp1d:
    mults = pd.read_csv(DATA_DIR / "skew_multipliers.csv")
    return interp1d(
        mults["moneyness"].values,
        mults["skew_multiplier"].values,
        kind="linear", fill_value="extrapolate", bounds_error=False,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Data fetching and extension
# ══════════════════════════════════════════════════════════════════════════════

def _flatten_yf(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns from yfinance download."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(c).strip().lower() for c in df.columns]
    else:
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df


def fetch_extended_history() -> pd.DataFrame:
    """
    Load historical data from disk, then extend to today via yfinance.
    Returns a unified DataFrame with columns:
        close, vix_close, vix9d_close, log_return
    indexed by date (business days only).
    """
    # Load existing history
    hist = pd.read_csv(DATA_DIR / "spy_vix_daily.csv", index_col="date", parse_dates=True)
    close_col = next(c for c in hist.columns if "spy" in c and "close" in c)
    hist = hist.rename(columns={close_col: "close"})
    hist = hist[["close", "vix_close", "vix9d_close", "log_return"]].copy()
    last_hist_date = hist.index.max()

    # Fetch fresh data for any days since the last saved date
    today = pd.Timestamp.today().normalize()
    if last_hist_date < today:
        fetch_start = (last_hist_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        spy_new   = _flatten_yf(yf.download("SPY",   start=fetch_start, auto_adjust=True, progress=False))
        vix_new   = _flatten_yf(yf.download("^VIX",  start=fetch_start, auto_adjust=True, progress=False))
        vix9d_new = _flatten_yf(yf.download("^VIX9D",start=fetch_start, auto_adjust=True, progress=False))

        if len(spy_new) > 0:
            spy_close_col  = next(c for c in spy_new.columns  if "close" in c)
            vix_close_col  = next(c for c in vix_new.columns  if "close" in c)
            vix9d_close_col = next((c for c in vix9d_new.columns if "close" in c), None)

            fresh = pd.DataFrame(index=spy_new.index)
            fresh.index.name = "date"
            fresh["close"]      = spy_new[spy_close_col].values
            fresh["vix_close"]  = vix_new[vix_close_col].reindex(spy_new.index).values
            fresh["vix9d_close"] = (
                vix9d_new[vix9d_close_col].reindex(spy_new.index).values
                if vix9d_close_col else fresh["vix_close"].values
            )
            fresh["vix9d_close"] = pd.Series(fresh["vix9d_close"], index=fresh.index).fillna(fresh["vix_close"])

            # Log returns
            prev_close = float(hist["close"].iloc[-1])
            log_rets = []
            closes = fresh["close"].values
            for i, c in enumerate(closes):
                prev = prev_close if i == 0 else closes[i - 1]
                log_rets.append(np.log(c / prev) if prev > 0 else np.nan)
            fresh["log_return"] = log_rets

            fresh = fresh.dropna(subset=["close", "vix_close", "log_return"])
            hist = pd.concat([hist, fresh])
            hist = hist[~hist.index.duplicated(keep="last")].sort_index()

    # Exclude today's row (may be intraday/partial) — live signal handles today separately
    hist = hist[hist.index.normalize() < today]
    return hist


def fetch_live_quotes() -> dict:
    """
    Fetch current SPY, VIX, VIX9D prices via yfinance.
    During market hours: ~15-min delayed live price.
    Outside market hours: most recent close / last traded price.
    Returns a dict with price info and market status.
    """
    result = {}

    for ticker_sym, key in [("SPY", "spy"), ("^VIX", "vix"), ("^VIX9D", "vix9d")]:
        try:
            t = yf.Ticker(ticker_sym)
            fi = t.fast_info
            price = getattr(fi, "last_price", None)
            if price is None or np.isnan(price):
                price = t.history(period="5d")["Close"].iloc[-1]
            result[f"{key}_live"] = round(float(price), 2)
        except Exception:
            result[f"{key}_live"] = None

    # Determine rough market status (NYSE hours: 9:30-16:00 ET)
    try:
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
    except ImportError:
        now_et = datetime.utcnow()  # fallback

    is_weekday = now_et.weekday() < 5
    market_open  = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0,  second=0, microsecond=0)
    result["market_open"] = is_weekday and (market_open <= now_et <= market_close)
    result["timestamp_et"] = now_et.strftime("%Y-%m-%d %H:%M:%S ET")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Signal computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_features(df: pd.DataFrame, p_calm_series: pd.Series) -> pd.DataFrame:
    """
    Build the 9 XGBoost features from the extended DataFrame.
    Mirrors build_features() in xgboost_train.py exactly.
    p_calm_series must be aligned to df.index.
    """
    r = df["log_return"]
    S = df["close"]

    feat = pd.DataFrame(index=df.index)
    feat["RV_5d"]    = r.rolling(5).std()  * np.sqrt(252)
    feat["RV_20d"]   = r.rolling(20).std() * np.sqrt(252)
    feat["RV_60d"]   = r.rolling(60).std() * np.sqrt(252)
    feat["RV_ratio"] = feat["RV_20d"] / feat["RV_60d"].replace(0, np.nan)
    feat["Mom_5d"]   = r.rolling(5).sum()
    feat["Mom_20d"]  = r.rolling(20).sum()
    feat["DD_60d"]   = (S - S.rolling(60).max()) / S.rolling(60).max()
    feat["RSkew_20d"]= r.rolling(20).skew()
    feat["p_calm"]   = p_calm_series.reindex(df.index)

    return feat.dropna(subset=FEATURE_COLS)


def build_live_df(df: pd.DataFrame, S_live: float,
                  vix_live: float, vix9d_live: float | None) -> pd.DataFrame:
    """
    Append a synthetic 'today' row built from live market prices.

    log_return is computed relative to yesterday's close (the last row in df
    that predates today), so it captures the full intraday move even when
    today's official close is already present in df.

    If the live prices equal the previous close (market closed, no movement),
    the appended row is a no-op and the two signals will be identical.
    """
    today = pd.Timestamp.today().normalize()

    # Find yesterday's close (most recent date strictly before today)
    hist_before_today = df[df.index.normalize() < today]
    if len(hist_before_today) == 0:
        # Fallback: use last available row
        prev_close = float(df["close"].iloc[-1])
    else:
        prev_close = float(hist_before_today["close"].iloc[-1])

    log_ret = np.log(S_live / prev_close) if prev_close > 0 else np.nan
    vix9d   = vix9d_live if (vix9d_live and not np.isnan(vix9d_live)) else vix_live

    live_row = pd.DataFrame(
        [{"close": S_live, "vix_close": vix_live,
          "vix9d_close": vix9d, "log_return": log_ret}],
        index=pd.DatetimeIndex([today], name="date"),
    )
    combined = pd.concat([df, live_row])
    # Keep the live row if today already exists (overwrites stale same-day data)
    return combined[~combined.index.duplicated(keep="last")].sort_index()


def compute_signal(hmm: GaussianHMM, xgb_model: xgb.XGBClassifier,
                   df: pd.DataFrame) -> tuple[float, float, pd.DataFrame]:
    """
    Run HMM → p_calm, then XGBoost → p_safe for the full series.
    Returns (p_calm_last, p_safe_last, feat_df_with_last_row_features)
    """
    X = df["log_return"].values.reshape(-1, 1)
    posteriors = hmm.predict_proba(X)    # shape (T, 2); state 0 = Calm
    p_calm_series = pd.Series(posteriors[:, 0], index=df.index, name="p_calm")

    feat = compute_features(df, p_calm_series)
    if len(feat) == 0:
        return np.nan, np.nan, feat

    X_feat = feat[FEATURE_COLS].values
    proba  = xgb_model.predict_proba(X_feat)    # P(class=0) = P(safe)
    p_safe_series = pd.Series(proba[:, 0], index=feat.index)

    return float(p_calm_series.iloc[-1]), float(p_safe_series.iloc[-1]), feat


def make_signal_block(p_calm: float, p_safe: float,
                      n_open: int, tc: float, ts: float,
                      feat: pd.DataFrame) -> dict:
    """Package one signal result into the JSON sub-dict returned to the frontend."""
    calm_pass   = bool(p_calm > tc)
    safe_pass   = bool(p_safe > ts)
    slots_avail = n_open < MAX_POSITIONS
    features    = {}
    if len(feat) > 0:
        last = feat[FEATURE_COLS].iloc[-1]
        features = {col: round(float(last[col]), 6) for col in FEATURE_COLS}
    return {
        "p_calm":      round(p_calm, 4),
        "p_safe":      round(p_safe, 4),
        "calm_pass":   calm_pass,
        "safe_pass":   safe_pass,
        "slots_avail": slots_avail,
        "signal_open": calm_pass and safe_pass and slots_avail,
        "features":    features,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Spread parameter calculation
# ══════════════════════════════════════════════════════════════════════════════

def _bs_put(S, K, T_yr, r, sigma):
    from scipy.stats import norm
    if T_yr <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_yr) / (sigma * np.sqrt(T_yr))
    d2 = d1 - sigma * np.sqrt(T_yr)
    return K * np.exp(-r * T_yr) * norm.cdf(-d2) - S * norm.cdf(-d1)


def option_price(S, K, T_days, vix, vix9d, skew_fn):
    """Mirror of get_option_price() from data_collection.py."""
    if T_days <= 0:
        return max(K - S, 0.0)
    total_var_30 = (vix / 100.0) ** 2 * 30.0
    if vix9d and vix9d > 0:
        total_var_9 = (vix9d / 100.0) ** 2 * 9.0
        slope = (total_var_30 - total_var_9) / 21.0
        total_var_T = max(total_var_9 + slope * (T_days - 9.0), 1e-8)
    else:
        total_var_T = max(total_var_30 * (T_days / 30.0), 1e-8)
    iv_atm    = np.sqrt(total_var_T / T_days)
    skew_mult = float(np.clip(skew_fn(K / S), 0.80, 3.0))
    return _bs_put(S, K, T_days / 365.0, R, iv_atm * skew_mult)


def next_nth_friday(date: pd.Timestamp, n: int = 4) -> pd.Timestamp:
    days_ahead = (4 - date.day_of_week) % 7
    if days_ahead == 0:
        days_ahead = 7
    return date + pd.Timedelta(days=days_ahead) + pd.Timedelta(weeks=n - 1)


def compute_spread_params(S_live, vix_live, vix9d_live, nav, skew_fn) -> dict:
    """Compute today's spread parameters using live market prices."""
    today = pd.Timestamp.today().normalize()
    expiry = next_nth_friday(today, n=N_WEEKS_OUT)
    T_days = (expiry - today).days

    K1 = round(S_live * SHORT_LEG_M, 2)
    K2 = round(S_live * LONG_LEG_M,  2)

    vix9d = vix9d_live if (vix9d_live and not np.isnan(vix9d_live)) else None

    p_short = option_price(S_live, K1, T_days, vix_live, vix9d, skew_fn)
    p_long  = option_price(S_live, K2, T_days, vix_live, vix9d, skew_fn)
    credit        = round(p_short - p_long, 4)
    max_loss_ps   = round((K1 - K2) - credit, 4)

    contracts     = round((CAR_FRACTION * nav) / (max_loss_ps * 100), 2) if max_loss_ps > 0.01 else 0.0
    credit_total  = round(credit * contracts * 100, 2)
    max_loss_total= round(max_loss_ps * contracts * 100, 2)

    return {
        "K1":                  K1,
        "K2":                  K2,
        "expiry":              expiry.strftime("%Y-%m-%d"),
        "T_days":              T_days,
        "p_short":             round(p_short, 4),
        "p_long":              round(p_long,  4),
        "credit_per_share":    credit,
        "max_loss_per_share":  max_loss_ps,
        "contracts":           contracts,
        "credit_total":        credit_total,
        "max_loss_total":      max_loss_total,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Flask routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("calculator.html")


@app.route("/api/signal")
def api_signal():
    """
    Main computation endpoint.
    Query params:
        nav   : float  Current portfolio NAV (default 100000)
        n_open: int    Number of currently open positions (default 0)

    Returns two independent signal blocks:
        sig_prev  — signal based on last official close (model-faithful)
        sig_live  — signal based on live intraday prices (situational awareness)
    """
    try:
        nav    = float(request.args.get("nav",    100_000))
        n_open = int(request.args.get("n_open", 0))

        # Load models and thresholds (once, shared by both signal runs)
        hmm       = reconstruct_hmm()
        xgb_model = load_xgb_model()
        skew_fn   = load_skew_fn()
        tc, ts    = load_thresholds()

        # Historical series (up to last official close)
        df = fetch_extended_history()
        prev_close_date = df.index[-1].strftime("%Y-%m-%d")
        spy_prev_close  = round(float(df["close"].iloc[-1]), 2)
        vix_prev_close  = round(float(df["vix_close"].iloc[-1]), 2)

        # Live market quotes
        live = fetch_live_quotes()
        S_live    = live["spy_live"]   or spy_prev_close
        vix_live  = live["vix_live"]   or vix_prev_close
        vix9d_live = live["vix9d_live"]

        # ── Signal 1: previous close ─────────────────────────────────────────
        p_calm_prev, p_safe_prev, feat_prev = compute_signal(hmm, xgb_model, df)
        sig_prev = make_signal_block(p_calm_prev, p_safe_prev, n_open, tc, ts, feat_prev)

        # ── Signal 2: live intraday ──────────────────────────────────────────
        df_live = build_live_df(df, S_live, vix_live, vix9d_live)
        p_calm_live, p_safe_live, feat_live = compute_signal(hmm, xgb_model, df_live)
        sig_live = make_signal_block(p_calm_live, p_safe_live, n_open, tc, ts, feat_live)

        # Intraday move relative to yesterday's close
        hist_before_today = df[df.index.normalize() < pd.Timestamp.today().normalize()]
        prev_close_ref = float(hist_before_today["close"].iloc[-1]) if len(hist_before_today) > 0 else spy_prev_close
        intraday_ret_pct = round((S_live / prev_close_ref - 1) * 100, 3)

        # Divergence flag: signals give opposite open/close decisions
        signals_diverge = sig_prev["signal_open"] != sig_live["signal_open"]

        # Spread parameters (always at live prices)
        spread = compute_spread_params(S_live, vix_live, vix9d_live, nav, skew_fn)

        return jsonify({
            "ok":                 True,
            "timestamp_et":       live["timestamp_et"],
            "market_open":        live["market_open"],
            "prev_close_date":    prev_close_date,
            "tc": tc, "ts": ts,
            "n_open":             n_open,
            # Two signal blocks
            "sig_prev":           sig_prev,
            "sig_live":           sig_live,
            "signals_diverge":    signals_diverge,
            # Market data
            "spy_live":           S_live,
            "spy_prev_close":     spy_prev_close,
            "vix_live":           vix_live,
            "vix_prev_close":     vix_prev_close,
            "vix9d_live":         vix9d_live,
            "intraday_ret_pct":   intraday_ret_pct,
            # Spread (live prices)
            "spread":             spread,
            "error":              None,
        })

    except Exception as exc:
        import traceback
        return jsonify({"ok": False, "error": str(exc),
                        "traceback": traceback.format_exc()}), 500


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Position management routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/positions")
def api_get_positions():
    """Return all positions; open ones are enriched with live MTM data."""
    try:
        positions = pm.load_positions()
        settings  = pm.load_settings()

        live      = fetch_live_quotes()
        skew_fn   = load_skew_fn()
        S_live    = live["spy_live"]
        vix_live  = live["vix_live"]
        vix9d_live = live["vix9d_live"]

        enriched = []
        for pos in positions:
            if pos["status"] == "open" and S_live and vix_live:
                enriched.append(pm.enrich_open_position(
                    pos, S_live, vix_live, vix9d_live, skew_fn, option_price
                ))
            else:
                enriched.append(pos)

        stats = pm.compute_portfolio_stats(enriched, settings["initial_capital"])
        return jsonify({"ok": True, "positions": enriched,
                        "settings": settings, "stats": stats})
    except Exception as exc:
        import traceback
        return jsonify({"ok": False, "error": str(exc),
                        "traceback": traceback.format_exc()}), 500


@app.route("/api/positions/open", methods=["POST"])
def api_open_position():
    """Create a new position using current live market prices."""
    try:
        settings  = pm.load_settings()
        nav       = settings["initial_capital"]

        # Use actual portfolio value as NAV for contract sizing
        positions = pm.load_positions()
        live      = fetch_live_quotes()
        skew_fn   = load_skew_fn()
        S_live    = live["spy_live"]
        vix_live  = live["vix_live"]
        vix9d_live = live["vix9d_live"]

        if not S_live or not vix_live:
            return jsonify({"ok": False, "error": "Live prices unavailable"}), 503

        # Check open position count
        n_open = sum(1 for p in positions if p["status"] == "open")
        if n_open >= MAX_POSITIONS:
            return jsonify({"ok": False, "error": "Maximum positions reached"}), 400

        # Enrich existing positions for portfolio value calculation
        enriched = []
        for pos in positions:
            if pos["status"] == "open":
                enriched.append(pm.enrich_open_position(
                    pos, S_live, vix_live, vix9d_live, skew_fn, option_price
                ))
            else:
                enriched.append(pos)
        stats = pm.compute_portfolio_stats(enriched, nav)

        spread = compute_spread_params(
            S_live, vix_live, vix9d_live, stats["portfolio_value"], skew_fn
        )
        new_pos = pm.create_position(spread, S_live, spread["p_short"], spread["p_long"])
        pm.add_position(new_pos)
        return jsonify({"ok": True, "position": new_pos})
    except Exception as exc:
        import traceback
        return jsonify({"ok": False, "error": str(exc),
                        "traceback": traceback.format_exc()}), 500


@app.route("/api/positions/<pos_id>/edit", methods=["PUT"])
def api_edit_position(pos_id):
    """Update editable fields on an existing position."""
    try:
        data = request.get_json(force=True)
        # Only allow whitelisted fields to be updated
        allowed = {
            "actual_spy", "actual_K1", "actual_K2", "actual_expiry",
            "actual_short_premium", "actual_long_premium", "actual_contracts",
            # Also allow updating close fields on closed positions
            "close_spy", "close_short_premium", "close_long_premium",
        }
        fields = {k: v for k, v in data.items() if k in allowed}
        # Cast numeric fields
        for k in fields:
            if k != "actual_expiry":
                try:
                    fields[k] = float(fields[k])
                except (TypeError, ValueError):
                    pass
        if "actual_contracts" in fields:
            fields["actual_contracts"] = int(fields["actual_contracts"])

        # Recompute locked_pnl if close premiums were edited
        positions = pm.load_positions()
        pos = next((p for p in positions if p["id"] == pos_id), None)
        updated = pm.update_position_fields(pos_id, fields)
        if updated is None:
            return jsonify({"ok": False, "error": "Position not found"}), 404

        # If it's a closed position and close premiums changed, recompute locked_pnl
        if updated["status"] == "closed" and any(
            k in fields for k in ("close_short_premium", "close_long_premium",
                                  "actual_contracts")
        ):
            credit_ps  = updated["actual_credit_per_share"]
            close_cost = updated["close_short_premium"] - updated["close_long_premium"]
            contracts  = updated["actual_contracts"]
            updated = pm.update_position_fields(pos_id, {
                "locked_pnl": round((credit_ps - close_cost) * contracts * 100, 2)
            })

        return jsonify({"ok": True, "position": updated})
    except Exception as exc:
        import traceback
        return jsonify({"ok": False, "error": str(exc),
                        "traceback": traceback.format_exc()}), 500


@app.route("/api/positions/<pos_id>/close", methods=["POST"])
def api_close_position(pos_id):
    """Lock in close prices and mark position as closed."""
    try:
        data = request.get_json(force=True)
        required = ("close_spy", "close_short_premium", "close_long_premium")
        for k in required:
            if k not in data:
                return jsonify({"ok": False, "error": f"Missing field: {k}"}), 400

        close_data = {
            "close_spy":           float(data["close_spy"]),
            "close_short_premium": float(data["close_short_premium"]),
            "close_long_premium":  float(data["close_long_premium"]),
        }
        if "actual_contracts" in data:
            close_data["actual_contracts"] = int(data["actual_contracts"])

        updated = pm.lock_close(pos_id, close_data)
        if updated is None:
            return jsonify({"ok": False, "error": "Position not found or already closed"}), 404

        return jsonify({"ok": True, "position": updated})
    except Exception as exc:
        import traceback
        return jsonify({"ok": False, "error": str(exc),
                        "traceback": traceback.format_exc()}), 500


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    """GET or POST user settings (initial capital)."""
    try:
        if request.method == "POST":
            data = request.get_json(force=True)
            settings = pm.load_settings()
            if "initial_capital" in data:
                settings["initial_capital"] = float(data["initial_capital"])
            pm.save_settings(settings)
            return jsonify({"ok": True, "settings": settings})
        else:
            return jsonify({"ok": True, "settings": pm.load_settings()})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/positions/reset", methods=["POST"])
def api_reset_positions():
    """Clear all position history and reset portfolio to initial capital."""
    try:
        pm.save_positions([])
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


def main():
    import webbrowser, threading, time
    def _open():
        time.sleep(1.2)
        webbrowser.open("http://127.0.0.1:5719")
    threading.Thread(target=_open, daemon=True).start()
    print("PRISM Live Calculator — http://127.0.0.1:5719")
    app.run(debug=False, host="127.0.0.1", port=5719)

if __name__ == "__main__":
    main()
