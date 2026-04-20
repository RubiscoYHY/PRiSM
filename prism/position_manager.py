"""
position_manager.py
===================
Local position storage and P/L calculation for PRISM.

All data is stored in ~/.prism/ — never inside the project directory,
so it is automatically excluded from git and stays private.

  ~/.prism/positions.json   — trade history
  ~/.prism/settings.json    — user preferences (initial capital, etc.)
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

PRISM_HOME     = Path.home() / ".prism"
POSITIONS_FILE = PRISM_HOME / "positions.json"
SETTINGS_FILE  = PRISM_HOME / "settings.json"

DEFAULT_SETTINGS = {"initial_capital": 100_000.0}

# Close-signal thresholds — mirror the backtest in threshold_grid.py:
#   "Close: DTE <= 5 | profit >= 80% of max | loss >= 50% of max"
TAKE_PROFIT_PCT = 0.80
STOP_LOSS_PCT   = 0.50
DTE_CLOSE       = 5


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ensure_dir() -> None:
    PRISM_HOME.mkdir(exist_ok=True)


# ── Settings ──────────────────────────────────────────────────────────────────

def load_settings() -> dict:
    _ensure_dir()
    if not SETTINGS_FILE.exists():
        return dict(DEFAULT_SETTINGS)
    try:
        return json.loads(SETTINGS_FILE.read_text())
    except Exception:
        return dict(DEFAULT_SETTINGS)


def save_settings(settings: dict) -> None:
    _ensure_dir()
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2))


# ── Position CRUD ─────────────────────────────────────────────────────────────

def load_positions() -> list:
    _ensure_dir()
    if not POSITIONS_FILE.exists():
        return []
    try:
        return json.loads(POSITIONS_FILE.read_text())
    except Exception:
        return []


def save_positions(positions: list) -> None:
    _ensure_dir()
    POSITIONS_FILE.write_text(json.dumps(positions, indent=2))


def create_position(spread_params: dict, spy_live: float,
                    p_short: float, p_long: float) -> dict:
    """
    Build a new position dict from spread_params + live option prices.
    Actual fills are pre-populated with model values; user can edit later.
    """
    contracts = int(max(1, round(spread_params["contracts"])))
    credit_ps = round(p_short - p_long, 4)
    max_loss_ps = round(spread_params["K1"] - spread_params["K2"] - credit_ps, 4)

    return {
        "id":          str(uuid.uuid4())[:8],
        "status":      "open",
        "opened_at":   datetime.today().strftime("%Y-%m-%d"),
        # Model defaults (for reference)
        "model_spy":              round(spy_live, 2),
        "model_K1":               spread_params["K1"],
        "model_K2":               spread_params["K2"],
        "model_expiry":           spread_params["expiry"],
        "model_short_premium":    round(p_short, 4),
        "model_long_premium":     round(p_long,  4),
        "model_credit_per_share": round(p_short - p_long, 4),
        "model_contracts":        contracts,
        # Actual fills — editable by user; pre-filled with model values
        "actual_spy":             round(spy_live, 2),
        "actual_K1":              spread_params["K1"],
        "actual_K2":              spread_params["K2"],
        "actual_expiry":          spread_params["expiry"],
        "actual_short_premium":   round(p_short, 4),
        "actual_long_premium":    round(p_long,  4),
        "actual_credit_per_share":  credit_ps,
        "actual_max_loss_per_share": max_loss_ps,
        "actual_contracts":       contracts,
        # Close data — null until position is closed
        "closed_at":              None,
        "close_spy":              None,
        "close_short_premium":    None,
        "close_long_premium":     None,
        "locked_pnl":             None,
    }


def add_position(position: dict) -> None:
    """Insert a new position at the front of the list (newest first)."""
    positions = load_positions()
    positions.insert(0, position)
    save_positions(positions)


def update_position_fields(pos_id: str, fields: dict) -> dict | None:
    """
    Update arbitrary editable fields on a position.
    Automatically recomputes derived fields (credit_per_share, max_loss_per_share)
    when premiums or strikes are changed.
    Returns the updated position dict, or None if not found.
    """
    positions = load_positions()
    for pos in positions:
        if pos["id"] != pos_id:
            continue
        pos.update(fields)
        # Recompute credit if either premium changed
        if "actual_short_premium" in fields or "actual_long_premium" in fields:
            pos["actual_credit_per_share"] = round(
                pos["actual_short_premium"] - pos["actual_long_premium"], 4
            )
        # Recompute max loss if strikes or credit changed
        if any(k in fields for k in ("actual_K1", "actual_K2",
                                     "actual_short_premium", "actual_long_premium")):
            pos["actual_max_loss_per_share"] = round(
                pos["actual_K1"] - pos["actual_K2"] - pos["actual_credit_per_share"], 4
            )
        save_positions(positions)
        return pos
    return None


def lock_close(pos_id: str, close_data: dict) -> dict | None:
    """
    Mark a position as closed and compute the locked P/L.

    close_data keys:
        close_spy             float — SPY price at close
        close_short_premium   float — price paid to buy back the short put
        close_long_premium    float — price received to sell the long put
        actual_contracts      int   — optional override (for partial closes)
    """
    positions = load_positions()
    for pos in positions:
        if pos["id"] != pos_id or pos["status"] != "open":
            continue
        contracts  = close_data.get("actual_contracts", pos["actual_contracts"])
        credit_ps  = pos["actual_credit_per_share"]
        close_cost = close_data["close_short_premium"] - close_data["close_long_premium"]
        locked_pnl = round((credit_ps - close_cost) * contracts * 100, 2)

        pos.update({
            "status":              "closed",
            "closed_at":           datetime.today().strftime("%Y-%m-%d"),
            "close_spy":           close_data["close_spy"],
            "close_short_premium": close_data["close_short_premium"],
            "close_long_premium":  close_data["close_long_premium"],
            "actual_contracts":    contracts,
            "locked_pnl":          locked_pnl,
        })
        save_positions(positions)
        return pos
    return None


# ── Mark-to-market ────────────────────────────────────────────────────────────

def enrich_open_position(pos: dict, S_live: float, vix_live: float,
                          vix9d_live, skew_fn, option_price_fn) -> dict:
    """
    Compute current mark-to-market P/L and close signals for an open position.

    Returns a shallow copy of pos with additional keys:
        dte, cur_short_price, cur_long_price, cur_spread_cost,
        pnl_per_share, pnl_total, pnl_pct,
        close_signal, signal_reasons
    """
    expiry_dt = datetime.strptime(pos["actual_expiry"], "%Y-%m-%d")
    dte = max(0, (expiry_dt - datetime.today()).days)

    K1        = pos["actual_K1"]
    K2        = pos["actual_K2"]
    contracts = pos["actual_contracts"]
    credit_ps = pos["actual_credit_per_share"]
    max_loss_ps = max(pos.get("actual_max_loss_per_share",
                               K1 - K2 - credit_ps), 0.01)

    cur_short = option_price_fn(S_live, K1, dte, vix_live, vix9d_live, skew_fn)
    cur_long  = option_price_fn(S_live, K2, dte, vix_live, vix9d_live, skew_fn)
    cur_cost  = cur_short - cur_long     # net debit to close
    pnl_ps    = credit_ps - cur_cost     # positive = profit
    pnl_total = round(pnl_ps * contracts * 100, 2)
    pnl_pct   = round(pnl_ps / credit_ps * 100, 1) if credit_ps > 1e-6 else 0.0

    take_profit = pnl_ps >= TAKE_PROFIT_PCT * credit_ps
    stop_loss   = pnl_ps <= -(STOP_LOSS_PCT * max_loss_ps)
    near_expiry = dte <= DTE_CLOSE

    reasons: list[str] = []
    if take_profit:  reasons.append(f"take profit (≥{int(TAKE_PROFIT_PCT*100)}% max credit)")
    if stop_loss:    reasons.append(f"stop loss (≥{int(STOP_LOSS_PCT*100)}% max loss)")
    if near_expiry:  reasons.append(f"near expiry ({dte}d remaining)")

    return {
        **pos,
        "dte":             dte,
        "cur_short_price": round(cur_short, 4),
        "cur_long_price":  round(cur_long,  4),
        "cur_spread_cost": round(cur_cost,  4),
        "pnl_per_share":   round(pnl_ps,    4),
        "pnl_total":       pnl_total,
        "pnl_pct":         pnl_pct,
        "close_signal":    take_profit or stop_loss or near_expiry,
        "signal_reasons":  reasons,
    }


# ── Portfolio statistics ──────────────────────────────────────────────────────

def compute_portfolio_stats(positions: list, initial_capital: float) -> dict:
    """
    Aggregate portfolio-level statistics from a list of (possibly enriched) positions.
    """
    locked_pnl   = sum(p["locked_pnl"] or 0 for p in positions if p["status"] == "closed")
    unrealized   = sum(p.get("pnl_total", 0) for p in positions if p["status"] == "open")
    portfolio_val = initial_capital + locked_pnl + unrealized

    open_pos = [p for p in positions if p["status"] == "open"]
    car_total = sum(
        p.get("actual_max_loss_per_share", 0) * p["actual_contracts"] * 100
        for p in open_pos
    )
    car_pct = round(car_total / portfolio_val * 100, 1) if portfolio_val > 0 else 0.0

    return {
        "portfolio_value": round(portfolio_val, 2),
        "initial_capital": round(initial_capital, 2),
        "locked_pnl":      round(locked_pnl, 2),
        "unrealized_pnl":  round(unrealized, 2),
        "car_total":       round(car_total, 2),
        "car_pct":         car_pct,
        "n_open":          len(open_pos),
        "n_total":         len(positions),
    }
