"""
plot_psafe_analysis.py
=====================
Generate diagnostic plots for P(Safe) and combined P(Calm)+P(Safe) signals.

Plot 1: SPY price + P(Safe) time series (mirroring the HMM P(Calm) layout)
Plot 2: SPY price with combined filter (PCalm > tc AND PSafe > ts) highlighted
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

from prism.paths import DATA_DIR
from prism.xgboost_train import build_features, FEATURE_COLS

# ── Config ──
TC = 0.71   # threshold_calm
TS = 0.95   # threshold_safe

XGB_DIR  = DATA_DIR / "XGBoost"
OUT_DIR  = DATA_DIR / "analysis"
OUT_DIR.mkdir(exist_ok=True)


# ── Load data ──
print("[1/4] Loading data...")
spy = pd.read_csv(DATA_DIR / "spy_vix_daily.csv", index_col="date", parse_dates=True)
hmm = pd.read_csv(DATA_DIR / "HMM" / "hmm_pcalm_daily.csv", index_col="date", parse_dates=True)

close_col = next(c for c in spy.columns if "spy" in c and "close" in c)
spy = spy.rename(columns={close_col: "close"})
df = spy.join(hmm[["p_calm"]], how="inner").sort_index()

feat = build_features(df)
feat = feat.join(df[["close", "vix_close", "vix9d_close"]], how="left")

# ── Compute p_safe ──
print("[2/4] Computing P(Safe) via XGBoost...")
model = xgb.XGBClassifier()
model.load_model(str(XGB_DIR / "xgb_model.json"))
proba = model.predict_proba(feat[FEATURE_COLS].values)
feat["p_safe"] = proba[:, 0]

print(f"  Dataset: {len(feat)} rows ({feat.index[0].date()} – {feat.index[-1].date()})")
print(f"  P(Safe) range: [{feat['p_safe'].min():.4f}, {feat['p_safe'].max():.4f}]")


# ══════════════════════════════════════════════════════════════
# PLOT 1: SPY + P(Safe) — mirroring the HMM P(Calm) layout
# ══════════════════════════════════════════════════════════════
print("[3/4] Generating Plot 1: SPY + P(Safe)...")

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(18, 9), sharex=True,
    gridspec_kw={"height_ratios": [2, 1.2], "hspace": 0.08},
)

dates = feat.index

# ── Top panel: SPY Close with turbulent shading ──
ax1.plot(dates, feat["close"], color="#1f77b4", linewidth=1.0, label="SPY Close")

# Shade regions where P(Safe) < TS (danger zones)
danger = feat["p_safe"] < TS
danger_starts = []
danger_ends = []
in_danger = False
for i, (d, v) in enumerate(zip(dates, danger)):
    if v and not in_danger:
        danger_starts.append(d)
        in_danger = True
    elif not v and in_danger:
        danger_ends.append(d)
        in_danger = False
if in_danger:
    danger_ends.append(dates[-1])

for s, e in zip(danger_starts, danger_ends):
    ax1.axvspan(s, e, alpha=0.18, color="salmon", zorder=0)

ax1.set_ylabel("SPY Price (USD)", fontsize=12)
ax1.legend(
    [plt.Line2D([0], [0], color="#1f77b4", lw=1.5),
     plt.Rectangle((0, 0), 1, 1, fc="salmon", alpha=0.3)],
    ["SPY Close", f"P(Safe) < {TS} (danger)"],
    loc="upper left", fontsize=10,
)
ax1.grid(True, alpha=0.3)
ax1.set_title(
    f"XGBoost P(Safe) Analysis — Threshold: {TS}\n"
    f"Training: 2015–2020 | Display: {dates[0].year}–{dates[-1].year}",
    fontsize=14, fontweight="bold",
)

# Add vertical lines for train/val/test boundaries
for label, d in [("Train end", "2020-01-01"), ("Val end", "2022-01-01")]:
    ax1.axvline(pd.Timestamp(d), color="black", linestyle=":", linewidth=1, alpha=0.6)
    ax1.text(pd.Timestamp(d), ax1.get_ylim()[1] * 0.95, label,
             ha="center", va="top", fontsize=9, color="black",
             bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

# ── Bottom panel: P(Safe) time series ──
safe_mask = feat["p_safe"] >= TS
ax2.fill_between(dates, feat["p_safe"], alpha=0.6,
                 where=safe_mask, color="#2ca02c", label=f"P(Safe) ≥ {TS}")
ax2.fill_between(dates, feat["p_safe"], alpha=0.6,
                 where=~safe_mask, color="salmon", label=f"P(Safe) < {TS}")
ax2.axhline(TS, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax2.text(dates[-1], TS + 0.02, f"ts = {TS}", fontsize=9, color="gray",
         ha="right", va="bottom")

ax2.set_ylabel("P(Safe)", fontsize=12)
ax2.set_ylim(-0.05, 1.05)
ax2.legend(loc="lower left", fontsize=10)
ax2.grid(True, alpha=0.3)

ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig(OUT_DIR / "psafe_spy_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {OUT_DIR / 'psafe_spy_plot.png'}")


# ══════════════════════════════════════════════════════════════
# PLOT 2: SPY + Combined filter (PCalm + PSafe both pass)
# ══════════════════════════════════════════════════════════════
print("[4/4] Generating Plot 2: Combined PCalm + PSafe filter...")

both_pass = (feat["p_calm"] > TC) & (feat["p_safe"] > TS)
calm_only = (feat["p_calm"] > TC) & (feat["p_safe"] <= TS)
safe_only = (feat["p_calm"] <= TC) & (feat["p_safe"] > TS)
neither   = (feat["p_calm"] <= TC) & (feat["p_safe"] <= TS)

fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(18, 12), sharex=True,
    gridspec_kw={"height_ratios": [2, 1, 1], "hspace": 0.08},
)

# ── Top: SPY with combined shading ──
ax1.plot(dates, feat["close"], color="#1f77b4", linewidth=1.0)

# Shade: green = both pass, red = neither pass, leave rest unshaded
both_starts, both_ends = [], []
in_both = False
for i, (d, v) in enumerate(zip(dates, both_pass)):
    if v and not in_both:
        both_starts.append(d)
        in_both = True
    elif not v and in_both:
        both_ends.append(d)
        in_both = False
if in_both:
    both_ends.append(dates[-1])

neither_starts, neither_ends = [], []
in_neither = False
for i, (d, v) in enumerate(zip(dates, neither)):
    if v and not in_neither:
        neither_starts.append(d)
        in_neither = True
    elif not v and in_neither:
        neither_ends.append(d)
        in_neither = False
if in_neither:
    neither_ends.append(dates[-1])

for s, e in zip(both_starts, both_ends):
    ax1.axvspan(s, e, alpha=0.2, color="#2ca02c", zorder=0)
for s, e in zip(neither_starts, neither_ends):
    ax1.axvspan(s, e, alpha=0.18, color="salmon", zorder=0)

ax1.set_ylabel("SPY Price (USD)", fontsize=12)
ax1.legend(
    [plt.Line2D([0], [0], color="#1f77b4", lw=1.5),
     plt.Rectangle((0, 0), 1, 1, fc="#2ca02c", alpha=0.3),
     plt.Rectangle((0, 0), 1, 1, fc="salmon", alpha=0.3)],
    ["SPY Close",
     f"Both pass (PCalm>{TC} & PSafe>{TS})",
     f"Neither pass"],
    loc="upper left", fontsize=10,
)
ax1.grid(True, alpha=0.3)
ax1.set_title(
    f"Combined Signal: P(Calm) > {TC} AND P(Safe) > {TS}\n"
    f"Training: 2015–2020 | Display: {dates[0].year}–{dates[-1].year}",
    fontsize=14, fontweight="bold",
)

for label, d in [("Train end", "2020-01-01"), ("Val end", "2022-01-01")]:
    ax1.axvline(pd.Timestamp(d), color="black", linestyle=":", linewidth=1, alpha=0.6)
    ax1.text(pd.Timestamp(d), ax1.get_ylim()[1] * 0.95, label,
             ha="center", va="top", fontsize=9, color="black",
             bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

# ── Middle: P(Calm) ──
calm_mask = feat["p_calm"] > TC
ax2.fill_between(dates, feat["p_calm"], alpha=0.6,
                 where=calm_mask, color="#2ca02c", label=f"P(Calm) > {TC}")
ax2.fill_between(dates, feat["p_calm"], alpha=0.6,
                 where=~calm_mask, color="salmon", label=f"P(Calm) ≤ {TC}")
ax2.axhline(TC, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax2.text(dates[-1], TC + 0.02, f"tc = {TC}", fontsize=9, color="gray",
         ha="right", va="bottom")
ax2.set_ylabel("P(Calm)", fontsize=12)
ax2.set_ylim(-0.05, 1.05)
ax2.legend(loc="lower left", fontsize=10)
ax2.grid(True, alpha=0.3)

# ── Bottom: P(Safe) ──
safe_mask = feat["p_safe"] > TS
ax3.fill_between(dates, feat["p_safe"], alpha=0.6,
                 where=safe_mask, color="#2ca02c", label=f"P(Safe) > {TS}")
ax3.fill_between(dates, feat["p_safe"], alpha=0.6,
                 where=~safe_mask, color="salmon", label=f"P(Safe) ≤ {TS}")
ax3.axhline(TS, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax3.text(dates[-1], TS + 0.02, f"ts = {TS}", fontsize=9, color="gray",
         ha="right", va="bottom")
ax3.set_ylabel("P(Safe)", fontsize=12)
ax3.set_ylim(-0.05, 1.05)
ax3.legend(loc="lower left", fontsize=10)
ax3.grid(True, alpha=0.3)

ax3.xaxis.set_major_locator(mdates.YearLocator())
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig(OUT_DIR / "combined_filter_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {OUT_DIR / 'combined_filter_plot.png'}")


# ══════════════════════════════════════════════════════════════
# STATISTICS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STATISTICS SUMMARY")
print("=" * 60)

total_days = len(feat)

# Overall
n_calm = (feat["p_calm"] > TC).sum()
n_safe = (feat["p_safe"] > TS).sum()
n_both = both_pass.sum()
n_neither = neither.sum()

print(f"\nTotal trading days: {total_days}")
print(f"  P(Calm) > {TC}  : {n_calm:4d} days ({n_calm/total_days:.1%})")
print(f"  P(Safe) > {TS}  : {n_safe:4d} days ({n_safe/total_days:.1%})")
print(f"  Both pass        : {n_both:4d} days ({n_both/total_days:.1%})")
print(f"  Neither pass     : {n_neither:4d} days ({n_neither/total_days:.1%})")
print(f"  Calm only        : {calm_only.sum():4d} days ({calm_only.sum()/total_days:.1%})")
print(f"  Safe only        : {safe_only.sum():4d} days ({safe_only.sum()/total_days:.1%})")

# By period
for period, start, end in [
    ("Train (2015-2020)", "2015-01-01", "2020-01-01"),
    ("Val   (2020-2022)", "2020-01-01", "2022-01-01"),
    ("Test  (2022-2025)", "2022-01-01", "2025-01-01"),
]:
    mask = (feat.index >= start) & (feat.index < end)
    sub = feat[mask]
    n = len(sub)
    if n == 0:
        continue
    bp = ((sub["p_calm"] > TC) & (sub["p_safe"] > TS)).sum()
    sc = (sub["p_safe"] > TS).sum()
    cc = (sub["p_calm"] > TC).sum()
    print(f"\n  {period}: {n} days")
    print(f"    P(Calm) pass: {cc:4d} ({cc/n:.1%})")
    print(f"    P(Safe) pass: {sc:4d} ({sc/n:.1%})")
    print(f"    Both pass   : {bp:4d} ({bp/n:.1%})")

# Recent 60 trading days
recent = feat.iloc[-60:]
print(f"\n  Recent 60 days ({recent.index[0].date()} – {recent.index[-1].date()}):")
print(f"    P(Safe) mean: {recent['p_safe'].mean():.4f}")
print(f"    P(Safe) > {TS}: {(recent['p_safe'] > TS).sum()}/{len(recent)} days ({(recent['p_safe'] > TS).mean():.1%})")
print(f"    P(Calm) > {TC}: {(recent['p_calm'] > TC).sum()}/{len(recent)} days ({(recent['p_calm'] > TC).mean():.1%})")
print(f"    Both pass: {((recent['p_calm'] > TC) & (recent['p_safe'] > TS)).sum()}/{len(recent)} days")

# Daily P(Safe) for last 20 trading days
print(f"\n  Last 20 trading days P(Safe):")
last20 = feat.iloc[-20:]
for d, row in last20.iterrows():
    flag_c = "V" if row["p_calm"] > TC else "X"
    flag_s = "V" if row["p_safe"] > TS else "X"
    flag_b = "PASS" if (row["p_calm"] > TC and row["p_safe"] > TS) else "----"
    print(f"    {d.date()}  P(Safe)={row['p_safe']:.4f}  P(Calm)={row['p_calm']:.4f}  "
          f"Calm:{flag_c}  Safe:{flag_s}  → {flag_b}")

print("\n" + "=" * 60)
print(f"Plots saved to: {OUT_DIR}")
print("=" * 60)
