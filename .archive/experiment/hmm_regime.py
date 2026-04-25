"""
hmm_regime.py
=============
Layer 1: 2-state Gaussian HMM trained on daily SPY log-returns.

Pipeline:
  1. Load spy_vix_daily.csv, restrict to training window (2015-2020).
  2. Run Baum-Welch with N_RESTARTS random initialisations.
     - Each restart uses a different random seed → different starting
       (A, mu, sigma, pi). Baum-Welch converges to the nearest local max.
  3. Select the restart with the highest final log-likelihood.
  4. Apply the best model to the full dataset (2015-2024) to produce
     daily P(Calm) posteriors for the backtest engine.
  5. Visualise P(Calm) time series with annotated market events.
  6. Compute BIC for 2-state vs 3-state as a model-selection sanity check.

Outputs (all in data/HMM/):
  hmm_all_runs.csv      — LL, params, convergence info for every restart
  hmm_best_params.txt   — human-readable summary of the winning model
  hmm_pcalm_daily.csv   — daily P(Calm) on the full dataset (2015-2024)
  hmm_pcalm_plot.png    — P(Calm) time series with market event annotations
  hmm_bic_comparison.txt — BIC comparison: 2-state vs 3-state
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from hmmlearn.hmm import GaussianHMM
import matplotlib
matplotlib.use("Agg")          # non-interactive backend, safe for scripts
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

from experiment.paths import DATA_DIR

# ── Output directory ───────────────────────────────────────────────────────────
HMM_DIR = DATA_DIR / "HMM"
HMM_DIR.mkdir(exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
TRAIN_START  = "2015-01-01"
TRAIN_END    = "2020-12-31"
N_STATES     = 2
N_RESTARTS   = 20
N_ITER       = 200    # max EM iterations per restart (convergence usually < 50)
TOL          = 1e-4   # stop when LL improvement < tol between iterations


def _fit_one(X: np.ndarray, seed: int) -> tuple[GaussianHMM, float]:
    """Fit a single HMM from one random initialisation. Returns (model, LL)."""
    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="diag",   # equivalent to full for 1-D data
        n_iter=N_ITER,
        tol=TOL,
        random_state=seed,
        init_params="stmc",       # hmmlearn randomly inits: s=startprob,
                                  # t=transmat, m=means, c=covars
    )
    model.fit(X)
    ll = model.score(X)           # total log-likelihood (not per-sample)
    return model, ll


def train_hmm(df: pd.DataFrame) -> tuple[GaussianHMM, pd.DataFrame]:
    """
    Run N_RESTARTS random initialisations on the training window.

    Returns
    -------
    best_model : GaussianHMM
    runs_df    : DataFrame with one row per restart — for manual inspection
    """
    mask  = (df.index >= TRAIN_START) & (df.index <= TRAIN_END)
    X_train = df.loc[mask, "log_return"].values.reshape(-1, 1)
    print(f"Training window: {TRAIN_START} to {TRAIN_END}  ({len(X_train)} days)")

    seeds = np.random.default_rng(42).integers(0, 10_000, size=N_RESTARTS)
    records = []
    best_ll, best_model = -np.inf, None

    for i, seed in enumerate(seeds):
        model, ll = _fit_one(X_train, int(seed))

        # Extract parameters — sort states so State 0 = lower vol (Calm)
        means  = model.means_.flatten()          # [mu_0, mu_1]
        stds   = np.sqrt(model.covars_.flatten()) # [sigma_0, sigma_1]
        order  = np.argsort(stds)                # Calm = smaller sigma
        A      = model.transmat_[np.ix_(order, order)]
        mu     = means[order]
        sigma  = stds[order]
        pi     = model.startprob_[order]

        records.append({
            "restart":       i + 1,
            "seed":          int(seed),
            "log_likelihood": round(ll, 4),
            "converged":     model.monitor_.converged,
            "n_iter_used":   len(model.monitor_.history),
            # Calm state (lower vol)
            "calm_mu":       round(mu[0], 6),
            "calm_sigma":    round(sigma[0], 6),
            "calm_stay_prob":round(A[0, 0], 4),  # P(Calm→Calm)
            "calm_pi":       round(pi[0], 4),
            # Turbulent state (higher vol)
            "turb_mu":       round(mu[1], 6),
            "turb_sigma":    round(sigma[1], 6),
            "turb_stay_prob":round(A[1, 1], 4),  # P(Turb→Turb)
            "turb_pi":       round(pi[1], 4),
        })

        if ll > best_ll:
            best_ll    = ll
            best_model = model
            best_seed  = int(seed)

        print(f"  Restart {i+1:2d}  seed={int(seed):5d}  LL={ll:10.2f}"
              f"  converged={model.monitor_.converged}"
              f"  iters={len(model.monitor_.history):3d}")

    runs_df = pd.DataFrame(records).sort_values("log_likelihood", ascending=False)
    print(f"\nBest restart: seed={best_seed}  LL={best_ll:.2f}")
    return best_model, runs_df


def apply_model(model: GaussianHMM, df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute posterior P(Calm) for every day in df using the best model.
    'Calm' is defined as the state with the smaller emission sigma.
    """
    X_full = df["log_return"].values.reshape(-1, 1)
    posteriors = model.predict_proba(X_full)   # shape (T, 2)

    # Identify which column is Calm (lower sigma)
    stds      = np.sqrt(model.covars_.flatten())
    calm_idx  = int(np.argmin(stds))

    out = df[["log_return", "vix_close"]].copy()
    out["p_calm"]  = posteriors[:, calm_idx]
    out["p_turb"]  = posteriors[:, 1 - calm_idx]
    out["state"]   = np.where(out["p_calm"] >= 0.5, "Calm", "Turbulent")
    return out


def save_outputs(best_model: GaussianHMM, runs_df: pd.DataFrame,
                 pcalm_df: pd.DataFrame) -> None:
    """Write all three output files to data/HMM/."""

    # 1. All runs summary
    runs_path = HMM_DIR / "hmm_all_runs.csv"
    runs_df.to_csv(runs_path, index=False)
    print(f"\nSaved all-runs table → {runs_path}")

    # 2. Best model human-readable summary
    stds     = np.sqrt(best_model.covars_.flatten())
    means    = best_model.means_.flatten()
    order    = np.argsort(stds)
    A        = best_model.transmat_[np.ix_(order, order)]
    mu       = means[order]
    sigma    = stds[order]
    pi       = best_model.startprob_[order]
    best_ll  = runs_df["log_likelihood"].iloc[0]

    lines = [
        "=" * 52,
        "Best HMM Model — Parameter Summary",
        "=" * 52,
        f"Training window : {TRAIN_START} to {TRAIN_END}",
        f"Best log-likelihood : {best_ll:.4f}",
        f"Restarts run        : {N_RESTARTS}",
        f"Max EM iterations   : {N_ITER}   tol={TOL}",
        "",
        "── State 0: CALM ──────────────────────────────",
        f"  mu (daily return)  : {mu[0]:+.6f}",
        f"  sigma (daily vol)  : {sigma[0]:.6f}",
        f"  annualised vol     : {sigma[0]*np.sqrt(252):.4f}",
        f"  P(Calm → Calm)     : {A[0,0]:.4f}",
        f"  expected duration  : {1/(1-A[0,0]):.1f} days",
        f"  start probability  : {pi[0]:.4f}",
        "",
        "── State 1: TURBULENT ─────────────────────────",
        f"  mu (daily return)  : {mu[1]:+.6f}",
        f"  sigma (daily vol)  : {sigma[1]:.6f}",
        f"  annualised vol     : {sigma[1]*np.sqrt(252):.4f}",
        f"  P(Turb → Turb)     : {A[1,1]:.4f}",
        f"  expected duration  : {1/(1-A[1,1]):.1f} days",
        f"  start probability  : {pi[1]:.4f}",
        "",
        "── Transition Matrix ──────────────────────────",
        f"  [Calm→Calm  {A[0,0]:.4f}]  [Calm→Turb  {A[0,1]:.4f}]",
        f"  [Turb→Calm  {A[1,0]:.4f}]  [Turb→Turb  {A[1,1]:.4f}]",
        "=" * 52,
    ]
    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    params_path = HMM_DIR / "hmm_best_params.txt"
    params_path.write_text(summary_text)
    print(f"Saved best-model summary → {params_path}")

    # 3. Daily P(Calm)
    pcalm_path = HMM_DIR / "hmm_pcalm_daily.csv"
    pcalm_df.to_csv(pcalm_path)
    print(f"Saved daily P(Calm) → {pcalm_path}  ({len(pcalm_df)} rows)")

    # 4. Quick regime distribution
    print("\nRegime distribution (full dataset):")
    print(pcalm_df["state"].value_counts().to_string())
    calm_frac = (pcalm_df["state"] == "Calm").mean()
    print(f"  Calm fraction: {calm_frac:.1%}")


# ── Visualisation ─────────────────────────────────────────────────────────────

# Key market events to annotate (date, label, vertical position hint)
_EVENTS = [
    ("2015-08-24", "2015 Flash\nCrash"),
    ("2016-02-11", "2016 Oil\nSelloff"),
    ("2018-02-05", "VIXplosion"),
    ("2018-12-24", "2018 Q4\nSelloff"),
    ("2020-02-20", "COVID\nCrash"),
    ("2020-03-23", "COVID\nBottom"),
    ("2022-01-03", "2022 Bear\nMarket"),
    ("2022-10-12", "2022\nBottom"),
]

def plot_pcalm(pcalm_df: pd.DataFrame, spy_df: pd.DataFrame) -> None:
    """
    Three-panel figure saved to data/HMM/hmm_pcalm_plot.png:
      Panel 1: SPY close price with Turbulent periods shaded
      Panel 2: P(Calm) posterior probability
      Panel 3: VIX close (reference)
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.5, 1]})
    fig.suptitle("HMM Regime Detection — 2-State Gaussian HMM on SPY Log-Returns\n"
                 "Training: 2015–2020  |  Display: 2015–2024",
                 fontsize=13, fontweight="bold", y=0.98)

    dates = pcalm_df.index
    turb_mask = pcalm_df["state"] == "Turbulent"

    # Identify the SPY close column
    spy_close_col = next(c for c in spy_df.columns if "close" in c.lower())
    spy_close = spy_df[spy_close_col].reindex(dates)

    # ── Panel 1: SPY price ────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(dates, spy_close, color="#1f77b4", linewidth=0.9, label="SPY Close")
    # Shade Turbulent periods
    _shade_turbulent(ax1, dates, turb_mask)
    ax1.set_ylabel("SPY Price (USD)", fontsize=10)
    ax1.legend(handles=[
        plt.Line2D([0], [0], color="#1f77b4", lw=1.5, label="SPY Close"),
        Patch(facecolor="#e74c3c", alpha=0.25, label="Turbulent regime"),
    ], fontsize=8, loc="upper left")
    ax1.grid(axis="y", alpha=0.3)

    # Annotate events on Panel 1
    spy_min = spy_close.min()
    spy_max = spy_close.max()
    for date_str, label in _EVENTS:
        dt = pd.Timestamp(date_str)
        if dt < dates[0] or dt > dates[-1]:
            continue
        ax1.axvline(dt, color="gray", linewidth=0.7, linestyle="--", alpha=0.6)
        # Alternate label height to reduce overlap
        ypos = spy_min + (spy_max - spy_min) * 0.05
        ax1.text(dt, ypos, label, fontsize=6, color="gray",
                 ha="center", va="bottom", rotation=90,
                 bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.5, ec="none"))

    # Train/test boundary
    for boundary, blabel in [("2020-12-31", "Train end"), ("2022-01-03", "Val end")]:
        ax1.axvline(pd.Timestamp(boundary), color="black",
                    linewidth=1.2, linestyle=":", alpha=0.8)
        ax1.text(pd.Timestamp(boundary), spy_max * 0.97, blabel,
                 fontsize=7, ha="right", color="black")

    # ── Panel 2: P(Calm) ─────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.fill_between(dates, pcalm_df["p_calm"], 0,
                     where=pcalm_df["p_calm"] >= 0.5,
                     color="#2ecc71", alpha=0.5, label="P(Calm) ≥ 0.5")
    ax2.fill_between(dates, pcalm_df["p_calm"], 0,
                     where=pcalm_df["p_calm"] < 0.5,
                     color="#e74c3c", alpha=0.4, label="P(Calm) < 0.5")
    ax2.plot(dates, pcalm_df["p_calm"], color="black", linewidth=0.5, alpha=0.6)
    ax2.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("P(Calm)", fontsize=10)
    ax2.legend(fontsize=8, loc="lower left")
    ax2.grid(axis="y", alpha=0.3)
    _shade_turbulent(ax2, dates, turb_mask)

    # ── Panel 3: VIX ─────────────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.plot(dates, pcalm_df["vix_close"], color="#e67e22", linewidth=0.8)
    ax3.axhline(20, color="gray", linewidth=0.7, linestyle="--", alpha=0.6)
    ax3.axhline(30, color="#e74c3c", linewidth=0.7, linestyle="--", alpha=0.6)
    ax3.set_ylabel("VIX", fontsize=10)
    ax3.text(dates[-1], 20, " VIX=20", fontsize=7, color="gray", va="center")
    ax3.text(dates[-1], 30, " VIX=30", fontsize=7, color="#e74c3c", va="center")
    ax3.grid(axis="y", alpha=0.3)

    # X-axis formatting
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0, ha="center")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = HMM_DIR / "hmm_pcalm_plot.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved P(Calm) plot → {out_path}")


def _shade_turbulent(ax, dates, turb_mask):
    """Fill Turbulent contiguous spans as red bands on the given axis."""
    in_span = False
    span_start = None
    for i, (dt, is_turb) in enumerate(zip(dates, turb_mask)):
        if is_turb and not in_span:
            span_start = dt
            in_span = True
        elif not is_turb and in_span:
            ax.axvspan(span_start, dt, color="#e74c3c", alpha=0.15, linewidth=0)
            in_span = False
    if in_span:
        ax.axvspan(span_start, dates[-1], color="#e74c3c", alpha=0.15, linewidth=0)


# ── BIC comparison ─────────────────────────────────────────────────────────────

def compute_bic(ll: float, n_params: int, n_obs: int) -> float:
    return -2 * ll + n_params * np.log(n_obs)


def bic_comparison(df: pd.DataFrame) -> None:
    """
    Fit best-of-20 2-state and 3-state HMMs on training data,
    compare BIC, save result to data/HMM/hmm_bic_comparison.txt.

    Parameter counts:
      2-state: 2 mu + 2 sigma + 2 free transition probs + 1 free init = 7
      3-state: 3 mu + 3 sigma + 6 free transition probs + 2 free init = 14
    """
    print("\nRunning BIC comparison (2-state vs 3-state)...")
    mask = (df.index >= TRAIN_START) & (df.index <= TRAIN_END)
    X    = df.loc[mask, "log_return"].values.reshape(-1, 1)
    n    = len(X)
    seeds = np.random.default_rng(42).integers(0, 10_000, size=N_RESTARTS)

    results = {}
    for n_states in (2, 3):
        best_ll = -np.inf
        for seed in seeds:
            m = GaussianHMM(n_components=n_states, covariance_type="diag",
                            n_iter=N_ITER, tol=TOL, random_state=int(seed))
            m.fit(X)
            ll = m.score(X)
            if ll > best_ll:
                best_ll = ll
        # Free params: n_states means + n_states variances
        #              + n_states*(n_states-1) free transition probs
        #              + (n_states-1) free initial probs
        k = (n_states          # mu
             + n_states        # sigma^2
             + n_states * (n_states - 1)   # transition (each row: n_states-1 free)
             + (n_states - 1)) # initial prob
        bic = compute_bic(best_ll, k, n)
        results[n_states] = {"best_ll": best_ll, "k": k, "bic": bic}
        print(f"  {n_states}-state: best LL={best_ll:.4f}  k={k}  BIC={bic:.4f}")

    delta_bic = results[2]["bic"] - results[3]["bic"]
    # Positive delta → 2-state has higher BIC → 3-state preferred by BIC
    # Negative delta → 2-state has lower BIC  → 2-state preferred by BIC
    if delta_bic < -10:
        verdict = "2-state strongly preferred (ΔBIC = {:.1f})".format(delta_bic)
    elif delta_bic < 0:
        verdict = "2-state weakly preferred (ΔBIC = {:.1f})".format(delta_bic)
    elif delta_bic < 10:
        verdict = "3-state weakly preferred (ΔBIC = {:.1f})".format(delta_bic)
    else:
        verdict = "3-state strongly preferred (ΔBIC = {:.1f})".format(delta_bic)

    lines = [
        "=" * 52,
        "BIC Model Comparison: 2-State vs 3-State HMM",
        "=" * 52,
        f"Training data: {TRAIN_START} to {TRAIN_END}  (n={n})",
        f"Restarts per model: {N_RESTARTS}",
        "",
        f"{'Model':<12} {'Best LL':>12} {'k (params)':>12} {'BIC':>12}",
        "-" * 52,
        f"{'2-state':<12} {results[2]['best_ll']:>12.4f} {results[2]['k']:>12d} {results[2]['bic']:>12.4f}",
        f"{'3-state':<12} {results[3]['best_ll']:>12.4f} {results[3]['k']:>12d} {results[3]['bic']:>12.4f}",
        "-" * 52,
        f"ΔBIC (2-state minus 3-state) = {delta_bic:.4f}",
        f"Verdict: {verdict}",
        "",
        "Interpretation guide:",
        "  |ΔBIC| < 2   : negligible difference",
        "  |ΔBIC| 2-6   : weak evidence",
        "  |ΔBIC| 6-10  : moderate evidence",
        "  |ΔBIC| > 10  : strong evidence",
        "=" * 52,
    ]
    text = "\n".join(lines)
    print("\n" + text)
    bic_path = HMM_DIR / "hmm_bic_comparison.txt"
    bic_path.write_text(text)
    print(f"Saved BIC comparison → {bic_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 52)
    print("PRISM — HMM Regime Detection")
    print("=" * 52)

    # Load full dataset
    df = pd.read_csv(DATA_DIR / "spy_vix_daily.csv", index_col="date", parse_dates=True)

    # Train
    best_model, runs_df = train_hmm(df)

    # Apply to full dataset
    pcalm_df = apply_model(best_model, df)

    # Save params + CSV
    save_outputs(best_model, runs_df, pcalm_df)

    # Visualise P(Calm) time series
    plot_pcalm(pcalm_df, df)

    # BIC comparison: 2-state vs 3-state
    bic_comparison(df)

    print("\n=== Done ===")
