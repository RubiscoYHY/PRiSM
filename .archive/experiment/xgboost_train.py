"""
xgboost_train.py
================
XGBoost Layer 2: signal refinement for the PRISM short put spread strategy.

Pipeline:
  1. Load spy_vix_daily.csv + hmm_pcalm_daily.csv
  2. Engineer 9 features: RV_5d, RV_20d, RV_60d, RV_ratio, Mom_5d, Mom_20d,
     DD_60d, RSkew_20d, P(Calm)
  3. Construct binary label: did SPY drop >5% in the next 30 calendar days?
  4. Train/val/test split: 2015-2020 / 2020-2022 / 2022-2024
  5. Bayesian optimization via Optuna (100 trials, 5-fold time-series CV, F1)
  6. Retrain on full training set with best params
  7. Save model, convergence plot, hyperparameter importance

Outputs (data/XGBoost/):
  xgb_model.json              - trained XGBoost model
  xgb_best_params.txt         - best hyperparameters + CV F1
  xgb_convergence.png         - best F1 vs. trial number
  xgb_param_importance.png    - Optuna parameter importance bar chart
  xgb_features_train.csv      - feature matrix used for training (for audit)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, classification_report
from experiment.paths import DATA_DIR

# ─────────────────────────────────────────────────────────────
# Output directory
# ─────────────────────────────────────────────────────────────

XGB_DIR = DATA_DIR / "XGBoost"
XGB_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# SECTION 1: Load data
# ─────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """
    Merge spy_vix_daily.csv with hmm_pcalm_daily.csv on date index.
    Returns a single DataFrame with all columns needed for feature engineering.
    """
    spy = pd.read_csv(DATA_DIR / "spy_vix_daily.csv", index_col="date", parse_dates=True)
    hmm = pd.read_csv(DATA_DIR / "HMM" / "hmm_pcalm_daily.csv", index_col="date", parse_dates=True)

    # Keep only p_calm from HMM output
    df = spy.join(hmm[["p_calm"]], how="inner")

    # Identify close column (handles column name variations)
    close_col = next(c for c in df.columns if "spy" in c and "close" in c)
    df = df.rename(columns={close_col: "close"})

    df = df.sort_index()
    print(f"  Loaded {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")
    return df


# ─────────────────────────────────────────────────────────────
# SECTION 2: Feature engineering
# ─────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 9 XGBoost features from the merged DataFrame.

    Features:
        RV_5d      : 5-day rolling realized vol (annualised)
        RV_20d     : 20-day rolling realized vol (annualised)
        RV_60d     : 60-day rolling realized vol (annualised)
        RV_ratio   : RV_20d / RV_60d  (vol acceleration)
        Mom_5d     : 5-day cumulative log return
        Mom_20d    : 20-day cumulative log return
        DD_60d     : drawdown from 60-day rolling high (always <= 0)
        RSkew_20d  : 20-day rolling realized skewness
        p_calm     : HMM posterior P(Calm) from Layer 1

    Label:
        label      : 1 if SPY drops >5% in the next 30 calendar days, else 0
                     (look-ahead uses future prices, computed on full df before splitting)
    """
    r = df["log_return"]
    S = df["close"]

    feat = pd.DataFrame(index=df.index)

    # Realized volatility (annualised std of log returns)
    feat["RV_5d"]  = r.rolling(5).std()  * np.sqrt(252)
    feat["RV_20d"] = r.rolling(20).std() * np.sqrt(252)
    feat["RV_60d"] = r.rolling(60).std() * np.sqrt(252)

    # Vol acceleration: short-term vs long-term vol ratio
    feat["RV_ratio"] = feat["RV_20d"] / feat["RV_60d"].replace(0, np.nan)

    # Momentum: cumulative log return over window
    feat["Mom_5d"]  = r.rolling(5).sum()
    feat["Mom_20d"] = r.rolling(20).sum()

    # Drawdown from 60-day rolling high
    rolling_max_60 = S.rolling(60).max()
    feat["DD_60d"] = (S - rolling_max_60) / rolling_max_60

    # Realized skewness over 20 days
    feat["RSkew_20d"] = r.rolling(20).skew()

    # HMM Layer 1 output
    feat["p_calm"] = df["p_calm"]

    # ── Label: did SPY drop >5% in the next 30 calendar days? ──
    # Use future_min = min close over the next ~21 trading days (~30 cal days)
    future_min = S.shift(-1).rolling(21, min_periods=1).min().shift(-(21-1))
    future_ret = (future_min - S) / S
    feat["label"] = (future_ret < -0.05).astype(int)

    # Drop rows with NaN features (first 60 days) or NaN label (last 21 days)
    feat = feat.dropna()

    print(f"  Feature matrix: {len(feat)} rows, {feat.shape[1]-1} features + label")
    print(f"  Label distribution: {feat['label'].value_counts().to_dict()}")
    crash_rate = feat['label'].mean()
    print(f"  Crash rate: {crash_rate:.1%}")

    return feat


# ─────────────────────────────────────────────────────────────
# SECTION 3: Train / val / test split
# ─────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "RV_5d", "RV_20d", "RV_60d", "RV_ratio",
    "Mom_5d", "Mom_20d", "DD_60d", "RSkew_20d", "p_calm"
]

def split_data(feat: pd.DataFrame):
    """
    Strict temporal split (pre-registered, must not change):
        Train : 2015-01-01 -- 2019-12-31
        Val   : 2020-01-01 -- 2021-12-31
        Test  : 2022-01-01 -- 2024-12-31

    Returns X_train, y_train, X_val, y_val, X_test, y_test as numpy arrays
    plus the corresponding date index slices for reference.
    """
    train = feat[feat.index < "2020-01-01"]
    val   = feat[(feat.index >= "2020-01-01") & (feat.index < "2022-01-01")]
    test  = feat[feat.index >= "2022-01-01"]

    print(f"  Train: {len(train)} rows  ({train.index[0].date()} – {train.index[-1].date()})")
    print(f"  Val  : {len(val)}  rows  ({val.index[0].date()} – {val.index[-1].date()})")
    print(f"  Test : {len(test)} rows  ({test.index[0].date()} – {test.index[-1].date()})")

    X_train, y_train = train[FEATURE_COLS].values, train["label"].values
    X_val,   y_val   = val[FEATURE_COLS].values,   val["label"].values
    X_test,  y_test  = test[FEATURE_COLS].values,  test["label"].values

    # Save feature matrix for audit trail
    train[FEATURE_COLS + ["label"]].to_csv(XGB_DIR / "xgb_features_train.csv")

    return X_train, y_train, X_val, y_val, X_test, y_test


# ─────────────────────────────────────────────────────────────
# SECTION 4: Bayesian optimisation via Optuna
# ─────────────────────────────────────────────────────────────

def run_optuna(X_train: np.ndarray, y_train: np.ndarray,
               n_trials: int = 200, n_splits: int = 5) -> optuna.Study:
    """
    Optimise XGBoost hyperparameters using Optuna TPE sampler.

    Inner loop: 5-fold time-series cross-validation (no data leakage).
    Objective: macro F1 score on validation folds.

    Search space:
        max_depth         : int   [3, 10]
        n_estimators      : int   [50, 500]
        learning_rate     : float [0.01, 0.3]  (log scale)
        subsample         : float [0.5, 1.0]
        colsample_bytree  : float [0.5, 1.0]
        min_child_weight  : int   [1, 10]
        scale_pos_weight  : float [1.0, 20.0]  (handles class imbalance)
        reg_alpha         : float [1e-8, 1.0]  (log scale, L1)
        reg_lambda        : float [1e-8, 1.0]  (log scale, L2)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    default_spw = neg_count / max(pos_count, 1)
    print(f"  Default scale_pos_weight (neg/pos ratio): {default_spw:.1f}")

    def objective(trial):
        params = {
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "n_estimators":      trial.suggest_int("n_estimators", 50, 500),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight":  trial.suggest_float("scale_pos_weight", 1.0, 20.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "objective":   "binary:logistic",
            "eval_metric": "logloss",
            "verbosity":   0,
            "random_state": 42,
        }

        f1_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_vl = X_train[train_idx], X_train[val_idx]
            y_tr, y_vl = y_train[train_idx], y_train[val_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr,
                      eval_set=[(X_vl, y_vl)],
                      verbose=False)
            y_pred = model.predict(X_vl)
            f1_scores.append(f1_score(y_vl, y_pred, average="macro", zero_division=0))

        return np.mean(f1_scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n  Best CV F1 : {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study


# ─────────────────────────────────────────────────────────────
# SECTION 5: Plots
# ─────────────────────────────────────────────────────────────

def plot_convergence(study: optuna.Study) -> None:
    """
    Plot best F1 score vs. trial number and save to XGB_DIR.
    """
    trials_df = study.trials_dataframe()
    # Keep only completed trials with a valid value
    trials_df = trials_df[trials_df["value"].notna()].reset_index(drop=True)
    best_so_far = trials_df["value"].cummax()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(trials_df.index, trials_df["value"],
               alpha=0.35, s=18, color="steelblue", label="Trial F1")
    ax.plot(best_so_far.index, best_so_far.values,
            color="crimson", linewidth=2, label="Best so far")

    ax.axhline(study.best_value, color="crimson", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(len(trials_df) * 0.02, study.best_value + 0.002,
            f"Best F1 = {study.best_value:.4f}", color="crimson", fontsize=9)

    ax.set_xlabel("Trial number")
    ax.set_ylabel("CV macro-F1")
    ax.set_title("Optuna convergence — XGBoost hyperparameter search")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(XGB_DIR / "xgb_convergence.png", dpi=150)
    plt.close()
    print(f"  Saved convergence plot → {XGB_DIR / 'xgb_convergence.png'}")


def plot_param_importance(study: optuna.Study) -> None:
    """
    Compute and plot Optuna parameter importance, save to XGB_DIR.
    Also prints the top parameters to stdout.
    """
    importances = optuna.importance.get_param_importances(study)

    params  = list(importances.keys())
    values  = list(importances.values())
    colors  = ["#c0392b" if v == max(values) else "steelblue" for v in values]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(params[::-1], values[::-1], color=colors[::-1])
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xlabel("Relative importance (fANOVA)")
    ax.set_title("Hyperparameter importance — Optuna")
    ax.set_xlim(0, max(values) * 1.2)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(XGB_DIR / "xgb_param_importance.png", dpi=150)
    plt.close()
    print(f"  Saved importance plot  → {XGB_DIR / 'xgb_param_importance.png'}")

    # Print ranked importance
    print("\n  ── Hyperparameter Importance (ranked) ──")
    cumulative = 0.0
    for rank, (p, v) in enumerate(importances.items(), 1):
        cumulative += v
        print(f"  #{rank:2d}  {p:<22s}  {v:.4f}  (cumulative: {cumulative:.2%})")


# ─────────────────────────────────────────────────────────────
# SECTION 6: Final model — retrain on full training set
# ─────────────────────────────────────────────────────────────

def train_final_model(
    best_params: dict,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
) -> xgb.XGBClassifier:
    """
    Retrain XGBoost on the full training set using the best Optuna params.
    Evaluate on val and test sets (read-only report — no further tuning).
    """
    params = {
        **best_params,
        "objective":   "binary:logistic",
        "eval_metric": "logloss",
        "verbosity":   0,
        "random_state": 42,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    print("\n  ── Validation set (2020–2022) ──")
    y_val_pred = model.predict(X_val)
    print(classification_report(y_val, y_val_pred,
                                 target_names=["Safe", "Crash"],
                                 zero_division=0))

    print("  ── Test set (2022–2024, read-only) ──")
    y_test_pred = model.predict(X_test)
    print(classification_report(y_test, y_test_pred,
                                  target_names=["Safe", "Crash"],
                                  zero_division=0))

    model.save_model(str(XGB_DIR / "xgb_model.json"))
    print(f"  Saved model → {XGB_DIR / 'xgb_model.json'}")
    return model


# ─────────────────────────────────────────────────────────────
# SECTION 7: Save best params summary
# ─────────────────────────────────────────────────────────────

def save_params_summary(study: optuna.Study) -> None:
    lines = [
        "XGBoost Best Hyperparameters",
        "=" * 40,
        f"Best CV macro-F1 : {study.best_value:.4f}",
        f"Trials completed  : {len(study.trials)}",
        "",
        "Parameters:",
    ]
    for k, v in study.best_params.items():
        lines.append(f"  {k:<24s}: {v}")

    summary = "\n".join(lines)
    (XGB_DIR / "xgb_best_params.txt").write_text(summary)
    print(f"\n  Saved params summary → {XGB_DIR / 'xgb_best_params.txt'}")
    print("\n" + summary)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("PRISM — XGBoost Layer 2 Training")
    print("=" * 60)

    print("\n[1/6] Loading data...")
    df = load_data()

    print("\n[2/6] Engineering features...")
    feat = build_features(df)

    print("\n[3/6] Splitting data...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(feat)

    print("\n[4/6] Running Optuna (200 trials, 5-fold time-series CV)...")
    study = run_optuna(X_train, y_train, n_trials=200, n_splits=5)

    print("\n[5/6] Generating plots...")
    plot_convergence(study)
    plot_param_importance(study)
    save_params_summary(study)

    print("\n[6/6] Training final model on full training set...")
    model = train_final_model(
        study.best_params,
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
    )

    print("\n" + "=" * 60)
    print("XGBoost training complete.")
    print(f"All outputs saved to: {XGB_DIR}")
    print("=" * 60)
