"""
run_all.py — Run the full PRiSM pipeline (SPY, options, 200 Optuna trials).
Identical to original prism except: 200 trials + equity timing comparison.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print("=" * 60)
    print("PRiSM Experiment — 200-trial consistency check")
    print("=" * 60)

    from experiment.paths import DATA_DIR, RESULTS_DIR
    print(f"  Data dir   : {DATA_DIR}")
    print(f"  Results dir: {RESULTS_DIR}")

    # Step 1: Data collection (SPY + VIX + skew + option prices)
    print("\n" + "=" * 60 + "\nSTEP 1/5: Data Collection\n" + "=" * 60)
    from experiment.data_collection import download_spy_vix, calibrate_skew_multipliers
    from experiment.data_collection import build_option_price_history, skew_correction_diagnostic
    spy_vix = download_spy_vix()
    skew_fn = calibrate_skew_multipliers()
    build_option_price_history(spy_vix, skew_fn)
    skew_correction_diagnostic(spy_vix, skew_fn)

    # Step 2: HMM
    print("\n" + "=" * 60 + "\nSTEP 2/5: HMM Regime Detection\n" + "=" * 60)
    from experiment.hmm_regime import train_hmm, apply_model, save_outputs, plot_pcalm, bic_comparison
    import pandas as pd
    df = pd.read_csv(DATA_DIR / "spy_vix_daily.csv", index_col="date", parse_dates=True)
    best_model, runs_df = train_hmm(df)
    pcalm_df = apply_model(best_model, df)
    save_outputs(best_model, runs_df, pcalm_df)
    plot_pcalm(pcalm_df, df)
    bic_comparison(df)

    # Step 3: XGBoost (200 trials)
    print("\n" + "=" * 60 + "\nSTEP 3/5: XGBoost (200 trials)\n" + "=" * 60)
    from experiment.xgboost_train import (
        load_data, build_features, split_data, run_optuna,
        plot_convergence, plot_param_importance, save_params_summary, train_final_model,
    )
    xdf = load_data()
    feat = build_features(xdf)
    X_tr, y_tr, X_v, y_v, X_te, y_te = split_data(feat)
    study = run_optuna(X_tr, y_tr, n_trials=200, n_splits=5)
    plot_convergence(study)
    plot_param_importance(study)
    save_params_summary(study)
    train_final_model(study.best_params, X_tr, y_tr, X_v, y_v, X_te, y_te)

    # Step 4: Threshold grid search with block bootstrap
    print("\n" + "=" * 60 + "\nSTEP 4/5: Robust Threshold Grid Search\n" + "=" * 60)
    from experiment.threshold_grid import (
        load_backtest_data, load_skew_fn, attach_p_safe,
        plot_psafe_distribution, run_threshold_grid_robust,
        _contour_plot, run_backtest, GRID_DIR,
    )
    import numpy as np
    feat_bt = load_backtest_data()
    skew_fn2 = load_skew_fn()
    feat_bt = attach_p_safe(feat_bt)
    feat_val = feat_bt[(feat_bt.index >= "2020-01-01") & (feat_bt.index < "2022-01-01")]
    plot_psafe_distribution(feat_val)
    tc_grid = np.round(np.arange(0.50, 0.96, 0.01), 2)
    ts_grid = np.round(np.arange(0.50, 0.96, 0.01), 2)
    results = run_threshold_grid_robust(
        feat_val, skew_fn2, tc_grid, ts_grid,
        block_size=21, n_bootstrap=200,
    )
    results.to_csv(GRID_DIR / "threshold_grid_results.csv", index=False)

    best_raw = results.loc[results["sharpe"].idxmax()]
    best_rob = results.loc[results["robust_sharpe"].idxmax()]

    _contour_plot(results, tc_grid, ts_grid, metric="sharpe",
                  title="Raw Sharpe — Threshold Grid (Val 2020–2022)",
                  cmap="RdYlGn", fname="threshold_sharpe_raw_contour.png",
                  best_tc=best_raw["tc"], best_ts=best_raw["ts"])
    _contour_plot(results, tc_grid, ts_grid, metric="robust_sharpe",
                  title="Robust Sharpe (5th pctl) — Val 2020–2022",
                  cmap="RdYlGn", fname="threshold_sharpe_robust_contour.png",
                  best_tc=best_rob["tc"], best_ts=best_rob["ts"])

    print(f"\n  Raw Sharpe best:    tc={best_raw['tc']:.2f}  ts={best_raw['ts']:.2f}  "
          f"Sharpe={best_raw['sharpe']:.4f}")
    print(f"  Robust Sharpe best: tc={best_rob['tc']:.2f}  ts={best_rob['ts']:.2f}  "
          f"Robust={best_rob['robust_sharpe']:.4f}  Raw={best_rob['sharpe']:.4f}")

    summary = (
        f"Best threshold pair (Robust Sharpe — block bootstrap 5th pctl)\n"
        f"{'='*55}\n"
        f"  threshold_calm   : {best_rob['tc']:.2f}\n"
        f"  threshold_safe   : {best_rob['ts']:.2f}\n"
        f"  Robust Sharpe    : {best_rob['robust_sharpe']:.4f}\n"
        f"  Raw Sharpe       : {best_rob['sharpe']:.4f}\n"
        f"  Annual return    : {best_rob['annual_return']:.2%}\n"
        f"  Total return     : {best_rob['total_return']:.2%}\n"
        f"  Ann. volatility  : {best_rob['vol']:.2%}\n"
        f"  Max drawdown     : {best_rob['max_dd']:.2%}\n"
    )
    print(summary)
    (GRID_DIR / "threshold_best.txt").write_text(summary)

    # Step 5: Test evaluation (options + equity timing + B&H)
    print("\n" + "=" * 60 + "\nSTEP 5/5: Test Set Evaluation\n" + "=" * 60)
    # Run test_evaluation as script
    import subprocess
    subprocess.run([sys.executable, "-m", "experiment.test_evaluation"], check=True)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
