import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FINA4713 baseline model estimation and evaluation.")
    parser.add_argument(
        "--train_path",
        type=str,
        default="f:/Files/code/2026_spring_term/FINA4713/Project/data/train/train_processed.parquet",
    )
    parser.add_argument(
        "--valid_path",
        type=str,
        default="f:/Files/code/2026_spring_term/FINA4713/Project/data/val/val_processed.parquet",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="f:/Files/code/2026_spring_term/FINA4713/Project/data/test/test_processed.parquet",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="f:/Files/code/2026_spring_term/FINA4713/Project/output/baseline",
    )
    parser.add_argument("--id_col", type=str, default="id")
    parser.add_argument("--date_col", type=str, default="eom")
    parser.add_argument("--target_col", type=str, default="ret_exc_lead1m")
    parser.add_argument(
        "--ridge_alphas",
        type=str,
        default="0.01,0.1,1.0,10.0,100.0",
        help="Comma-separated candidate alphas for Ridge.",
    )
    return parser.parse_args()


def load_data(path: str, date_col: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df[date_col] = pd.to_datetime(df[date_col])
    return df.sort_values([date_col]).reset_index(drop=True)


def get_feature_cols(df: pd.DataFrame, id_col: str, date_col: str, target_col: str) -> list[str]:
    excluded = {id_col, date_col, target_col}
    return [c for c in df.columns if c not in excluded]


def prepare_xy(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> tuple[np.ndarray, np.ndarray]:
    x = df[feature_cols].to_numpy(dtype=np.float64)
    y = df[target_col].to_numpy(dtype=np.float64)
    return x, y


def clean_for_modeling(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    n_before = int(len(out))

    # Drop rows with missing target first (required by sklearn and metric computation).
    mask_target_ok = out[target_col].notna().to_numpy()
    out = out.loc[mask_target_ok].copy()

    # Keep only rows with finite features and finite target.
    x = out[feature_cols].to_numpy(dtype=np.float64)
    y = out[target_col].to_numpy(dtype=np.float64)
    mask_finite = np.isfinite(y) & np.isfinite(x).all(axis=1)
    out = out.loc[mask_finite].copy()

    stats = {
        "n_before": n_before,
        "n_after": int(len(out)),
        "n_dropped": int(n_before - len(out)),
    }
    return out.reset_index(drop=True), stats


def oos_r2(y_true: np.ndarray, y_pred: np.ndarray, y_train_mean: float) -> float:
    sse_model = float(np.sum((y_true - y_pred) ** 2))
    sse_null = float(np.sum((y_true - y_train_mean) ** 2))
    if sse_null <= 0:
        return np.nan
    return 1.0 - sse_model / sse_null


def fit_historical_average(train_df: pd.DataFrame, id_col: str, target_col: str) -> tuple[pd.Series, float]:
    stock_means = train_df.groupby(id_col, sort=False)[target_col].mean()
    global_mean = float(train_df[target_col].mean())
    return stock_means, global_mean


def predict_historical_average(
    df: pd.DataFrame,
    stock_means: pd.Series,
    global_mean: float,
    id_col: str,
) -> np.ndarray:
    preds = df[id_col].map(stock_means).fillna(global_mean)
    return preds.to_numpy(dtype=np.float64)


def choose_ridge_alpha(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    alphas: list[float],
) -> tuple[float, dict]:
    val_mse_by_alpha = {}
    best_alpha = alphas[0]
    best_mse = float("inf")
    for alpha in alphas:
        model = Ridge(alpha=alpha, fit_intercept=True, random_state=42)
        model.fit(x_train, y_train)
        pred_valid = model.predict(x_valid)
        mse = mean_squared_error(y_valid, pred_valid)
        val_mse_by_alpha[str(alpha)] = float(mse)
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
    return best_alpha, val_mse_by_alpha


def compute_portfolio_weights(month_df: pd.DataFrame, pred_col: str) -> pd.Series:
    centered = month_df[pred_col] - month_df[pred_col].mean()
    denom = centered.abs().sum()
    if denom <= 0:
        return pd.Series(0.0, index=month_df.index)
    return centered / denom


def evaluate_portfolio(
    test_df: pd.DataFrame,
    date_col: str,
    target_col: str,
    pred_col: str,
) -> tuple[pd.DataFrame, dict]:
    rets = []
    for month, group in test_df.groupby(date_col, sort=True):
        weights = compute_portfolio_weights(group, pred_col)
        ret = float(np.sum(weights.to_numpy() * group[target_col].to_numpy()))
        rets.append({"month": month, "portfolio_ret": ret})

    ret_df = pd.DataFrame(rets).sort_values("month").reset_index(drop=True)
    mean_m = float(ret_df["portfolio_ret"].mean())
    std_m = float(ret_df["portfolio_ret"].std(ddof=0))
    ann_mean = 12.0 * mean_m
    ann_vol = np.sqrt(12.0) * std_m
    sharpe = ann_mean / ann_vol if ann_vol > 0 else np.nan
    metrics = {
        "annualized_mean_excess_return": ann_mean,
        "annualized_volatility": ann_vol,
        "annualized_sharpe": sharpe,
    }
    return ret_df, metrics


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_data(args.train_path, args.date_col)
    valid_df = load_data(args.valid_path, args.date_col)
    test_df = load_data(args.test_path, args.date_col)

    feature_cols = get_feature_cols(train_df, args.id_col, args.date_col, args.target_col)
    train_df, train_clean_stats = clean_for_modeling(train_df, feature_cols, args.target_col)
    valid_df, valid_clean_stats = clean_for_modeling(valid_df, feature_cols, args.target_col)
    test_df, test_clean_stats = clean_for_modeling(test_df, feature_cols, args.target_col)

    x_train, y_train = prepare_xy(train_df, feature_cols, args.target_col)
    x_valid, y_valid = prepare_xy(valid_df, feature_cols, args.target_col)
    x_test, y_test = prepare_xy(test_df, feature_cols, args.target_col)
    y_train_mean = float(y_train.mean())

    # (i) Historical-average benchmark.
    hist_stock_means, hist_global_mean = fit_historical_average(train_df, args.id_col, args.target_col)
    pred_valid_hist = predict_historical_average(valid_df, hist_stock_means, hist_global_mean, args.id_col)
    pred_test_hist = predict_historical_average(test_df, hist_stock_means, hist_global_mean, args.id_col)

    # (ii) OLS baseline.
    ols = LinearRegression(fit_intercept=True)
    ols.fit(x_train, y_train)
    pred_valid_ols = ols.predict(x_valid)
    pred_test_ols = ols.predict(x_test)

    # (iii) Ridge as ML model with validation-set hyperparameter selection.
    alpha_grid = [float(x.strip()) for x in args.ridge_alphas.split(",") if x.strip()]
    best_alpha, val_mse_grid = choose_ridge_alpha(x_train, y_train, x_valid, y_valid, alpha_grid)
    ridge = Ridge(alpha=best_alpha, fit_intercept=True, random_state=42)
    ridge.fit(x_train, y_train)
    pred_valid_ridge = ridge.predict(x_valid)
    pred_test_ridge = ridge.predict(x_test)

    # OOS R^2 with train-mean null model.
    model_rows = []
    model_rows.append(
        {
            "model": "historical_average",
            "valid_mse": float(mean_squared_error(y_valid, pred_valid_hist)),
            "test_oos_r2": oos_r2(y_test, pred_test_hist, y_train_mean),
        }
    )
    model_rows.append(
        {
            "model": "ols",
            "valid_mse": float(mean_squared_error(y_valid, pred_valid_ols)),
            "test_oos_r2": oos_r2(y_test, pred_test_ols, y_train_mean),
        }
    )
    model_rows.append(
        {
            "model": "ridge",
            "valid_mse": float(mean_squared_error(y_valid, pred_valid_ridge)),
            "test_oos_r2": oos_r2(y_test, pred_test_ridge, y_train_mean),
        }
    )
    metrics_df = pd.DataFrame(model_rows)

    # Save test predictions for portfolio construction.
    pred_df = test_df[[args.id_col, args.date_col, args.target_col]].copy()
    pred_df["pred_historical_average"] = pred_test_hist
    pred_df["pred_ols"] = pred_test_ols
    pred_df["pred_ridge"] = pred_test_ridge

    # Portfolio metrics by model.
    port_rows = []
    port_ret_dict = {}
    for model_name, pred_col in [
        ("historical_average", "pred_historical_average"),
        ("ols", "pred_ols"),
        ("ridge", "pred_ridge"),
    ]:
        ret_series_df, port_metrics = evaluate_portfolio(pred_df, args.date_col, args.target_col, pred_col)
        ret_series_df.to_csv(out_dir / f"portfolio_monthly_returns_{model_name}.csv", index=False)
        port_ret_dict[model_name] = ret_series_df
        row = {"model": model_name}
        row.update(port_metrics)
        port_rows.append(row)
    portfolio_df = pd.DataFrame(port_rows)

    # Lightweight textual interpretation template for section 3.4 and 3.5.
    neg_oos_models = metrics_df.loc[metrics_df["test_oos_r2"] < 0, "model"].tolist()
    interpretation = {
        "hyperparameter_choice": {
            "ridge_alphas_tried": alpha_grid,
            "ridge_best_alpha": best_alpha,
            "selection_metric": "validation_mse",
            "explanation": "Ridge alpha controls L2 shrinkage strength; selected alpha minimizes validation MSE.",
        },
        "oos_r2_note": (
            "Out-of-sample R^2 is computed as 1 - SSE_model / SSE_null, where null uses train-period mean return."
        ),
        "negative_oos_r2_explanation": (
            "Negative OOS R^2 means model forecast error exceeds the train-mean benchmark, "
            "which is common in noisy monthly return prediction with low signal-to-noise ratio."
            if neg_oos_models
            else "No model has negative OOS R^2 in this run."
        ),
        "qualitative_trading_constraints": (
            "A monthly long-short strategy based on cross-sectional predictions may face material frictions: "
            "transaction costs from frequent rebalancing, liquidity limits for small-cap names, and turnover-induced slippage. "
            "Practical deployment should include turnover penalties, liquidity filters, and capacity controls."
        ),
    }

    # Persist outputs.
    metrics_df.to_csv(out_dir / "model_oos_metrics.csv", index=False)
    portfolio_df.to_csv(out_dir / "portfolio_metrics.csv", index=False)
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)
    with (out_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "train_path": args.train_path,
                "valid_path": args.valid_path,
                "test_path": args.test_path,
                "n_train": int(len(train_df)),
                "n_valid": int(len(valid_df)),
                "n_test": int(len(test_df)),
                "n_features": int(len(feature_cols)),
                "data_cleaning": {
                    "train": train_clean_stats,
                    "valid": valid_clean_stats,
                    "test": test_clean_stats,
                },
                "models": ["historical_average", "ols", "ridge"],
                "ridge_alpha_grid": alpha_grid,
                "ridge_best_alpha": best_alpha,
                "oos_r2_by_model": dict(
                    zip(metrics_df["model"].tolist(), metrics_df["test_oos_r2"].astype(float).tolist())
                ),
                "portfolio_metrics": portfolio_df.set_index("model").to_dict(orient="index"),
                "interpretation": interpretation,
            },
            f,
            ensure_ascii=True,
            indent=2,
            default=str,
        )

    print("Baseline pipeline finished.")
    print(f"Output directory: {out_dir}")
    print("Saved files:")
    print("- model_oos_metrics.csv")
    print("- portfolio_metrics.csv")
    print("- test_predictions.csv")
    print("- portfolio_monthly_returns_historical_average.csv")
    print("- portfolio_monthly_returns_ols.csv")
    print("- portfolio_monthly_returns_ridge.csv")
    print("- run_summary.json")
    print("\nData cleaning summary:")
    print(f"train: {train_clean_stats}")
    print(f"valid: {valid_clean_stats}")
    print(f"test : {test_clean_stats}")
    print("\nModel OOS metrics:")
    print(metrics_df.to_string(index=False))
    print("\nPortfolio metrics:")
    print(portfolio_df.to_string(index=False))


if __name__ == "__main__":
    main()
