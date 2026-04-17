import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TRAIN_START = "2005-01-01"
TRAIN_END = "2015-12-31"
VALID_START = "2016-01-01"
VALID_END = "2018-12-31"
TEST_START = "2019-01-01"
TEST_END = "2024-12-31"


@dataclass
class PreprocessConfig:
    input_path: Path
    output_root: Path
    target_col: str = "ret_exc_lead1m"
    date_col: str = "eom"
    id_col: str = "id"
    skew_threshold: float = 1.0
    winsor_lower_q: float = 0.01
    winsor_upper_q: float = 0.99


def parse_args() -> PreprocessConfig:
    parser = argparse.ArgumentParser(description="Preprocess JKP slim dataset for FINA4713 project.")
    parser.add_argument(
        "--input_path",
        type=str,
        default="f:/Files/code/2026_spring_term/FINA4713/Project/data/jkp_data_slim.parquet",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="f:/Files/code/2026_spring_term/FINA4713/Project/data",
    )
    parser.add_argument("--target_col", type=str, default="ret_exc_lead1m")
    parser.add_argument("--date_col", type=str, default="eom")
    parser.add_argument("--id_col", type=str, default="id")
    parser.add_argument("--skew_threshold", type=float, default=1.0)
    parser.add_argument("--winsor_lower_q", type=float, default=0.01)
    parser.add_argument("--winsor_upper_q", type=float, default=0.99)
    args = parser.parse_args()
    return PreprocessConfig(
        input_path=Path(args.input_path),
        output_root=Path(args.output_root),
        target_col=args.target_col,
        date_col=args.date_col,
        id_col=args.id_col,
        skew_threshold=args.skew_threshold,
        winsor_lower_q=args.winsor_lower_q,
        winsor_upper_q=args.winsor_upper_q,
    )


def temporal_split(df: pd.DataFrame, date_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    date_series = pd.to_datetime(df[date_col])
    train = df[(date_series >= TRAIN_START) & (date_series <= TRAIN_END)].copy()
    valid = df[(date_series >= VALID_START) & (date_series <= VALID_END)].copy()
    test = df[(date_series >= TEST_START) & (date_series <= TEST_END)].copy()
    return train, valid, test


def select_predictors(df: pd.DataFrame, target_col: str, date_col: str, id_col: str) -> list[str]:
    excluded = {target_col, date_col, id_col}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    predictors = [col for col in numeric_cols if col not in excluded]
    if not predictors:
        raise ValueError("No numeric predictors available after exclusion.")
    return predictors


def fit_preprocess_params(
    train_df: pd.DataFrame,
    predictors: list[str],
    skew_threshold: float,
    winsor_lower_q: float,
    winsor_upper_q: float,
) -> dict:
    train_x = train_df[predictors].copy()

    skewness = train_x.skew(numeric_only=True)
    log_cols = skewness[skewness.abs() >= skew_threshold].index.tolist()

    transformed = train_x.copy()
    for col in log_cols:
        transformed[col] = np.sign(transformed[col]) * np.log1p(np.abs(transformed[col]))

    lower = transformed.quantile(winsor_lower_q)
    upper = transformed.quantile(winsor_upper_q)
    winsorized = transformed.clip(lower=lower, upper=upper, axis=1)

    medians = winsorized.median()
    imputed = winsorized.fillna(medians)

    means = imputed.mean()
    stds = imputed.std(ddof=0).replace(0, 1.0)

    params = {
        "predictors": predictors,
        "log_cols": log_cols,
        "winsor_lower": lower.to_dict(),
        "winsor_upper": upper.to_dict(),
        "impute_median": medians.to_dict(),
        "scale_mean": means.to_dict(),
        "scale_std": stds.to_dict(),
    }
    return params


def apply_preprocess(df: pd.DataFrame, params: dict, date_col: str, id_col: str, target_col: str) -> pd.DataFrame:
    out = df.copy()
    predictors = params["predictors"]
    x = out[predictors].copy()

    for col in params["log_cols"]:
        x[col] = np.sign(x[col]) * np.log1p(np.abs(x[col]))

    lower = pd.Series(params["winsor_lower"])
    upper = pd.Series(params["winsor_upper"])
    x = x.clip(lower=lower, upper=upper, axis=1)

    medians = pd.Series(params["impute_median"])
    x = x.fillna(medians)

    means = pd.Series(params["scale_mean"])
    stds = pd.Series(params["scale_std"]).replace(0, 1.0)
    x = (x - means) / stds

    keep_cols = [id_col, date_col, target_col]
    out_final = out[keep_cols].copy()
    out_final[predictors] = x[predictors]
    return out_final


def main() -> None:
    cfg = parse_args()
    train_out_dir = cfg.output_root / "train"
    valid_out_dir = cfg.output_root / "val"
    test_out_dir = cfg.output_root / "test"
    train_out_dir.mkdir(parents=True, exist_ok=True)
    valid_out_dir.mkdir(parents=True, exist_ok=True)
    test_out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cfg.input_path)
    train_df, valid_df, test_df = temporal_split(df, cfg.date_col)
    predictors = select_predictors(train_df, cfg.target_col, cfg.date_col, cfg.id_col)

    params = fit_preprocess_params(
        train_df=train_df,
        predictors=predictors,
        skew_threshold=cfg.skew_threshold,
        winsor_lower_q=cfg.winsor_lower_q,
        winsor_upper_q=cfg.winsor_upper_q,
    )

    train_processed = apply_preprocess(train_df, params, cfg.date_col, cfg.id_col, cfg.target_col)
    valid_processed = apply_preprocess(valid_df, params, cfg.date_col, cfg.id_col, cfg.target_col)
    test_processed = apply_preprocess(test_df, params, cfg.date_col, cfg.id_col, cfg.target_col)

    train_processed.to_parquet(train_out_dir / "train_processed.parquet", index=False)
    valid_processed.to_parquet(valid_out_dir / "val_processed.parquet", index=False)
    test_processed.to_parquet(test_out_dir / "test_processed.parquet", index=False)

    summary = {
        "input_path": str(cfg.input_path),
        "n_rows_total": int(len(df)),
        "n_train": int(len(train_processed)),
        "n_valid": int(len(valid_processed)),
        "n_test": int(len(test_processed)),
        "output_root": str(cfg.output_root),
        "train_output_path": str(train_out_dir / "train_processed.parquet"),
        "valid_output_path": str(valid_out_dir / "val_processed.parquet"),
        "test_output_path": str(test_out_dir / "test_processed.parquet"),
        "n_predictors": int(len(params["predictors"])),
        "predictors": params["predictors"],
        "log_transformed_predictors": params["log_cols"],
        "split": {
            "train": [TRAIN_START, TRAIN_END],
            "valid": [VALID_START, VALID_END],
            "test": [TEST_START, TEST_END],
        },
    }

    with (cfg.output_root / "preprocess_params.json").open("w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=True, indent=2)

    with (cfg.output_root / "preprocess_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    print("Preprocessing completed.")
    print(f"Input: {cfg.input_path}")
    print(f"Output root: {cfg.output_root}")
    print(f"Train output: {train_out_dir / 'train_processed.parquet'}")
    print(f"Valid output: {valid_out_dir / 'val_processed.parquet'}")
    print(f"Test output: {test_out_dir / 'test_processed.parquet'}")
    print(f"Train rows: {len(train_processed)}")
    print(f"Valid rows: {len(valid_processed)}")
    print(f"Test rows: {len(test_processed)}")
    print(f"Predictors: {len(params['predictors'])}")
    print(f"Log transformed predictors: {len(params['log_cols'])}")


if __name__ == "__main__":
    main()
