import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pandas as pd

class PreprocessConfig:
    input_path: Path
    output_root: Path
    target_col: str = "ret_exc_lead1m"
    date_col: str = "eom"
    id_col: str = "id"
    invalid_val : float = -1e8
    skew_threshold: float = 10.0
    winsor_lower_q: float = 0.01
    winsor_upper_q: float = 0.99
    train_end : pd.Timestamp = pd.Timestamp("2015-12-31")
    valid_end : pd.Timestamp = pd.Timestamp("2018-12-31")
    test_end : pd.Timestamp = pd.Timestamp("2024-12-31")

    def __init__(self,
                 input_path: Path,
                 output_root: Path,
                 target_col: str = "ret_exc_lead1m",
                 date_col: str = "eom",
                 id_col: str = "id",
                 invalid_val: float = -1e8,
                 skew_threshold: float = 10.0,
                 winsor_lower_q: float = 0.01,
                 winsor_upper_q: float = 0.99) -> None:
        self.input_path = input_path
        self.output_root = output_root
        self.target_col = target_col
        self.date_col = date_col
        self.id_col = id_col
        self.invalid_val = invalid_val
        self.skew_threshold = skew_threshold
        self.winsor_lower_q = winsor_lower_q
        self.winsor_upper_q = winsor_upper_q

    def save_json(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)
    
    @staticmethod
    def load_json(path: Path) -> Self :
        with open(path, "r") as f:
            data = json.load(f)
        return PreprocessConfig(**data)

def parse_args() -> PreprocessConfig:
    parser = argparse.ArgumentParser(description="Preprocess JKP slim dataset for FINA4713 project.")
    parser.add_argument(
        "--input_path",
        type=str,
        default="../jkp_data_slim.parquet",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="../split_data",
    )
    parser.add_argument("--target_col", type=str, default="ret_exc_lead1m")
    parser.add_argument("--date_col", type=str, default="eom")
    parser.add_argument("--id_col", type=str, default="id")
    parser.add_argument("--clip_value", type=float, default=None)
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
        clip_value=args.clip_value,
        skew_threshold=args.skew_threshold,
        winsor_lower_q=args.winsor_lower_q,
        winsor_upper_q=args.winsor_upper_q,
    )

def _split(df: pd.DataFrame, config: PreprocessConfig):
    df[config.date_col] = pd.to_datetime(df[config.date_col])
    train_df = df[df[config.date_col] <= config.train_end]
    valid_df = df[(df[config.date_col] > config.train_end) & (df[config.date_col] <= config.valid_end)]
    test_df = df[df[config.date_col] > config.valid_end]
    return train_df, valid_df, test_df

def load_data(config : PreprocessConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] :
    df = pd.read_parquet(config.input_path)
    # remove data with missing target
    df = df[df[config.target_col].notna()].copy()
    # transform non-numeric columns to numeric if possible, otherwise keep as is
    for column in df.select_dtypes(exclude=['number']).columns :
        def _cvt_item(x : None | object | str | pd.Timestamp) -> float | str | pd.Timestamp:
            if x == None : return -config.invalid_val
            elif isinstance(x, str) or isinstance(x, pd.Timestamp) : return x
            else : return float(x)
        df[column] = df[column].apply(_cvt_item)

    # select predictors
    num_cols = df.select_dtypes(include=[np.number]).drop(columns=[config.target_col, config.id_col]).columns.tolist()
    str_cols = df.select_dtypes(exclude=[np.number]).drop(columns=[config.date_col]).columns.tolist()
    print(f"Selected columns: ", *num_cols, sep="\n- ")

    # split data set
    train_df, valid_df, test_df = _split(df, config)

    # transform train_df and use the same transformation for valid_df and test_df
    skewness = train_df[num_cols].skew(numeric_only=True)
    log_cols = skewness[skewness.abs() >= config.skew_threshold].index.tolist()
    print(f"Log-transforming {len(log_cols)} features with skewness above {config.skew_threshold}: ", *log_cols, sep="\n- ")
    def log_transform(df: pd.DataFrame) -> pd.DataFrame:
        output = df.copy()
        for col in log_cols:
            output[col] = np.sign(output[col]) * np.log1p(np.abs(output[col]))
        return df

    train_df, valid_df, test_df = log_transform(train_df), log_transform(valid_df), log_transform(test_df)

    # winsorize features
    low = train_df[num_cols].quantile(config.winsor_lower_q)
    high = train_df[num_cols].quantile(config.winsor_upper_q)
    print(f"Winsorizing features to [{config.winsor_lower_q}, {config.winsor_upper_q}] quantiles: ", *num_cols, sep="\n- ")
    def winsorize(df: pd.DataFrame) :
        output = df.copy()
        output[num_cols] = output[num_cols].clip(lower=low, upper=high, axis=1)
        return output
    
    train_df, valid_df, test_df = winsorize(train_df), winsorize(valid_df), winsorize(test_df)
    
    scaler = StandardScaler()
    train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
    valid_df[num_cols] = scaler.transform(valid_df[num_cols])
    test_df[num_cols] = scaler.transform(test_df[num_cols])

    # convert string columns to category dtype
    for column in str_cols :
        le = LabelEncoder()
        train_df[column] = le.fit_transform(train_df[column])
        valid_df[column] = le.transform(valid_df[column])
        test_df[column] = le.transform(test_df[column])

        print(f"Encoded column '{column}' with {len(le.classes_)} unique values: ", *le.classes_, sep="\n- ")

    return train_df, valid_df, test_df