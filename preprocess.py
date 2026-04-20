import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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

def split(df: pd.DataFrame, config: PreprocessConfig):
    df[config.date_col] = pd.to_datetime(df[config.date_col])
    train_df = df[df[config.date_col] <= config.train_end]
    valid_df = df[(df[config.date_col] > config.train_end) & (df[config.date_col] <= config.valid_end)]
    test_df = df[df[config.date_col] > config.valid_end]
    return train_df, valid_df, test_df

def transform(train : pd.DataFrame, val : pd.DataFrame, test : pd.DataFrame, config : PreprocessConfig,
    include : list[str] | None = None,
    exclude : list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] :
    if include is None :
        if exclude is None :
            include = [col for col in train.columns if col not in [config.id_col, config.date_col, config.target_col]]
        else :
            include = [col for col in train.columns if col not in [*exclude, config.id_col, config.date_col, config.target_col]]

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    imputer.fit(train[include])
    
    train[include] = imputer.transform(train[include])
    val[include] = imputer.transform(val[include])
    test[include] = imputer.transform(test[include])

    # log transform for highly skewed features
    skewness = train[include].skew()
    skewed_cols = skewness[skewness.abs() > config.skew_threshold].index.tolist()
    
    def _log_transform(x):
        return np.sign(x) * np.log1p(np.abs(x))

    train[skewed_cols] = train[skewed_cols].apply(_log_transform)
    val[skewed_cols] = val[skewed_cols].apply(_log_transform)
    test[skewed_cols] = test[skewed_cols].apply(_log_transform)

    print(f"Applied log transform to {len(skewed_cols)} skewed features: ", *skewed_cols, sep='\n- ')

    # winsorize to handle outliers
    low = train[include].quantile(config.winsor_lower_q)
    high = train[include].quantile(config.winsor_upper_q)
    train[include] = train[include].clip(low, high, axis=1)
    val[include] = val[include].clip(low, high, axis=1)
    test[include] = test[include].clip(low, high, axis=1)

    scaler.fit(train[include])
    train[include] = scaler.transform(train[include])
    val[include] = scaler.transform(val[include])
    test[include] = scaler.transform(test[include])

    return train, val, test

def load_data(config : PreprocessConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] :
    df = pd.read_parquet(config.input_path)
    # remove data with missing target
    df = df[df[config.target_col].notna()].copy()
    # transform non-numeric columns to numeric if possible, otherwise keep as is
    for column in df.select_dtypes(exclude=['number']).columns :
        def _cvt_item(x : None | object | str | pd.Timestamp) -> float | str | pd.Timestamp:
            if x == None : return config.invalid_val
            elif isinstance(x, str) or isinstance(x, pd.Timestamp) : return x
            else : return float(x)
        df[column] = df[column].apply(_cvt_item)
    df.fillna(config.invalid_val, inplace=True)
    return df

def get_xy(df : pd.DataFrame, config : PreprocessConfig) -> tuple[pd.DataFrame, pd.Series] :
    X = df.drop(columns=[config.target_col, config.id_col, config.date_col])
    y = df[config.target_col]

    return X, y

def onehot(train : pd.DataFrame, val : pd.DataFrame, test : pd.DataFrame, column : str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] :
    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

    train_ohe = ohe.fit_transform(train[[column]])
    val_ohe = ohe.transform(val[[column]])
    test_ohe = ohe.transform(test[[column]])

    idx = ohe.get_feature_names_out([column]).tolist()

    train = pd.concat([train.drop(columns=[column]), pd.DataFrame(train_ohe, index=train.index, columns=idx)], axis=1)
    val = pd.concat([val.drop(columns=[column]), pd.DataFrame(val_ohe, index=val.index, columns=idx)], axis=1)
    test = pd.concat([test.drop(columns=[column]), pd.DataFrame(test_ohe, index=test.index, columns=idx)], axis=1)

    return train, val, test