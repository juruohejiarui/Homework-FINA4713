import argparse
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from typing import Self
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import pandas as pd

class Config:
    input_path: Path
    output_root: Path
    skewed_cols : list[str] | str | None = None
    onehot_cols : list[str] = []
    target_col: str = "ret_exc_lead1m"
    date_col: str = "eom"
    id_col: str = "id"
    grp_idx : pd.Series = pd.Series({})
    sel_num : int = 2
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
                 skewed_cols : list[str] | str | None = None,
                 onehot_cols : list[str] = [],
                 target_col: str = "ret_exc_lead1m",
                 date_col: str = "eom",
                 id_col: str = "id",
                 grp_idx : dict[str, int] = {},
                 sel_num : int = 2,
                 invalid_val: float = -1e8,
                 skew_threshold: float = 10.0,
                 winsor_lower_q: float = 0.01,
                 winsor_upper_q: float = 0.99) -> None:
        self.input_path = input_path
        self.output_root = output_root
        self.skewed_cols = skewed_cols
        self.onehot_cols = onehot_cols
        self.target_col = target_col
        self.date_col = date_col
        self.id_col = id_col
        # grp_idx is a dict mapping group name to group index, used for group-aware splitting
        self.grp_idx = pd.Series(grp_idx, index=grp_idx.keys())
        self.sel_num = sel_num
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
        return Config(**data)

def split(df: pd.DataFrame, config: Config):
    train_df = df[df[config.date_col] <= config.train_end]
    valid_df = df[(df[config.date_col] > config.train_end) & (df[config.date_col] <= config.valid_end)]
    test_df = df[df[config.date_col] > config.valid_end]
    return train_df, valid_df, test_df

def onehot(train : pd.DataFrame, val : pd.DataFrame, test : pd.DataFrame, columns : list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] :
    if len(columns) == 0 : return train, val, test

    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

    train_ohe = ohe.fit_transform(train[columns])
    val_ohe = ohe.transform(val[columns])
    test_ohe = ohe.transform(test[columns])

    idx = ohe.get_feature_names_out(columns).tolist()

    train = pd.concat([train.drop(columns=columns), pd.DataFrame(train_ohe, index=train.index, columns=idx)], axis=1)
    val = pd.concat([val.drop(columns=columns), pd.DataFrame(val_ohe, index=val.index, columns=idx)], axis=1)
    test = pd.concat([test.drop(columns=columns), pd.DataFrame(test_ohe, index=test.index, columns=idx)], axis=1)

    return train, val, test

def transform(train : pd.DataFrame, val : pd.DataFrame, test : pd.DataFrame, config : Config,
    include : list[str] | None = None,
    exclude : list[str] | None = None,
    log : bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] :
    
    train = train.copy()
    val = val.copy()
    test = test.copy()

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
    if config.skewed_cols == None :
        skewness = train[include].skew()
        skewed_cols = skewness[skewness.abs() > config.skew_threshold].index.tolist()
    else :
        skewed_cols = config.skewed_cols
    
    def _log_transform(x):
        return np.sign(x) * np.log1p(np.abs(x))

    if log :
        print(f"Applied log transform to {len(skewed_cols)} skewed features: ", 
          *[f"- {col}: {train[col].skew()}" for col in skewed_cols], sep='\n')

    # plot top 10 (abs) skewed features before and after log transform
    if log :
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        skewness_before = train[include].skew().abs().sort_values(ascending=False)
        skewness_before.head(10).plot(kind='bar')
        plt.title('Top 10 (abs) Skewed Features Before Log Transform')
        plt.ylabel('Absolute Skewness')
        plt.subplot(1, 2, 2)
        skewness_after = train[include].apply(_log_transform).skew().abs().sort_values(ascending=False)
        skewness_after.head(10).plot(kind='bar')
        plt.title('Top 10 (abs) Skewed Features After Log Transform')
        plt.ylabel('Absolute Skewness')
        plt.tight_layout()
        plt.show()

    train.loc[:, skewed_cols] = train[skewed_cols].apply(_log_transform)
    val.loc[:, skewed_cols] = val[skewed_cols].apply(_log_transform)
    test.loc[:, skewed_cols] = test[skewed_cols].apply(_log_transform)

    # winsorize to handle outliers
    low = train[include].quantile(config.winsor_lower_q)
    high = train[include].quantile(config.winsor_upper_q)
    train.loc[:, include] = train[include].clip(low, high, axis=1)
    val.loc[:, include] = val[include].clip(low, high, axis=1)
    test.loc[:, include] = test[include].clip(low, high, axis=1)

    scaler.fit(train[include])
    train.loc[:, include] = scaler.transform(train[include])
    val.loc[:, include] = scaler.transform(val[include])
    test.loc[:, include] = scaler.transform(test[include])

    # transform onehot columns
    train, val, test = onehot(train, val, test, config.onehot_cols)

    return train, val, test

def load_data(config : Config) -> pd.DataFrame :
    df = pd.read_parquet(config.input_path)
    # remove data with missing target
    df = df[df[config.target_col].notna()].copy()
    df[config.date_col] = pd.to_datetime(df[config.date_col])
    return df

def get_xy(df : pd.DataFrame, config : Config, excludes : list[str] = []) -> tuple[pd.DataFrame, pd.Series] :
    X = df.drop(columns=[config.target_col, config.id_col, config.date_col, *excludes])
    y = df[config.target_col]

    return X, y

def make_intersactions(df : pd.DataFrame, sel_feat : list[str]) -> pd.DataFrame :
    from itertools import combinations
    out = pd.DataFrame(index=df.index)
    for feat1, feat2 in combinations(sel_feat, 2) :
        out[f"{feat1}_x_{feat2}"] = df[feat1] * df[feat2]
    return out