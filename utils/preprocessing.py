"""Shared preprocessing utilities for time-series models.

Keep helpers small and dependency-light; used by model training
and evaluation pipelines.
"""
from typing import List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def train_test_split_time_series(df: pd.DataFrame, date_col: str, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sort by `date_col` and split into train/test by proportion.

    Returns (train_df, test_df).
    """
    if date_col not in df.columns:
        raise ValueError(f"date_col '{date_col}' not found in dataframe")
    df_sorted = df.sort_values(by=date_col).reset_index(drop=True)
    n_test = int(len(df_sorted) * test_size)
    if n_test < 1:
        return df_sorted.copy(), df_sorted.iloc[0:0].copy()
    split_idx = len(df_sorted) - n_test
    return df_sorted.iloc[:split_idx].copy(), df_sorted.iloc[split_idx:].copy()


def create_lag_features(df: pd.DataFrame, value_col: str, lags: Sequence[int]) -> pd.DataFrame:
    """Return a copy of `df` with additional lag features for `value_col`.

    New columns are named like `{value_col}_lag1`, `{value_col}_lag2`, ...
    """
    out = df.copy()
    for lag in sorted(set(lags)):
        out[f"{value_col}_lag{lag}"] = out[value_col].shift(lag)
    return out


def rolling_features(df: pd.DataFrame, value_col: str, windows: Sequence[int] = (3, 7), funcs: Sequence[str] = ("mean", "std")) -> pd.DataFrame:
    """Add rolling-window statistics for `value_col`.

    Supported funcs: 'mean', 'std', 'min', 'max', 'median'
    """
    out = df.copy()
    for w in sorted(set(windows)):
        roll = out[value_col].rolling(window=w)
        for f in funcs:
            col = f"{value_col}_r{w}_{f}"
            if f == "mean":
                out[col] = roll.mean()
            elif f == "std":
                out[col] = roll.std()
            elif f == "min":
                out[col] = roll.min()
            elif f == "max":
                out[col] = roll.max()
            elif f == "median":
                out[col] = roll.median()
            else:
                raise ValueError(f"unsupported rolling func: {f}")
    return out


def impute_missing(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """Impute missing values. Methods: 'ffill', 'bfill', 'interpolate', 'zero'."""
    out = df.copy()
    if method == "ffill":
        return out.fillna(method="ffill").fillna(method="bfill")
    if method == "bfill":
        return out.fillna(method="bfill").fillna(method="ffill")
    if method == "interpolate":
        return out.interpolate().fillna(method="bfill").fillna(method="ffill")
    if method == "zero":
        return out.fillna(0)
    raise ValueError(f"unknown imputation method: {method}")


def scale_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: Sequence[str], scaler: Optional[str] = "standard") -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    """Fit scaler on `train_df[feature_cols]` and transform both train and test.

    `scaler` may be 'standard' (StandardScaler) or 'minmax' (MinMaxScaler),
    or an already-instantiated scaler object.
    Returns (train_scaled_df, test_scaled_df, fitted_scaler).
    """
    if scaler is None or scaler == "standard":
        s = StandardScaler()
    elif scaler == "minmax":
        s = MinMaxScaler()
    elif hasattr(scaler, "fit"):
        s = scaler
    else:
        raise ValueError("scaler must be 'standard', 'minmax', or a scaler instance")

    train_vals = train_df[list(feature_cols)].astype(float)
    test_vals = test_df[list(feature_cols)].astype(float)

    s.fit(train_vals)
    train_scaled = pd.DataFrame(s.transform(train_vals), columns=list(feature_cols), index=train_df.index)
    test_scaled = pd.DataFrame(s.transform(test_vals), columns=list(feature_cols), index=test_df.index)

    train_out = train_df.copy()
    test_out = test_df.copy()
    train_out.update(train_scaled)
    test_out.update(test_scaled)
    return train_out, test_out, s


__all__ = [
    "train_test_split_time_series",
    "create_lag_features",
    "rolling_features",
    "impute_missing",
    "scale_train_test",
]
