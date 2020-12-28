import random
import os

from typing import List

import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def seed_everything(seed_value: int = 42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)


def continious_stratification(
    X_data: np.ndarray,
    y_data: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    shuffle: bool = True,
    n_bins: int = 10,
):
    stratify_col = np.digitize(y_data, np.histogram(y_data, n_bins)[1])
    return train_test_split(
        X_data,
        y_data,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify_col,
    )


def get_and_process_boston_dataset(
    random_state: int = 42, normalize_y: bool = True, normalize_X: bool = True
):
    # Load
    X, y = load_boston(return_X_y=True)
    # Split into train/test
    X_train, X_test, y_train, y_test = continious_stratification(
        X, y, random_state=random_state
    )
    # Normalize target
    if normalize_y:
        tgt_trans = QuantileTransformer(
            n_quantiles=300, output_distribution="normal", random_state=random_state
        )
        y_train = tgt_trans.fit_transform(y_train[:, None])
        y_test = tgt_trans.transform(y_test[:, None])
    else:
        y_train = y_train[:, None]
        y_test = y_test[:, None]
    # Normalize features
    if normalize_X:
        feature_trans = StandardScaler()
        X_train = feature_trans.fit_transform(X_train)
        X_test = feature_trans.transform(X_test)

    return X_train, X_test, y_train, y_test


def visualise_boston(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    only_target: bool = True,
):
    if not only_target:
        for i in range(X_train.shape[1]):
            plt.title(f"Train feature {i}")
            plt.hist(X_train[:, i])
            plt.show()

            plt.title(f"Test feature {i}")
            plt.hist(X_test[:, i])
            plt.show()

    plt.title("Train target")
    plt.hist(y_train[:, 0])
    plt.show()

    plt.title("Test target")
    plt.hist(y_test[:, 0])
    plt.show()

def normalize_df(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    featutre_to_normalize: List[str]
):
    feature_trans = StandardScaler()
    train_df[featutre_to_normalize] = feature_trans.fit_transform(train_df[featutre_to_normalize])
    test_df[featutre_to_normalize] = feature_trans.transform(test_df[featutre_to_normalize])
    
    return train_df, test_df

def create_lags(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    lag_cols: List[str],
    n_lags: int,
    drop_nan: bool =True
):

    for col in lag_cols:
        for sh in range(1,n_lags+1):
            train_df[f'{col}_{sh}'] = train_df.shift(sh)[col]
            test_df[f'{col}_{sh}'] = test_df.shift(sh)[col] 
            
    if drop_nan:
        train_df = train_df.dropna()
        test_df = test_df.dropna()
            
    return train_df, test_df

