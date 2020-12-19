import typing as tp

import autograd.numpy as np


def MSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Loss"""
    return np.mean((y_true - y_pred) ** 2, 0)


def MAE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    "Mean Average Loss"
    return np.mean(np.abs(y_true - y_pred), 0)


def Huber(y_true: np.ndarray, y_pred: np.ndarray, d: tp.Optional[float] = 1.0) -> float:
    "Huber Loss"
    condlist = [np.abs(y_true - y_pred) <= d, True]
    funclist = [0.5 * (y_true - y_pred) ** 2, d * (np.abs(y_true - y_pred) - 0.5 * d)]
    return np.mean(np.select(condlist, funclist), 0)
