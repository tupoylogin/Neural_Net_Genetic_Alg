import typing as tp

import autograd.numpy as np


def MSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Loss"""
    return np.mean((y_true - y_pred) ** 2, 0)

def RMSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Loss"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2, 0))

def FMSE(pred: tp.Tuple[np.ndarray], y_true: np.ndarray) -> float:
    """Fuzzy-Mean Squared Loss"""
    y_pred, intervals = pred
    return np.sum(intervals, 0) + 0.5 * (np.mean((y_true - y_pred) ** 2, 0))


def MAE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    "Mean Average Loss"
    return np.mean(np.abs(y_true - y_pred), 0)

def MAPE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    "Mean Absolute Precentage Loss"
    len_interval = y_pred.shape[0]
    return np.mean(np.abs((y_true - y_pred)/y_pred), 0)/len_interval

def Huber(y_true: np.ndarray, y_pred: np.ndarray, d: tp.Optional[float] = 1.0) -> float:
    "Huber Loss"
    condlist = [np.abs(y_true - y_pred) <= d, True]
    funclist = [0.5 * (y_true - y_pred) ** 2, d * (np.abs(y_true - y_pred) - 0.5 * d)]
    return np.mean(np.select(condlist, funclist), 0)
