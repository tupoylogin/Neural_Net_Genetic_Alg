import typing as tp

import autograd.numpy as np

EPS = 1e-8


def ReLU(x: np.ndarray) -> np.ndarray:

    return np.maximum(0, x)


def Linear(x: np.ndarray) -> np.ndarray:
    return x


def Sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1 + np.exp(-1 * x))


def Tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def Sum(x: np.ndarray) -> np.ndarray:
    return np.sum(x, axis=-1, keepdims=True)


def LogSigmoid(x: np.ndarray) -> np.ndarray:
    return np.clip(x - np.logaddexp(0, x), 1e-6, 1e6)


def GaussianRBF(x: np.ndarray, c: np.ndarray, r: np.ndarray) -> np.ndarray:
    return np.clip(
        np.exp(-0.5 * np.multiply(np.linalg.norm(x - c, 2, axis=1), np.power(r, -2))),
        1e-6,
        1e6,
    )


def BellMembership(x: np.ndarray, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.clip(1 / (1 + (np.power((x - c), 2) / (np.power(a, 2) + EPS))), EPS, None)


def GausMembership(x: np.ndarray, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.clip(np.exp(-(np.power((x - c), 2) / (np.power(a, 2) + EPS))), EPS, None)
