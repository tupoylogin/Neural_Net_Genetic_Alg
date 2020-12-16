import typing as tp
import inspect
import math
from itertools import repeat
import numpy as np


def gen_batch(dataset: tp.Tuple[np.ndarray, np.ndarray], batch_size: int):
    start_position = 0
    X, y = dataset
    while start_position < len(X):
        to_position = min(start_position + batch_size, len(X))
        yield (X[start_position:to_position], y[start_position:to_position])
        start_position = to_position

def nCr(n, r):
    f = math.factorial
    return int(f(n) / f(r) / f(n - r))


def concat_and_multiply(weights, *args):
    cat_state = np.hstack(args + (np.ones((args[0].shape[0], 1)),))
    return np.dot(cat_state, weights)
