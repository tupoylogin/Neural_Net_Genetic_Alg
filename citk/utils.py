import typing as tp
from itertools import repeat
import numpy as np

def gen_batch(dataset: tp.Tuple[np.ndarray, np.ndarray], batch_size: int):
    start_position = 0
    X, y = dataset
    while start_position<len(X):
        to_position = min(start_position + batch_size, len(X))
        yield (X[start_position: to_position],
            y[start_position: to_position])
        start_position += to_position



