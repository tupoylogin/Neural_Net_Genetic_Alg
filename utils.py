import numpy as np

def cast_to_same_shape(input: np.ndarray, target: np.ndarray) -> None:
    try:
        input.reshape(target.shape)
    except:
        raise Exception(f'Can\'t cast `input` (shape = {input.shape}) to `target` (shape = {target.shape})')

def weights_on_batch(weights: np.ndarray, X_batch: np.ndarray) -> np.ndarray:
    return np.array([np.array([w*X for w in weights.T]) for X in X_batch])