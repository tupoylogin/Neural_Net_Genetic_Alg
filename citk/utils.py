import typing as tp
import inspect
import math
from itertools import repeat

from scipy.optimize import linprog
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

def centroid(x, mfx):
    """
    Defuzzification using centroid (`center of gravity`) method.

    Parameters
    ----------
    x : 1d array, length M
        Independent variable
    mfx : 1d array, length M
        Fuzzy membership function

    Returns
    -------
    u : 1d array, length M
        Defuzzified result

    """

    sum_moment_area = 0.0
    sum_area = 0.0

    # If the membership function is a singleton fuzzy set:
    if len(x) == 1:
        return x[0]*mfx[0] / np.fmax(mfx[0], np.finfo(float).eps).astype(float)

    # else return the sum of moment*area/sum of area
    for i in range(1, len(x)):
        x1 = x[i - 1]
        x2 = x[i]
        y1 = mfx[i - 1]
        y2 = mfx[i]

        # if y1 == y2 == 0.0 or x1==x2: --> rectangle of zero height or width
        if not(y1 == y2 == 0.0 or x1 == x2):
            if y1 == y2:  # rectangle
                moment = 0.5 * (x1 + x2)
                area = (x2 - x1) * y1
            elif y1 == 0.0 and y2 != 0.0:  # triangle, height y2
                moment = 2.0 / 3.0 * (x2-x1) + x1
                area = 0.5 * (x2 - x1) * y2
            elif y2 == 0.0 and y1 != 0.0:  # triangle, height y1
                moment = 1.0 / 3.0 * (x2 - x1) + x1
                area = 0.5 * (x2 - x1) * y1
            else:
                moment = (2.0 / 3.0 * (x2-x1) * (y2 + 0.5*y1)) / (y1+y2) + x1
                area = 0.5 * (x2 - x1) * (y1 + y2)

            sum_moment_area += moment * area
            sum_area += area

    return sum_moment_area / np.fmax(sum_area,
                                     np.finfo(float).eps).astype(float)

def step_simplex(weights: np.ndarray, 
                margin: np.ndarray, 
                absolute: np.ndarray, 
                inputs: np.ndarray, 
                y: np.ndarray):
    lenl = weights.shape[-1]
    inputs_w_b = np.column_stack((inputs, np.ones(inputs.shape[0])))
    abs_w_b = np.column_stack((absolute, np.ones(absolute.shape[0])))
    initial_x = np.append(weights, margin)
    obj_w = np.zeros(2*lenl)
    obj_w[lenl:] = np.sum(abs_w_b, axis=0).ravel()
    ineq_A_1 = np.column_stack((inputs_w_b, -1*abs_w_b))
    ineq_A_2 = np.column_stack((-1*inputs_w_b, -1*abs_w_b))
    ineq_A = np.vstack((ineq_A_1, ineq_A_2))
    ineq_y = np.vstack((y,-1*y))
    bounds = [(None, None) for _ in range(lenl)] + [(0, None) for _ in range(lenl)]
    lp = linprog(obj_w, ineq_A, ineq_y, bounds=bounds, method='interior-point', x0=initial_x)
    return lp.x


