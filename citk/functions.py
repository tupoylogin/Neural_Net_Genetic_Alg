import typing as tp

import autograd.numpy as np

EPS = 1e-8


def ReLU(x: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit (ReLU) activation

    Parameters
    ----------

    x : ndarray
        Input array.

    Returns
    -------
    res : ndarray
        Array of element-wise maximum(0, x_i) for all x_i in a.

    See Also
    --------
    Linear : linear activation function.
    Sigmoid : sigmoid activation function.
    Tanh : hyperbolic tangent activation function.
 
    Notes
    -----
    When scalar is passed, scalar is returned, so it is recommended to conert scalar into 1-d array instead.

    Examples
    --------
    >>> x = np.array([[1, -2], [-3, 4]])
    >>> y = np.array([-3.])
    >>> ReLU(x)
    array([[1, 0],
           [0, 4]])
    >>> ReLU(y)
    array([0.])

    """

    return np.maximum(0, x)


def Linear(x: np.ndarray) -> np.ndarray:
    """
    Linear activation

    Parameters
    ----------

    x : ndarray
        Input Array.

    Returns
    -------
    res : ndarray
        Copy of input.

    See Also
    --------
    ReLU : rectified linear unit activation function.
    Sigmoid : sigmoid activation function.
    Tanh : hyperbolic tangent activation function.

    Notes
    -----
    When scalar is passed, scalar is returned, so it is recommended to conert scalar into 1-d array instead.

    Examples
    --------
    >>> x = np.array([[1, -2], [-3, 4]])
    >>> y = np.array([-3.])
    >>> Linear(x)
    array([[1, -2],
           [-3, 4]])
    >>> Linear(y)
    array([-3.])

    """

    return x


def Sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation

    Parameters
    ----------

    x : ndarray
        Input Array.

    Returns
    -------
    res : ndarray
        res = 1/(1+exp(-x_i)) for x_i in x.

    See Also
    --------
    ReLU : rectified linear unit activation function.
    Linear : identity activation function.
    Tanh : hyperbolic tangent activation function.

    Notes
    -----
    When scalar is passed, scalar is returned, so it is recommended to conert scalar into 1-d array instead.

    Examples
    --------
    >>> x = np.array([[0, np.inf], [-np.inf, 0]])
    >>> y = np.array([-0.])
    >>> Sigmoid(x)
    array([[0.5, 1.],
           [0., 0.5]])
    >>> Sigmoid(y)
    array([0.5])

    """

    return 1.0 / (1 + np.exp(-1 * x))


def Tanh(x: np.ndarray) -> np.ndarray:
    """
    Hyperbolic tangent activation

    Parameters
    ----------

    x : ndarray
        Input Array.

    Returns
    -------
    res : ndarray
        res = (exp(x_i)-exp(-x_i))/(exp(x_i)+exp(-x_i)) for x_i in x.

    See Also
    --------
    ReLU : rectified linear unit activation function.
    Linear : identity activation function.
    Sigmoid : sigmoid activation function.

    Notes
    -----
    When scalar is passed, scalar is returned, so it is recommended to conert scalar into 1-d array instead.

    Examples
    --------
    >>> x = np.array([[0, np.inf], [-np.inf, 0]])
    >>> y = np.array([-0.])
    >>> Tanh(x)
    array([[0., 1.],
           [-1., 0.]])
    >>> Tanh(y)
    array([0.5])

    """
    return np.tanh(x)


def Sum(x: np.ndarray) -> np.ndarray:
    return np.sum(x, axis=-1, keepdims=True)


def LogSigmoid(x: np.ndarray) -> np.ndarray:
    return np.clip(x - np.logaddexp(0, x), 1e-6, 1e6)


def GaussianRBF(x: np.ndarray, c: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Gaussian radial basis activation

    Parameters
    ----------

    x : ndarray
        Input Array.

    c: ndarray
        Centroid Array.
    
    r: ndarray
        Standard deviation.

    Returns
    -------
    res : ndarray
        res = np.exp(-||x-c||**2/(2/*r**2))

    """
    return np.exp(
        -0.5
        * np.power(np.linalg.norm(x - c, 2, axis=1), 2)
        / np.clip(np.power(r, 2), EPS, None)
    )


def BellMembership(x: np.ndarray, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.clip(
        1 / (1 + (np.power((x - c), 2) / np.clip(np.power(a, 2), EPS, None))), EPS, None
    )


def GaussianMembership(x: np.ndarray, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.clip(
        np.exp(-(np.power((x - c), 2) / np.clip(np.power(a, 2), EPS, None))), EPS, None
    )
