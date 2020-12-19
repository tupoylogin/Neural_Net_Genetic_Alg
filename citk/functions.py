import typing as tp
from itertools import combinations, combinations_with_replacement

import autograd.numpy as np

EPS = 1e-8


def ReLU(x: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit (ReLU) activation

    Parameters
    ----------

    :param x: Input array.
    :type x: np.ndarray

    Returns
    -------
    :return: Array of element-wise maximum(0, x_i) for all x_i in a.
    :rtype: np.ndarray

    See Also
    --------
    Linear : linear activation function.
    Sigmoid : sigmoid activation function.
    Tanh : hyperbolic tangent activation function.
 
    Note
    -----
    When scalar is passed, scalar is returned, so it is recommended to convert scalar into 1-d array instead.

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

    :param x: Input array.
    :type x: np.ndarray

    Returns
    -------
    :return: Copy of input.
    :rtype: np.ndarray

    See Also
    --------
    ReLU : rectified linear unit activation function.
    Sigmoid : sigmoid activation function.
    Tanh : hyperbolic tangent activation function.

    Note
    -----
    When scalar is passed, scalar is returned, so it is recommended to convert scalar into 1-d array instead.

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

    :param x: Input array.
    :type x: np.ndarray

    Returns
    -------
    :return: res = 1/(1+exp(-x_i)) for x_i in x.
    :rtype: np.ndarray

    See Also
    --------
    ReLU : rectified linear unit activation function.
    Linear : identity activation function.
    Tanh : hyperbolic tangent activation function.

    Note
    -----
    When scalar is passed, scalar is returned, so it is recommended to convert scalar into 1-d array instead.

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

    :param x: Input array.
    :type x: np.ndarray

    Returns
    -------
    
    :return: res = (exp(x_i)-exp(-x_i))/(exp(x_i)+exp(-x_i)) for x_i in x.
    :rtype: np.ndarray

    See Also
    --------
    ReLU : rectified linear unit activation function.
    Linear : identity activation function.
    Sigmoid : sigmoid activation function.

    Note
    -----
    When scalar is passed, scalar is returned, so it is recommended to convert scalar into 1-d array instead.

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
    """
    Basic sum along `rows`.
    """
    return np.sum(x, axis=-1, keepdims=True)


def LogSigmoid(x: np.ndarray) -> np.ndarray:
    """
    Natural log of sigmoid function.
    """
    return np.clip(x - np.logaddexp(0, x), 1e-6, 1e6)


def GaussianRBF(x: np.ndarray, c: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Gaussian radial basis activation

    Parameters
    ----------

    :param x: Input array.
    :type x: np.ndarray

    :param c: Centroid array.
    :type c: np.ndarray
    
    :param r: Standard deviation array.
    :type r: np.ndarray

    Returns
    -------
    :return: res = res = np.exp(-||x-c||**2/(2*r**2)).
    :rtype: np.ndarray

    """
    return np.exp(
        -0.5
        * np.power(np.linalg.norm(x - c, 2, axis=1), 2)
        / np.clip(np.power(r, 2), EPS, None)
    )


def BellMembership(x: np.ndarray, c: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Bell Membership Function

    Parameters
    ----------

    :param x: Input array.
    :type x: np.ndarray

    :param c: Centroid array.
    :type c: np.ndarray
    
    :param a: Bandwith.
    :type a: np.ndarray

    Returns
    -------
    :return: 1 / (1 + ((x - c)**2)/a**2).
    :rtype: np.ndarray

    """
    return np.clip(
        1 / (1 + (np.power((x - c), 2) / np.clip(np.power(a, 2), EPS, None))), EPS, None
    )


def GaussianMembership(x: np.ndarray, c: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Gaussian Membership Function

    Parameters
    ----------

    :param x: Input array.
    :type x: np.ndarray

    :param c: Centroid array.
    :type c: np.ndarray
    
    :param a: Bandwith.
    :type a: np.ndarray

    Returns
    -------
    :return: exp(-((x - c)**2)/a**2).
    :rtype: np.ndarray

    """
    return np.clip(
        np.exp(-(np.power((x - c), 2) / np.clip(np.power(a, 2), EPS, None))), EPS, None
    )


def Poly(x: np.ndarray, deg: int, type: tp.Optional[str] = 'full'):
    inp_size = x.shape[1]
    X_p = np.empty(x.shape)
    #arrange every pair of input vectors
    indices_n_wise = combinations(range(inp_size),2)
    if deg <2 :
        for idx in indices_n_wise:
            X_p = np.append(x[:, idx], axis=1)
    else:
        for idx in indices_n_wise:
            #full degree polymone
            indices_to_mul = sum([list(combinations_with_replacement(idx, repeat=i)) for i in range(2, deg+1)], [])
            if type=='partial':
                #partial degree polynome
                indices_to_mul = filter(lambda x: len(set(x))>1, indices_to_mul)
            for ind in indices_to_mul:
                pf = np.prod(x[:, ind], axis=1)[:, np.newaxis]
                X_p = np.append(X_p, pf, axis=1)
    return X_p
