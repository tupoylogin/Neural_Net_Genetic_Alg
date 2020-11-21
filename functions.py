import typing as tp 

import autograd.numpy as np
from autograd import elementwise_grad

def ReLU(x: np.ndarray) -> np.ndarray:
    
    return np.maximum(0, x)

def Linear(x: np.ndarray) -> np.ndarray:
    return x

def Sigmoid(x: np.ndarray) -> np.ndarray:
    return 1./(1+np.exp(-1*x))

def Tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def LogSigmoid(x: np.ndarray) -> np.ndarray:
    return x - np.logaddexp(0, x)

def GaussianRBF(x: np.ndarray, c: tp.Optional[float] = 0., 
                r: tp.Optional[float] = 1.) -> np.ndarray:
    return np.exp(-1*(x-c)**2/r**2)
