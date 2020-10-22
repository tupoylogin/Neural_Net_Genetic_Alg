import typing as tp 

from scipy.special import expit
import numpy as np
from sympy import *
from sympy.core.symbol import Symbol
from sympy.utilities.lambdify import lambdify

class Variable():
    def __init__(self, name: str, value=None, shape=None):
        """
        class Variable
        Implements dummy function variable
        Arguments:
            name: str - name alias
        """
        if (shape is not None) or (value is not None):
            shp = value.shape if shape is None else shape
        else:
            shp=(1,1)
        self._variable = MatrixSymbol(name, *shp)
        self.value = value
    
    @property
    def value(self) -> np.ndarray:
        return self._value
    
    @value.setter
    def value(self, val: tp.Union[float, np.ndarray]):
        self._value = val
    
    @property
    def variable(self) -> Symbol:
        return self._variable
    

class BaseFunction():
    def __init__(self, func: Mul, **kwargs):
        self._func = func

    def __call__(self, at: tp.List[Variable]) -> tp.Union[FunctionClass, np.ndarray]:
        return lambdify(list(map(lambda x: x.variable, at)), self._func, 'numpy')(*[x.value for x in at])

    def deriv(self, wrt: Variable):
        return BaseFunction(self._func.diff(wrt.variable))

class Linear(BaseFunction):
    def __init__(self, x: Variable, w: Variable):
        """
        Linear 
        """
        func = x.variable@w.variable
        super().__init__(func)

class ReLU(BaseFunction):
    def __init__(self, x: Variable, w: Variable):
        func = Piecewise((0, x.variable@w.variable<=0),(x.variable@w.variable, True))
        super().__init__(func)

class Tanh(BaseFunction):
    def __init__(self, x: Variable, w: Variable):
        func = tanh(x.variable@w.variable)
        super().__init__(func)

class MeanSquared(BaseFunction):
    def __init__(self, y_hat: Variable, y: Variable):
        func = np.mean(np.power(y_hat.variable-y.variable, 2))
        super().__init__(func)