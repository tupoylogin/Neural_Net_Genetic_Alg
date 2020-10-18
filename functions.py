import typing

import numpy as np
from sympy import *
from sympy.core.symbol import Symbol
from sympy.utilities.lambdify import lambdify

class Variable():
    def __init__(self, name: str):
        """
        class Variable
        Implements dummy function variable
        Arguments:
            name: str - name alias
        """
        self._variable = Symbol(name)
        self.value = None
    
    @property
    def value(self) -> np.ndarray:
        return self._value
    
    @value.setter
    def value(self, val: typing.Union[float, np.ndarray]):
        self._value = val
    
    @property
    def variable(self) -> Symbol:
        return self._variable
    

class BaseFunction():
    def __init__(self, func: typing.Callable[[typing.Sequence[Variable]], Mul], **kwargs):
        self._func = func

    def __call__(self, at: typing.List[Variable]) -> np.ndarray:
        return lambdify(list(map(lambda x: x.variable, at)), self._func, 'numpy')(*[x.value for x in at])

    def deriv(self, wrt: Variable):
        return BaseFunction(self._func.diff(wrt.variable))

class Linear(BaseFunction):
    def __init__(self, x: Variable, w: Variable):
        """
        Linear 
        """
        func = np.dot(w, x)
        super().__init__(func)

class ReLU(BaseFunction):
    def __init__(self, x: Variable, w: Variable):
        func = Piecewise((0, np.dot(w,x)<=0),(np.dot(w,x), True))
        super().__init__(func)

class Sigmoid(BaseFunction):
    def __init__(self, x: Variable, w: Variable):
        func = 1./(1+np.math.exp(np.dot(w,x)))
        super().__init__(func)