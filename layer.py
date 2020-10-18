import typing

import deap
import numpy as np

from functions import Variable, BaseFunction, ReLU

class BaseLayer():
    def __init__(self, *args, **kwargs):
        self.weights = None
        self.activation = None
    
    @property
    def weights(self):
        return self._weights
    
    @weights.setter
    def weights(self, value: typing.Union[int, float]):
        self._weights = weights
    
    @property
    def activation(self):
        return self._activation
    
    @weights.setter
    def weights(self, value: BaseFunction):
        self._activation = activation

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

class Dense(BaseLayer):
    def __init__(self,
        input_size: int,
        num_units: int, 
        weights_initializer: typing.Callable[[..., typing.Tuple[int, int]], np.ndarray],
        activation: typing.Union[BaseFunction, None],
        **kwargs):
        super().__init__(**kwargs)
        self.weights = Variable('w')
        self.x = Variable('x')
        if weights_initializer is None:
            self.weights.value = np.random.Generator().normal(size=(num_units, input_size+1))
        else:
            self.weights.value = weights_initializer((num_units, input_size+1))
        self.activation = ReLU(x=self.x, w=self.weights) if activation is None else activation
    
    def forward(self, X: np.ndarray):
        X_plus_bias = np.stack((X, np.ones(X.shape[0])), axis=-1)
        self.x.value = X_plus_bias
        return self.activation([self.x, self.weights])
    
    def backward(self):
        return self.activation.deriv(wrt=self.x)(at=self.weights)
