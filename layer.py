import typing as tp

import deap
import numpy as np

from functions import Variable, BaseFunction, ReLU

class BaseLayer():
    def __init__(self, *args, **kwargs):
        print(f'invoking init from {str(self)}')
        self.weights = None
    
    @property
    def weights(self):
        return self._weights
    
    @weights.setter
    def weights(self, value: tp.Union[int, float]):
        self._weights = value

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Dense(BaseLayer):
    def __init__(self,
        input_size: int,
        batch_size: int,
        num_units: int, 
        weights_initializer: tp.Union[None, tp.Callable[..., np.ndarray]] = None,
        activation: tp.Union[tp.Callable[..., BaseFunction], None] = None,
        **kwargs):
        super().__init__(**kwargs)
        self._number = kwargs.pop('number', 0)
        #self.weights = Variable(f'w_{self._number}')
        if weights_initializer is None:
            weights_value = np.random.default_rng().normal(size=(input_size+1, num_units))
        else:
            weights_value = weights_initializer((input_size+1, num_units))
        self.weights = Variable(f'w_{self._number}', value=weights_value.astype('float32'))
        self.x = Variable(f'x_{self._number}', shape=(batch_size, input_size+1))
        self.activation = ReLU(x=self.x, w=self.weights) if activation is None else activation(x=self.x, w=self.weights)
    
    def forward(self, X: np.ndarray):
        X_plus_bias = np.hstack((X,np.ones(X.shape[0]).reshape(-1,1)))
        self.x.value = X_plus_bias.astype('float32')
        return self.activation(at=[self.x, self.weights])
    
    def backward(self):
        return self.activation.deriv(wrt=self.x)(at=self.weights)
