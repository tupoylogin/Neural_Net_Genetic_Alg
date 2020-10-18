import typing

import numpy as np
from layer import BaseLayer, Dense
from functions import BaseFunction, ReLU, Sigmoid

class BaseNetwork():
    def __init__(self, layers: typing.Union[None, typing.List[BaseLayer]]):
        self._layers=layers
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        out = X.copy()
        for l in self.layers:
            out = l(out)
        return out
    
    @property
    def layers(self) -> typing.Union[None, typing.List[BaseLayer]]:
        return self._layers

    def add(self, layer: BaseLayer, idx: typing.Union[None, int] = None):
        if idx is None:
            self._layers.append(layer)
        else:
            self._layers.insert(idx, layer)

class FeedForwardRegressor(BaseNetwork):
    def __init__(self, layer_sizes: typing.List[int]):
        self._layer_sizes = layer_sizes
        super().__init__(layers=None)
        for i_s, outp_s in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.add(Dense(input_size=i_s, num_units=outp_s, activation=ReLU))
        self.add(Dense(input_size=layer_sizes[-1], num_units=1, activation=Sigmoid))
    
    def compile(self, input_size: np.ndarray, loss: BaseFunction, optimiser):
        self.add(Dense(input_size=input_size, 
            num_units=self._layer_sizes[0], activation=ReLU), 0)
        self._loss = loss
        self._optimiser = optimiser
    
    def fit(self, X: np.ndarray, y: np.ndarray, iters=150):
        for i in range(iters):
            y_hat = self.__call__(X)
            loss = self._loss(y, y_hat)
            self._optimiser.apply(loss)
        return self

        

        