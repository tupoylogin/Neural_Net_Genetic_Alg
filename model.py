import typing as tp

import numpy as np
from layer import BaseLayer, Dense
from functions import BaseFunction, ReLU, Tanh, Linear
from optimisers import BaseOptimizer

class BaseNetwork():
    def __init__(self, layers: tp.Union[None, tp.List[BaseLayer]]):
        self._layers=layers
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        out = X.copy()
        for l in self.layers:
            out = l.forward(out)
        return out
    
    @property
    def layers(self) -> tp.Union[None, tp.List[BaseLayer]]:
        return self._layers

    def add(self, layer: BaseLayer, idx: tp.Union[None, int] = None):
        if idx is None:
            self._layers.append(layer)
        else:
            self._layers.insert(idx, layer)

class FeedForwardRegressor(BaseNetwork):
    def __init__(self, layer_sizes: tp.List[int], batch_size: int):
        self._layer_sizes = layer_sizes
        super().__init__(layers=[])
        for idx, (i_s, outp_s) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.add(Dense(input_size=i_s, batch_size=batch_size,  num_units=outp_s, activation=Linear, number=idx+1))
        self.add(Dense(input_size=layer_sizes[-1], batch_size=batch_size, num_units=1, activation=Tanh, number=len(layer_sizes)))
    
    def compile(self, input_size: int, batch_size: int, loss: tp.Callable[..., BaseFunction], optimiser: BaseOptimizer):
        self.add(Dense(input_size=input_size, batch_size=batch_size,
            num_units=self._layer_sizes[0], activation=Linear, number=0), 0)
        self._loss = loss
        self._optimiser = optimiser
    
    def fit(self, X: np.ndarray, y: np.ndarray, iters=150):
        for i in range(iters):
            inst, to_stop = self._optimiser.apply(self._loss, X, y, self.layers)
            if to_stop and i//100:
                break
        for layer_self,layer_inst  in zip(self.layers, inst):
            layer_self.weights.value = layer_inst.weights.value
        return self

        

        