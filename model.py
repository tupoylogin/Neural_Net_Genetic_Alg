import typing as tp

import autograd.numpy as np
from tqdm.auto import tqdm

from .layer import BaseLayer, Dense
from .functions import ReLU, Tanh, Sigmoid, Linear
from .optimisers import BaseOptimizer
from .utils import cast_to_same_shape

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
    def __init__(self):
        """
        `Dummy` Multilayer Perceptron
        """
        super().__init__(layers=[])
    
    def compile(self, 
                layer_sizes: tp.List[int],
                input_size: int, 
                output_size: int,
                ):
        """
        Construct Feed-Forward Net
        
        Args:
            layer_sizes (list(int)): hidden layer sizes
            input_size (int): input dimension of first layer, i.e. your `X` variable
            output_size (int): output dimension of last layer, i.e. your `Y` variable
        """
        self._layer_sizes = layer_sizes+[output_size]
        for idx, (i_s, outp_s) in enumerate(zip(self._layer_sizes[:-1], self._layer_sizes[1:])):
            self.add(Dense(input_size=i_s,
                num_units=outp_s,
                activation=ReLU,
                regularization='l2',
                lambda_reg=0.1,
                number=idx+1))
        self.add(Dense(input_size=input_size,
            bias_initializer=np.ones,
            num_units=self._layer_sizes[0], activation=ReLU, number=0), 0)

    def eval(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate Network on batch
        
        Args:
            X (ndarray): input batch
            y (ndarray): desired output
        Returns:
            lossfn (float): loss function value
        """
        outp = self(X).ravel()
        cast_to_same_shape(outp, y)
        lossfn = self._loss(outp, y)
        return lossfn

    def fit(self, 
                  loss: tp.Callable[..., tp.Any],
                  optimiser: BaseOptimizer,
                  train_sample: tp.Tuple[np.ndarray], 
                  validation_sample: tp.Tuple[np.ndarray], 
                  batch_size: int,
                  iters: tp.Optional[int] = None,
                  verbose: tp.Optional[bool] = None):
        self._loss = loss
        self._optimiser = optimiser
        verbose = verbose if verbose else False
        (X, y) = train_sample
        batch_s = min(batch_size, X.shape[0])
        iters = iters if iters else 2*len(X)//batch_s
        inst = None
        history = dict(iteration=[], train_loss=[], validation_loss=[])
        for i in tqdm(range(iters), desc="Training "):
            history['iteration'].append(i)
            batch = np.random.default_rng().choice(range(X.shape[0]), batch_s)
            X_b, y_b = X[batch], y[batch]
            inst, tr_loss = self._optimiser.apply(self._loss, X_b, y_b, self.layers, verbose=verbose)
            history['train_loss'].append(tr_loss)
            for layer_self, layer_inst  in zip(self.layers, inst):
                layer_self.weights = layer_inst.weights
            val_loss = self.eval(*validation_sample)
            history['validation_loss'].append(val_loss)
            if verbose:
                print(f'validation loss - {val_loss}')
        return self, history

        

        