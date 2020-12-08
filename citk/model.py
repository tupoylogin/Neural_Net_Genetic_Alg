import typing as tp
from datetime import datetime

import autograd.numpy as np
from tqdm.auto import tqdm

from .layer import BaseLayer, Dense, WeightsParser, Fuzzify
from .functions import GaussianRBF, ReLU, Tanh, Sigmoid, Linear, BellMembership
from .optimisers import BaseOptimizer
from .utils import gen_batch


class FFN(object):
    def __init__(
        self,
        input_shape: tp.Tuple[int],
        layer_specs: tp.List[BaseLayer],
        loss: tp.Callable[..., np.ndarray],
        **kwargs,
    ) -> None:
        """
        Feed Forward Network
        Args:
            input_shape (int): shape of your `X` variable
            layer_specs (list(BaseLayer)): layer list
            loss (callable): loss function
        """
        self.parser = WeightsParser()
        self.regularization = kwargs.get("regularization", "l2")
        self.reg_coef = kwargs.get("reg_coef", 0)
        self.layer_specs = layer_specs
        cur_shape = input_shape
        for num, layer in enumerate(self.layer_specs):
            layer.number = num
            N_weights, cur_shape = layer.build_weights_dict(cur_shape)
            self.parser.add_weights(str(layer), (N_weights,))
        self._loss = loss
        self.W_vect = 0.1 * np.random.default_rng(
            int(datetime.utcnow().timestamp() * 1e5)
        ).normal(size=(self.parser.N,))

    def loss(self, W_vect: np.ndarray, X: np.ndarray, y: np.ndarray):
        if self.regularization == "l2":
            reg = np.power(np.linalg.norm(W_vect, 2), 2)
        elif self.regularization == "l1":
            reg = np.linalg.norm(W_vect, 1)
        else:
            reg = 0.0
        return self._loss(self._predict(W_vect, X), y) + self.reg_coef * reg

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self._predict(self.W_vect, inputs)

    def _predict(self, W_vect: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        cur_units = inputs
        for layer in self.layer_specs:
            cur_weights = self.parser.get(W_vect, str(layer))
            cur_units = layer.forward(cur_units, cur_weights)
        return cur_units

    def eval(self, input: np.ndarray, output: np.ndarray) -> float:
        return self.loss(self.W_vect, input, output)

    def frac_err(self, X, T):
        return np.mean(
            np.argmax(T, axis=1) != np.argmax(self.predict(self.W_vect, X), axis=1)
        )

    def fit(
        self,
        optimiser: BaseOptimizer,
        train_sample: tp.Tuple[np.ndarray],
        validation_sample: tp.Tuple[np.ndarray],
        batch_size: int,
        epochs: tp.Optional[int] = None,
        verbose: tp.Optional[bool] = None,
    ):
        self._optimiser = optimiser
        verbose = verbose if verbose else False
        epochs = epochs if epochs else 1
        inst = None
        history = dict(epoch=[], train_loss=[], validation_loss=[])
        for i in tqdm(range(epochs), desc="Training "):
            tr_loss = np.inf
            to_stop = False
            for (X,y) in gen_batch(train_sample,batch_size):
                to_stop, inst, tr_loss = self._optimiser.apply(
                    self.loss, X, y, self.W_vect, verbose=verbose
                )
                self.W_vect = inst
            self.W_vect = inst
            
            history["epoch"].append(i)
            history["train_loss"].append(tr_loss)
            val_loss = self.eval(*validation_sample)[0]
            history["validation_loss"].append(val_loss)
            if verbose:
                print(f"validation loss - {val_loss}")
            if to_stop:
                break
        return self, history
