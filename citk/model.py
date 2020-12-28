import typing as tp
from datetime import datetime
from copy import deepcopy
from os import system
from itertools import combinations

import autograd.numpy as np
from tqdm.auto import tqdm

from .layer import BaseLayer, Dense, FuzzyGMDHLayer, GMDHLayer, WeightsParser, Fuzzify
from .functions import GaussianMembership, GaussianRBF, ReLU, Tanh, Sigmoid, Linear, BellMembership
from .optimisers import BaseOptimizer
from .utils import gen_batch, step_simplex
from .losses import MSE


class FFN(object):
    """
    Feed Forward Network
    --------------------
    """
    def __init__(
        self,
        input_shape: tp.Tuple[int],
        layer_specs: tp.List[BaseLayer],
        loss: tp.Callable[..., np.ndarray],
        **kwargs,
    ) -> None:
        """
        Constructor method

        Parameters
        ----------

        :param input_shape: Input shape.
        :type input_shape: tuple
        
        :param layer_specs: List containing layers.
        :type layer_specs: list
        
        :param loss: Loss function.
        :type mode: callable
        """
        self.parser = WeightsParser()
        self.regularization = kwargs.get("regularization", "l2")
        self.reg_coef = kwargs.get("reg_coef", 0)
        self.layer_specs = layer_specs
        cur_shape = input_shape
        W_vect = np.array([])
        for num, layer in enumerate(self.layer_specs):
            layer.number = num
            N_weights, cur_shape = layer.build_weights_dict(cur_shape)
            self.parser.add_weights(str(layer), (N_weights,))
            W_vect = np.append(W_vect, layer.initializer(size=(N_weights,)))
        self._loss = loss
        self.W_vect = 0.1 * W_vect

    def loss(
        self, W_vect: np.ndarray, X: np.ndarray, y: np.ndarray, omit_reg: bool = False
    ) -> np.ndarray:
        """
        Loss function constructor

        Parameters
        ----------

        :W_vect: Network weights vector.
        :type W_vect: np.ndarray
        
        :X: Input vector.
        :type X: np.ndarray
        
        :y: Desired network response.
        :type y: np.ndarray

        :omit_reg: Omit regularization flag. Default is `False`
        :type omit_reg: bool
        """
        if self.regularization == "l2" and not omit_reg:
            reg = np.power(np.linalg.norm(W_vect, 2), 2)
        elif self.regularization == "l1" and not omit_reg:
            reg = np.linalg.norm(W_vect, 1)
        else:
            reg = 0.0
        return self._loss(self._predict(W_vect, X), y) + self.reg_coef * reg

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Predict method

        Parameters
        ----------

        :param inputs: Input vector.
        :type inputs: np.ndarray

        Returns
        -------

        :returns: Network response.
        :rtype: np.ndarray
        """
        return self._predict(self.W_vect, inputs)

    def _predict(self, W_vect: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        cur_units = inputs
        for layer in self.layer_specs:
            cur_weights = self.parser.get(W_vect, str(layer))
            cur_units = layer.forward(cur_units, cur_weights)
        return cur_units

    def eval(self, input: np.ndarray, output: np.ndarray) -> float:
        """
        Evaluate network on given input

        Parameters
        ----------
        :param inputs: Input vector.
        :type inputs: np.ndarray

        :param output: Desired output
        :type output: np.ndarray

        Returns
        -------
        :returns: Loss value
        :rtype: float
        """
        return self.loss(self.W_vect, input, output, omit_reg=True)

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
        load_best_model_on_end: bool = True,
        minimize_metric: bool = True,
    ):
        """
        Fit network on given input

        Parameters
        ----------
        :optimiser: Algorithm to use for minimuzing loss.
        :type optimiser: BaseOptimiser

        :param train_sample: Train pair (X, y).
        :type train_sample: tuple

        :param validation_sample: Validation pair (X, y).
        :type validation_sample: tuple

        :batch_size: Batch size.
        :type batch_size: int

        :epochs: Number of epochs.
        :type epochs: int

        Returns
        -------
        :returns: Tuple (trained_model, loss_history)
        :rtype: union[FFN, dict]
        """
        self._optimiser = optimiser

        verbose = verbose if verbose else False
        epochs = epochs if epochs else 1

        inst = None
        best_inst = None
        best_score = np.inf if minimize_metric else -np.inf
        best_epoch = 0

        history = dict(epoch=[], train_loss=[], validation_loss=[])

        for i in tqdm(range(epochs), desc="Training "):

            tr_accum_loss = []
            tr_loss = np.inf
            to_stop = False

            for (X, y) in gen_batch(train_sample, batch_size):
                to_stop, inst, tr_loss = self._optimiser.apply(
                    self.loss, X, y, self.W_vect, verbose=verbose
                )
                self.W_vect = inst
                tr_accum_loss.append(tr_loss)

            tr_accum_loss = np.mean(tr_accum_loss)

            history["epoch"].append(i)
            history["train_loss"].append(tr_accum_loss)

            val_loss = self.eval(*validation_sample)[0]
            history["validation_loss"].append(val_loss)

            if minimize_metric and val_loss < best_score:
                best_score = val_loss
                best_inst = deepcopy(self.W_vect)
                best_epoch = i
            elif (not minimize_metric) and val_loss > best_score:
                best_score = val_loss
                best_inst = deepcopy(self.W_vect)
                best_epoch = i
            else:
                pass

            if verbose:
                print(f"validation loss - {val_loss}")
            if to_stop:
                break

        if load_best_model_on_end:
            self.W_vect = best_inst
            if verbose:
                print(f"best validation loss - {best_score}")
                print(f"best epoch - {best_epoch}")

        return self, history

class GMDH(object):
    """
    Group Method of Data Handling
    -----------------------------
    """
    def __init__(
        self,
        method_type: str,
        poli_type: str,
        loss: tp.Callable[..., np.ndarray],
        **kwargs,
    ) -> None:
        """
        Constructor method

        Parameters
        ----------

        :param method_type: Type of algorithm, `fuuzy` or `crisp`.
        :type method_type: str

        :param poli_type: Type of polinome, `linear`, 'quadratiic` or `partial_quadratic`.
        :type poli_type: str
        
        :param loss: Loss function.
        :type mode: callable
        """
        self.parser = WeightsParser()
        self._method_type = method_type,
        self._poli_type = poli_type
        self._confidence = kwargs.get('confidence', 0.8)
        self.layer_specs = self._construct_initial()
        cur_shape = 2
        W_vect = np.array([])
        for num, layer in enumerate(self.layer_specs):
            layer.number = num
            N_weights, cur_shape = layer.build_weights_dict(cur_shape)
            self.parser.add_weights(str(layer), (N_weights,))
            W_vect = np.append(W_vect, np.abs(layer.initializer(size=(N_weights,))))
        self._loss = loss
        self.W_vect = 0.1 * W_vect
    
    def _construct_initial(self):
        if self._method_type == 'crisp':
            return [GMDHLayer(poli_type=self._poli_type)]
        else:
            return [FuzzyGMDHLayer(poli_type=self._poli_type, 
                        msf=BellMembership, 
                        confidence=self._confidence,
                        return_defuzzify=True)]

    def predict_one(self, inputs: np.ndarray, return_uncertanity: bool = False) -> np.ndarray:
        """
        Predict one gmdh submodule method

        Parameters
        ----------

        :param inputs: Input vector.
        :type inputs: np.ndarray

        Returns
        -------

        :returns: Network response.
        :rtype: np.ndarray
        """
        if return_uncertanity:
            return self._predict(self.W_vect, inputs)
        return self._predict(self.W_vect, inputs)[0]

    def _predict(self, W_vect: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        cur_units = inputs
        for layer in self.layer_specs:
            cur_weights = self.parser.get(W_vect, str(layer))
            cur_units = layer.forward(cur_units, cur_weights)
        return cur_units

    def frac_err(self, X, T):
        return np.mean(
            np.argmax(T, axis=1) != np.argmax(self.predict(self.W_vect, X), axis=1)
        )
    
    def fit_simplex(self,
                    train_sample: tp.Tuple[np.ndarray]):
        X_train, y_train = train_sample
        weights, margin, absolute, inputs = self.layer_specs[0].forward(X_train, self.W_vect, True)
        w = step_simplex(weights, margin, absolute, inputs, y_train)
        self.W_vect = np.array(w).reshape(self.W_vect.shape)
        return self, None

    def fit(
        self,
        train_sample: tp.Tuple[np.ndarray],
        validation_sample: tp.Tuple[np.ndarray],
        max_gmdh_layers: int,
        n_best_to_take: int,
        minimize_metric: bool = True,
        verbose: tp.Optional[bool] = None
    ):
        """
        Fit network on given input

        Parameters
        ----------
        :param train_sample: Train pair (X, y).
        :type train_sample: tuple

        :param validation_sample: Validation pair (X, y).
        :type validation_sample: tuple

        :max_gmdh_layers: Maximum number of GMDH layers.
        :type max_gmdh_layers: int

        :n_best_to_take: Number of best GMDH outputs that go to the next layer.
        :type n_best_to_take: int

        :n_best_to_take: Number of best GMDH outputs that go to the next layer.
        :type n_best_to_take: int

        :minimize_metric: Whether we minimize target metric.
        :type minimize_metric: bool

        :verbose: Whether we turn on verbosity.
        :type verbose: bool

        Returns
        -------
        :returns: Tuple (trained_model, loss_history, best_test_pred, best_train_pred)
        :rtype: union[FFN, dict, np.array, np.array]
        """

        verbose = verbose if verbose else False
        
        all_possible_pairs = list(combinations(range(train_sample[0].shape[1]),2))

        overall_best_metric = np.inf if minimize_metric else -np.inf
        best_test_pred = None
        best_train_pred = None

        history = dict(metric=[])

        for r in tqdm(range(max_gmdh_layers), desc="Training "):

            layer_metrics = []
            layer_val_preds = []
            layer_train_preds = []

            for pair in all_possible_pairs:

                self.fit_simplex((train_sample[0][:,pair], train_sample[1]))

                prediction_val = self.predict_one(validation_sample[0][:, pair])
                prediction_train =  self.predict_one(train_sample[0][:,pair])

                metric_val = self._loss(prediction_val, validation_sample[1])[0]

                layer_metrics.append(metric_val)
                layer_val_preds.append(prediction_val)
                layer_train_preds.append(prediction_train)

            layer_metrics = np.array(layer_metrics)
            layer_val_preds = np.concatenate(layer_val_preds, axis=-1)
            layer_train_preds = np.concatenate(layer_train_preds, axis=-1)

            if minimize_metric:
                sorted_indices = np.argsort(layer_metrics)
            else:
                sorted_indices = np.argsort(-layer_metrics)

            best_metric = layer_metrics[sorted_indices[0]]
            history['metric'].append(best_metric)

            layer_val_preds = layer_val_preds[:,sorted_indices]
            validation_sample = (layer_val_preds[:,:n_best_to_take], validation_sample[1])

            layer_train_preds = layer_train_preds[:,sorted_indices]
            train_sample = (layer_train_preds[:,:n_best_to_take], train_sample[1])

            all_possible_pairs = list(combinations(range(train_sample[0].shape[1]),2))

            if verbose:
                print(f"Layer: {r}. Metric: {best_metric}")

            if minimize_metric and best_metric < overall_best_metric:
                overall_best_metric = best_metric
                best_test_pred = layer_val_preds[:,sorted_indices[0]][...,np.newaxis]
                best_train_pred = layer_train_preds[:,sorted_indices[0]][...,np.newaxis]
            elif (not minimize_metric) and best_metric > overall_best_metric:
                overall_best_metric = best_metric
                best_test_pred = layer_val_preds[:,sorted_indices[0]][...,np.newaxis]
                best_train_pred = layer_train_preds[:,sorted_indices[0]][...,np.newaxis]
            else:
                break


        return self, history, best_test_pred, best_train_pred

