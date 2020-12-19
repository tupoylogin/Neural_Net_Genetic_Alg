from copy import deepcopy
from datetime import datetime
from functools import partial
from threading import setprofile
import typing as tp

from autograd import grad
import numpy as np

from .layer import BaseLayer


class BaseOptimizer:
    """
    Base Optimizer
    --------------
    All custom optimizers should inherit this class
    """
    def __init__(self, *args, **kwargs):
        """
        Counstructor method
        """
        pass

    def apply(self, loss: tp.Callable[..., float], graph: tp.List[BaseLayer]):
        """
        Apply optimizer
        """
        raise NotImplementedError


class GeneticAlgorithmOptimizer(BaseOptimizer):
    """
    `Vanilla` Genetic Algorithm.
    """
    def __init__(self, num_population: int, k: int = 5, **kwargs):
        """
        Constructor method

        Parameters
        ----------
        :param num_population: Number of individuals in each population
        :type num_population: int

        :param k: Number of crossover rounds to perform
        :type k: int
        """
        self._num_population = num_population
        self._k = k
        self._population = None
        self._best_iter = None
        self._last_score = np.inf
        self._iter = 0
        self._tol = kwargs.pop("tol", 1e-2)
        super().__init__(**kwargs)

    @staticmethod
    def construct_genome(W: np.ndarray, weight_init: tp.Callable[..., np.ndarray]):
        """
        Construct random population

        Parameters
        ----------
        :param layers_list: Genotype, i.e. FFN template to mimic to.
        :type layers_list: list
        :param weight_init: Weight distribution function.
        :type weight_init: callable
        
        Returns
        -------
        :returns: Initalized weights.
        :rtype: np.ndarray
        """
        return 0.1 * weight_init(0, 1, size=W.shape)

    @staticmethod
    def crossover(ind_1: np.ndarray, ind_2: np.ndarray) -> np.ndarray:
        """
        Perform simple crossover

        Parameters
        ----------
        :param ind_1: FFN layers weights, first individual.
        :type ind_1: np.ndarray
        :param ind_2: FFN layers weights, second individual
        :type ind_2: np.ndarray
        
        Returns
        -------

        :returns: Generated offsprings
        :rtype: np.ndarray
        """
        assert len(ind_1) == len(ind_2), "individuals must have same len"
        index = np.random.default_rng().integers(len(ind_1))
        ind_12 = np.concatenate((ind_1[:index], ind_2[index:]), axis=None)
        ind_21 = np.concatenate((ind_2[:index], ind_1[index:]), axis=None)
        return ind_12, ind_21

    @staticmethod
    def mutate(
        ind: np.ndarray, mu: float = 0.1, sigma: float = 1.0, factor: float = 0.01
    ) -> tp.List[BaseLayer]:
        """
        Perform simple mutation

        Parameters
        ----------
        :param ind: FFN layers weights
        :type ind: np.ndarray

        :param mu: mean of distribution
        :type mu: float

        :param sigma: scale of distribution
        :type sigma: float

        :param factor: scale factor of mutation
        :type factor: float
        
        Returns
        -------
        :returns: Generated individual
        :rtype: np.ndarray
        """
        seed = int(datetime.utcnow().timestamp() * 1e5)
        ind += factor * np.random.default_rng(seed).normal(
            loc=mu, scale=sigma, size=len(ind)
        )
        return ind

    def apply(
        self,
        loss: tp.Callable[
            [np.ndarray, np.ndarray, tp.Optional[tp.Dict[str, float]]], float
        ],
        input_tensor: np.ndarray,
        output_tensor: np.ndarray,
        W: np.ndarray,
        **kwargs,
    ):
        """
        Perform one step of GA

        Parameters
        ----------

        :param loss: `Inverse` fitness function to minimize
        :type loss: callable

        :param input_tensor: Global input to FFN, i.e. your `X` variable
        :type input_tensor:  np.ndarray

        :param output_tensor: Desired FFN response, i.e. your `Y` variable 
        :type output_tensor: np.ndarray
        
        :param W: Initial network weights
        :type W: np.ndarray
        
        Returns
        -------

        :returns: tuple (best individual so far, lowest loss so far)
        :rtype: Union[np.ndarray, float]
        """
        verbose = kwargs.pop("verbose", False)
        seed = int(datetime.utcnow().timestamp() * 1e5)
        to_stop = False
        if not (self._population):
            population = [
                self.construct_genome(W, np.random.default_rng(seed + 42 * i).normal)
                for i in range(self._num_population)
            ]
        else:
            population = self._population[:]
        scores = []
        for g in population:
            scores.append(loss(g, input_tensor, output_tensor)[0])
        scores, scores_idx = np.sort(scores), np.argsort(scores)
        if verbose:
            print(f"best individual - {scores[0]}")

        if scores[0] < self._tol:
            to_stop = True
        self._population = np.array(population)[scores_idx][
            : self._num_population - self._k * 3
        ].tolist()
        probas = 1.0 - (scores - np.min(scores)) / np.ptp(scores)
        probas /= sum(probas)
        for _ in range(self._k):
            indices = np.random.default_rng(seed).choice(scores_idx, 2, p=probas)
            ind_1, ind_2 = self.crossover(
                population[indices[0]], population[indices[1]]
            )
            self._population.append(
                self.mutate(ind_1, factor=np.random.default_rng(seed).normal(0.01, 0.1))
            )
            self._population.append(
                self.mutate(ind_2, factor=np.random.default_rng(seed).normal(0.01, 0.1))
            )
        idx_survived = np.random.default_rng().choice(
            scores_idx[: len(population)], self._k
        )
        for idx in idx_survived:
            self._population.append(
                self.mutate(
                    population[idx], factor=np.random.default_rng(seed).normal(0.1, 0.1)
                )
            )
        self._iter += 1
        return to_stop, np.array(population[scores_idx[0]]), scores[0]


class SGDOptimizer(BaseOptimizer):
    """
    Stochastic Gradient Descent Optimizer
    """
    def __init__(self, alpha: float = 0.0, eta: float = 1e-3, **kwargs):
        """
        Constructor method
        Parameters
        ----------
        :param alpha: momentum constant, `0` by default
        :type alpha: float
        :param eta: learning rate constant, `0.001` by default
        :type eta: float
        """
        self._alpha = alpha
        self._eta = eta
        self._score = []
        self._tol = kwargs.pop("tol", 1e-2)
        self._v_t = []
        super().__init__(**kwargs)

    def apply(
        self,
        loss: tp.Callable[
            [np.ndarray, np.ndarray, np.ndarray, tp.Optional[tp.Dict[str, float]]],
            float,
        ],
        input_tensor: np.ndarray,
        output_tensor: np.ndarray,
        W: np.ndarray,
        **kwargs,
    ):
        """
        Perform one step of SGD

        Parameters
        ----------

        :param loss: Loss fitness function to minimize
        :type loss: callable

        :param input_tensor: Global input to FFN, i.e. your `X` variable
        :type input_tensor:  np.ndarray

        :param output_tensor: Desired FFN response, i.e. your `Y` variable 
        :type output_tensor: np.ndarray
        
        :param W: Initial network weights
        :type W: np.ndarray
        
        Returns
        -------

        :returns: tuple (reached tolerance flag, corrected weights, loss value)
        :rtype: Union [np.ndarray, float]
        """
        verbose = kwargs.pop("verbose", False)
        to_stop = False
        loss_grad = grad(loss)
        if not (len(self._v_t)):
            self._v_t = np.zeros_like(W)
        self._score.append(loss(W, input_tensor, output_tensor)[0])
        if verbose:
            print(f"train score - {self._score[-1]}")
        grad_W = np.clip(loss_grad(W, input_tensor, output_tensor), -1e6, 1e6,)
        if self._score[-1] <= self._tol:
            to_stop = True
        self._v_t = self._alpha * self._v_t + (1.0 - self._alpha) * grad_W
        W -= self._eta * self._v_t
        return to_stop, W, self._score[-1]


class ConjugateSGDOptimizer(BaseOptimizer):
    """
    Conjugate Stochastic Gradient Descent Optimiser
    """
    def __init__(self, eta: float = 1e-3, **kwargs):
        """
        Constructor method

        Parameters
        ----------
        :param eta: learning rate constant, `0.001` by default
        :type eta: float
        """
        self._eta = eta
        self._score = []
        self._tol = kwargs.pop("tol", 1e-2)
        self._g_prev = None
        self._p_prev = None
        self._k = 0
        super().__init__(**kwargs)

    def apply(
        self,
        loss: tp.Callable[
            [np.ndarray, np.ndarray, np.ndarray, tp.Optional[tp.Dict[str, float]]],
            float,
        ],
        input_tensor: np.ndarray,
        output_tensor: np.ndarray,
        W: np.ndarray,
        **kwargs,
    ):
        """
        Perform one step of Conjugate SGD

        Parameters
        ----------

        :param loss: Loss fitness function to minimize
        :type loss: callable

        :param input_tensor: Global input to FFN, i.e. your `X` variable
        :type input_tensor:  np.ndarray

        :param output_tensor: Desired FFN response, i.e. your `Y` variable 
        :type output_tensor: np.ndarray
        
        :param W: Initial network weights
        :type W: np.ndarray
        
        Returns
        -------

        :returns: union (reached tolerance flag, corrected weights, loss value)
        :rtype: Union [np.ndarray, float]
        """
        verbose = kwargs.pop("verbose", False)
        loss_grad = grad(loss)

        self._score.append(loss(W, input_tensor, output_tensor)[0])
        if verbose:
            print(f"train score - {self._score[-1]}")

        g = np.clip(loss_grad(W, input_tensor, output_tensor), -1e6, 1e6,)
        d = W.shape[0]

        if (self._k == 0) or (d % self._k == 0):
            p = -(g / np.linalg.norm(g))
        else:
            beta = np.dot(g, g) / np.dot(self._g_prev, self._g_prev)
            p = -(g + beta * self._p_prev) / np.linalg.norm(-g + beta * self._p_prev)

        self._g_prev = g
        self._p_prev = p

        W += self._eta * p

        if self._score[-1] <= self._tol:
            to_stop = True
        else:
            to_stop = False

        return to_stop, W, self._score[-1]
