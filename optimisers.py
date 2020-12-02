from copy import deepcopy
from functools import partial
from threading import setprofile
import typing as tp

from autograd import grad
import numpy as np

from layer import BaseLayer
from utils import cast_to_same_shape

class BaseOptimizer():
    def __init__(self, *args, **kwargs):
        """
        Estimator Wrapper
        """
        pass

    def apply(self, loss: tp.Callable[..., float], graph: tp.List[BaseLayer]):
        raise NotImplementedError

class GeneticAlgorithmOptimizer(BaseOptimizer):
    def __init__(self, num_population: int, k: int = 5, **kwargs):
        """
        Genetic Algorithm Optimiser

        Args:
            num_population (int): number of individuals in each population
            k (int): number of crossover rounds to perform
        """
        self._num_population = num_population
        self._k = k
        self._population = None
        self._best_iter = None
        self._last_score = np.inf
        self._iter = 0
        self._tol = kwargs.pop('tol', 1e-2)
        super().__init__(**kwargs)
    
    @staticmethod
    def construct_genome(W: np.ndarray,
                    weight_init: tp.Callable[..., np.ndarray]):
        """
        Construct random population

        Args:
            layers_list (list(BaseLayer)): genotype, i.e. FFN template to mimic to
            weight_init (callable): weight distribution function
        
        Returns:
            genome (np.ndarray): initalized weights
        """
        return weight_init(0, 0.1, size=W.shape)
        

    @staticmethod
    def crossover(ind_1: np.ndarray, 
                ind_2: np.ndarray) -> np.ndarray:
        """
        Perform simple crossover

        Args:
            ind_1 (np.ndarray): FFN layers weights, first individual
            ind_2 (np.ndarray): FFN layers weights, second individual
        
        Returns:
            ind_12, ind_21 (np.ndarray): generated offsprings
        """
        assert len(ind_1) == len(ind_2), 'individuals must have same len'
        index = np.random.default_rng().integers(len(ind_1))
        ind_12 = np.concatenate((ind_1[:index], ind_2[index:]), axis=None)
        ind_21 = np.concatenate((ind_2[:index], ind_1[index:]), axis=None)
        return ind_12, ind_21

    @staticmethod
    def mutate(ind: np.ndarray, 
                    mu: float = 0.,
                    sigma: float = 1.,
                    factor: float = 0.01) -> tp.List[BaseLayer]:
        """
        Perform simple mutation

        Args:
            ind (np.ndarray): FFN layers weights
            mu (float): mean of distribution
            sigma (float): scale of distribution
            factor (float): scale factor of mutation
        
        Returns:
            ind (np.ndarray): generated individual
        """
        ind += factor*np.random.default_rng().normal(loc=mu, 
                scale=sigma, size=len(ind))
        return ind
    
    def apply(self, 
            loss: tp.Callable[[np.ndarray, np.ndarray, tp.Optional[tp.Dict[str, float]]], float], 
            input_tensor: np.ndarray, 
            output_tensor: np.ndarray, 
            W: np.ndarray,
            **kwargs):
        """
        Perform one step of GA

        Args:
            loss (callable): `inverse` fitness function to minimize
            input_tensor (ndarray): global input to FFN, i.e. your `X` variable
            output_tensor (ndarray): desired FFN response, i.e. your `Y` variable 
            graph (list(BaseLayer)): your FFN structure
        
        Returns:
            best_iter (list(BaseLayer)): best individual so far
            score (float): lowest loss so far
        """
        verbose = kwargs.pop('verbose', False)
        to_stop = False
        if not(self._population):
            population =[self.construct_genome(W, np.random.default_rng().normal) 
            for i in range(self._num_population)]
        else:
            population = self._population[:]
        scores=[]
        for g in population:
            scores.append(loss(g, input_tensor, output_tensor)[0])
        scores, scores_idx = np.sort(scores), np.argsort(scores)
        if verbose:
            print(f'best individual - {scores[0]}')
        if self._last_score>scores[0]:
            self._last_score = scores[0]
            self._best_iter = population[scores_idx[0]]
        else:
            self._best_iter = population[0]
        if scores[0]<self._tol:
            to_stop = True
        self._population = np.array(population)[scores_idx][:self._num_population-self._k*3].tolist()
        probas = np.exp(-1*scores)/np.sum(np.exp(-1*scores))
        for _ in range(self._k):
            indices = np.random.default_rng().choice(scores_idx, 2, p=probas)
            ind_1, ind_2 = self.crossover(population[indices[0]], population[indices[1]])
            self._population.append(self.mutate(ind_1, factor=np.random.default_rng().normal(0,0.001)))
            self._population.append(self.mutate(ind_2, factor=np.random.default_rng().normal(0,0.001)))
        idx_survived = np.random.default_rng().choice(scores_idx[:len(population)], self._k)
        for idx in idx_survived:
            self._population.append(self.mutate(population[idx], factor=np.random.default_rng().normal(0,0.01)))
        self._iter += 1
        return to_stop, np.array(self._best_iter), scores[0]

class SGDOptimizer(BaseOptimizer):
    def __init__(self, alpha: float = 0., eta: float = 1e-3, **kwargs):
        """
        Stochastic Gradient Descent Optimiser

        Args:
            alpha (float): momentum constant, 0 by default
            eta (float): learning rate constant, 0.001 by default
            batch_size (int): batch_size for calculations
        """
        self._alpha = alpha
        self._eta = eta
        self._score = []
        self._tol = kwargs.pop('tol', 1e-2)
        self._batch_size = kwargs.pop('batch_size', 1)
        self._v_t = []
        super().__init__(**kwargs)
    
    def apply(self, 
            loss: tp.Callable[[np.ndarray, np.ndarray, np.ndarray, 
                                tp.Optional[tp.Dict[str, float]]], float], 
            input_tensor: np.ndarray, 
            output_tensor: np.ndarray, 
            W: np.ndarray,
            **kwargs):
        """
        Perform one step of SGD

        Args:
            loss (callable): loss function to minimize
            input_tensor (ndarray): global input to FFN, i.e. your `X` variable
            output_tensor (ndarray): desired FFN response, i.e. your `Y` variable 
            graph (list(BaseLayer)): your FFN structure
        
        Returns:
            graph (list(BaseLayer)): FFN struture with corrected weights
            score (float): loss on current iteration
        """
        verbose = kwargs.pop('verbose', False)
        to_stop = False
        loss_grad = grad(loss)
        if not(len(self._v_t)):
            self._v_t = np.zeros_like(W)
        rand_subset = np.random.default_rng().choice(range(input_tensor.shape[0]), self._batch_size)
        self._score.append(loss(W, input_tensor, output_tensor)[0])
        if verbose:
            print(f'train score - {self._score[-1]}')
        grad_W = loss_grad(W, input_tensor[rand_subset], 
                            output_tensor[rand_subset])
        if self._score[-1]<=self._tol:
            to_stop = True
        self._v_t = self._alpha * self._v_t + (1.0 - self._alpha) * grad_W
        param = W - self._eta * self._v_t
        return to_stop, param, self._score[-1]

class ConjugateSGDOptimizer(BaseOptimizer):
    def __init__(self, eta: float = 1e-3, **kwargs):
        """
        Conjugate Stochastic Gradient Descent Optimiser

        Args:
            eta (float): learning rate constant, 0.01 by default
            batch_size (int): batch_size for calculations
        """
        self._eta = eta
        self._score = []
        self._tol = kwargs.pop('tol', 1e-2)
        self._batch_size = kwargs.pop('batch_size', 1)
        self._g_prev = None
        self._p_prev = None
        self._k = 0
        super().__init__(**kwargs)
    
    def apply(self, 
            loss: tp.Callable[[np.ndarray, np.ndarray, np.ndarray, 
                                tp.Optional[tp.Dict[str, float]]], float], 
            input_tensor: np.ndarray, 
            output_tensor: np.ndarray, 
            W: np.ndarray,
            **kwargs):
        """
        Perform one step of ConjugateSGDOptimizer

        Args:
            loss (callable): loss function to minimize
            input_tensor (ndarray): global input to FFN, i.e. your `X` variable
            output_tensor (ndarray): desired FFN response, i.e. your `Y` variable 
            W (ndarray): NN weight matrix
        
        Returns:
            to_stop (bool): Flag, which indicates that tollerance was reached 
            W (ndarray): updated NN weight matrix
            score (float): loss on current iteration
        """
        W_vect = W
        verbose = kwargs.pop('verbose', False)
        loss_grad = grad(loss)
        
        rand_subset = np.random.default_rng().choice(range(input_tensor.shape[0]), self._batch_size)
        self._score.append(loss(W, input_tensor, output_tensor)[0])
        if verbose:
            print(f'train score - {self._score[-1]}')

        g = loss_grad(W_vect, input_tensor[rand_subset], output_tensor[rand_subset])
        d = W_vect.shape[0]

        if (self._k == 0) or (d % self._k == 0):
            p = - (g / np.linalg.norm(g))
        else:
            beta = (g @ g) / (self._g_prev @ self._g_prev)
            p = - (g + beta*self._p_prev) / np.linalg.norm( -g + beta*self._p_prev)

        self._g_prev = g
        self._p_prev = p
        
        W_vect += self._eta * p

        if self._score[-1]<=self._tol:
            to_stop = True
        else:
            to_stop = False

        return to_stop, np.array(W_vect), self._score[-1]
