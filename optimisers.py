from copy import deepcopy
from functools import partial
from threading import setprofile
import typing as tp

from autograd import elementwise_grad
import numpy as np

from .layer import BaseLayer
from .utils import cast_to_same_shape

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
        self._tol = kwargs.pop('tol', 1e-5)
        super().__init__(**kwargs)
    
    @staticmethod
    def construct_genome(layers_list: tp.List[BaseLayer],
                    weight_init: tp.Callable[..., np.ndarray]):
        """
        Construct random population

        Args:
            layers_list (list(BaseLayer)): genotype, i.e. FFN template to mimic to
            weight_init (callable): weight distribution function
        
        Returns:
            layers (list(BaseLayer)): FFN layers list with initalized weights
        """
        layers = deepcopy(layers_list)
        for l in layers:
            l.weights = weight_init(0, 0.1, size=l.weights.shape)
        return layers

    @staticmethod
    def crossover(ind_1: tp.List[BaseLayer], ind_2: tp.List[BaseLayer]) -> tp.List[BaseLayer]:
        """
        Perform simple crossover

        Args:
            ind_1 (list(BaseLayer)): FFN layers list, first individual
            ind_2 (list(BaseLayer)): FFN layers list, second individual
        
        Returns:
            ind_12, ind_21 (tuple(list(BaseLayer))): generated offsprings
        """
        assert len(ind_1) == len(ind_2), 'individuals must have same len'
        index = np.random.default_rng().integers(len(ind_1))
        ind_12 =  ind_1[:index]+ind_2[index:]
        ind_21 =  ind_2[:index]+ind_1[index:]
        return ind_12, ind_21

    @staticmethod
    def mutate(ind: tp.List[BaseLayer], 
                    mu: float = 0.,
                    sigma: float = 1.,
                    factor: float = 0.01) -> tp.List[BaseLayer]:
        """
        Perform simple mutation

        Args:
            ind (list(BaseLayer)): FFN layers list
            mu (float): mean of distribution
            sigma (float): scale of distribution
            factor (float): scale factor of mutation
        
        Returns:
            ind (list(BaseLayer)): generated individual
        """
        for l in ind:
            l.weights += factor*np.random.default_rng().normal(loc=mu, 
                scale=sigma, size=l.weights.shape)
        return ind
    
    def apply(self, 
            loss: tp.Callable[[np.ndarray, np.ndarray, tp.Optional[tp.Dict[str, float]]], float], 
            input_tensor: np.ndarray, 
            output_tensor: np.ndarray, 
            graph: tp.List[BaseLayer],
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
        if not(self._population):
            population =[self.construct_genome(graph, np.random.default_rng().normal) 
            for i in range(self._num_population)]
        else:
            population = self._population[:]
        scores=[]
        for g in population:
            res = input_tensor
            for idx, layer in enumerate(g):
                res = layer.forward(res)
            y_pred = res.ravel()
            cast_to_same_shape(y_pred, output_tensor)
            scores.append(loss(output_tensor, y_pred))
        scores, scores_idx = np.sort(scores), np.argsort(scores)
        if verbose:
            print(f'best individual - {scores[0]}')
        if self._last_score>scores[0]:
            self._last_score = scores[0]
            self._best_iter = population[scores_idx[0]]
        else:
            self._best_iter = population[0]
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
        return self._best_iter, scores[0]

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
        self._tol = kwargs.pop('tol', 1e-3)
        self._batch_size = kwargs.pop('batch_size', 1)
        self._v_t = []
        super().__init__(**kwargs)
    
    def apply(self, 
            loss: tp.Callable[[np.ndarray, np.ndarray, tp.Optional[tp.Dict[str, float]]], float], 
            input_tensor: np.ndarray, 
            output_tensor: np.ndarray, 
            graph: tp.List[BaseLayer],
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
        if not(self._v_t):
            self._v_t = [np.zeros_like(l.weights) for l in graph][::-1]
        rand_subset = np.random.default_rng().choice(range(input_tensor.shape[0]), self._batch_size)
        res, grads = [], []
        res.append(input_tensor)
        for layer in graph:
            res.append(layer.forward(res[-1]))
        cast_to_same_shape(res[-1], output_tensor)
        self._score.append(loss(output_tensor, res[-1].ravel()))
        if verbose:
            print(f'train score - {self._score[-1]}')
        for idx, layer in enumerate(graph[::-1]):
            if idx==0:
                loss_grad = np.sum(elementwise_grad(loss, 1)(output_tensor[rand_subset], 
                                                            res[-1].ravel()[rand_subset]))
                grads.append(loss_grad*layer.backward(res[-2][rand_subset]))
            else:
                grd = layer.backward(res[-idx-2][rand_subset])
                grads.append(grads[idx-1][:, :-1].T*grd)
            self._v_t[idx] = self._alpha*self._v_t[idx]-(1-self._alpha)*grads[idx].T
            layer.weights += self._eta*self._v_t[idx]
        return graph, self._score[-1]

class ConjugateSGDOptimizer(BaseOptimizer):
    def __init__(self, alpha: float, eta: float = 1e-3, **kwargs):
        self._alpha = alpha if alpha else 0.
        self._eta = eta
        self._score = []
        self._tol = kwargs.pop('tol', 1e-3)
        self._batch_size = kwargs.pop('batch_size', 1)
        super().__init__(**kwargs)
    
    def apply(self, 
            loss: tp.Callable[[np.ndarray, np.ndarray, tp.Optional[tp.Dict[str, float]]], float], 
            input_tensor: np.ndarray, 
            output_tensor: np.ndarray, 
            graph: tp.List[BaseLayer],
            **kwargs): 
        verbose = kwargs.pop('verbose', False)
        rand_subset = np.random.default_rng().choice(range(input_tensor.shape[0]), self._batch_size)
        res, grads = [], []
        res.append(input_tensor)
        for layer in graph:
            res.append(layer.forward(res[-1]))
        cast_to_same_shape(res[-1], output_tensor)
        self._score.append(loss(output_tensor, res[-1].ravel()))
        if verbose:
            print(f'train score - {self._score[-1]}')
        for idx, layer in enumerate(graph[::-1]):
            if idx==0:
                loss_grad = np.sum(elementwise_grad(loss, 1)(output_tensor[rand_subset], 
                                                            res[-1].ravel()[rand_subset]))
                grads.append(loss_grad*layer.backward(res[-2][rand_subset]))
            else:
                grd = layer.backward(res[-idx-2][rand_subset])
                grads.append(grads[idx-1][:, :-1].T*grd)
            self._v_t[idx] = self._alpha*self._v_t[idx]-(1-self._alpha)*grads[idx].T
            layer.weights += self._eta*self._v_t[idx]
        return graph, self._score[-1]
