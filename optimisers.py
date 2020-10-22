from functions import BaseFunction, Variable
from layer import BaseLayer
import typing as tp

import numpy as np

class BaseOptimizer():
    def __init__(self, *args, **kwargs):
        """
        Estimator Wrapper
        """
        pass

    def apply(self, loss: BaseFunction, graph: tp.List[BaseLayer]):
        #TODO: develop some code logic
        raise NotImplementedError
    
    def create_initial_point(self, graph: tp.List[BaseLayer]) -> tp.List[tp.List[BaseLayer]]:
        return [graph]

class GeneticAlgorithmOptimizer(BaseOptimizer):
    def __init__(self, num_population: int, k: int = 5, **kwargs):
        self._num_population = num_population
        self._k = k
        self._population = None
        self._best_score = []
        self._best_ind = []
        self._tol = kwargs.pop('tol', 1e-3)
        super().__init__(**kwargs)
    
    @staticmethod
    def construct_genome(layers_list: tp.List[BaseLayer],
                    weight_init: tp.Callable[..., np.ndarray]):
        layers = layers_list[:]
        for layer in layers:
            layer.weights.value = weight_init(size=layer.weights.value.shape)
        return layers

    @staticmethod
    def crossover(ind_1: tp.List[BaseLayer], ind_2: tp.List[BaseLayer]) -> tp.List[BaseLayer]:
            assert len(ind_1) == len(ind_2), 'individuals must have same len'
            index = np.random.default_rng().integers(len(ind_1))
            ind_12 =  ind_1[:index]+ind_2[index:]
            ind_21 =  ind_2[:index]+ind_1[index:]
            return ind_12, ind_21

    @staticmethod
    def mutate(ind: tp.List[BaseLayer], 
                    mu: float = 0.,
                    sigma: float = 1.,
                    factor: float = 0.3) -> tp.List[BaseLayer]:
        for l in ind:
            l.weights.value += factor*np.random.default_rng().normal(loc=mu, 
                scale=sigma**2, size=l.weights.value.shape)
        return ind

    def create_point(self, graph: tp.List[BaseLayer]) -> tp.List[tp.List[BaseLayer]]:
        return [construct_genome(graph, 
                        np.random.default_rng().normal) 
                        for _ in range(self._num_population)]
    
    def apply(self, loss: tp.Callable[..., BaseFunction], 
            input_tensor: np.ndarray, 
            output_tensor: np.ndarray, 
            graph: tp.List[BaseLayer]):
        np.random.seed(42)
        to_stop = False
        if self._population is None:
            population = [self.construct_genome(graph, np.random.default_rng().normal) 
                            for _ in range(self._num_population)]
        else:
            population = self._population[:]
        scores = []
        y_true = Variable('y_true') 
        y_true.value = output_tensor.reshape(-1,1)
        y_pred = Variable('y_pred')
        lossfn = loss(y_true, y_pred)
        for g in population:
            res = input_tensor
            for layer in g:
                res = layer.forward(res)
            y_pred.value = np.array(res).reshape(-1,1)
            scores.append(lossfn(at=[y_true, y_pred]))
        
        scores, scores_idx = np.sort(scores), np.argsort(scores)
        print(scores[0])
        print(f'best individual - {scores_idx[0]}')
        self._best_score.append(scores[0])
        self._best_ind.append([l.weights for l in population[scores_idx[0]]])
        if (len(self._best_score)>1) and (self._best_score[-2]-self._best_score[-1]<self._tol):
            to_stop=True
        best_population = []
        for mt in range(self._k):
            indices = np.random.default_rng().choice(scores_idx, 2)
            ind_1, ind_2 = self.crossover(population[indices[0]], population[indices[1]])
            best_population.append(self.mutate(ind_1, factor=np.random.default_rng().uniform(0,1)))
            best_population.append(self.mutate(ind_2, factor=np.random.default_rng().uniform(0,1)))
        idx_survived = np.random.default_rng().choice(scores_idx[:len(population)//2],
                                                    len(population)-len(best_population))
        for idx in idx_survived:
            best_population.append(self.mutate(population[idx], factor=np.random.default_rng().uniform(0,5)))
        self._population = best_population
        return population[0], to_stop 


        

        
        
        
        
        
        
        
        
            
        


        
        
