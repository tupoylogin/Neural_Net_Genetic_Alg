import typing as tp

import autograd.numpy as np
from autograd.differential_operators import elementwise_grad

from .functions import ReLU
from .utils import weights_on_batch

class BaseLayer():
    def __init__(self, *args, **kwargs):
        """
        Base Layer Structure
        """
        self.weights = None
    
    @property
    def weights(self):
        return self._weights
    
    @weights.setter
    def weights(self, value: tp.Union[int, float]):
        self._weights = value

    def forward(self, *args, **kwargs):
        """
        Performs forward pass logic of layer
        """
        raise NotImplementedError


class Dense(BaseLayer):
    def __init__(self,
        input_size: int,
        num_units: int, 
        kernel_initializer: tp.Optional[tp.Callable[..., np.ndarray]] = None,
        bias_initializer: tp.Optional[tp.Callable[..., np.ndarray]] = None,
        activation: tp.Optional[tp.Callable[..., tp.Any]] = None,
        lambda_reg: tp.Optional[float] = None,
        regularization: tp.Optional[str] = None,
        **kwargs):
        """
        Dense Layer

        Args:
            input_size (int): number of input neurons.
            output_size (int): number of onput neurons.
            kernel_initializer (callable): probability distribution of weights.
            bias_initializer (callable): probability distribution of bias coefficients.
            acivation (callable): activation function.
            lambda_reg (float): regularization coefficient.
            regularization (callable): regularization function.
        """
        super().__init__(**kwargs)
        self._number = kwargs.pop('number', 0)
        if kernel_initializer is None:
            weights = np.random.default_rng(42).normal(0, 0.1, size=(input_size, num_units))
        else:
            weights = kernel_initializer((input_size, num_units))
        if bias_initializer is None:
            bias = np.random.default_rng().uniform(size=(1, num_units))
        else:
            bias = bias_initializer((1, num_units))
        self.weights = np.vstack((weights, bias))
        self.activation = ReLU if activation is None else activation
        self.lambda_reg = 0. if lambda_reg is None else lambda_reg
        if regularization:
            if regularization == 'l1':
                self.regularization = lambda x: self.lambda_reg*np.sum(np.abs(x))
            elif regularization == 'l2':
                self.regularization = lambda x: self.lambda_reg*np.sum(np.power(x, 2))
        else:
            self.regularization = None
    
    def forward(self, X: np.ndarray):
        """
        Forward pass of layer

        Args:
            X (ndarray): input tensor
        Returns:
            output (ndarray): response of layer
        """
        X_plus_bias = np.hstack((X,np.ones(X.shape[0]).reshape(-1,1)))
        output = self.activation(np.sum(weights_on_batch(self.weights, X_plus_bias),-1))
        if self.regularization:
            output += self.regularization(self.weights)
        return output
    
    def backward(self, X: np.ndarray):
        """
        Backward pass of layer

        Args:
            X (ndarray): input tensor
        Returns:
            output (ndarray): gradient of layer with respect to weights
        """
        X_plus_bias = np.hstack((X,np.ones(X.shape[0]).reshape(-1,1)))
        grad = elementwise_grad(self.activation)(weights_on_batch(self.weights, X_plus_bias))
        output = np.mean(np.array([grad[i]*X_plus_bias[i] for i in range(grad.shape[0])]), axis=0)
        if self.regularization:
            reg_grad = elementwise_grad(self.regularization)(self.weights)
            output += reg_grad.T
        return output
