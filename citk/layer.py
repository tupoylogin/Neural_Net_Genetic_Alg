import typing as tp

import autograd.numpy as np
from autograd.scipy.signal import convolve, compute_conv_size
from autograd.differential_operators import elementwise_grad

from .functions import GaussianRBF, ReLU, Linear
from .utils import weights_on_batch


class WeightsParser(object):
    """A helper class to index into a parameter vector."""

    def __init__(self):
        self.idxs_and_shapes = {}
        self.N = 0

    def add_weights(self, name, shape):
        start = self.N
        self.N += np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, self.N), shape)

    def get(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)


class BaseLayer:
    def __init__(
        self, nonlinearity: tp.Callable[[tp.Any], np.ndarray], *args, **kwargs
    ):
        """
        Base Layer Structure
        """
        self.parser = WeightsParser()
        self.nonlinearity = nonlinearity
        self.number = kwargs.get("number", 0)

    @property
    def parser(self):
        return self._parser

    @parser.setter
    def parser(self, value: WeightsParser):
        self._parser = value

    def build_weights_dict(self, *args):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """
        Performs forward pass logic of layer
        """
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__ + str(self.number)


class Conv2D(BaseLayer):
    def __init__(
        self,
        kernel_shape: tp.Tuple[int],
        num_filters: int,
        mode: str,
        nonlinearity: tp.Callable[[tp.Any], np.ndarray],
    ):
        self.kernel_shape = kernel_shape
        self.num_filters = num_filters
        self._mode = mode
        super().__init__(nonlinearity=nonlinearity)

    def forward(self, inputs: np.ndarray, param_vector: np.ndarray):
        # Input dimensions:  [data, color_in, y, x]
        # Params dimensions: [color_in, color_out, y, x]
        # Output dimensions: [data, color_out, y, x]
        params = self.parser.get(param_vector, "params")
        biases = self.parser.get(param_vector, "biases")
        conv = convolve(
            inputs, params, axes=([2, 3], [2, 3]), dot_axes=([1], [0]), mode=self._mode
        )
        return conv + biases

    def build_weights_dict(self, input_shape: tp.Tuple[int]):
        # Input shape : [color, y, x] (don't need to know number of data yet)
        self.parser.add_weights(
            "params", (input_shape[0], self.num_filters) + self.kernel_shape
        )
        self.parser.add_weights("biases", (1, self.num_filters, 1, 1))
        output_shape = (self.num_filters,) + self.conv_output_shape(
            input_shape[1:], self.kernel_shape
        )
        return self.parser.N, output_shape

    def conv_output_shape(self, A, B):
        return compute_conv_size(A, B, self._mode)


class MaxPool(BaseLayer):
    def __init__(self, pool_shape, nonlinearity: tp.Callable[[tp.Any], np.ndarray]):
        self.pool_shape = pool_shape
        super().__init__(nonlinearity=nonlinearity)

    def build_weights_dict(self, input_shape):
        # input_shape dimensions: [color, y, x]
        output_shape = list(input_shape)
        for i in [0, 1]:
            assert (
                input_shape[i + 1] % self.pool_shape[i] == 0
            ), "maxpool shape should tile input exactly"
            output_shape[i + 1] = input_shape[i + 1] / self.pool_shape[i]
        return 0, output_shape

    def forward(self, inputs, param_vector):
        new_shape = inputs.shape[:2]
        for i in [0, 1]:
            pool_width = self.pool_shape[i]
            img_width = inputs.shape[i + 2]
            new_shape += (img_width // pool_width, pool_width)
        result = inputs.reshape(new_shape)
        return np.max(np.max(result, axis=3), axis=4)


class Dense(BaseLayer):
    def __init__(self, size, nonlinearity: tp.Callable[[tp.Any], np.ndarray]):
        """
        Dense Layer

        Args:
            size (int): number of units.
            nonlinearity (callable): activation function.
        """
        self.size = size
        super().__init__(nonlinearity=nonlinearity)

    def build_weights_dict(self, input_shape):
        # Input shape is anything (all flattened)
        input_size = np.prod(input_shape, dtype=int)
        self.parser.add_weights("params", (input_size, self.size))
        self.parser.add_weights("biases", (self.size,))
        return self.parser.N, (self.size,)

    def forward(self, inputs, param_vector):
        params = self.parser.get(param_vector, "params")
        biases = self.parser.get(param_vector, "biases")
        if inputs.ndim > 2:
            inputs = inputs.reshape((inputs.shape[0], np.prod(inputs.shape[1:])))
        return self.nonlinearity(np.dot(inputs[:, :], params) + biases)


class RBFDense(BaseLayer):
    def __init__(self, size: int):
        """
        Gaussian RBF Dense Layer

        Args:
            size (int): numberof units.
            nonlinearity (callable): activation function.
        """
        self.size = size
        self.rbf = GaussianRBF
        super().__init__(nonlinearity=Linear)

    def build_weights_dict(self, input_shape):
        # Input shape is anything (all flattened)
        input_size = np.prod(input_shape, dtype=int)
        self.parser.add_weights("mu", (input_size, self.size))
        self.parser.add_weights("sigma", (self.size,))
        return self.parser.N, (self.size,)

    def forward(self, inputs, param_vector):
        mu = self.parser.get(param_vector, "mu")
        sigma = self.parser.get(param_vector, "sigma")
        if inputs.ndim > 2:
            inputs = inputs.reshape((inputs.shape[0], np.prod(inputs.shape[1:])))
        inputs = np.tile(np.expand_dims(inputs[:, :], axis=-1), self.size)
        return self.nonlinearity(self.rbf(inputs, mu, sigma))


class Fuzzify(BaseLayer):
    def __init__(
        self,
        num_rules: int,
        msf: tp.Callable[[tp.Any], np.ndarray],
        nonlinearity: tp.Callable[[tp.Any], np.ndarray] = Linear,
    ):
        """
        Fuzzification Layer

        Args:
            num_rules (int): number of rules.
            msf (callable): membership function.
        """
        self.size = num_rules
        self.msf = msf
        super().__init__(nonlinearity=nonlinearity)

    def build_weights_dict(self, input_shape):
        # Input shape is anything (all flattened)
        input_size = np.prod(input_shape, dtype=int)
        self.parser.add_weights("a", (input_size, self.size,))
        self.parser.add_weights("c", (input_size, self.size,))
        self.parser.add_weights("r", (self.size,))
        return self.parser.N, (self.size,)

    def forward(self, inputs, param_vector):
        a = self.parser.get(param_vector, "a")[np.newaxis, :]
        c = self.parser.get(param_vector, "c")[np.newaxis, :]
        r = self.parser.get(param_vector, "r")[np.newaxis, :]
        if inputs.ndim > 2:
            inputs = inputs.reshape((inputs.shape[0], np.prod(inputs.shape[1:])))
        inputs = inputs[..., np.newaxis]
        w = self.msf(inputs, a, c)
        w = np.prod(w, axis=1)
        f = np.sum(a * inputs, axis=1) + r
        o = w * f / np.sum(w, axis=1, keepdims=True)
        return self.nonlinearity(o)
