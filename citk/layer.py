from datetime import datetime
from itertools import product, combinations
import typing as tp

import autograd.numpy as np
from autograd.scipy.signal import convolve, compute_conv_size
from numpy.core.function_base import linspace
from scipy.optimize import fsolve, root
from autograd.differential_operators import elementwise_grad

from .functions import GaussianRBF, ReLU, Linear, Poly, Sigmoid, Tanh
from .utils import centroid, concat_and_multiply, nCr

SEED = int(datetime.utcnow().timestamp() * 1e5)
POSSIBLE_POLI_TYPES = ["linear", "partial_quadratic", "quadratic"]


class WeightsParser(object):
    """
    A helper class to index into a parameter vector.
    ------------------------------------------------
    """

    def __init__(self):
        """
        Constructor method
        """
        self.idxs_and_shapes = {}
        self.N = 0

    def add_weights(self, name: str, shape: tp.Tuple[int]):
        """
        Helper tool to add weights to ANN Layers

        Parameters
        ----------

        :param name: name of layer/weights set.
        :type name: str

        :param shape: shape of layer/weights set.
        :type shape: tuple

        """
        start = self.N
        self.N += np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, self.N), shape)

    def get(self, vect: np.ndarray, name: str):
        """
        Helper tool to parse weights from ANN Layers

        Parameters
        ----------

        :param vect: vector of weights.
        :type vect: np.ndarray

        :param name: name of layer/weights set.
        :type name: str

        """
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)


class BaseLayer:
    """
    Base Layer class
    -----------------
    All custom layers should be inherited from this class.

    :param parser: Weights Parser
    :type parser: WeightsParser
    """

    def __init__(
        self, nonlinearity: tp.Callable[[tp.Any], np.ndarray], *args, **kwargs
    ):
        """
        Base Layer Constructor
        """
        self.parser = WeightsParser()
        self.nonlinearity = nonlinearity
        self.number = kwargs.get("number", 0)
        self.initializer = kwargs.get("init", np.random.default_rng(SEED).normal)

    @property
    def parser(self):
        return self._parser

    @parser.setter
    def parser(self, value: WeightsParser):
        self._parser = value

    def build_weights_dict(self, *args):
        """
        Builds Weight Dictionary
        """
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """
        Performs forward pass logic of layer
        """
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__ + f"_{self.number}"


class Conv2D(BaseLayer):
    """
    2D Convolutional Layer
    ----------------------
    Useful for image classification tasks.
    """

    def __init__(
        self,
        kernel_shape: tp.Tuple[int],
        num_filters: int,
        mode: str,
        nonlinearity: tp.Callable[[tp.Any], np.ndarray],
        **kwargs,
    ):
        """
        Constructor method

        Parameters
        ----------

        :kernel_shape: Convolution kernel shape.
        :type kernel_shape: tuple

        :num_filters: Number of convolutional filters
        :type num_filters: int

        :mode: Convolution mode aka padding. Must be either `valid` or `same`
        :type mode: str

        :nonlinearity: Activation function.
        :type nonlinearity: callable
        """
        self.kernel_shape = kernel_shape
        self.num_filters = num_filters
        self._mode = mode
        super().__init__(nonlinearity=nonlinearity, **kwargs)

    def forward(self, inputs: np.ndarray, param_vector: np.ndarray) -> np.ndarray:
        """
        Forward pass method

        Parameters
        ----------

        :inputs: Input matrix.
        :type inputs: np.ndarray

        :param_vector: Vector of network's weights.
        :type param_vector: np.ndarray

        Returns
        -------

        :return: Result of convolution
        :rtype: np.ndarray

        """
        # Input dimensions:  [data, color_in, y, x]
        # Params dimensions: [color_in, color_out, y, x]
        # Output dimensions: [data, color_out, y, x]
        params = self.parser.get(param_vector, "params")
        biases = self.parser.get(param_vector, "biases")
        conv = convolve(
            inputs, params, axes=([2, 3], [2, 3]), dot_axes=([1], [0]), mode=self._mode
        )
        return conv + biases

    def build_weights_dict(
        self, input_shape: tp.Tuple[int]
    ) -> tp.Union[int, tp.Tuple[int]]:
        """
        Weights builder

        Parameters
        ----------

        :input_shape: Input shape.
        :type input_shape: tuple

        Returns
        -------

        :return: union object (number_of_weights, _output_shape)
        :rtype: union
        """
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
    """
    Max Pooling layer
    """

    def __init__(
        self, pool_shape, nonlinearity: tp.Callable[[tp.Any], np.ndarray], **kwargs
    ):
        """
        Constructor method

        Parameters
        ----------

        :pool_shape: Max pooling shape.
        :type kernel_shape: tuple

        :nonlinearity: Activation function.
        :type nonlinearity: callable
        """
        self.pool_shape = pool_shape
        super().__init__(nonlinearity=nonlinearity, **kwargs)

    def build_weights_dict(self, input_shape: tp.Tuple[int]):
        """
        Weights builder

        Parameters
        ----------

        :input_shape: Input shape.
        :type input_shape: tuple

        Returns
        -------

        :return: union object (number_of_weights, _output_shape)
        :rtype: union
        """
        # input_shape dimensions: [color, y, x]
        output_shape = list(input_shape)
        for i in [0, 1]:
            assert (
                input_shape[i + 1] % self.pool_shape[i] == 0
            ), "maxpool shape should tile input exactly"
            output_shape[i + 1] = input_shape[i + 1] / self.pool_shape[i]
        return 0, output_shape

    def forward(self, inputs: np.ndarray, param_vector: np.ndarray):
        """
        Forward pass method

        Parameters
        ----------

        :inputs: Input matrix.
        :type inputs: np.ndarray

        :param_vector: Vector of network's weights. (ingored)
        :type param_vector: np.ndarray

        Returns
        -------

        :return: Result of pooling
        :rtype: np.ndarray

        """
        new_shape = inputs.shape[:2]
        for i in [0, 1]:
            pool_width = self.pool_shape[i]
            img_width = inputs.shape[i + 2]
            new_shape += (img_width // pool_width, pool_width)
        result = inputs.reshape(new_shape)
        return np.max(np.max(result, axis=3), axis=4)


class Dense(BaseLayer):
    """
    Dense Layer
    -----------
    The essential building block of an ANN.
    """

    def __init__(
        self, size: int, nonlinearity: tp.Callable[[tp.Any], np.ndarray], **kwargs
    ):
        """
        Constructor method

        Parameters
        ----------

        :size: number of units.
        :type size: int

        :nonlinearity: Activation function.
        :type nonlinearity: callable
        """
        self.size = size
        super().__init__(nonlinearity=nonlinearity, **kwargs)

    def build_weights_dict(self, input_shape):
        """
        Weights builder

        Parameters
        ----------

        :input_shape: Input shape.
        :type input_shape: tuple

        Returns
        -------

        :return: Union object (number_of_weights, _output_shape)
        :rtype: union
        """
        # Input shape is anything (all flattened)
        input_size = np.prod(input_shape, dtype=int)
        self.parser.add_weights("params", (input_size, self.size))
        self.parser.add_weights("biases", (self.size,))
        return self.parser.N, (self.size,)

    def forward(self, inputs, param_vector):
        """
        Forward pass method

        Parameters
        ----------

        :inputs: Input matrix.
        :type inputs: np.ndarray

        :param_vector: Vector of network's weights.
        :type param_vector: np.ndarray

        Returns
        -------

        :return: Nonlinearity applied to matrix multiplicationbetween weights and input
        :rtype: np.ndarray

        """
        params = self.parser.get(param_vector, "params")
        biases = self.parser.get(param_vector, "biases")
        if inputs.ndim > 2:
            inputs = inputs.reshape((inputs.shape[0], np.prod(inputs.shape[1:])))
        return self.nonlinearity(np.dot(inputs[:, :], params) + biases)


class RBFDense(BaseLayer):
    """
    Gaussian RBF Dense Layer
    ------------------------
    Building block of RBF-network
    """

    def __init__(self, hidden: int, out: int, **kwargs):
        """
        Constructor method

        Parameters
        ----------

        :hidden: Number of hidden units.
        :type hidden: int

        :out: Number of output units.
        :type out: int
        """
        self.hidden = hidden
        self.size = out
        self.rbf = GaussianRBF
        super().__init__(nonlinearity=Linear, **kwargs)

    def build_weights_dict(self, input_shape):
        """
        Weights builder

        Parameters
        ----------

        :input_shape: Input shape.
        :type input_shape: tuple

        Returns
        -------

        :return: Union object (number_of_weights, _output_shape)
        :rtype: union
        """
        # Input shape is anything (all flattened)
        input_size = np.prod(input_shape, dtype=int)
        self.parser.add_weights("mu", (input_size, self.hidden))
        self.parser.add_weights("sigma", (self.hidden,))
        self.parser.add_weights("params", (self.hidden, self.size))
        # self.parser.add_weights("biases", (self.size,))
        return self.parser.N, (self.size,)

    def forward(self, inputs, param_vector):
        """
        Forward pass method

        Parameters
        ----------

        :inputs: Input matrix.
        :type inputs: np.ndarray

        :param_vector: Vector of network's weights.
        :type param_vector: np.ndarray

        Returns
        -------

        :return: Nonlinearity applied to matrix multiplicationbetween weights and input
        :rtype: np.ndarray

        """
        mu = self.parser.get(param_vector, "mu")[np.newaxis, :]
        sigma = self.parser.get(param_vector, "sigma")[np.newaxis, :]
        params = self.parser.get(param_vector, "params")
        # biases = self.parser.get(param_vector, "biases")
        if inputs.ndim > 2:
            inputs = inputs.reshape((inputs.shape[0], np.prod(inputs.shape[1:])))
        inputs = inputs[..., np.newaxis]
        rbf = self.rbf(inputs, mu, sigma)
        return self.nonlinearity(np.dot(rbf, params))


class Fuzzify(BaseLayer):
    """
    Takagi-Sugeno controller
    ------------------------
    Main block for ANFIS-type networks
    """

    def __init__(
        self,
        num_rules: int,
        msf: tp.Callable[[tp.Any], np.ndarray],
        nonlinearity: tp.Callable[[tp.Any], np.ndarray] = Linear,
        **kwargs,
    ):
        """
        Constructor method

        Parameters
        ----------

        :num_rules: Number of rules to be developed.
        :type num_rules: int

        :msf: Membership function.
        :type out: callable

        :nonlinearity: Activation function.
        :type nonlinearity: callable

        """
        self.size = num_rules
        self.msf = msf
        super().__init__(nonlinearity=nonlinearity, **kwargs)

    def build_weights_dict(self, input_shape):
        """
        Weights builder

        Parameters
        ----------

        :input_shape: Input shape.
        :type input_shape: tuple

        Returns
        -------

        :return: Union object (number_of_weights, _output_shape)
        :rtype: union
        """
        # Input shape is anything (all flattened)
        input_size = np.prod(input_shape, dtype=int)
        self.parser.add_weights(
            "a",
            (
                input_size,
                self.size,
            ),
        )
        self.parser.add_weights(
            "c",
            (
                input_size,
                self.size,
            ),
        )
        self.parser.add_weights("r", (self.size,))
        return self.parser.N, (self.size,)

    def forward(self, inputs, param_vector):
        """
        Forward pass method

        Parameters
        ----------

        :inputs: Input matrix.
        :type inputs: np.ndarray

        :param_vector: Vector of network's weights.
        :type param_vector: np.ndarray

        Returns
        -------

        :return: Result of fuzzy-consequence
        :rtype: np.ndarray

        """
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


class NeoFuzzyLayer(BaseLayer):
    def __init__(
        self, num_rules: int, msf: tp.Callable[[tp.Any], np.ndarray], **kwargs
    ):
        self.size = num_rules
        self.msf = msf
        super().__init__(nonlinearity=Linear, **kwargs)

    def build_weights_dict(self, input_shape):
        """
        Weights builder

        Parameters
        ----------

        :input_shape: Input shape.
        :type input_shape: tuple

        Returns
        -------

        :return: Union object (number_of_weights, _output_shape)
        :rtype: union
        """
        # Input shape is anything (all flattened)
        input_size = np.prod(input_shape, dtype=int)
        self.parser.add_weights(
            "a",
            (
                1,
                input_size,
                self.size,
            ),
        )
        self.parser.add_weights(
            "c",
            (
                1,
                input_size,
                self.size,
            ),
        )
        self.parser.add_weights(
            "w",
            (
                1,
                input_size,
                self.size,
            ),
        )
        return self.parser.N, (self.size,)

    def forward(self, inputs, param_vector):
        """
        Forward pass method

        Parameters
        ----------

        :inputs: Input matrix.
        :type inputs: np.ndarray

        :param_vector: Vector of network's weights.
        :type param_vector: np.ndarray

        Returns
        -------

        :return: Result of fuzzy-consequence
        :rtype: np.ndarray

        """
        a = self.parser.get(param_vector, "a")
        c = self.parser.get(param_vector, "c")
        w = self.parser.get(param_vector, "w")
        if inputs.ndim > 2:
            inputs = inputs.reshape((inputs.shape[0], np.prod(inputs.shape[1:])))
        inputs = inputs[..., np.newaxis]
        msfs = self.msf(inputs, c, a)
        f = np.sum(msfs * w, axis=-1)
        o = np.sum(f, axis=1, keepdims=True)
        return self.nonlinearity(o)


class GMDHLayer(BaseLayer):
    """
    Group Method of Data Handling Layer
    -----------------------------------
    Building block of GMDH pipeline.
    """

    def __init__(
        self,
        poli_type: str,
        # nonlinearity: tp.Callable[[tp.Any], np.ndarray],
        **kwargs,
    ):
        """
        Constructor method

         Parameters
        ----------

        :poli_type: Type of GMDH polinome. Should be either `linear`, `partial_quadratic` or `quadratic`
        :type poli_type: str

        :nonlinearity: Activation function.
        :type nonlinearity: callable

        Raises
        ------
        :raises ValueError: Incorrect poli_type

        """
        if poli_type == "linear":
            self.n_weights = 2
        elif poli_type == "partial_quadratic":
            self.n_weights = 3
        elif poli_type == "quadratic":
            self.n_weights = 5
        else:
            raise ValueError("Incorrect poli_type")

        self.input_size = None
        super().__init__(nonlinearity=Linear, **kwargs)

    def build_weights_dict(self, input_shape):
        """
        Weights builder

        Parameters
        ----------

        :input_shape: Input shape.
        :type input_shape: tuple

        Returns
        -------

        :return: Union object (number_of_weights, _output_shape)
        :rtype: union
        """
        # Input shape is anything (all flattened)
        input_size = np.prod(input_shape, dtype=int)
        input_size = nCr(input_size, 2)

        self.input_size = input_size
        self.parser.add_weights("params", (1, input_size, self.n_weights))
        self.parser.add_weights("biases", (1, input_size))
        return self.parser.N, (input_size,)

    def _compute_grouped_arguments(self, inputs):
        grouped_indices = list(combinations(list(range(inputs.shape[1])), 2))
        grouped_inputs = []
        for group_ids in grouped_indices:
            group_ids = list(group_ids)
            temp_inputs = [inputs[:, group_ids]]
            if self.n_weights > 2:
                x_i_m_x_j = inputs[:, group_ids[0]] * inputs[:, group_ids[1]]
                temp_inputs.append(x_i_m_x_j[..., np.newaxis])
            if self.n_weights > 3:
                x_i_square = inputs[:, group_ids[0]] ** 2
                x_j_square = inputs[:, group_ids[1]] ** 2
                temp_inputs.append(x_i_square[..., np.newaxis])
                temp_inputs.append(x_j_square[..., np.newaxis])

            grouped_inputs.append(np.concatenate(temp_inputs, axis=-1))
        grouped_inputs = np.stack(grouped_inputs, axis=1)
        return grouped_inputs

    def forward(self, inputs, param_vector, return_trans_input=False):
        """
        Forward pass method

        Parameters
        ----------

        :inputs: Input matrix.
        :type inputs: np.ndarray

        :param_vector: Vector of network's weights.
        :type param_vector: np.ndarray

        Returns
        -------

        :return: Polynome of input.
        :rtype: np.ndarray

        """
        params = self.parser.get(param_vector, "params")
        biases = self.parser.get(param_vector, "biases")
        if inputs.ndim > 2:
            inputs = inputs.reshape((inputs.shape[0], np.prod(inputs.shape[1:])))
        inputs = self._compute_grouped_arguments(inputs)
        if return_trans_input:
            return inputs
        outputs = np.sum(inputs * params, axis=-1) + biases
        return outputs


class FuzzyGMDHLayer(BaseLayer):
    """
    Fuzzy Group Method of Data Handling Layer
    -----------------------------------
    Building block of FGMDH pipeline. Here we combined GMDH functionality and embed it into TSK controller
    """

    def __init__(
        self,
        poli_type: str,
        # nonlinearity: tp.Callable[[tp.Any], np.ndarray],
        msf: tp.Callable[[tp.Any], np.ndarray],
        **kwargs,
    ):
        """
        Constructor method

        Parameters
        ----------

        :poli_type: Type of GMDH polinome. Should be either `linear`, `partial_quadratic` or `quadratic`
        :type poli_type: str

        :msf: Membership function
        :type msf: callable

        Raises
        ------
        :raises ValueError: Incorrect poli_type
        """
        if poli_type == "linear":
            self.n_weights = 2
        elif poli_type == "partial_quadratic":
            self.n_weights = 3
        elif poli_type == "quadratic":
            self.n_weights = 5
        else:
            raise ValueError("Incorrect poli_type")

        self.msf = msf

        self.input_size = None
        self._confidence = kwargs.get("confidence", 0.8)
        self._return_defuzzify = kwargs.get("return_defuzzify", False)
        super().__init__(nonlinearity=Linear, **kwargs)

    def build_weights_dict(self, input_shape):
        """
        Weights builder

        Parameters
        ----------

        :input_shape: Input shape.
        :type input_shape: tuple

        Returns
        -------

        :return: Union object (number_of_weights, _output_shape)
        :rtype: union
        """
        # Input shape is anything (all flattened)
        input_size = np.prod(input_shape, dtype=int)
        input_size = nCr(input_size, 2)

        self.input_size = input_size
        self.parser.add_weights("a", (1, input_size * self.n_weights + 1))
        self.parser.add_weights("c", (1, input_size * self.n_weights + 1))
        return self.parser.N, (input_size,)

    def _compute_grouped_arguments(self, inputs):
        grouped_indices = list(combinations(list(range(inputs.shape[1])), 2))
        grouped_inputs = []
        for group_ids in grouped_indices:
            group_ids = list(group_ids)
            temp_inputs = [inputs[:, group_ids]]
            if self.n_weights > 2:
                x_i_m_x_j = inputs[:, group_ids[0]] * inputs[:, group_ids[1]]
                temp_inputs.append(x_i_m_x_j[..., np.newaxis])
            if self.n_weights > 3:
                x_i_square = inputs[:, group_ids[0]] ** 2
                x_j_square = inputs[:, group_ids[1]] ** 2
                temp_inputs.append(x_i_square[..., np.newaxis])
                temp_inputs.append(x_j_square[..., np.newaxis])
            grouped_inputs.append(np.concatenate(temp_inputs, axis=-1))
        grouped_inputs = np.stack(grouped_inputs, axis=1)
        return grouped_inputs

    def forward(self, inputs, param_vector, simplex=False):
        """
        Forward pass method

        Parameters
        ----------

        :inputs: Input matrix.
        :type inputs: np.ndarray

        :param_vector: Vector of network's weights.
        :type param_vector: np.ndarray

        Returns
        -------

        :return: Result of fuzzy-consequence over polynome of input
        :rtype: np.ndarray

        """
        a = self.parser.get(param_vector, "a")
        c = self.parser.get(param_vector, "c")
        inputs = self._compute_grouped_arguments(inputs)
        if inputs.ndim > 2:
            inputs = inputs.reshape((inputs.shape[0], np.prod(inputs.shape[1:])))

        if simplex:
            return a, c, np.abs(inputs)[:, :], inputs[:, :]
        else:
            vals = concat_and_multiply(a.T, inputs[:, :])
            cvals = (1 - self._confidence) * concat_and_multiply(
                c.T, np.abs(inputs[:, :])
            )
            return vals, cvals


class GMDHDense(BaseLayer):
    def __init__(
        self, size, degree, nonlinearity: tp.Callable[[tp.Any], np.ndarray], **kwargs
    ):
        self.size = size
        self.degree = degree
        super().__init__(nonlinearity=nonlinearity, **kwargs)

    def build_weights_dict(self, input_shape):
        # Input shape is anything (all flattened)
        input_size = np.prod(input_shape, dtype=int)
        input_size = self.calc_input_shape(input_size, self.degree)
        print(input_size)
        self.parser.add_weights("params", (input_size, self.size))
        self.parser.add_weights("biases", (self.size,))
        return self.parser.N, (self.size,)

    def forward(self, inputs, param_vector):
        params = self.parser.get(param_vector, "params")
        biases = self.parser.get(param_vector, "biases")
        if inputs.ndim > 2:
            inputs = inputs.reshape((inputs.shape[0], np.prod(inputs.shape[1:])))
        inputs = Poly(inputs, self.degree)
        return self.nonlinearity(np.dot(inputs[:, :], params) + biases)

    @staticmethod
    def calc_input_shape(input_size: int, deg: int) -> int:
        return sum([input_size ** i for i in range(1, deg + 1)])


class SimpleRNN(BaseLayer):
    """
    `Vanilla` Reccurent Neural Net
    ------------------------------
    """

    def __init__(self, units, size, **kwargs):
        """
        Constructor method

        Parameters
        ----------

        :units: Number of units.
        :type units: int

        :size: Output shape.
        :type size: int
        """
        self.units = units
        self.size = size
        super().__init__(nonlinearity=Tanh, **kwargs)

    @staticmethod
    def _update_rnn(param, input, hiddens, nonlinearity):
        return nonlinearity(concat_and_multiply(param, input, hiddens))

    def build_weights_dict(self, input_shape):
        """
        Weights builder

        Parameters
        ----------

        :input_shape: Input shape.
        :type input_shape: tuple

        Returns
        -------

        :return: Union object (number_of_weights, _output_shape)
        :rtype: union
        """
        self.parser.add_weights("init_hiddens", (1, self.units))
        self.parser.add_weights("change", (input_shape + self.units + 1, self.units))
        self.parser.add_weights("predict", (self.units + 1, self.size))
        return self.parser.N, (self.size,)

    def forward(self, inputs, param_vector):
        """
        Forward pass method

        Parameters
        ----------

        :inputs: Input matrix.
        :type inputs: np.ndarray

        :param_vector: Vector of network's weights.
        :type param_vector: np.ndarray

        Returns
        -------

        :return: Result of RNN operations.
        :rtype: np.ndarray
        """
        init_hiddens = self.parser.get(param_vector, "init_hiddens")
        change = self.parser.get(param_vector, "change")
        predict = self.parser.get(param_vector, "predict")
        num_sequences = inputs.shape[1]
        hiddens = np.repeat(init_hiddens, num_sequences, axis=0)
        output = [self.nonlinearity(concat_and_multiply(predict, hiddens))]

        for input in inputs:  # Iterate over time steps.
            hiddens = self._update_rnn(change, input, hiddens, self.nonlinearity)
            output.append(self.nonlinearity(concat_and_multiply(predict, hiddens)))
        return output


class LSTM(BaseLayer):
    """
    Long Short-Term Memory cell
    ---------------------------
    """

    def __init__(self, units, size, **kwargs):
        """
        Constructor method

        Parameters
        ----------

        :units: Number of units.
        :type units: int

        :size: Output shape.
        :type size: int
        """
        self.units = units
        self.size = size
        super().__init__(nonlinearity=Tanh, **kwargs)

    @staticmethod
    def _update_rnn(params, input, hiddens, cells, nonlinearity):
        change = nonlinearity(concat_and_multiply(params["change"], input, hiddens))
        forget = Sigmoid(concat_and_multiply(params["forget"], input, hiddens))
        ingate = Sigmoid(concat_and_multiply(params["ingate"], input, hiddens))
        outgate = Sigmoid(concat_and_multiply(params["outgate"], input, hiddens))
        cells = cells * forget + ingate * change
        hiddens = outgate * nonlinearity(cells)
        return hiddens, cells

    def build_weights_dict(self, input_shape):
        """
        Weights builder

        Parameters
        ----------

        :input_shape: Input shape.
        :type input_shape: tuple

        Returns
        -------

        :return: Union object (number_of_weights, _output_shape)
        :rtype: union
        """
        self.parser.add_weights("init_cells", (1, self.units))
        self.parser.add_weights("init_hiddens", (1, self.units))
        self.parser.add_weights("change", (input_shape + self.units + 1, self.units))
        self.parser.add_weights("forget", (input_shape + self.units + 1, self.units))
        self.parser.add_weights("ingate", (input_shape + self.units + 1, self.units))
        self.parser.add_weights("outgate", (input_shape + self.units + 1, self.units))
        self.parser.add_weights("predict", (self.units + 1, self.size))
        return self.parser.N, (self.size,)

    def forward(self, inputs, param_vector):
        """
        Forward pass method

        Parameters
        ----------

        :inputs: Input matrix.
        :type inputs: np.ndarray

        :param_vector: Vector of network's weights.
        :type param_vector: np.ndarray

        Returns
        -------

        :return: Result of LSTM operations.
        :rtype: np.ndarray

        """
        init_hiddens = self.parser.get(param_vector, "init_hiddens")
        init_cells = self.parser.get(param_vector, "init_cells")
        change = self.parser.get(param_vector, "change")
        forgate = self.parser.get(param_vector, "forgate")
        ingate = self.parser.get(param_vector, "ingate")
        outgate = self.parser.get(param_vector, "outgate")
        predict = self.parser.get(param_vector, "predict")
        num_sequences = inputs.shape[1]
        dictp = {
            "change": change,
            "forgate": forgate,
            "ingate": ingate,
            "outgate": outgate,
        }
        hiddens = np.repeat(init_hiddens, num_sequences, axis=0)
        cells = np.repeat(init_cells, num_sequences, axis=0)

        output = [self.nonlinearity(concat_and_multiply(predict, hiddens))]

        for input in inputs:  # Iterate over time steps.
            hiddens, cells = self._update_rnn(
                dictp, input, hiddens, cells, self.nonlinearity
            )
            output.append(self.nonlinearity(concat_and_multiply(predict, hiddens)))

        return output
