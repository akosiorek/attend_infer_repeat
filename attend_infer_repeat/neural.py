import math
import tensorflow as tf
from tensorflow.python.util import nest
import sonnet as snt


def create_linear_initializer(input_size):
    """Returns a default initializer for weights of a linear module."""
    stddev = 1. / math.sqrt(input_size * 2)
    return tf.truncated_normal_initializer(stddev=stddev)


def selu(x):
    with tf.name_scope('selu'):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


default_activation = tf.nn.elu

default_init = {
    'w': create_linear_initializer,
    'b': tf.zeros_initializer()
}


def activation_based_init(nonlinearity):
    """Returns initialiaation based on a nonlinearlity"""

    init = tf.uniform_unit_scaling_initializer()
    if nonlinearity == tf.nn.relu:
        init = tf.contrib.layers.xavier_initializer()
    elif nonlinearity == tf.nn.elu:
        init = tf.contrib.layers.variance_scaling_initializer()
    elif nonlinearity == selu:
        init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')

    return init


class Affine(snt.Linear):
    """Layer implementing an affine non-linear transformation"""

    def __init__(self, n_output, transfer=default_activation, initializers=None, transfer_based_init=False):

        if initializers is None:
            initializers = default_init

        if transfer_based_init and 'w' not in initializers:
            initializers['w'] = activation_based_init(transfer)

        super(Affine, self).__init__(n_output, initializers)
        self._transfer = transfer

    def _build(self, inpt):
        output = super(Affine, self)._build(inpt)
        if self._transfer is not None:
            output = self._transfer(output)
        return output


class MLP(snt.AbstractModule):
    """Implements a multi-layer perceptron"""

    def __init__(self, n_hiddens, hidden_transfer=default_activation, n_out=None, transfer=None, initializers=default_init):
        """Initialises the MLP

        :param n_hiddens: int or an interable of ints, number of hidden units in layers
        :param hidden_transfer: callable or iterable; a transfer function for hidden layers or an interable thereof. If it's an iterable its length should be the same as length of `n_hiddens`
        :param n_out: int or None, number of output units
        :param transfer: callable or None, a transfer function for the output
        """

        super(MLP, self).__init__(self.__class__.__name__)
        self._n_hiddens = nest.flatten(n_hiddens)
        transfers = nest.flatten(hidden_transfer)
        if len(transfers) > 1:
            assert len(transfers) == len(self._n_hiddens)
        else:
            transfers *= len(self._n_hiddens)
        self._hidden_transfers = nest.flatten(transfers)
        self._n_out = n_out
        self._transfer = transfer
        self._initializers = initializers

    @property
    def output_size(self):
        if self._n_out is not None:
            return self._n_out
        return self._n_hiddens[-1]

    def _build(self, inpt):
            layers = []
            for n_hidden, hidden_transfer in zip(self._n_hiddens, self._hidden_transfers):
                layers.append(Affine(n_hidden, hidden_transfer, self._initializers))

            if self._n_out is not None:
                layers.append(Affine(self._n_out, self._transfer, self._initializers))

            module = snt.Sequential(layers)
            return module(inpt)