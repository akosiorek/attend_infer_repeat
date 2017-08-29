import math
import tensorflow as tf
from tensorflow.python.util import nest
import sonnet as snt


def create_linear_initializer(input_size):
    """Returns a default initializer for weights of a linear module."""
    stddev = 1. / math.sqrt(input_size * 2)
    return tf.truncated_normal_initializer(stddev=stddev)
    # return tf.contrib.layers.variance_scaling_initializer()
    # return tf.contrib.layers.python.layers.initializers.variance_scaling_initializer()
    # return tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')


snt.python.modules.basic.create_linear_initializer = create_linear_initializer


# def create_bias_initializer(unused_bias_shape):
#     """Returns a default initializer for the biases of a linear/AddBias module."""
#     return tf.truncated_normal_initializer(stddev=1e-3)

# snt.python.modules.basic.create_bias_initializer = create_bias_initializer


def selu(x):
    with tf.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))

default_activation = tf.nn.elu


# default_init = {
#     'w': tf.uniform_unit_scaling_initializer(),
#     'b': tf.truncated_normal_initializer(stddev=1e-2)
# }
default_init = dict()


def activation_based_init(func):
    init = tf.uniform_unit_scaling_initializer()
    if func == tf.nn.relu:
        init = tf.contrib.layers.xavier_initializer()
    elif func == tf.nn.elu:
        init = tf.contrib.layers.variance_scaling_initializer()
    elif func == selu:
        init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')

    return init


class Affine(snt.Linear):
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


def MLP(n_hiddens, hidden_transfer=default_activation, n_out=None, transfer=None):
    n_hiddens = nest.flatten(n_hiddens)
    transfers = nest.flatten(hidden_transfer)
    if len(transfers) > 1:
        assert len(transfers) == len(n_hiddens)
    else:
        transfers *= len(n_hiddens)

    layers = []
    for n_hidden, hidden_transfer in zip(n_hiddens, transfers):
        layers.append(snt.Linear(n_hidden))
        layers.append(hidden_transfer)

    if n_out is not None:
        layers.append(snt.Linear(n_out))

    if transfer is not None:
        layers.append(transfer)

    module = snt.Sequential(layers)
    module.output_size = n_out if n_out is not None else n_hiddens[-1]

    return module