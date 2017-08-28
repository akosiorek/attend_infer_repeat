import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
import sonnet as snt

import math


def create_linear_initializer(input_size):
    """Returns a default initializer for weights of a linear module."""
    stddev = 1 / math.sqrt(input_size * 2)
    return tf.truncated_normal_initializer(stddev=stddev)
    # return tf.contrib.layers.variance_scaling_initializer()
    # return tf.contrib.layers.python.layers.initializers.variance_scaling_initializer()


def create_bias_initializer(unused_bias_shape):
  """Returns a default initializer for the biases of a linear/AddBias module."""
  return tf.truncated_normal_initializer(stddev=1e-2)


snt.python.modules.basic.create_linear_initializer = create_linear_initializer
snt.python.modules.basic.create_bias_initializer = create_bias_initializer


default_init = None
# default_init = {
#     'w': tf.uniform_unit_scaling_initializer(),
#     'b': tf.truncated_normal_initializer(stddev=1e-2)
# }





def epsilon_greedy(events, eps):
    shape = tf.shape(events)
    do_explore = tf.less(tf.random_uniform(shape, dtype=tf.float32), tf.ones(shape, dtype=tf.float32) * eps)
    random = tf.cast(tf.round(tf.random_uniform(shape, dtype=tf.float32)), events.dtype)
    events = tf.where(do_explore, random, events)
    return events


class TransformParam(snt.AbstractModule):

    def __init__(self, n_hidden, n_param, max_crop_size=1.0):
        super(TransformParam, self).__init__(self.__class__.__name__)
        self._n_hidden = n_hidden
        self._n_param = n_param
        self._max_crop_size = max_crop_size

    def _embed(self, inpt):
        # flat = snt.BatchFlatten()
        # linear1 = snt.Linear(256, initializers=default_init)
        # linear2 = snt.Linear(256, initializers=default_init)
        #
        # # init = {
        # #     'w': tf.uniform_unit_scaling_initializer(.1),
        # #     'b': tf.zeros_initializer()
        # # }
        # init = default_init
        # linear3 = snt.Linear(self._n_param, initializers=init)
        # seq = snt.Sequential([flat, linear1, tf.nn.elu, linear2, tf.nn.elu, linear3])
        # output = seq(inpt)

        flatten = snt.BatchFlatten()
        mlp = MLP(self._n_hidden, n_out=self._n_param)
        seq = snt.Sequential([flatten, mlp])
        return seq(inpt)

    def _transform(self, inpt):
        # output *= 1e-4
        sx, tx, sy, ty = tf.split(inpt, 4, 1)
        # sx, sy = (.5e-4 + (1 - 1e-4) * self._max_crop_size * tf.nn.sigmoid(s) for s in (sx, sy))
        sx, sy = (self._max_crop_size * tf.nn.sigmoid(s) for s in (sx, sy))
        tx, ty = (tf.nn.tanh(t) for t in (tx, ty))
        output = tf.concat((sx, tx, sy, ty), -1)
        return output

    def _build(self, inpt):
        embedding = self._build(inpt)
        return self._transform(embedding)


class StochasticTransformParam(TransformParam):
    def __init__(self, n_hidden, n_param, max_crop_size=1.0, scale_bias=-2.):
        super(StochasticTransformParam, self).__init__(n_hidden, n_param * 2, max_crop_size)
        self._scale_bias = scale_bias

    def _build(self, inpt):
        embedding = self._embed(inpt)
        n_params = self._n_param / 2
        locs = self._transform(embedding[..., :n_params])
        scales = embedding[..., n_params:]
        return locs, scales #+ self._scale_bias


class Encoder(snt.AbstractModule):

    def __init__(self, n_hidden):
        super(Encoder, self).__init__(self.__class__.__name__)
        self._n_hidden = n_hidden

    def _build(self, inpt):
        flat = snt.BatchFlatten()
        # linear1 = snt.Linear(256, initializers=default_init)
        # linear2 = snt.Linear(256, initializers=default_init)
        mlp = MLP(self._n_hidden)
        # linear2 = snt.Linear(2 * self._n_appearance, initializers=default_init)
        # seq = snt.Sequential([flat, linear1, tf.nn.elu, linear2, tf.nn.elu])
        seq = snt.Sequential([flat, mlp])
        return seq(inpt)


class Decoder(snt.AbstractModule):

    def __init__(self, n_hidden, output_size):
        super(Decoder, self).__init__(self.__class__.__name__)
        self._n_hidden = n_hidden
        self._output_size = output_size

    def _build(self, inpt):
        n = np.prod(self._output_size)
        mlp = MLP(self._n_hidden, n_out=n)
        reshape = snt.BatchReshape(self._output_size)
        seq = snt.Sequential([mlp, reshape])
        return seq(inpt)


class SpatialTransformer(snt.AbstractModule):

    def __init__(self, img_size, crop_size, constraints=None, inverse=False):
        super(SpatialTransformer, self).__init__(self.__class__.__name__)

        with self._enter_variable_scope():
            self._warper = snt.AffineGridWarper(img_size, crop_size, constraints)
            if inverse:
                self._warper = self._warper.inverse()

    def _build(self, img, transform_params):
        if len(img.get_shape()) == 3:
            img = img[..., tf.newaxis]

        grid_coords = self._warper(transform_params)
        return snt.resampler(img, grid_coords)


def MLP(n_hiddens, hidden_transfer=tf.nn.elu, n_out=None, transfer=None):
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


class BaselineMLP(snt.AbstractModule):

    def __init__(self, n_hidden):
        super(BaselineMLP, self).__init__(self.__class__.__name__)
        self._n_hidden = n_hidden

    def _build(self, img, what, where, presence_prob):

        batch_size = int(img.get_shape()[0])
        parts = [tf.reshape(tf.transpose(i, (1, 0, 2)), (batch_size, -1)) for i in (what, where, presence_prob)]
        img_flat = tf.reshape(img, (batch_size, -1))
        baseline_inpts = [img_flat] + parts
        baseline_inpts = tf.concat(baseline_inpts, -1)
        mlp = MLP(self._n_hidden, n_out=1)
        baseline = mlp(baseline_inpts)
        return baseline