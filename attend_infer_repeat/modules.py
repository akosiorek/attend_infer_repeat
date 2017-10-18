import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import NormalWithSoftplusScale
from tensorflow.python.util import nest

import sonnet as snt

from neural import MLP


class ParametrisedGaussian(snt.AbstractModule):

    def __init__(self, n_params, scale_offset=0., *args, **kwargs):
        super(ParametrisedGaussian, self).__init__(self.__class__.__name__)
        self._n_params = n_params
        self._scale_offset = scale_offset
        self._create_distrib = lambda x, y: NormalWithSoftplusScale(x, y, *args, **kwargs)

    def _build(self, inpt):
        transform = snt.Linear(2 * self._n_params)
        params = transform(inpt)
        loc, scale = tf.split(params, 2, len(params.get_shape()) - 1)
        distrib = self._create_distrib(loc, scale + self._scale_offset)
        return distrib


class TransformParam(snt.AbstractModule):

    def __init__(self, n_hidden, n_param, max_crop_size=1.0):
        super(TransformParam, self).__init__(self.__class__.__name__)
        self._n_hidden = n_hidden
        self._n_param = n_param
        self._max_crop_size = max_crop_size

    def _embed(self, inpt):
        flatten = snt.BatchFlatten()
        mlp = MLP(self._n_hidden, n_out=self._n_param)
        seq = snt.Sequential([flatten, mlp])
        return seq(inpt)

    def _transform(self, inpt):
        sx, tx, sy, ty = tf.split(inpt, 4, 1)
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
        return locs, scales + self._scale_bias


class Encoder(snt.AbstractModule):

    def __init__(self, n_hidden):
        super(Encoder, self).__init__(self.__class__.__name__)
        self._n_hidden = n_hidden

    def _build(self, inpt):
        flat = snt.BatchFlatten()
        mlp = MLP(self._n_hidden)
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

    def _sample_image(self, img, transform_params):
        grid_coords = self._warper(transform_params)
        return snt.resampler(img, grid_coords)

    def _build(self, img, transform_params):
        if len(img.get_shape()) == 3:
            img = img[..., tf.newaxis]

        if len(transform_params.get_shape()) == 2:
            return self._sample_image(img, transform_params)
        else:
            transform_params = tf.unstack(transform_params, axis=1)
            samples = [self._sample_image(img, tp) for tp in transform_params]
            return tf.stack(samples, axis=1)


class StepsPredictor(snt.AbstractModule):

    def __init__(self, n_hidden, steps_bias=0.):
        super(StepsPredictor, self).__init__(self.__class__.__name__)
        self._n_hidden = n_hidden
        self._steps_bias = steps_bias

    def _build(self, inpt):
        mlp = MLP(self._n_hidden, n_out=1)
        logit = mlp(inpt) + self._steps_bias
        return logit


class BaselineMLP(snt.AbstractModule):

    def __init__(self, n_hidden):
        super(BaselineMLP, self).__init__(self.__class__.__name__)
        self._n_hidden = n_hidden

    def _build(self, *inpts):

        inpts = nest.flatten(inpts)
        min_dim = min([i.shape.ndims for i in inpts])
        flatten = snt.FlattenTrailingDimensions(dim_from=min_dim-1)
        inpts = map(flatten, inpts)
        baseline_inpts = tf.concat(inpts, -1)

        mlp = MLP(self._n_hidden, n_out=1)
        if min_dim > 2:
            mlp = snt.BatchApply(mlp)
        baseline = mlp(baseline_inpts)

        return baseline[..., 0]