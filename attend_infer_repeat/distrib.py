import sonnet as snt

import tensorflow as tf
from tensorflow.contrib.distributions import NormalWithSoftplusScale


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

