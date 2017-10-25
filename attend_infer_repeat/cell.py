import numpy as  np
import sonnet as snt
import tensorflow as tf
from tensorflow.contrib.distributions import Bernoulli, NormalWithSoftplusScale

from modules import SpatialTransformer, ParametrisedGaussian


class AIRCell(snt.RNNCore):
    """RNN cell that implements the core features of Attend, Infer, Repeat, as described here:
    https://arxiv.org/abs/1603.08575
    """
    _n_transform_param = 4

    def __init__(self, img_size, crop_size, n_what,
                 transition, input_encoder, glimpse_encoder, transform_estimator, steps_predictor,
                 discrete_steps=True, debug=False):
        """Creates the cell

        :param img_size: int tuple, size of the image
        :param crop_size: int tuple, size of the attention glimpse
        :param n_what: number of latent units describing the "what"
        :param transition: an RNN cell for maintaining the internal hidden state
        :param input_encoder: callable, encodes the original input image before passing it into the transition
        :param glimpse_encoder: callable, encodes the glimpse into latent representation
        :param transform_estimator: callabe, transforms the hidden state into parameters for the spatial transformer
        :param steps_predictor: callable, predicts whether to take a step
        :param discrete_steps: boolean, steps are samples from a Bernoulli distribution if True; if False, all steps are
         taken and are weighted by the step probability
        :param debug: boolean, adds checks for NaNs in the inputs to distributions
        """

        super(AIRCell, self).__init__(self.__class__.__name__)
        self._img_size = img_size
        self._n_pix = np.prod(self._img_size)
        self._crop_size = crop_size
        self._n_what = n_what
        self._transition = transition
        self._n_hidden = self._transition.output_size[0]

        self._sample_presence = discrete_steps
        self._debug = debug

        with self._enter_variable_scope():

            self._spatial_transformer = SpatialTransformer(img_size, crop_size)
            # self._inverse_transformer = SpatialTransformer(img_size, crop_size, inverse=True)

            self._transform_estimator = transform_estimator(self._n_transform_param)
            self._input_encoder = input_encoder()
            self._glimpse_encoder = glimpse_encoder()

            self._what_distrib = ParametrisedGaussian(n_what, scale_offset=0.5,
                                                      validate_args=self._debug, allow_nan_stats=not self._debug)
            self._steps_predictor = steps_predictor()

    @property
    def state_size(self):
        return [
            np.prod(self._img_size),  # image
            self._n_what,  # what
            self._n_transform_param,  # where
            self._transition.state_size,  # hidden state of the rnn
            1,  # presence
        ]

    @property
    def output_size(self):
        return [
            self._n_what,  # what code
            self._n_what,  # what loc
            self._n_what,  # what scale
            self._n_transform_param,  # where code
            self._n_transform_param,  # where loc
            self._n_transform_param,  # where scale
            1,  # presence prob
            1  # presence
        ]

    @property
    def output_names(self):
        return 'what what_loc what_scale where where_loc where_scale presence_prob presence'.split()

    def initial_state(self, img):
        batch_size = img.get_shape().as_list()[0]
        hidden_state = self._transition.initial_state(batch_size, tf.float32, trainable=True)

        where_code = tf.zeros([1, self._n_transform_param], dtype=tf.float32, name='where_init')
        what_code = tf.zeros([1, self._n_what], dtype=tf.float32, name='what_init')

        where_code, what_code = (tf.tile(i, (batch_size, 1)) for i in (where_code, what_code))

        flat_img = tf.reshape(img, (batch_size, self._n_pix))
        init_presence = tf.ones((batch_size, 1), dtype=tf.float32)
        return [flat_img, what_code, where_code, hidden_state, init_presence]

    def _build(self, inpt, state):
        """Input is unused; it's only to force a maximum number of steps"""

        img_flat, what_code, where_code, hidden_state, presence = state

        img_inpt = img_flat
        img = tf.reshape(img_inpt, (-1,) + tuple(self._img_size))

        inpt_encoding = self._input_encoder(img)
        with tf.variable_scope('rnn_inpt'):
            hidden_output, hidden_state = self._transition(inpt_encoding, hidden_state)

        where_param = self._transform_estimator(hidden_output)
        where_distrib = NormalWithSoftplusScale(*where_param,
                                                validate_args=self._debug, allow_nan_stats=not self._debug)
        where_loc, where_scale = where_distrib.loc, where_distrib.scale
        where_code = where_distrib.sample()

        cropped = self._spatial_transformer(img, where_code)

        with tf.variable_scope('presence'):
            presence_logit = self._steps_predictor(hidden_output)
            presence_prob = tf.nn.sigmoid(presence_logit)

            if self._sample_presence:
                presence_distrib = Bernoulli(probs=presence_prob, dtype=tf.float32,
                                             validate_args=self._debug, allow_nan_stats=not self._debug)

                new_presence = presence_distrib.sample()
                presence *= new_presence

            else:
                presence = presence_prob

        what_params = self._glimpse_encoder(cropped)
        what_distrib = self._what_distrib(what_params)
        what_loc, what_scale = what_distrib.loc, what_distrib.scale
        what_code = what_distrib.sample()

        # decoded = self._glimpse_decoder(what_code)
        # inversed = self._inverse_transformer(decoded, where_code)

        # with tf.variable_scope('rnn_outputs'):
            # inversed_flat = tf.reshape(inversed, (-1, self._n_pix))
            # canvas_flat += presence * inversed_flat
            # decoded_flat = tf.reshape(decoded, (-1, np.prod(self._crop_size)))

        output = [what_code, what_loc, what_scale, where_code, where_loc, where_scale,
                  presence_prob, presence]
        state = [img_flat, what_code, where_code, hidden_state, presence]
        return output, state