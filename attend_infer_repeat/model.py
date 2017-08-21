import numpy as  np
import tensorflow as tf
import sonnet as snt

from tensorflow.contrib.distributions import Bernoulli, NormalWithSoftplusScale


default_init = {
    'w': tf.uniform_unit_scaling_initializer(),
    'b': tf.truncated_normal_initializer(stddev=1e-2)}


def epsilon_greedy(events, eps):
    shape = tf.shape(events)
    do_explore = tf.less(tf.random_uniform(shape, dtype=tf.float32), tf.ones(shape, dtype=tf.float32) * eps)
    random = tf.cast(tf.round(tf.random_uniform(shape, dtype=tf.float32)), events.dtype)
    # events = tf.where(do_explore, random, events)

    do_explore = tf.to_float(do_explore)
    events = do_explore * random + (1. - do_explore) * events

    return events


class TransformParam(snt.AbstractModule):

    def __init__(self, n_param, max_crop_size=1.0):
        super(TransformParam, self).__init__(self.__class__.__name__)
        self._n_param = n_param
        self._max_crop_size = max_crop_size

    def _build(self, inpt):

        flat = snt.BatchFlatten()
        linear1 = snt.Linear(20, initializers=default_init)
        linear2 = snt.Linear(self._n_param, initializers=default_init)
        seq = snt.Sequential([flat, linear1, tf.nn.elu, linear2])
        output = seq(inpt)
        output *= 1e-4
        sx, tx, sy, ty = tf.split(output, 4, 1)
        sx, sy = (.5e-4 + (1 - 1e-4) * self._max_crop_size * tf.nn.sigmoid(s) for s in (sx, sy))
        tx, ty = (tf.nn.tanh(t) for t in (tx, ty))
        output = tf.concat((sx, tx, sy, ty), -1)
        return output


class Encoder(snt.AbstractModule):

    def __init__(self, n_latent):
        super(Encoder, self).__init__(self.__class__.__name__)
        self._n_latent = n_latent

    def _build(self, inpt):
        flat = snt.BatchFlatten()
        linear1 = snt.Linear(100, initializers=default_init)
        linear2 = snt.Linear(self._n_latent, initializers=default_init)
        seq = snt.Sequential([flat, linear1, tf.nn.elu, linear2, tf.nn.elu])
        return seq(inpt)


class StochasticEncoder(Encoder):
    def _build(self, inpt):
        inpt = super(StochasticEncoder, self)._build(inpt)
        linear = snt.Linear(self._n_latent * 2, initializers=default_init)
        output = linear(inpt)
        loc, scale = output[..., :self._n_latent], output[..., self._n_latent:]
        return loc, scale


class Decoder(snt.AbstractModule):

    def __init__(self, output_size):
        super(Decoder, self).__init__(self.__class__.__name__)
        self._output_size = output_size

    def _build(self, inpt):
        n = np.prod(self._output_size)
        linear1 = snt.Linear(100, initializers=default_init)
        linear2 = snt.Linear(n, initializers=default_init)
        reshape = snt.BatchReshape(self._output_size)
        seq = snt.Sequential([linear1, tf.nn.elu, linear2, reshape])
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


class AIRCell(snt.RNNCore):
    _n_transform_param = 4

    def __init__(self, img_size, crop_size, n_latent, transition, max_crop_size=1.0,
                 sample_presence=True, canvas_init=-10., presence_bias=0., explore_eps=None, debug=False):

        super(AIRCell, self).__init__(self.__class__.__name__)
        self._img_size = img_size
        self._n_pix = np.prod(self._img_size)
        self._crop_size = crop_size
        self._n_latent = n_latent
        self._transition = transition
        self._n_hidden = self._transition.output_size[0]

        self._sample_presence = sample_presence
        self._presence_bias = presence_bias
        self._explore_eps = explore_eps
        self._debug = debug

        with self._enter_variable_scope():
            self._canvas = tf.zeros(self._img_size, dtype=tf.float32)
            if canvas_init is not None:
                self._canvas_value = tf.get_variable('canvas_value', dtype=tf.float32, initializer=canvas_init)
                self._canvas += self._canvas_value

            transform_constraints = snt.AffineWarpConstraints.no_shear_2d()

            self._transform_param = TransformParam(self._n_transform_param, max_crop_size)

            self._spatial_transformer = SpatialTransformer(img_size, crop_size, transform_constraints)
            self._input_encoder = Encoder(self._transition.output_size[0])
            # self._encoder = Encoder(n_latent)
            self._encoder = StochasticEncoder(n_latent)

            self._decoder = Decoder(crop_size)
            self._inverse_transformer = SpatialTransformer(img_size, crop_size, transform_constraints, inverse=True)

    @property
    def state_size(self):
        return [
            np.prod(self._img_size),        # image
            np.prod(self._img_size),        # canvas
            np.prod(self._img_size),        # explainability
            self._n_latent,                 # what
            self._n_transform_param,        # where
            self._transition.state_size,    # hidden state of the rnn
            1,                              # presence
        ]

    @property
    def output_size(self):
        return [np.prod(self._img_size), np.prod(self._crop_size), self._n_latent, self._n_latent, self._n_latent, self._n_transform_param, 1, 1]

    def initial_state(self, img):
        batch_size = img.get_shape().as_list()[0]
        hidden_state = self._transition.initial_state(batch_size, tf.float32, trainable=True)

        where_code = tf.get_variable('where_init', shape=[1, self._n_transform_param], dtype=tf.float32)

        what_code = tf.get_variable('what_init', shape=[1, self._n_latent], dtype=tf.float32)

        flat_canvas = tf.reshape(self._canvas, (1, self._n_pix))

        where_code, what_code, flat_canvas = (tf.tile(i, (batch_size, 1)) for i in (where_code, what_code, flat_canvas))
        # flat_explain = tf.zeros_like(flat_canvas)

        flat_img = tf.reshape(img, (batch_size, self._n_pix))
        init_presence = tf.ones((batch_size, 1), dtype=tf.float32)
        return [flat_img, flat_canvas,
                # flat_explain,
                what_code, where_code, hidden_state, init_presence]

    def _build(self, inpt, state):

        # img_flat, canvas_flat, explain_flat, what_code, where_code, hidden_state, presence = state
        img_flat, canvas_flat, what_code, where_code, hidden_state, presence = state
        img = tf.reshape(img_flat, (-1,) + tuple(self._img_size))

        inpt_encoding = img
        inpt_encoding = self._input_encoder(inpt_encoding)

        with tf.variable_scope('rnn_inpt'):
            rnn_inpt = tf.concat((inpt_encoding, what_code, where_code, presence), -1)
            rnn_inpt = snt.Linear(self._n_hidden, initializers=default_init)(rnn_inpt)
            rnn_inpt = tf.nn.elu(rnn_inpt)
            hidden_output, hidden_state = self._transition(rnn_inpt, hidden_state)

        where_code = self._transform_param(hidden_output)
        cropped = self._spatial_transformer(img, where_code)

        with tf.variable_scope('presence'):
            presence_model = snt.Linear(self._n_latent, initializers=default_init), tf.nn.elu, snt.Linear(1, initializers=default_init)
            presence_model = snt.Sequential(presence_model)
            presence_logit = presence_model(hidden_output) + self._presence_bias
            presence_prob = tf.nn.sigmoid(presence_logit)

            # if self._explore_eps is not None:
            #     presence_prob = self._explore_eps + (1. - 2 * self._explore_eps) * presence_prob

            if self._explore_eps is not None:
                clipped_prob = tf.clip_by_value(presence_prob, self._explore_eps, 1. - self._explore_eps)
                presence_prob = tf.stop_gradient(clipped_prob - presence_prob) + presence_prob
                # presence_prob = tf.clip_by_value(presence_prob, self._explore_eps, 1. - self._explore_eps)

            if self._sample_presence:
                presence_distrib = Bernoulli(probs=presence_prob, dtype=tf.float32,
                                             validate_args=self._debug, allow_nan_stats=not self._debug)

                new_presence = presence_distrib.sample()
                # if self._explore_eps is not None:
                #    new_presence = epsilon_greedy(presence, self._explore_eps)
                presence *= new_presence

            else:
                presence = presence_prob

        # what_code = self._encoder(cropped)
        what_loc, what_scale = self._encoder(cropped)
        what_distrib = NormalWithSoftplusScale(what_loc, what_scale,
                                                validate_args=self._debug, allow_nan_stats=not self._debug)
        what_code = what_distrib.sample()

        decoded = self._decoder(what_code)
        inversed = self._inverse_transformer(decoded, where_code)

        # with tf.variable_scope('novelty'):
        #     explained_now = self._inverse_transformer(tf.ones_like(decoded), where_code)
        #     explained_now = tf.cast(explained_now, tf.bool)
        #
        #     explained = tf.reshape(explain_flat, tf.shape(explained_now))
        #     explained = tf.cast(explained, tf.bool)
        #
        #     explained_new = tf.logical_or(explained, explained_now)
        #     explained_new_flat = tf.reshape(tf.to_float(explained_new), tf.shape(explain_flat))
        #
        #     novelty_neg = tf.logical_or(tf.logical_not(explained_new), explained)
        #     novelty = tf.logical_not(novelty_neg)
        #     novelty = tf.to_float(novelty)
        #     novelty_flat = tf.reshape(novelty, tf.shape(explain_flat))


        with tf.variable_scope('rnn_outputs'):
            inversed_flat = tf.reshape(inversed, (-1, self._n_pix))

            canvas_flat = canvas_flat + presence * inversed_flat# * novelty_flat
            decoded_flat = tf.reshape(decoded, (-1, np.prod(self._crop_size)))

        output = [canvas_flat, decoded_flat, what_code, what_loc, what_scale, where_code, presence_prob, presence]
        state = [img_flat, canvas_flat,
                 # explained_new_flat,
                 what_code, where_code, hidden_state, presence]
        return output, state


if __name__ == '__main__':
    learning_rate = 1e-4
    batch_size = 10
    img_size = 50, 50
    crop_size = 20, 20
    n_latent = 10
    n_steps = 3

    x = tf.placeholder(tf.float32, (batch_size,) + img_size, name='inpt')

    transition = snt.GRU(n_latent)
    air = AIRCell(img_size, crop_size, n_latent, transition)
    initial_state = air.initial_state(x)

    dummy_sequence = tf.zeros((n_steps, batch_size, 1), name='dummy_sequence')
    outputs, state = tf.nn.dynamic_rnn(air, dummy_sequence, initial_state=initial_state, time_major=True)
    canvas, crop, what, what_loc, what_scale, where, presence_logit, presence = outputs

    canvas = tf.reshape(canvas, (n_steps, batch_size,) + tuple(img_size))
    final_canvas = canvas[-1]

    loss = tf.nn.l2_loss(x - final_canvas)

    opt = tf.train.AdamOptimizer(learning_rate)
    train_step = opt.minimize(loss)

    print 'Constructed model'


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    xx = np.random.rand(*x.get_shape().as_list())
    res, l = sess.run([outputs, loss], {x: xx})

    for r in res:
        print r.shape

    print res

    print 'loss = {}'.format(l)
    print 'Done'
