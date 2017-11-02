import functools

import tensorflow as tf
from tensorflow.contrib.distributions import Normal
from tensorflow.python.util import nest

import ops
from cell import AIRCell
from evaluation import gradient_summaries
from prior import NumStepsDistribution
from modules import AIRDecoder


# TODO: extract things related to NVIL to a separate class, possibly a mixin
# TODO: implement IWAE
# TODO: implement VIMCO as a mixin


class AIRModel(object):
    """Generic AIR model

    :param analytic_kl_expectation: bool, computes expectation over conditional-KL analytically if True
    """

    analytic_kl_expectation = False

    def __init__(self, obs, max_steps, glimpse_size,
                 n_what, transition, input_encoder, glimpse_encoder, glimpse_decoder, transform_estimator,
                 steps_predictor,
                 output_std=1., discrete_steps=True, output_multiplier=1., iw_samples=1,
                 debug=False, **cell_kwargs):
        """Creates the model.

        :param obs: tf.Tensor, images
        :param max_steps: int, maximum number of steps to take (or objects in the image)
        :param glimpse_size: tuple of ints, size of the attention glimpse
        :param n_what: int, number of latent variables describing an object
        :param transition: see :class: AIRCell
        :param input_encoder: see :class: AIRCell
        :param glimpse_encoder: see :class: AIRCell
        :param glimpse_decoder: callable, decodes the glimpse from latent representation
        :param transform_estimator: see :class: AIRCell
        :param steps_predictor: see :class: AIRCell
        :param output_std: float, std. dev. of the output Gaussian distribution
        :param discrete_steps: see :class: AIRCell
        :param output_multiplier: float, a factor that multiplies the reconstructed glimpses
        :param debug: see :class: AIRCell
        :param **cell_kwargs: all other parameters are passed to AIRCell
        """

        self.obs = obs

        self.max_steps = max_steps
        self.glimpse_size = glimpse_size
        self.n_what = n_what
        self.output_std = output_std
        self.discrete_steps = discrete_steps
        self.iw_samples = iw_samples
        self.debug = debug

        print 'iw_samples', iw_samples

        tiles = [iw_samples] + [1] * (obs.shape.ndims - 1)
        self.used_obs = tf.tile(self.obs, tiles)

        with tf.variable_scope(self.__class__.__name__):
            self.output_multiplier = tf.Variable(output_multiplier, dtype=tf.float32, trainable=False, name='canvas_multiplier')

            shape = self.obs.get_shape().as_list()
            self.batch_size = shape[0]
            self.effective_batch_size = self.batch_size * self.iw_samples
            self.img_size = shape[1:]
            self._build(transition, input_encoder, glimpse_encoder, glimpse_decoder, transform_estimator,
                        steps_predictor, cell_kwargs)

    def _build(self, transition, input_encoder, glimpse_encoder, glimpse_decoder, transform_estimator,
               steps_predictor, cell_kwargs):
        """Build the model. See __init__ for argument description"""
        # save existing variables to know later what we've created
        previous_vars = tf.trainable_variables()

        self.decoder = AIRDecoder(self.img_size, self.glimpse_size, glimpse_decoder)
        self.cell = AIRCell(self.img_size, self.glimpse_size, self.n_what, transition,
                            input_encoder, glimpse_encoder, transform_estimator, steps_predictor,
                            discrete_steps=self.discrete_steps,
                            debug=self.debug,
                            **cell_kwargs)

        initial_state = self.cell.initial_state(self.used_obs)

        dummy_sequence = tf.zeros((self.max_steps, self.effective_batch_size, 1), name='dummy_sequence')
        outputs, state = tf.nn.dynamic_rnn(self.cell, dummy_sequence, initial_state=initial_state, time_major=True)

        for name, output in zip(self.cell.output_names, outputs):
            output = tf.transpose(output, (1, 0, 2))
            setattr(self, name, output)

        self.canvas, self.glimpse = self.decoder(self.what, self.where, self.presence)

        self.final_state = state[-2]
        self.num_step_per_sample = tf.to_float(tf.reduce_sum(tf.squeeze(self.presence), -1))

        self.output_distrib, self.num_steps_posterior, self.scale_posterior, self.shift_posterior, self.what_posterior\
            = self._make_posteriors()

        self.num_step_prior_prob, self.num_step_prior,\
        self.scale_prior, self.shift_prior, self.what_prior = self._make_priors()

        # group variables
        model_vars = set(tf.trainable_variables()) - set(previous_vars)
        self.decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         scope=self.decoder.variable_scope.name)
        self.encoder_vars = list(model_vars - set(self.decoder_vars))
        self.model_vars = list(model_vars)

    def _make_posteriors(self):
        """Builds posterior distributions. This is fairly standard and shouldn't be changed.

        :return:
        """
        output_distrib = Normal(self.canvas, self.output_std)

        posterior_step_probs = tf.squeeze(self.presence_prob)
        num_steps_posterior = NumStepsDistribution(posterior_step_probs)

        ax = self.where_loc.shape.ndims - 1
        us, ut = tf.split(self.where_loc, 2, ax)
        ss, st = tf.split(self.where_scale, 2, ax)
        scale_posterior = Normal(us, ss)
        shift_posterior = Normal(ut, st)
        what_posterior = Normal(self.what_loc, self.what_scale)

        return output_distrib, num_steps_posterior, scale_posterior, shift_posterior, what_posterior

    def train_step(self, learning_rate, l2_weight=0., nums=None,
                   optimizer=tf.train.RMSPropOptimizer, opt_kwargs=dict(momentum=.9, centered=True)):
        """Creates the train step and the global_step

        :param learning_rate: float or tf.Tensor
        :param l2_weight: float or tf.Tensor, if > 0. then adds l2 regularisation to the model
        :param use_reinforce: boolean, if False doesn't compute gradients for the number of steps
        :param baseline: callable or None, baseline for variance reduction of REINFORCE
        :param decay_rate: float, decay rate to use for exp-moving average for NVIL
        :param nums: tf.Tensor, number of objects in images
        :return: train step and global step
        """

        self.l2_weight = l2_weight

        with tf.variable_scope('loss'):
            self.learning_rate = tf.Variable(learning_rate, name='learning_rate', trainable=False)
            make_opt = functools.partial(optimizer, **opt_kwargs)

            # Reconstruction Loss: - \E_q [ p(x | z, n) ]
            self.rec_loss_per_sample = self._log_likelihood()

            # Prior Loss: KL[ q(z, n | x) || p(z, n) ]
            self.kl_div_per_sample = self._kl_divergence()

        with tf.variable_scope('grad'):
            self._train_step, self.gvs = self._make_train_step(make_opt, self.rec_loss_per_sample, self.kl_div_per_sample)

        # Metrics
        gradient_summaries(self.gvs)
        if nums is not None:
            self.gt_num_steps = tf.squeeze(tf.reduce_sum(nums, 0))
            num_step_per_sample = self.resample(self.num_step_per_sample)
            self.num_step_accuracy = tf.reduce_mean(tf.to_float(tf.equal(self.gt_num_steps, num_step_per_sample)))

        # negative ELBO
        self.nelbo = self.make_nelbo()

        # hack for reporting
        with tf.variable_scope('loss'):
            for name in ['kl_div', 'kl_num_steps', 'kl_where', 'kl_scale', 'kl_shift', 'kl_what', 'rec_loss', 'num_step']:
                per_sample_name = name + '_per_sample'
                per_sample = getattr(self, per_sample_name, None)
                if per_sample is not None:
                    self._log_and_resample(per_sample, name)

        # hack for visualsiations
        for name in 'canvas glimpse presence where what'.split():
            setattr(self, 'resampled_' + name, self.resample(getattr(self, name)))

        return self._train_step, tf.train.get_or_create_global_step()

    def make_nelbo(self):
        try:
            return self._make_nelbo()
        except AttributeError:
            return self.rec_loss.value + self.kl_div.value

    def resample(self, *args):
        res = args
        try:
            res = self._resample(*args)
        except AttributeError:
            pass

        if nest.is_sequence(res) and len(res) == 1:
            res = res[0]
        return res

    def _l2_loss(self):

        l2_loss = 0.
        # L2 reg
        if self.l2_weight > 0.:
            # don't penalise biases
            weights = [w for w in self.model_vars if len(w.get_shape()) == 2]
            l2_loss = self.l2_weight * sum(map(tf.nn.l2_loss, weights))
            tf.summary.scalar('l2', l2_loss)
        return l2_loss

    def _log_and_resample(self, per_sample, name):
        per_sample = self._resample(per_sample)
        value = tf.reduce_mean(per_sample)
        setattr(self, name, value)
        tf.summary.scalar(name, value)

    def _kl_divergence(self):
        """Creates KL-divergence term of the loss"""

        with tf.variable_scope('KL_divergence'):
            kl_div = 0.

            with tf.variable_scope('num_steps'):

                self.kl_num_steps_per_sample = self._kl_num_steps()
                kl_div += self.kl_num_steps_per_sample

            self.ordered_step_prob = self._ordered_step_prob()
            with tf.variable_scope('what'):
                self.kl_what_per_sample = self._kl_what()
                kl_div += self.kl_what_per_sample

            with tf.variable_scope('where'):
                self.kl_where_per_sample, self.kl_scale_per_sample, self.kl_shift_per_sample = self._kl_where()
                kl_div += self.kl_where_per_sample

        return kl_div
