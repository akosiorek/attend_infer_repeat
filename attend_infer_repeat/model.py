import functools

import tensorflow as tf
from tensorflow.contrib.distributions import Normal
from tensorflow.contrib.distributions.python.ops.kullback_leibler import kl as _kl


import ops
from cell import AIRCell
from evaluation import gradient_summaries
from prior import geometric_prior, NumStepsDistribution, tabular_kl
from modules import AIRDecoder


class AIRPriorMixin(object):

    def _geom_success_prob(self, **kwargs):
        return 1e-5

    def _make_priors(self, **kwargs):
        """Defines prior distributions

        :return: prior over num steps, scale, shift and what
        """

        num_step_prior_prob, num_step_prior = geometric_prior(self._geom_success_prob(**kwargs), self.max_steps)
        scale = Normal(.5, 1.)
        shift = Normal(self.shift_posterior.loc, 1.)
        what = Normal(0., 1.)
        return num_step_prior_prob, num_step_prior, scale, shift, what


class AIRModel(object):
    """Generic AIR model"""

    def __init__(self, obs, max_steps, glimpse_size,
                 n_what, transition, input_encoder, glimpse_encoder, glimpse_decoder, transform_estimator,
                 steps_predictor,
                 output_std=1., discrete_steps=True, output_multiplier=1.,
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
        self.debug = debug

        with tf.variable_scope(self.__class__.__name__):
            self.output_multiplier = tf.Variable(output_multiplier, dtype=tf.float32, trainable=False, name='canvas_multiplier')

            shape = self.obs.get_shape().as_list()
            self.batch_size = shape[0]
            self.img_size = shape[1:]
            self._build(transition, input_encoder, glimpse_encoder, glimpse_decoder, transform_estimator,
                        steps_predictor, cell_kwargs)

    def _build(self, transition, input_encoder, glimpse_encoder, glimpse_decoder, transform_estimator,
               steps_predictor, cell_kwargs):
        """Build the model. See __init__ for argument description"""

        self.decoder = AIRDecoder(self.img_size, self.glimpse_size, glimpse_decoder)
        self.cell = AIRCell(self.img_size, self.glimpse_size, self.n_what, transition,
                            input_encoder, glimpse_encoder, transform_estimator, steps_predictor,
                            discrete_steps=self.discrete_steps,
                            debug=self.debug,
                            **cell_kwargs)

        initial_state = self.cell.initial_state(self.obs)

        dummy_sequence = tf.zeros((self.max_steps, self.batch_size, 1), name='dummy_sequence')
        outputs, state = tf.nn.dynamic_rnn(self.cell, dummy_sequence, initial_state=initial_state, time_major=True)

        for name, output in zip(self.cell.output_names, outputs):
            output = tf.transpose(output, (1, 0, 2))
            setattr(self, name, output)

        self.canvas, self.glimpse = self.decoder(self.what, self.where, self.presence)

        self.final_state = state[-2]
        self.num_step_per_sample = tf.to_float(tf.reduce_sum(tf.squeeze(self.presence), -1))
        self.num_step = tf.reduce_mean(self.num_step_per_sample)

        self.output_distrib, self.num_steps_posterior, self.scale_posterior, self.shift_posterior, self.what_posterior\
            = self._make_posteriors()

        self.num_step_prior_prob, self.num_step_prior,\
        self.scale_prior, self.shift_prior, self.what_prior = self._make_priors()

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

    def _kl_divergence(self, analytic_kl_expectation):
        """Creates KL-divergence term of the loss"""

        with tf.variable_scope('KL_divergence'):
            kl_divergence = ops.Loss()

            with tf.variable_scope('num_steps'):
                num_steps_posterior_prob = self.num_steps_posterior.prob()
                steps_kl = tabular_kl(num_steps_posterior_prob, self.num_step_prior_prob)
                self.kl_num_steps_per_sample = tf.squeeze(tf.reduce_sum(steps_kl, 1))

                self.kl_num_steps = tf.reduce_mean(self.kl_num_steps_per_sample)
                tf.summary.scalar('kl_num_steps', self.kl_num_steps)

                kl_divergence.add(self.kl_num_steps, self.kl_num_steps_per_sample)

            if analytic_kl_expectation:
                # reverse cumsum of q(n) needed to compute \E_{q(n)} [ KL[ q(z|n) || p(z|n) ]]
                ordered_step_prob = num_steps_posterior_prob[..., 1:]
                ordered_step_prob = tf.cumsum(ordered_step_prob, axis=-1, reverse=True)
            else:
                ordered_step_prob = tf.squeeze(self.presence)

            self.ordered_step_prob = ordered_step_prob
            with tf.variable_scope('what'):
                what_kl = _kl(self.what_posterior, self.what_prior)
                what_kl = tf.reduce_sum(what_kl, -1) * self.ordered_step_prob
                what_kl_per_sample = tf.reduce_sum(what_kl, -1)
                self.kl_what = tf.reduce_mean(what_kl_per_sample)
                tf.summary.scalar('kl_what', self.kl_what)
                kl_divergence.add(self.kl_what, what_kl_per_sample)

            with tf.variable_scope('where'):
                scale_kl = _kl(self.scale_posterior, self.scale_prior)
                shift_kl = _kl(self.shift_posterior, self.shift_prior)
                where_kl = tf.reduce_sum(scale_kl + shift_kl, -1) * self.ordered_step_prob
                where_kl_per_sample = tf.reduce_sum(where_kl, -1)
                self.kl_where = tf.reduce_mean(where_kl_per_sample)
                tf.summary.scalar('kl_where', self.kl_where)
                kl_divergence.add(self.kl_where, where_kl_per_sample)

        return kl_divergence

    def _reinforce(self, importance_weight, decay_rate):
        """Implements REINFORCE for training the discrete probability distribution over number of steps and train-step
         for the baseline"""

        log_prob = self.num_steps_posterior.log_prob(self.num_step_per_sample)

        if self.baseline is not None:
            if not isinstance(self.baseline, tf.Tensor):
                self.baseline_module = self.baseline
                self.baseline = self.baseline_module(self.obs, self.what_loc, self.where_loc, self.presence_prob, self.final_state)
                self.baseline_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                       scope=self.baseline_module.variable_scope.name)
            importance_weight -= self.baseline

        if decay_rate is not None:
            axes = range(len(importance_weight.get_shape()))
            mean, var = tf.nn.moments(tf.squeeze(importance_weight), axes=axes)
            self.imp_weight_moving_mean = ops.make_moving_average('imp_weight_moving_mean', mean, 0., decay_rate)
            self.imp_weight_moving_var = ops.make_moving_average('imp_weight_moving_var', var, 1., decay_rate)

            factor = tf.maximum(tf.sqrt(self.imp_weight_moving_var), 1.)
            importance_weight = (importance_weight - self.imp_weight_moving_mean) / factor

        self.importance_weight = importance_weight
        axes = range(len(self.importance_weight.get_shape()))
        imp_weight_mean, imp_weight_var = tf.nn.moments(self.importance_weight, axes)
        tf.summary.scalar('imp_weight_mean', imp_weight_mean)
        tf.summary.scalar('imp_weight_var', imp_weight_var)
        reinforce_loss_per_sample = tf.stop_gradient(self.importance_weight) * log_prob
        self.reinforce_loss = tf.reduce_mean(reinforce_loss_per_sample)
        tf.summary.scalar('reinforce_loss', self.reinforce_loss)

        return self.reinforce_loss

    def _make_baseline_train_step(self, opt, loss, baseline, baseline_vars):
        baseline_target = tf.stop_gradient(loss)

        self.baseline_loss = .5 * tf.reduce_mean(tf.square(baseline_target - baseline))
        tf.summary.scalar('baseline_loss', self.baseline_loss)
        train_step = opt.minimize(self.baseline_loss, var_list=baseline_vars)
        return train_step

    def train_step(self, learning_rate, l2_weight=0., analytic_kl_expectation=False,
                   use_reinforce=True, baseline=None, decay_rate=None, nums=None,
                   optimizer=tf.train.RMSPropOptimizer, opt_kwargs=dict(momentum=.9, centered=True)):
        """Creates the train step and the global_step

        :param learning_rate: float or tf.Tensor
        :param l2_weight: float or tf.Tensor, if > 0. then adds l2 regularisation to the model
        :param analytic_kl_expectation: bool, computes expectation over conditional-KL analytically if True
        :param use_reinforce: boolean, if False doesn't compute gradients for the number of steps
        :param baseline: callable or None, baseline for variance reduction of REINFORCE
        :param decay_rate: float, decay rate to use for exp-moving average for NVIL
        :param nums: tf.Tensor, number of objects in images
        :return: train step and global step
        """

        self.l2_weight = l2_weight
        if not hasattr(self, 'baseline'):
            self.baseline = baseline

        self.use_reinforce = use_reinforce and self.discrete_steps
        with tf.variable_scope('loss'):
            loss = ops.Loss()
            self._train_step = []
            self.learning_rate = tf.Variable(learning_rate, name='learning_rate', trainable=False)
            make_opt = functools.partial(optimizer, **opt_kwargs)

            # Reconstruction Loss, - \E_q [ p(x | z, n) ]
            rec_loss_per_sample = -self.output_distrib.log_prob(self.obs)
            self.rec_loss_per_sample = tf.reduce_sum(rec_loss_per_sample, axis=(1, 2))
            self.rec_loss = tf.reduce_mean(self.rec_loss_per_sample)
            tf.summary.scalar('rec', self.rec_loss)
            loss.add(self.rec_loss, self.rec_loss_per_sample)

            # Prior Loss, KL[ q(z, n | x) || p(z, n) ]
            self.kl_div = self._kl_divergence(analytic_kl_expectation)
            tf.summary.scalar('prior', self.kl_div.value)
            loss.add(self.kl_div)

            # REINFORCE
            opt_loss = loss.value
            if use_reinforce:

                self.reinforce_imp_weight = self.rec_loss_per_sample
                if not analytic_kl_expectation:
                    self.reinforce_imp_weight += self.kl_div.per_sample

                reinforce_loss = self._reinforce(self.reinforce_imp_weight, decay_rate)
                opt_loss += reinforce_loss

            baseline_vars = getattr(self, 'baseline_vars', [])
            model_vars = list(set(tf.trainable_variables()) - set(baseline_vars))
            # L2 reg
            if l2_weight > 0.:
                # don't penalise biases
                weights = [w for w in model_vars if len(w.get_shape()) == 2]
                self.l2_loss = l2_weight * sum(map(tf.nn.l2_loss, weights))
                opt_loss += self.l2_loss
                tf.summary.scalar('l2', self.l2_loss)

            opt = make_opt(self.learning_rate)
            gvs = opt.compute_gradients(opt_loss, var_list=model_vars)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            global_step = tf.train.get_or_create_global_step()
            with tf.control_dependencies(update_ops):
                self._train_step = opt.apply_gradients(gvs, global_step=global_step)

            if self.use_reinforce and self.baseline is not None:
                baseline_opt = make_opt(10 * learning_rate)
                self._baseline_tran_step = self._make_baseline_train_step(baseline_opt, self.reinforce_imp_weight,
                                                                          self.baseline, self.baseline_vars)
                self._true_train_step = self._train_step
                self._train_step = tf.group(self._true_train_step, self._baseline_tran_step)

            tf.summary.scalar('num_step', self.num_step)
        # Metrics
        gradient_summaries(gvs)
        if nums is not None:
            self.gt_num_steps = tf.squeeze(tf.reduce_sum(nums, 0))
            self.num_step_accuracy = tf.reduce_mean(tf.to_float(tf.equal(self.gt_num_steps, self.num_step_per_sample)))

        self.loss = loss
        self.opt_loss = opt_loss
        return self._train_step, global_step