import tensorflow as tf
from tensorflow.contrib.distributions import Normal
from tensorflow.contrib.distributions.python.ops.kullback_leibler import kl as _kl

from cell import AIRCell
from ops import Loss
from prior import geometric_prior, NumStepsDistribution, tabular_kl
from tf_tools.eval import gradient_summaries


class AIRModel(object):
    def __init__(self, obs, nums, max_steps, glimpse_size,
                 n_appearance, transition, input_encoder, glimpse_encoder, glimpse_decoder, transform_estimator,
                 output_std=1., discrete_steps=True,
                 step_bias=0., explore_eps=None, debug=False):

        self.obs = obs
        self.nums = nums
        self.max_steps = max_steps
        self.glimpse_size = glimpse_size

        self.n_appearance = n_appearance

        self.output_std = output_std
        self.discrete_steps = discrete_steps
        self.step_bias = step_bias
        self.explore_eps = explore_eps
        self.debug = debug

        with tf.variable_scope(self.__class__.__name__):
            shape = self.obs.get_shape().as_list()
            self.batch_size = shape[0]
            self.img_size = shape[1:]
            self._build(transition, input_encoder, glimpse_encoder, glimpse_decoder, transform_estimator)

    def _build(self, transition, input_encoder, glimpse_encoder, glimpse_decoder, transform_estimator):
        if self.explore_eps is not None:
            self.explore_eps = tf.get_variable('explore_eps', initializer=self.explore_eps, trainable=False)

        air = AIRCell(self.img_size, self.glimpse_size, self.n_appearance, transition,
                      input_encoder, glimpse_encoder, glimpse_decoder, transform_estimator,
                      canvas_init=None,
                      discrete_steps=self.discrete_steps,
                      step_bias=self.step_bias,
                      explore_eps=self.explore_eps,
                      debug=self.debug)

        initial_state = air.initial_state(self.obs)

        dummy_sequence = tf.zeros((self.max_steps, self.batch_size, 1), name='dummy_sequence')
        outputs, state = tf.nn.dynamic_rnn(air, dummy_sequence, initial_state=initial_state, time_major=True)
        for name, output in zip(air.output_names, outputs):
            setattr(self, name, output)
        # canvas, glimpse, what, what_loc, what_scale, where, where_loc, where_scale, presence_prob, presence = outputs

        self.glimpse = tf.reshape(self.presence * tf.nn.sigmoid(self.glimpse),
                                  (self.max_steps, self.batch_size,) + tuple(self.glimpse_size))
        self.canvas = tf.reshape(self.canvas, (self.max_steps, self.batch_size,) + tuple(self.img_size))
        self.final_canvas = self.canvas[-1]

        self.output_distrib = Normal(self.final_canvas, self.output_std)

        posterior_step_probs = tf.transpose(tf.squeeze(self.presence_prob))
        self.num_steps_distrib = NumStepsDistribution(posterior_step_probs)

        self.num_step_per_sample = tf.to_float(tf.squeeze(tf.reduce_sum(self.presence, 0)))
        self.num_step = tf.reduce_mean(self.num_step_per_sample)
        self.gt_num_steps = tf.squeeze(tf.reduce_sum(self.nums, 0))

    def _prior_loss(self, appearance_prior, where_scale_prior, where_shift_prior,
                    num_steps_prior, global_step):

        with tf.variable_scope('prior_loss'):
            prior_loss = Loss()
            if num_steps_prior is not None:
                if num_steps_prior.anneal is not None:
                    with tf.variable_scope('num_steps_prior'):
                        nsp = num_steps_prior
                        val = tf.get_variable('value', initializer=num_steps_prior.init, dtype=tf.float32,
                                              trainable=False)

                        if num_steps_prior.anneal == 'exp':
                            decay_rate = (nsp.final / nsp.init) ** (float(nsp.steps_div) / nsp.steps)
                            val = tf.train.exponential_decay(val, global_step, nsp.steps_div, decay_rate)

                        elif num_steps_prior.anneal == 'linear':
                            val = nsp.final + (nsp.init - nsp.final) * (1. - tf.to_float(global_step) / nsp.steps)

                        num_steps_prior_value = tf.maximum(nsp.final, val)
                else:
                    num_steps_prior_value = num_steps_prior.init

                prior = geometric_prior(num_steps_prior_value, 3)
                steps_kl = tabular_kl(self.num_steps_distrib.prob(), prior)
                num_steps_prior_loss_per_sample = tf.squeeze(tf.reduce_sum(steps_kl, 1))

                self.num_steps_prior_loss = tf.reduce_mean(num_steps_prior_loss_per_sample)
                tf.summary.scalar('num_steps_prior', self.num_steps_prior_loss)
                prior_loss.add(self.num_steps_prior_loss, num_steps_prior_loss_per_sample)

            if appearance_prior is not None:
                prior = Normal(appearance_prior.loc, appearance_prior.scale)
                posterior = Normal(self.what_loc, self.what_scale)

                what_kl = _kl(posterior, prior)
                what_kl = tf.reduce_sum(what_kl, -1, keep_dims=True) * self.presence
                appearance_prior_loss_per_sample = tf.squeeze(tf.reduce_sum(what_kl, 0))

                #         n_samples_with_encoding = tf.reduce_sum(tf.to_float(tf.greater(num_step_per_sample, 0.)))
                #         div = tf.maximum(n_samples_with_encoding, 1.)
                #         appearance_prior_loss = tf.reduce_sum(latent_code_prior_loss_per_sample) / div
                self.appearance_prior_loss = tf.reduce_mean(appearance_prior_loss_per_sample)
                tf.summary.scalar('latent_code_prior', self.appearance_prior_loss)
                prior_loss.add(self.appearance_prior_loss, appearance_prior_loss_per_sample)

                usx, utx, usy, uty = tf.split(self.where_loc, 4, 2)
                ssx, stx, ssy, sty = tf.split(self.where_scale, 4, 2)
                us = tf.concat((usx, usy), -1)
                ss = tf.concat((ssx, ssy), -1)

                scale_distrib = Normal(us, ss)
                scale_prior = Normal(where_scale_prior.loc, where_scale_prior.scale)
                scale_kl = _kl(scale_distrib, scale_prior)

                ut = tf.concat((utx, uty), -1)
                st = tf.concat((stx, sty), -1)
                shift_distrib = Normal(ut, st)

                if 'loc' in where_shift_prior:
                    shift_mean = where_shift_prior.loc
                else:
                    shift_mean = ut
                shift_prior = Normal(shift_mean, where_shift_prior.scale)

                shift_kl = _kl(shift_distrib, shift_prior)
                where_kl = tf.reduce_sum(scale_kl + shift_kl, -1, keep_dims=True) * self.presence
                where_kl_per_sample = tf.reduce_sum(tf.squeeze(where_kl), 0)
                self.where_kl = tf.reduce_mean(where_kl_per_sample)
                tf.summary.scalar('where_prior', self.where_kl)
                prior_loss.add(self.where_kl, where_kl_per_sample)

        return prior_loss

    def _reinforce(self, loss, make_opt, baseline=None):
        if baseline is None:
            baseline = getattr(self, 'baseline', None)

        if callable(baseline):
            baseline_module = baseline
            baseline = baseline(self.obs, self.what, self.where, self.presence_prob)

        log_prob = self.num_steps_distrib.log_prob(self.num_step_per_sample)
        log_prob = tf.clip_by_value(log_prob, -1e32, 1e32)

        #     log_prob *= -1 # cause we're maximising
        self.importance_weight = loss._per_sample
        if baseline is not None:
            self.importance_weight -= baseline

        reinforce_loss_per_sample = tf.stop_gradient(self.importance_weight) * log_prob
        self.reinforce_loss = tf.reduce_mean(reinforce_loss_per_sample)
        tf.summary.scalar('reinforce_loss', self.reinforce_loss)

        # Baseline Optimisation
        baseline_vars, baseline_train_step = [], None
        if baseline is not None:
            baseline_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=baseline_module.variable_scope.name)
            baseline_target = tf.stop_gradient(loss.per_sample)
            baseline_loss_per_sample = (baseline_target - baseline) ** 2
            self.baseline_loss = tf.reduce_mean(baseline_loss_per_sample)
            tf.summary.scalar('baseline_loss', self.baseline_loss)

            baseline_opt = make_opt(10 * self.learning_rate)
            baseline_train_step = baseline_opt.minimize(self.baseline_loss, var_list=baseline_vars)

        return self.reinforce_loss, baseline_vars, baseline_train_step

    def train_step(self, learning_rate, l2_weight=0., appearance_prior=None, where_scale_prior=None,
                   where_shift_prior=None,
                   num_steps_prior=None, use_prior=True,
                   use_reinforce=True, baseline=None):

        self.l2_weight = l2_weight
        self.appearance_prior = appearance_prior
        self.where_scale_prior = where_scale_prior
        self.where_shift_prior = where_shift_prior
        self.num_steps_prior = num_steps_prior
        self.use_prior = use_prior
        self.use_reinforce = use_reinforce

        with tf.variable_scope('loss'):
            global_step = tf.train.get_or_create_global_step()
            loss = Loss()
            self._train_step = []
            self.learning_rate = tf.Variable(learning_rate, name='learning_rate', trainable=False)
            make_opt = lambda lr: tf.train.RMSPropOptimizer(lr, momentum=.9, centered=True)

            # Reconstruction Loss
            rec_loss_per_sample = -self.output_distrib.log_prob(self.obs)
            self.rec_loss_per_sample = tf.reduce_sum(rec_loss_per_sample, axis=(1, 2))
            self.rec_loss = tf.reduce_mean(self.rec_loss_per_sample)
            tf.summary.scalar('rec', self.rec_loss)
            loss.add(self.rec_loss, self.rec_loss_per_sample)

            # Prior Loss
            if use_prior:
                self.prior_loss = self._prior_loss(appearance_prior, where_scale_prior,
                                              where_shift_prior, num_steps_prior, global_step)
                tf.summary.scalar('prior', self.prior_loss.value)
                loss.add(self.prior_loss)

            # REINFORCE
            opt_loss = loss.value
            baseline_vars = []
            if use_reinforce:
                reinforce_loss, baseline_vars, baseline_train_step = self._reinforce(loss, make_opt, baseline)
                if baseline_train_step is not None:
                    self._train_step.append(baseline_train_step)

                opt_loss += reinforce_loss

            model_vars = list(set(tf.trainable_variables()) - set(baseline_vars))
            # L2 reg
            if l2_weight > 0.:
                self.l2_loss = l2_weight * sum(map(tf.nn.l2_loss, model_vars))
                opt_loss += self.l2_loss
                tf.summary.scalar('l2', self.l2_loss)

            opt = make_opt(self.learning_rate)
            gvs = opt.compute_gradients(opt_loss, var_list=model_vars)
            true_train_step = opt.apply_gradients(gvs, global_step=global_step)
            self._train_step.append(true_train_step)

            # Metrics
            gradient_summaries(gvs)
            self.num_step_accuracy = tf.reduce_mean(tf.to_float(tf.equal(self.gt_num_steps, self.num_step_per_sample)))

            self.loss = loss
            return self._train_step, global_step
