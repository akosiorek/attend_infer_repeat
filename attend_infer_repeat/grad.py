import math
import tensorflow as tf

import ops


class EstimatorBaselineMixin(object):

    def make_baseline(self):
        res = None, []
        _make_baseline = getattr(self, '_make_baseline', None)
        if _make_baseline is not None:
            res = _make_baseline()

        return res

    def _make_baseline_train_step(self, opt, loss, baseline, baseline_vars):
        baseline_target = tf.stop_gradient(loss)

        self.baseline_loss = .5 * tf.reduce_mean(tf.square(baseline_target - baseline))
        tf.summary.scalar('baseline_loss', self.baseline_loss)
        train_step = opt.minimize(self.baseline_loss, var_list=baseline_vars)
        return train_step


class ImportanceWeightedMixin(object):
    importance_resample = False

    def _make_nelbo(self):
        return self.nelbo

    def _resample(self, *args):
        iw_sample_idx = self._iw_sample_index * self.batch_size + tf.range(self.batch_size)
        resampled = [tf.gather(arg, iw_sample_idx) for arg in args]

        return resampled

    def _estimate_importance_weighted_elbo(self, per_sample_elbo):

        per_sample_elbo = tf.reshape(per_sample_elbo, (self.batch_size, self.iw_samples))
        importance_weights = tf.nn.softmax(per_sample_elbo, -1)
        self.iw_distrib = tf.contrib.distributions.Categorical(per_sample_elbo)
        self._iw_sample_index = self.iw_distrib.sample()

        # tf.exp(tf.float32(89)) is inf, but if arg is 88 then it's not inf;
        # similarly on the negative, exp of -90 is 0;
        # when we subtract the max value, the dynamic range is about [-85, 0].
        # If we subtract 78 from control, it becomes [-85, 78], which is almost twice as big.
        control = tf.reduce_max(per_sample_elbo, -1, keep_dims=True) - 78.
        normalised = tf.exp(per_sample_elbo - control)
        elbo = tf.log(tf.reduce_sum(normalised, -1, keep_dims=True)) + control - tf.log(float(self.iw_samples))
        return elbo, importance_weights


class NVILEstimator(EstimatorBaselineMixin):

    decay_rate = None

    def _make_train_step(self, make_opt, rec_loss, kl_div):

        # loss used as a proxy for gradient computation
        self.proxy_loss = rec_loss.value + kl_div.value + self._l2_loss()

        # REINFORCE
        reinforce_imp_weight = rec_loss.per_sample
        if not self.analytic_kl_expectation:
            reinforce_imp_weight += kl_div.per_sample

        self.baseline, self.baseline_vars = self.make_baseline()
        self.reinforce_loss = self._reinforce(reinforce_imp_weight, self.decay_rate)
        self.proxy_loss += self.reinforce_loss

        opt = make_opt(self.learning_rate)
        gvs = opt.compute_gradients(self.proxy_loss, var_list=self.model_vars)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        global_step = tf.train.get_or_create_global_step()
        with tf.control_dependencies(update_ops):
            train_step = opt.apply_gradients(gvs, global_step=global_step)

        if self.baseline is not None:
            baseline_opt = make_opt(10 * self.learning_rate)
            self._baseline_train_step = self._make_baseline_train_step(baseline_opt, reinforce_imp_weight,
                                                                       self.baseline, self.baseline_vars)
            train_step = tf.group(train_step, self._baseline_train_step)
        return train_step, gvs

    def _reinforce(self, importance_weight, decay_rate):
        """Implements REINFORCE for training the discrete probability distribution over number of steps and train-step
         for the baseline"""

        log_prob = self.num_steps_posterior.log_prob(self.num_step_per_sample)

        if self.baseline is not None:
               importance_weight -= self.baseline

        # constant baseline and learning signal normalisation according to NVIL paper
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
        reinforce_loss = tf.reduce_mean(reinforce_loss_per_sample)
        tf.summary.scalar('reinforce_loss', reinforce_loss)

        return reinforce_loss


class ImportanceWeightedNVILEstimator(ImportanceWeightedMixin, EstimatorBaselineMixin):

    decay_rate = None
    use_r_imp_weight = True

    def _make_train_step(self, make_opt, rec_loss_per_sample, kl_div_per_sample):

        negative_per_sample_elbo = rec_loss_per_sample + kl_div_per_sample
        per_sample_elbo = -negative_per_sample_elbo
        iw_elbo_estimate, elbo_importance_weights = self._estimate_importance_weighted_elbo(per_sample_elbo)

        self.elbo_importance_weights = tf.stop_gradient(elbo_importance_weights)

        self.negative_weighted_per_sample_elbo = self.elbo_importance_weights \
                                            * tf.reshape(negative_per_sample_elbo, (self.batch_size, self.iw_samples))

        # loss used as a proxy for gradient computation
        self.baseline, self.baseline_vars = self.make_baseline()

        posterior_num_steps_log_prob = self.num_steps_posterior.log_prob(self.num_step_per_sample)
        if self.importance_resample:
            posterior_num_steps_log_prob = self._resample(posterior_num_steps_log_prob)
            posterior_num_steps_log_prob = tf.reshape(posterior_num_steps_log_prob, (self.batch_size, 1))

            # this could be constant e.g. 1, but the expectation of this is zero anyway, so there's no point in adding that.
            r_imp_weight = 0.
        else:
            posterior_num_steps_log_prob = tf.reshape(posterior_num_steps_log_prob, (self.batch_size, self.iw_samples))
            r_imp_weight = self.elbo_importance_weights

        if not self.use_r_imp_weight:
            r_imp_weight = 0.

        self.nelbo_per_sample = -tf.reshape(iw_elbo_estimate, (self.batch_size, 1))
        num_steps_learning_signal = self.nelbo_per_sample
        self.nelbo = tf.reduce_mean(self.nelbo_per_sample)
        self.proxy_loss = self.nelbo + self._l2_loss()

        self.reinforce_loss = self._reinforce(num_steps_learning_signal + r_imp_weight, posterior_num_steps_log_prob)
        self.proxy_loss += self.reinforce_loss

        opt = make_opt(self.learning_rate)
        gvs = opt.compute_gradients(self.proxy_loss, var_list=self.model_vars)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        global_step = tf.train.get_or_create_global_step()
        with tf.control_dependencies(update_ops):
            train_step = opt.apply_gradients(gvs, global_step=global_step)

        if self.baseline is not None:
            baseline_opt = make_opt(10 * self.learning_rate)
            self._baseline_train_step = self._make_baseline_train_step(baseline_opt, num_steps_learning_signal,
                                                                       self.baseline, self.baseline_vars)
            train_step = tf.group(train_step, self._baseline_train_step)
        return train_step, gvs

    def _reinforce(self, learning_signal, posterior_num_steps_log_prob):
        """Implements REINFORCE for training the discrete probability distribution over number of steps and train-step
         for the baseline"""

        self.num_steps_learning_signal = learning_signal
        if self.baseline is not None:
            self.num_steps_learning_signal -= self.baseline

        # constant baseline and learning signal normalisation according to NVIL paper
        if self.decay_rate is not None:
            axes = range(len(learning_signal.get_shape()))
            mean, var = tf.nn.moments(tf.squeeze(learning_signal), axes=axes)
            self.imp_weight_moving_mean = ops.make_moving_average('imp_weight_moving_mean', mean, 0., self.decay_rate)
            self.imp_weight_moving_var = ops.make_moving_average('imp_weight_moving_var', var, 1., self.decay_rate)

            factor = tf.maximum(tf.sqrt(self.imp_weight_moving_var), 1.)
            self.num_steps_learning_signal = (self.num_steps_learning_signal - self.imp_weight_moving_mean) / factor

        axes = range(len(self.num_steps_learning_signal.get_shape()))
        imp_weight_mean, imp_weight_var = tf.nn.moments(self.num_steps_learning_signal, axes)
        tf.summary.scalar('imp_weight_mean', imp_weight_mean)
        tf.summary.scalar('imp_weight_var', imp_weight_var)
        reinforce_loss_per_sample = tf.stop_gradient(self.num_steps_learning_signal) * posterior_num_steps_log_prob

        shape = reinforce_loss_per_sample.shape.as_list()
        assert len(shape) == 2 and shape[0] == self.batch_size and shape[1] in (1, self.iw_samples), 'shape is {}'.format(shape)

        reinforce_loss = tf.reduce_mean(tf.reduce_sum(reinforce_loss_per_sample, -1))
        tf.summary.scalar('reinforce_loss', reinforce_loss)

        return reinforce_loss


class VIMCOEstimator(ImportanceWeightedMixin):
    vimco_per_sample_control=False

    decay_rate = None
    use_r_imp_weight = True
    n_anneal_steps_loss = 1.

    def _make_train_step(self, make_opt, rec_loss_per_sample, kl_div_per_sample):
        assert self.iw_samples >= 2, 'VIMCO requires at least two importance samples'


        # 1) estimate the per-sample elbo
        negative_per_sample_elbo = rec_loss_per_sample + kl_div_per_sample
        per_sample_elbo = -negative_per_sample_elbo

        # 2) compute the multi-sample stochastic bound
        iw_elbo_estimate, elbo_importance_weights = self._estimate_importance_weighted_elbo(per_sample_elbo)

        self.elbo_importance_weights = tf.stop_gradient(elbo_importance_weights)

        self.negative_weighted_per_sample_elbo = self.elbo_importance_weights \
                                            * tf.reshape(negative_per_sample_elbo, (self.batch_size, self.iw_samples))

        self.baseline = self._make_baseline(per_sample_elbo)

        # loss used as a proxy for gradient computation
        posterior_num_steps_log_prob = self.num_steps_posterior.log_prob(self.num_step_per_sample)
        if self.importance_resample:
            posterior_num_steps_log_prob = self._resample(posterior_num_steps_log_prob)
            posterior_num_steps_log_prob = tf.reshape(posterior_num_steps_log_prob, (self.batch_size, 1))

            # this could be constant e.g. 1, but the expectation of this is zero anyway,
            #  so there's no point in adding that.
            r_imp_weight = 0.
        else:
            posterior_num_steps_log_prob = tf.reshape(posterior_num_steps_log_prob, (self.batch_size, self.iw_samples))
            r_imp_weight = self.elbo_importance_weights

        if not self.use_r_imp_weight:
            r_imp_weight = 0.

        self.nelbo_per_sample = -tf.reshape(iw_elbo_estimate, (self.batch_size, 1))
        num_steps_learning_signal = self.nelbo_per_sample
        self.nelbo = tf.reduce_mean(self.nelbo_per_sample)
        self.proxy_loss = self.nelbo + self._l2_loss()

        self.reinforce_loss = self._reinforce(num_steps_learning_signal - r_imp_weight, posterior_num_steps_log_prob)
        self.proxy_loss += self.reinforce_loss

        opt = make_opt(self.learning_rate)
        gvs = opt.compute_gradients(self.proxy_loss, var_list=self.model_vars)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        global_step = tf.train.get_or_create_global_step()
        with tf.control_dependencies(update_ops):
            train_step = opt.apply_gradients(gvs, global_step=global_step)

        return train_step, gvs

    def _make_baseline(self, per_sample_elbo):
        #####################

        # compute the baseline
        #########################
        # 3) precompute the sum of per-sample bounds
        reshaped_per_sample_elbo = tf.reshape(per_sample_elbo, (self.batch_size, self.iw_samples))
        summed_per_sample_elbo = tf.reduce_sum(reshaped_per_sample_elbo, -1, keep_dims=True)

        # 4) compute the baseline
        all_but_one_average = (summed_per_sample_elbo - reshaped_per_sample_elbo) / (self.iw_samples - 1.)

        if self.vimco_per_sample_control:
            baseline, control = self._exped_baseline_and_control(reshaped_per_sample_elbo, all_but_one_average)
        else:
            control = tf.reduce_max(reshaped_per_sample_elbo, -1, keep_dims=True) - .78
            exped_per_sample_elbo = tf.exp(reshaped_per_sample_elbo - control)
            summed_exped_per_sample_elbo = tf.reduce_sum(exped_per_sample_elbo, -1, keep_dims=True)
            baseline = summed_exped_per_sample_elbo - exped_per_sample_elbo + tf.exp(all_but_one_average - control)


        # the log-arg for the baseline can be zero when the elbo for that particular estimate is the max of estimates
        # for all other samples; when this happenes, and when the differences are big, then
        # `summed_exped_per_sample_elbo` takes the value of `exped_per_sample_elbo`; also then `all_but_one_average` is
        # smaller than `control` (average is always less extreme) and all goes to hell.
        #
        # adding an eps for numerical stability biases the baseline, so it's better to just set the value of the
        #  baseline equal to the learnig signal for that sample, which effectively takes this sample away from.
        #baseline_is_zero = tf.equal(baseline, 0.)

        baseline = tf.log(baseline) - math.log(self.iw_samples) + control
        #baseline = tf.where(baseline_is_zero, reshaped_per_sample_elbo, baseline)
        return -baseline

    def _reinforce(self, learning_signal, posterior_num_steps_log_prob):
        """Implements REINFORCE for training the discrete probability distribution over number of steps and train-step
         for the baseline"""

        self.num_steps_learning_signal = learning_signal
        if self.baseline is not None:
            self.num_steps_learning_signal -= self.baseline

        axes = range(len(self.num_steps_learning_signal.get_shape()))
        imp_weight_mean, imp_weight_var = tf.nn.moments(self.num_steps_learning_signal, axes)
        tf.summary.scalar('imp_weight_mean', imp_weight_mean)
        tf.summary.scalar('imp_weight_var', imp_weight_var)
        reinforce_loss_per_sample = tf.stop_gradient(self.num_steps_learning_signal) * posterior_num_steps_log_prob
    
        shape = reinforce_loss_per_sample.shape.as_list()
        assert len(shape) == 2 and shape[0] == self.batch_size and shape[1] in (1, self.iw_samples), 'shape is {}'.format(shape)

        reinforce_loss = tf.reduce_mean(tf.reduce_sum(reinforce_loss_per_sample, -1))
        tf.summary.scalar('reinforce_loss', reinforce_loss)

        if self.n_anneal_steps_loss > 1.:
            global_step = tf.to_float(tf.train.get_or_create_global_step())
            anneal_weight = tf.minimum(global_step / self.n_anneal_steps_loss, 1.)
            reinforce_loss *= anneal_weight

        return reinforce_loss

    @staticmethod
    def _raw_baseline(reshaped_per_sample_elbo, all_but_one_average):
        diag = tf.matrix_diag(all_but_one_average - reshaped_per_sample_elbo)
        baseline = reshaped_per_sample_elbo[..., tf.newaxis] + diag
        return baseline

    @staticmethod
    def _control(raw_baseline):
        control = tf.reduce_max(raw_baseline, -2)
        return control

    @staticmethod
    def _exped_baseline_and_control(reshaped_per_sample_elbo, all_but_one_average):

        baseline = VIMCOEstimator._raw_baseline(reshaped_per_sample_elbo, all_but_one_average)
        control = VIMCOEstimator._control(baseline) - 78.

        baseline = tf.reduce_sum(tf.exp(baseline - control[..., tf.newaxis, :]), -2)
        return baseline, control
