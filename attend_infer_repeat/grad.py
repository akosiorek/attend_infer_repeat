import tensorflow as tf

import ops


class NVILEstimator(object):

    decay_rate = None

    def _make_train_step(self, make_opt, rec_loss, kl_div):

        # loss used as a proxy for gradient computation
        self.proxy_loss = rec_loss.value + kl_div.value + self._l2_loss()

        # REINFORCE
        reinforce_imp_weight = rec_loss.per_sample
        if not self.analytic_kl_expectation:
            reinforce_imp_weight += kl_div.per_sample

        self.baseline, self.baseline_vars = self._make_baseline()
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

    def _make_baseline(self):
        return None, []

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

    def _make_baseline_train_step(self, opt, loss, baseline, baseline_vars):
        baseline_target = tf.stop_gradient(loss)

        self.baseline_loss = .5 * tf.reduce_mean(tf.square(baseline_target - baseline))
        tf.summary.scalar('baseline_loss', self.baseline_loss)
        train_step = opt.minimize(self.baseline_loss, var_list=baseline_vars)
        return train_step