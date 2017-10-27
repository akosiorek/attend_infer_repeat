import tensorflow as tf
from tensorflow.contrib.distributions import Normal
from tensorflow.contrib.distributions.python.ops.kullback_leibler import kl as _kl

from prior import geometric_prior, tabular_kl


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


class KLMixin(object):

    def _kl_num_steps(self):
        num_steps_posterior_prob = self.num_steps_posterior.prob()
        steps_kl = tabular_kl(num_steps_posterior_prob, self.num_step_prior_prob)
        kl_num_steps_per_sample = tf.squeeze(tf.reduce_sum(steps_kl, 1))
        kl_num_steps = tf.reduce_mean(kl_num_steps_per_sample)
        return kl_num_steps, kl_num_steps_per_sample

    def _ordered_step_prob(self):
        if self.analytic_kl_expectation:
            # reverse cumsum of q(n) needed to compute \E_{q(n)} [ KL[ q(z|n) || p(z|n) ]]
            ordered_step_prob = self.num_steps_posterior.prob()[..., 1:]
            ordered_step_prob = tf.cumsum(ordered_step_prob, axis=-1, reverse=True)
        else:
            ordered_step_prob = tf.squeeze(self.presence)
        return ordered_step_prob

    def _kl_what(self):
        what_kl = _kl(self.what_posterior, self.what_prior)
        what_kl = tf.reduce_sum(what_kl, -1) * self.ordered_step_prob
        what_kl_per_sample = tf.reduce_sum(what_kl, -1)
        kl_what = tf.reduce_mean(what_kl_per_sample)
        return kl_what, what_kl_per_sample

    def _kl_where(self):
        scale_kl = _kl(self.scale_posterior, self.scale_prior)
        shift_kl = _kl(self.shift_posterior, self.shift_prior)
        where_kl = tf.reduce_sum(scale_kl + shift_kl, -1) * self.ordered_step_prob
        where_kl_per_sample = tf.reduce_sum(where_kl, -1)
        kl_where = tf.reduce_mean(where_kl_per_sample)
        return kl_where, where_kl_per_sample


class LogLikelihoodMixin(object):

    def _log_likelihood(self):
        # Reconstruction Loss, - \E_q [ p(x | z, n) ]
        rec_loss_per_sample = -self.output_distrib.log_prob(self.obs)
        rec_loss_per_sample = tf.reduce_sum(rec_loss_per_sample, axis=(1, 2))
        rec_loss = tf.reduce_mean(rec_loss_per_sample)
        return rec_loss, rec_loss_per_sample