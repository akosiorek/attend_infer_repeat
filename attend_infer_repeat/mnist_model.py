from functools import partial

import tensorflow as tf
import sonnet as snt

from model import AIRModel
from elbo import *
from grad import NVILEstimator, ImportanceWeightedNVILEstimator
from seq_model import SeqAIRModel
from modules import BaselineMLP, Encoder, Decoder, StochasticTransformParam, StepsPredictor
from ops import anneal_weight


class BaselineMixin(object):

    def _make_baseline(self):
        baseline_module = BaselineMLP(self.baseline_hidden)
        final_state = [i[::self.iw_samples] for i in self.final_state]
        inpts = [self.obs, self.what_loc, self.where_loc, self.presence_prob, final_state]
        for i in xrange(1, len(inpts)-1):
            inpts[i] = inpts[i][::self.iw_samples]

        baseline = baseline_module(*inpts)[..., tf.newaxis]
        baseline_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope=baseline_module.variable_scope.name)

        print 'constructed baseline'
        return baseline, baseline_vars


class NVILEstimatorWithBaseline(NVILEstimator, BaselineMixin):
    pass


class ImportanceWeightedNVILEstimatorWithBaseline(ImportanceWeightedNVILEstimator, BaselineMixin):
    pass


class MNISTPriorMixin(AIRPriorMixin, LogLikelihoodMixin):
    init_step_success_prob = 1. - 1e-7

    def _geom_success_prob(self, **kwargs):

        hold_init = 1e3
        steps_div = 1e4
        anneal_steps = 1e5
        global_step = tf.train.get_or_create_global_step()
        steps_prior_success_prob = anneal_weight(self.init_step_success_prob, 1e-5, 'exp', global_step,
                                                     anneal_steps, hold_init, steps_div)
        self.steps_prior_success_prob = steps_prior_success_prob
        return self.steps_prior_success_prob


class AIRonMNIST(AIRModel, MNISTPriorMixin, LogLikelihoodMixin):
    """Implements AIR for the MNIST dataset"""

    def __init__(self, obs, glimpse_size=(20, 20),
                 inpt_encoder_hidden=[256]*2,
                 glimpse_encoder_hidden=[256]*2,
                 glimpse_decoder_hidden=[252]*2,
                 transform_estimator_hidden=[256]*2,
                 steps_pred_hidden=[50]*1,
                 baseline_hidden=[256, 128]*1,
                 transform_var_bias=-2.,
                 min_glimpse_size=0.,
                 step_bias=0.,
                 *args, **kwargs):

        self.transform_var_bias = tf.Variable(transform_var_bias, trainable=False, dtype=tf.float32,
                                                       name='transform_var_bias')
        self.min_glimpse_size = min_glimpse_size
        self.step_bias = tf.Variable(step_bias, trainable=False, dtype=tf.float32, name='step_bias')
        self.baseline_hidden = baseline_hidden

        super(AIRonMNIST, self).__init__(
            *args,
            obs=obs,
            glimpse_size=glimpse_size,
            n_what=50,
            transition=snt.LSTM(256),
            input_encoder=partial(Encoder, inpt_encoder_hidden),
            glimpse_encoder=partial(Encoder, glimpse_encoder_hidden),
            glimpse_decoder=partial(Decoder, glimpse_decoder_hidden),
            transform_estimator=partial(StochasticTransformParam, transform_estimator_hidden,
                                      scale_bias=self.transform_var_bias, min_glimpse_size=self.min_glimpse_size),
            steps_predictor=partial(StepsPredictor, steps_pred_hidden, self.step_bias),
            output_std=.3,
            **kwargs
        )


class SeqAIRonMNIST(SeqAIRModel):
    """Implements AIR for the MNIST dataset"""

    def __init__(self, obs, glimpse_size=(20, 20),
                 inpt_encoder_hidden=[256]*2,
                 glimpse_encoder_hidden=[256]*2,
                 glimpse_decoder_hidden=[252]*2,
                 transform_estimator_hidden=[256]*2,
                 steps_pred_hidden=[50]*1,
                 baseline_hidden=[256, 128]*1,
                 transform_var_bias=-2.,
                 step_bias=0.,
                 *args, **kwargs):

        self.transform_var_bias = tf.Variable(transform_var_bias, trainable=False, dtype=tf.float32,
                                                       name='transform_var_bias')
        self.step_bias = tf.Variable(step_bias, trainable=False, dtype=tf.float32, name='step_bias')
        self.baseline = BaselineMLP(baseline_hidden)

        super(SeqAIRonMNIST, self).__init__(
            *args,
            obs=obs,
            glimpse_size=glimpse_size,
            n_appearance=50,
            transition=snt.LSTM(256),
            input_encoder=partial(Encoder, inpt_encoder_hidden),
            glimpse_encoder=partial(Encoder, glimpse_encoder_hidden),
            glimpse_decoder=partial(Decoder, glimpse_decoder_hidden),
            transform_estimator=partial(StochasticTransformParam, transform_estimator_hidden,
                                      scale_bias=self.transform_var_bias),
            steps_predictor=partial(StepsPredictor, steps_pred_hidden, self.step_bias),
            output_std=.3,
            **kwargs
        )
