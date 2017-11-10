import tensorflow as tf

from attend_infer_repeat.mnist_model import (AIRonMNIST,
                                             KLBySamplingMixin)
from attend_infer_repeat.grad import VIMCOEstimator
from attend_infer_repeat.experiment_tools import optimizer_from_string

flags = tf.flags

tf.flags.DEFINE_float('step_bias', 1., '')
tf.flags.DEFINE_float('transform_var_bias', -3., '')
tf.flags.DEFINE_float('learning_rate', 1e-5, '')
tf.flags.DEFINE_float('output_multiplier', .25, '')
tf.flags.DEFINE_float('init_step_success_prob', 1. - 1e-7, '')
tf.flags.DEFINE_float('n_anneal_steps_loss', 1e3, '')
tf.flags.DEFINE_integer('n_iw_samples', 5, '')
tf.flags.DEFINE_integer('n_steps_per_image', 3, '')
tf.flags.DEFINE_boolean('importance_resample', False, '')
tf.flags.DEFINE_boolean('use_r_imp_weight', True, '')
tf.flags.DEFINE_boolean('vimco_per_sample_control', False, '')
tf.flags.DEFINE_string('opt', '', '')


def load(img, num):

    f = tf.flags.FLAGS

    n_hidden = 32 * 8
    n_layers = 2
    n_hiddens = [n_hidden] * n_layers

    class AIRwithVIMCO(AIRonMNIST, VIMCOEstimator, KLBySamplingMixin):
        importance_resample = f.importance_resample
        use_r_imp_weight = f.use_r_imp_weight
        vimco_per_sample_control = f.vimco_per_sample_control
        init_step_success_prob = f.init_step_success_prob
        n_anneal_steps_loss = f.n_anneal_steps_loss

    air = AIRwithVIMCO(img,
                      max_steps=f.n_steps_per_image,
                      inpt_encoder_hidden=n_hiddens,
                      glimpse_encoder_hidden=n_hiddens,
                      glimpse_decoder_hidden=n_hiddens,
                      transform_estimator_hidden=n_hiddens,
                      steps_pred_hidden=[128, 64],
                      baseline_hidden=[256, 128],
                      transform_var_bias=f.transform_var_bias,
                      step_bias=f.step_bias,
                      discrete_steps=True,
                      iw_samples=f.n_iw_samples,
                      output_multiplier=f.output_multiplier)

    kwargs = dict(learning_rate=f.learning_rate, nums=num)
    if f.opt:
        opt, opt_kwargs = optimizer_from_string(f.opt, build=False)
        kwargs['optimizer'] = opt
        kwargs['opt_kwargs'] = opt_kwargs

    train_step, global_step = air.train_step(**kwargs)

    return air, train_step, global_step
