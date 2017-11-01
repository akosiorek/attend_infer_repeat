import tensorflow as tf

from attend_infer_repeat.mnist_model import (AIRonMNIST,
                                             ImportanceWeightedNVILEstimatorWithBaseline,
                                             KLBySamplingMixin)

flags = tf.flags

tf.flags.DEFINE_float('step_bias', 1., '')
tf.flags.DEFINE_float('transform_var_bias', -3., '')
tf.flags.DEFINE_float('learning_rate', 1e-5, '')
tf.flags.DEFINE_integer('n_iw_samples', 5, '')
tf.flags.DEFINE_integer('n_steps_per_image', 3, '')
tf.flags.DEFINE_boolean('importance_resample', False, '')
tf.flags.DEFINE_boolean('use_r_imp_weight', True, '')


def load(img, num):

    f = tf.flags.FLAGS

    n_hidden = 32 * 8
    n_layers = 2
    n_hiddens = [n_hidden] * n_layers

    class ConcreteAIR(AIRonMNIST, ImportanceWeightedNVILEstimatorWithBaseline, KLBySamplingMixin):
        importance_resample = f.importance_resample
        use_r_imp_weight = f.use_r_imp_weight

    air = ConcreteAIR(img,
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
                      iw_samples=f.n_iw_samples)

    train_step, global_step = air.train_step(f.learning_rate, nums=num)

    print 'AIR tensors:'
    for k in dir(air):
        v = getattr(air, k)
        if isinstance(v, tf.Tensor):
            print k, v.shape

    return air, train_step, global_step