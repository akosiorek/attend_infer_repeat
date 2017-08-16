import numpy as np
import tensorflow as tf


def geometric_prior(success_prob, n_steps):
    assert (.0 < success_prob < 1.), 'Success probability has to be within (0., 1.)'
    probs = [1. - success_prob] + [success_prob ** i for i in xrange(1, n_steps + 1)]
    probs = np.asarray(probs, dtype=np.float32)
    probs /= probs.sum()
    return probs


def presence_prob_table(presence_prob):
    inv = 1. - presence_prob
    prob0 = inv[..., 0]
    prob1 = inv[..., 1] * presence_prob[..., 0]
    prob2 = inv[..., 2] * presence_prob[..., 1] * presence_prob[..., 0]
    prob3 = tf.reduce_prod(presence_prob, tf.rank(presence_prob) - 1)

    modified_prob = tf.stack((prob0, prob1, prob2, prob3), len(prob0.get_shape()))

    modified_prob /= tf.reduce_sum(modified_prob, -1, keep_dims=True)  # + 1e-7
    return modified_prob


def tabular_kl(p, q, zero_prob_value=0., logarg_clip=None):
    non_zero = tf.greater(p, zero_prob_value)
    logarg = p / q

    if logarg_clip is not None:
        logarg = tf.clip_by_value(logarg, 1. / logarg_clip, logarg_clip)

    logarg = tf.boolean_mask(logarg, non_zero)
    log = tf.log(logarg)
    idx = tf.to_int32(tf.where(non_zero))
    log = tf.scatter_nd(idx, log, tf.shape(p))
    kl = p * log

    return kl


def tabular_kl_sampling(p, q, samples):

    samples = tf.to_int32(samples)
    p, q = (tf.squeeze(i) for i in (p, q))
    q = tf.gather(q, samples)

    # TODO: implement this by flattening p and shifting `samples`
    shift = tf.range(tf.shape(p)[0]) * tf.shape(p)[1]

    p = tf.reshape(p, (-1,))
    samples = tf.reshape(samples, (-1,)) + shift

    p = tf.gather(p, samples)
    p = tf.reshape(p, tf.shape(q))

    kl = tf.where(tf.greater(p, 0.), tf.log(p / q), tf.zeros_like(p))
    return kl