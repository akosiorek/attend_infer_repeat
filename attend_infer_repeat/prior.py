import numpy as np
import tensorflow as tf


def masked_apply(tensor, op, mask):
    """Applies `op` to tensor only at locations indicated by `mask` and sets the rest to zero.

    Similar to doing tensor = tf.where(mask, op(tensor), tf.zeros_like(tensor)) behaves correctly
    when op(tensor) is NaN or inf while tf.where does nor.

    :param tensor:
    :param op:
    :param mask:
    :return:
    """
    chosen = tf.boolean_mask(tensor, mask)
    applied = op(chosen)
    idx = tf.to_int32(tf.where(mask))
    result = tf.scatter_nd(idx, applied, tf.shape(tensor))
    return result


def geometric_prior(success_prob, n_steps):
    if isinstance(success_prob, tf.Tensor):
        prob0 = 1 - success_prob
        probs = tf.ones(n_steps, dtype=tf.float32) * success_prob
        probs = tf.cumprod(probs)
        probs = tf.concat(([prob0], probs), 0)
        probs /= tf.reduce_sum(probs)
    else:
        assert (.0 < success_prob < 1.), 'Success probability has to be within (0., 1.)'
        probs = [1. - success_prob] + [success_prob ** i for i in xrange(1, n_steps + 1)]
        probs = np.asarray(probs, dtype=np.float32)
        probs /= probs.sum()
    return probs


def presence_prob_table(presence_prob):
    presence_prob = tf.cast(presence_prob, tf.float64)
    inv = 1. - presence_prob
    prob0 = inv[..., 0]
    prob1 = inv[..., 1] * presence_prob[..., 0]
    prob2 = inv[..., 2] * presence_prob[..., 1] * presence_prob[..., 0]
    prob3 = tf.reduce_prod(presence_prob, tf.rank(presence_prob) - 1)

    modified_prob = tf.stack((prob0, prob1, prob2, prob3), len(prob0.get_shape()))

    modified_prob /= tf.reduce_sum(modified_prob, -1, keep_dims=True)
    return tf.cast(modified_prob, tf.float32)


def tabular_kl(p, q, zero_prob_value=0., logarg_clip=None):
    p, q = (tf.cast(i, tf.float64) for i in (p, q))
    non_zero = tf.greater(p, zero_prob_value)
    logarg = p / q

    if logarg_clip is not None:
        logarg = tf.clip_by_value(logarg, 1. / logarg_clip, logarg_clip)

    log = masked_apply(logarg, tf.log, non_zero)
    kl = p * log

    return tf.cast(kl, tf.float32)


def sample_from_1d_tensor(arr, idx):
    arr = tf.convert_to_tensor(arr)
    assert len(arr.get_shape()) == 1, "shape is {}".format(arr.get_shape())

    idx = tf.to_int32(idx)
    arr = tf.gather(tf.squeeze(arr), idx)
    return arr


def sample_from_tensor(tensor, idx):
    tensor = tf.convert_to_tensor(tensor)
    if len(tensor.get_shape()) > 2:
        raise NotImplemented

    idx = tf.to_int32(idx)
    shift = tf.range(tf.shape(tensor)[0]) * tf.shape(tensor)[1]

    p_flat = tf.reshape(tensor, (-1,))
    idx_flat = tf.reshape(idx, (-1,)) + shift
    samples_flat = sample_from_1d_tensor(p_flat, idx_flat)
    samples = tf.reshape(samples_flat, tf.shape(idx))
    return samples


def tabular_kl_sampling(p, q, samples_from_p):

    p_samples = sample_from_tensor(p, samples_from_p)

    q_samples = sample_from_1d_tensor(q, samples_from_p)
    logarg = p_samples / q_samples
    # kl = tf.log(logarg)

    non_zero = tf.greater(p_samples, 1e-8)
    kl = masked_apply(logarg, tf.log, non_zero)

    return kl


class NumStepsDistribution(object):

    def __init__(self, steps_probs):
        self._steps_probs = steps_probs
        self._joint = presence_prob_table(steps_probs)

    def sample(self):
        pass

    def prob(self, samples=None):
        if samples is None:
            return self._joint
        return sample_from_tensor(self._joint, samples)

    def log_prob(self, samples):
        return tf.log(self.prob(samples))





