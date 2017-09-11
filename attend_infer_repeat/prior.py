import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import Bernoulli, Geometric


def masked_apply(tensor, op, mask):
    """Applies `op` to tensor only at locations indicated by `mask` and sets the rest to zero.

    Similar to doing `tensor = tf.where(mask, op(tensor), tf.zeros_like(tensor))` but it behaves correctly
    when `op(tensor)` is NaN or inf while tf.where does not.

    :param tensor: tf.Tensor
    :param op: tf.Op
    :param mask: tf.Tensor with dtype == bool
    :return: tf.Tensor
    """
    chosen = tf.boolean_mask(tensor, mask)
    applied = op(chosen)
    idx = tf.to_int32(tf.where(mask))
    result = tf.scatter_nd(idx, applied, tf.shape(tensor))
    return result


def geometric_prior(success_prob, n_steps):
    success_prob = tf.clip_by_value(success_prob, 1e-7, 1. - 1e-15)
    geom = Geometric(probs=1. - success_prob)
    events = tf.range(n_steps + 1, dtype=geom.dtype)
    probs = geom.prob(events)
    return probs


def _cumprod(tensor, axis=0):
    """A custom version of cumprod to prevent NaN gradients when there are zeros in `tensor`
    as reported here: https://github.com/tensorflow/tensorflow/issues/3862

    :param tensor: tf.Tensor
    :return: tf.Tensor
    """
    transpose_permutation = None
    n_dim = len(tensor.get_shape())
    if n_dim > 1 and axis != 0:

        if axis < 0:
            axis += n_dim

        transpose_permutation = np.arange(n_dim)
        transpose_permutation[-1], transpose_permutation[0] = 0, axis

    tensor = tf.transpose(tensor, transpose_permutation)

    def prod(acc, x):
        return acc * x

    prob = tf.scan(prod, tensor)
    tensor = tf.transpose(prob, transpose_permutation)
    return tensor


def bernoulli_to_modified_geometric(presence_prob):
    presence_prob = tf.cast(presence_prob, tf.float64)
    inv = 1. - presence_prob
    prob = _cumprod(presence_prob, axis=-1)
    modified_prob = tf.concat([inv[..., :1], inv[..., 1:] * prob[..., :-1], prob[..., -1:]], -1)
    modified_prob /= tf.reduce_sum(modified_prob, -1, keep_dims=True)
    return tf.cast(modified_prob, tf.float32)


def tabular_kl(p, q, zero_prob_value=0., logarg_clip=None):
    """Computes KL-divergence KL(p||q) for two probability mass functions (pmf) given in a tabular form.

    :param p: iterable
    :param q: iterable
    :param zero_prob_value: float; values below this threshold are treated as zero
    :param logarg_clip: float or None, clips the argument to the log to lie in [-logarg_clip, logarg_clip] if not None
    :return: iterable of brodcasted shape of (p * q), per-coordinate value of KL(p||q)
    """
    p, q = (tf.cast(i, tf.float64) for i in (p, q))
    non_zero = tf.greater(p, zero_prob_value)
    logarg = p / q

    if logarg_clip is not None:
        logarg = tf.clip_by_value(logarg, 1. / logarg_clip, logarg_clip)

    log = masked_apply(logarg, tf.log, non_zero)
    kl = p * log

    return tf.cast(kl, tf.float32)


def sample_from_1d_tensor(arr, idx):
    """Takes samples from `arr` indicated by `idx`"""
    arr = tf.convert_to_tensor(arr)
    assert len(arr.get_shape()) == 1, "shape is {}".format(arr.get_shape())

    idx = tf.to_int32(idx)
    arr = tf.gather(tf.squeeze(arr), idx)
    return arr


def sample_from_tensor(tensor, idx):
    """Takes sample from `tensor` indicated by `idx`, works for minibatches"""
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


class NumStepsDistribution(object):
    """Probability distribution used for the number of steps

    Transforms Bernoulli probabilities of an event = 1 into p(n) where n is the number of steps
    as described in the AIR paper."""

    def __init__(self, steps_probs):
        """

        :param steps_probs: tensor; Bernoulli success probabilities
        """
        self._steps_probs = steps_probs
        self._joint = bernoulli_to_modified_geometric(steps_probs)
        self._bernoulli = None

    def sample(self, n=None):
        if self._bernoulli is None:
            self._bernoulli = Bernoulli(self._steps_probs)

        sample = self._bernoulli.sample(n)
        sample = tf.cumprod(sample, tf.rank(sample) - 1)
        sample = tf.reduce_sum(sample, -1)
        return sample

    def prob(self, samples=None):
        if samples is None:
            return self._joint
        return sample_from_tensor(self._joint, samples)

    def log_prob(self, samples):
        return tf.log(self.prob(samples))





