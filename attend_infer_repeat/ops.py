import tensorflow as tf
from tensorflow.python.training import moving_averages


class Loss(object):
    """Helper class for keeping track of losses"""

    def __init__(self):
        self._value = None
        self._per_sample = None

    def add(self, loss=None, per_sample=None, weight=1.):
        if isinstance(loss, Loss):
            per_sample = loss.per_sample
            loss = loss.value

        self._update('_value', loss, weight)
        self._update('_per_sample', per_sample, weight)

    def _update(self, name, expr, weight):
        value = getattr(self, name)
        expr *= weight
        if value is None:
            value = expr
        else:
            assert value.get_shape().as_list() == expr.get_shape().as_list(), 'Shape should be {} but is {}'.format(value.get_shape(), expr.get_shape())
            value += expr

        setattr(self, name, value)

    def _get_value(self, name):
        v = getattr(self, name)
        if v is None:
            v = tf.zeros([])
        return v

    @property
    def value(self):
        return self._get_value('_value')

    @property
    def per_sample(self):
        return self._get_value('_per_sample')


def make_moving_average(name, value, init, decay, log=True):
    """Creates an exp-moving average of `value` and an update op, which is added to UPDATE_OPS collection.

    :param name: string, name of the created moving average tf.Variable
    :param value: tf.Tensor, the value to be averaged
    :param init: float, an initial value for the moving average
    :param decay: float between 0 and 1, exponential decay of the moving average
    :param log: bool, add a summary op if True
    :return: tf.Tensor, the moving average
    """
    var = tf.get_variable(name, shape=value.get_shape(),
                          initializer=tf.constant_initializer(init), trainable=False)

    update = moving_averages.assign_moving_average(var, value, decay, zero_debias=False)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update)
    if log:
        tf.summary.scalar(name, var)

    return var


def clip_preserve(expr, min, max):
    """Clips the immediate gradient but preserves the chain rule

    :param expr: tf.Tensor, expr to be clipped
    :param min: float
    :param max: float
    :return: tf.Tensor, clipped expr
    """
    clipped = tf.clip_by_value(expr, min, max)
    return tf.stop_gradient(clipped - expr) + expr


def anneal_weight(init_val, final_val, anneal_type, global_step, anneal_steps, hold_for=0., steps_div=1.,
                   dtype=tf.float64):
    val, final, step, hold_for, anneal_steps, steps_div = (tf.cast(i, dtype) for i in
                                                           (init_val, final_val, global_step, hold_for, anneal_steps,
                                                            steps_div))
    step = tf.maximum(step - hold_for, 0.)

    if anneal_type == 'exp':
        decay_rate = tf.pow(final / val, steps_div / anneal_steps)
        val = tf.train.exponential_decay(val, step, steps_div, decay_rate)

    elif anneal_type == 'linear':
        val = final + (val - final) * (1. - step / anneal_steps)
    else:
        raise NotImplementedError

    anneal_weight = tf.maximum(final, val)
    return anneal_weight


def sample_from_1d_tensor(arr, idx):
    """Takes samples from `arr` indicated by `idx`

    :param arr:
    :param idx:
    :return:
    """
    arr = tf.convert_to_tensor(arr)
    assert len(arr.get_shape()) == 1, "shape is {}".format(arr.get_shape())

    idx = tf.to_int32(idx)
    arr = tf.gather(tf.squeeze(arr), idx)
    return arr


def sample_from_tensor(tensor, idx):
    """Takes sample from `tensor` indicated by `idx`, works for minibatches

    :param tensor:
    :param idx:
    :return:
    """
    tensor = tf.convert_to_tensor(tensor)

    assert tensor.shape.ndims == (idx.shape.ndims + 1) \
           or ((tensor.shape.ndims == idx.shape.ndims) and (idx.shape[-1] == 1)), \
        'Shapes: tensor={} vs idx={}'.format(tensor.shape.ndims, idx.shape.ndims)

    batch_shape = tf.shape(tensor)[:-1]
    trailing_dim = int(tensor.shape[-1])
    n_elements = tf.reduce_prod(batch_shape)
    shift = tf.range(n_elements) * trailing_dim

    tensor_flat = tf.reshape(tensor, (-1,))
    idx_flat = tf.reshape(tf.to_int32(idx), (-1,)) + shift
    samples_flat = sample_from_1d_tensor(tensor_flat, idx_flat)
    samples = tf.reshape(samples_flat, batch_shape)

    return samples