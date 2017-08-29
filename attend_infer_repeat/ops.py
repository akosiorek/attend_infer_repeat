import tensorflow as tf


class Loss(object):
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
            assert value.get_shape() == expr.get_shape(), 'Shape should be {} but is {}'.format(value.get_shape(), expr.get_shape())
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


# def check_numerics():
#
# for k, v in o.iteritems():

#         if np.isnan(v).any():
#             print 'found NaN in {} in iter {}'.format(k, train_itr)
#             break

#         if np.isinf(v).any():
#             print 'found Inf in {} in iter {}'.format(k, train_itr)
#             break


def epsilon_greedy(events, eps):
    shape = tf.shape(events)
    do_explore = tf.less(tf.random_uniform(shape, dtype=tf.float32), tf.ones(shape, dtype=tf.float32) * eps)
    random = tf.cast(tf.round(tf.random_uniform(shape, dtype=tf.float32)), events.dtype)
    events = tf.where(do_explore, random, events)
    return events