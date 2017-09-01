import os
import unittest
import tensorflow as tf


class TFTestBase(unittest.TestCase):

    vars = {
        'x': [tf.float32, None],
        'y': [tf.float32, None],
        'm': [tf.float32, None]
    }

    @classmethod
    def setUpClass(cls):
        tf.reset_default_graph()

        for k, v in cls.vars.iteritems():
            setattr(cls, k, tf.placeholder(v[0], v[1], name=k))
        cls.sess = tf.Session()

    @classmethod
    def tearDownClass(cls):
        cls.sess.close()
        tf.reset_default_graph()

    @classmethod
    def init_vars(cls):
        cls.sess.run(tf.global_variables_initializer())
        cls.sess.run(tf.local_variables_initializer())

    def feed_dict(self, xx, yy=None, mm=None):
        fd = {self.x: xx}

        if yy is not None:
            fd[self.y] = yy

        if mm is not None:
            fd[self.m] = mm

        return fd

    def eval(self, expr, xx=None, yy=None, mm=None, feed_dict=None):
        if feed_dict is None:
            feed_dict = self.feed_dict(xx, yy, mm)

        try:
            expr = tf.convert_to_tensor(expr)
        except TypeError: pass

        return self.sess.run(expr, feed_dict)


def test_path(path=None):
    p = os.path.abspath(os.path.dirname(__file__))
    if path is not None:
        p = os.path.join(p, path)
    return p
