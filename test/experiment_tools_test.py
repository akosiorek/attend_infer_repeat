import unittest
import tensorflow as tf

from attend_infer_repeat.experiment_tools import optimizer_from_string


class OptimizerFromStringTest(unittest.TestCase):

    def test_no_args(self):
        opt = optimizer_from_string('Adam')
        self.assertTrue(isinstance(opt, tf.train.AdamOptimizer))

    def test_with_args(self):
        opt = optimizer_from_string('Adam(learning_rate=1e-3')
        self.assertTrue(isinstance(opt, tf.train.AdamOptimizer))

    def test_with_args_no_build(self):
        opt, args = optimizer_from_string('Adam(learning_rate=1e-3)', build=False)
        self.assertEquals(opt, tf.train.AdamOptimizer)
        self.assertEqual(args, dict(learning_rate=1e-3))