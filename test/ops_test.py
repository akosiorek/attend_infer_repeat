import tensorflow as tf
import numpy as np

from numpy.testing import assert_array_equal
from testing_tools import TFTestBase

from attend_infer_repeat.ops import sample_from_tensor


class SampleFromTensorTest(TFTestBase):

    @classmethod
    def setUpClass(cls):
        super(SampleFromTensorTest, cls).setUpClass()
        cls.n_dim = 3

        for i in xrange(1, 4):
            d = '{}d'.format(i)
            local = {
                'x'+d: tf.placeholder(tf.float32, [None] * (i-1) + [cls.n_dim], 'x' + d),
                'y'+d: tf.placeholder(tf.float32, [None] * (i-1), 'y' + d),
            }

            local['sample'+d] = sample_from_tensor(local['x'+d], local['y'+d])

            for k, v in local.iteritems():
                setattr(cls, k, v)

    def test_sample1d(self):
        """single vectors"""

        x = [5, 7, 11]
        for y in xrange(3):
            r = self.eval(self.sample1d, feed_dict={self.x1d: x, self.y1d: y})

            self.assertEqual(r, x[y])

    def test_sample2d(self):
        """minibatches of observations"""

        x = np.asarray([[5, 7, 11]])
        x = np.tile(x, (4, 1)) + np.arange(4)[:, np.newaxis]
        r = self.eval(self.sample2d, feed_dict={self.x2d: x, self.y2d: [0, 1, 2, 0]})
        assert_array_equal(r, [5, 8, 13, 8])

    def test_sample3d(self):
        """minibatches of timeseries"""

        x = np.asarray([[[5, 7, 11]]])
        x = np.tile(x, (2, 4, 1)) + np.arange(8).reshape(2, 4, 1)
        r = self.eval(self.sample3d, feed_dict={self.x3d: x, self.y3d: [[2, 1, 1, 2], [0, 1, 2, 0]]})
        assert_array_equal(r, [[11, 8, 9, 14], [9, 12, 17, 12]])
