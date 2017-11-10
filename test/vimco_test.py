from test import testing_tools

import tensorflow as tf
import numpy as np

from numpy.testing import assert_array_equal, assert_array_almost_equal

from attend_infer_repeat.grad import VIMCOEstimator


class ExpedVimcoBaselineTest(testing_tools.TFTestBase):

    vars = {
        'x': [tf.float32, [1, 3]],
        'y': [tf.float32, [1, 3]],
    }

    @classmethod
    def setUpClass(cls):
        super(ExpedVimcoBaselineTest, cls).setUpClass()

        cls.raw_baseline = VIMCOEstimator._raw_baseline(cls.x, cls.y)
        cls.raw_control = VIMCOEstimator._control(cls.raw_baseline)
        cls.baseline, cls.control = VIMCOEstimator._exped_baseline_and_control(cls.x, cls.y)

    def test_raw_control(self):

        l = np.asarray([1, 2, 3]).reshape(1, 3)
        lm1 = np.asarray([2.5, 2., 1.5]).reshape(1, 3)

        raw_control = self.eval(self.raw_control, l, lm1)
        expected = np.asarray([3., 3., 2.]).reshape(1, 3)
        assert_array_equal(raw_control, expected)

        control = self.eval(self.control, l, lm1)
        assert_array_equal(control, expected)

    def test_raw_baseline(self):

        l = np.asarray([1, 2, 3]).reshape(1, 3)
        lm1 = np.asarray([2.5, 2., 1.5]).reshape(1, 3)

        raw_baseline = self.eval(self.raw_baseline, l, lm1)

        expected = np.asarray([[2.5, 2, 3], [1, 2, 3], [1, 2, 1.5]]).T.reshape(1, 3, 3)
        assert_array_equal(raw_baseline, expected)

    def test_baseline(self):
        l = np.asarray([1, 2, 3]).reshape(1, 3)
        lm1 = np.asarray([2.5, 2., 1.5]).reshape(1, 3)

        baseline, control = self.eval([self.baseline, self.control], l, lm1)

        # control is:
        # [[3.  3.  2.]]
        # raw baseline is
        # [[[2.5  1.   1.]
        #   [2.   2.   2.]
        #  [3.   3.   1.5]]]

        b1 = np.exp(np.asarray([2.5, 2., 3.]) - 3.).sum()
        b2 = np.exp(np.asarray([1, 2., 3.]) - 3.).sum()
        b3 = np.exp(np.asarray([1, 2., 1.5]) - 2.).sum()

        expected = np.asarray([b1, b2, b3]).reshape(1, 3)
        assert_array_almost_equal(baseline, expected)