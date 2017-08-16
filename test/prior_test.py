import numpy as np
import unittest

from numpy.testing import assert_array_equal
from tf_tools.testing_tools import TFTestBase

from attend_infer_repeat.prior import *


class TabularKLTest(TFTestBase):

    @classmethod
    def setUpClass(cls):
        super(TabularKLTest, cls).setUpClass()

        cls.x.set_shape([None])
        cls.kl = tabular_kl(cls.x, cls.y, 0.)

    def test_same(self):
        p = [.25] * 4
        kl = self.eval(self.kl, p, p)
        self.assertEqual(sum(kl), 0.)

    def test_zero(self):
        p = [0., .25, .25, .5]
        q = [.25] * 4

        kl = self.eval(self.kl, p, q)
        self.assertGreater(sum(kl), 0.)

    def test_one(self):
        p = [0., 1., 0., 0.]
        q = [1. - 1e-7, 1e-7, 0., 0.]

        kl = self.eval(self.kl, p, q)
        self.assertGreater(sum(kl), 0.)

    def test_always_positive_on_random(self):
        def gen():
            a = abs(np.random.rand(4))
            a /= a.sum()
            return a

        for i in xrange(100):
            p = gen()
            q = gen()

            kl = self.eval(self.kl, p, q)
            self.assertGreater(sum(kl), 0.)


class GeometricPriorTest(unittest.TestCase):

    def test(self):
        p = geometric_prior(.25, 3)
        self.assertEqual(p.shape, (4, 1, 1))
        self.assertTrue(.5 < p[0] < .75)
        self.assertTrue(.2 < p[1] < .25)
        self.assertTrue(.24**2 < p[2] < .25**2)
        self.assertTrue(.24**3 < p[3] < .25**3)


class ConditionalPresencePosteriorTest(TFTestBase):

    @classmethod
    def setUpClass(cls):
        super(ConditionalPresencePosteriorTest, cls).setUpClass()
        cls.probs = presence_prob_table(cls.x)

    def test_obvious(self):
        p = [0., 0., 0.]
        p = self.eval(self.probs, p)
        assert_array_equal(p, [1., 0., 0., 0.])

        p = [1., 0., 0.]
        p = self.eval(self.probs, p)
        assert_array_equal(p, [0., 1., 0., 0.])

        p = [1., 1., 0.]
        p = self.eval(self.probs, p)
        assert_array_equal(p, [0., 0., 1., 0.])

        p = [1., 1., 1.]
        p = self.eval(self.probs, p)
        assert_array_equal(p, [0., 0., 0., 1.])

    def test_geom(self):
        p = [.5, .5, .5]
        p = self.eval(self.probs, p)
        assert_array_equal(p, [.5, .5**2, .5**3, .5**3])


class NumStepsKLTest(TFTestBase):

    vars = {'x': [tf.float32, [None]]}

    @classmethod
    def setUpClass(cls):
        super(NumStepsKLTest, cls).setUpClass()

        cls.prior = geometric_prior(.005, 3).squeeze()

        cls.posterior = presence_prob_table(cls.x)
        cls.posterior_grad = tf.gradients(cls.posterior, cls.x)

        cls.posterior_kl = tabular_kl(cls.posterior, cls.prior, 0.)
        cls.posterior_kl_grad = tf.gradients(tf.reduce_sum(cls.posterior_kl), cls.x)

        cls.free_kl = tabular_kl(cls.x, cls.prior, 0.)
        cls.free_kl_grad = tf.gradients(tf.reduce_sum(cls.free_kl), cls.x)

    def test_free_stress(self):
        for i in xrange(100):
            p = abs(np.random.rand(4))
            p /= p.sum()

            kl = self.eval(self.free_kl, p)
            self.assertGreater(kl.sum(), 0)
            self.assertFalse(np.isnan(kl).any())
            self.assertTrue(np.isfinite(kl).all())

            grad = self.eval(self.free_kl_grad, p)
            self.assertFalse(np.isnan(grad).any())
            self.assertTrue(np.isfinite(grad).all())

    def test_posterior_stress(self):
        for i in xrange(100):
            p = np.random.rand(3)
            kl = self.eval(self.posterior_kl, p)
            self.assertGreater(kl.sum(), 0)
            self.assertFalse(np.isnan(kl).any())
            self.assertTrue(np.isfinite(kl).all())

            grad = self.eval(self.posterior_kl_grad, p)
            self.assertFalse(np.isnan(grad).any())
            self.assertTrue(np.isfinite(grad).all())

    def test_posterior_zeros(self):
        p = np.asarray([.5, 0., 0.])

        posterior = self.eval(self.posterior, p)
        print 'posterior', posterior
        posterior_grad = self.eval(self.posterior_grad, p)
        print 'posterior grad', posterior_grad

        kl = self.eval(self.posterior_kl, p)
        print kl
        self.assertGreater(kl.sum(), 0)
        self.assertFalse(np.isnan(kl).any())
        self.assertTrue(np.isfinite(kl).all())

        grad = self.eval(self.posterior_kl_grad, p)
        print grad
        self.assertFalse(np.isnan(grad).any())
        self.assertTrue(np.isfinite(grad).all())


class NumStepsSamplingKLTest(TFTestBase):

    vars = {'x': [tf.float32, [None, None]], 'y': [tf.int32, [None, None]]}

    @classmethod
    def setUpClass(cls):
        super(NumStepsSamplingKLTest, cls).setUpClass()

        cls.prior = geometric_prior(.05, 3).squeeze()

        cls.posterior = presence_prob_table(cls.x)
        cls.posterior_kl = tabular_kl_sampling(cls.posterior, cls.prior, cls.y)
        cls.free_kl = tabular_kl_sampling(cls.x, cls.prior, cls.y)

    def test_free_stress(self):
        batch_size = 64

        for i in xrange(100):
            p = abs(np.random.rand(batch_size, 4))
            j = np.random.randint(1, 4)
            # p[j:] = 0
            # p /= p.sum(1, keepdims=True)

            samples = np.random.randint(0, 4, (batch_size, 1))

            kl = self.eval(self.free_kl, p, samples)
            self.assertGreater(kl.sum(), 0)
            self.assertFalse(np.isnan(kl).any())
            self.assertTrue(np.isfinite(kl).all())