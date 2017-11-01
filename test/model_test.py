import unittest
import tensorflow as tf
import sonnet as snt

from attrdict import AttrDict

from attend_infer_repeat.model import AIRModel
from attend_infer_repeat.elbo import AIRPriorMixin, KLMixin, LogLikelihoodMixin
from attend_infer_repeat.grad import NVILEstimator, ImportanceWeightedNVILEstimator
from attend_infer_repeat.modules import *


class AIRModelWithPriors(AIRModel, AIRPriorMixin, KLMixin, LogLikelihoodMixin, ImportanceWeightedNVILEstimator):
    importance_resample = True
    pass


def make_modules():

    return dict(
        transition=snt.GRU(3),
        input_encoder=(lambda: Encoder(5)),
        glimpse_encoder=(lambda: Encoder(7)),
        glimpse_decoder=(lambda x: Decoder(11, x)),
        transform_estimator=(lambda: StochasticTransformParam(13)),
        steps_predictor=(lambda: StepsPredictor(17))
    )


class ModelTest(unittest.TestCase):
    learning_rate = 1e-4
    batch_size = 10
    img_size = (5, 7)
    crop_size = (2, 2)
    n_what = 13
    n_steps_per_image = 3
    iw_samples = 2

    @classmethod
    def setUpClass(cls):
        cls.imgs = tf.placeholder(tf.float32, (cls.batch_size,) + cls.img_size, name='inpt')
        cls.nums = tf.placeholder(tf.float32, (cls.n_steps_per_image, cls.batch_size, 1), name='nums')

        print 'Building AIR'
        cls.modules = make_modules()
        cls.air = AIRModelWithPriors(cls.imgs, cls.n_steps_per_image, cls.crop_size, cls.n_what,
                                     iw_samples=cls.iw_samples, **cls.modules)
        cls.outputs = AttrDict({k: getattr(cls.air, k) for k in cls.air.cell.output_names})
        print 'Constructed model'

        cls.train_step = cls.air.train_step(cls.learning_rate, 1e-4, nums=cls.nums)
        cls.loss = cls.air.nelbo
        print 'Computed gradients'

    def test_forward(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        xx = np.random.rand(*self.imgs.get_shape().as_list())
        res, l = sess.run([self.outputs, self.loss], {self.imgs: xx})

        print 'outputs:'
        for k, v in res.iteritems():
            print k, v.shape
        print 'loss = {}'.format(l)

    def test_backward(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        xx = np.random.rand(*self.imgs.get_shape().as_list())

        print 'Running train step'
        sess.run(self.train_step, {self.imgs: xx})
        print 'Done'

    def test_shapes(self):
        learning_signal_shape = self.air.num_steps_learning_signal.shape.as_list()
        self.assertEqual(learning_signal_shape, [self.batch_size, 1])