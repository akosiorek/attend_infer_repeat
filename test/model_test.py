import unittest
import tensorflow as tf
import sonnet as snt

from attrdict import AttrDict

from attend_infer_repeat.model import AIRModel
from attend_infer_repeat.elbo import AIRPriorMixin, KLMixin, LogLikelihoodMixin
from attend_infer_repeat.modules import *


class AIRModelWithPriors(AIRModel, AIRPriorMixin, KLMixin, LogLikelihoodMixin): pass


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

    def test_instantiate(self):
        learning_rate = 1e-4
        batch_size = 10
        img_size = (5, 7)
        crop_size = (2, 2)
        n_what = 13
        n_steps_per_image = 3

        imgs = tf.placeholder(tf.float32, (batch_size,) + img_size, name='inpt')
        nums = tf.placeholder(tf.float32, (batch_size, 1), name='nums')

        modules = make_modules()
        air = AIRModelWithPriors(imgs, n_steps_per_image, crop_size, n_what, **modules)
        outputs = AttrDict({k: getattr(air, k) for k in air.cell.output_names})
        print 'Constructed model'

        train_step = air.train_step(learning_rate, 1e-4, nums=nums)
        loss = air.loss.value
        print 'Computed gradients'

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        xx = np.random.rand(*imgs.get_shape().as_list())
        res, l = sess.run([outputs, loss], {imgs: xx})

        print 'outputs:'
        for k, v in res.iteritems():
            print k, v.shape

        print 'loss = {}'.format(l)
        print 'Running train step'
        sess.run(train_step, {imgs: xx})

        print 'Done'