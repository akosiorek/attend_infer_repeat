import unittest
import tensorflow as tf
import sonnet as snt

from attrdict import AttrDict

from attend_infer_repeat.seq_model import SeqAIRModel, SeqAIRCell
from attend_infer_repeat.modules import *


def make_modules():

    return dict(
        transition=snt.GRU(3),
        input_encoder=(lambda: Encoder(5)),
        glimpse_encoder=(lambda: Encoder(7)),
        glimpse_decoder=(lambda x: Decoder(11, x)),
        transform_estimator=(lambda x: StochasticTransformParam(13, x)),
        steps_predictor=(lambda: StepsPredictor(17))
    )


class SeqModelTest(unittest.TestCase):

    def test_instantiate(self):
        learning_rate = 1e-4
        batch_size = 10
        img_size = (5, 7)
        crop_size = (2, 2)
        n_what = 13
        n_steps_per_image = 3
        n_timesteps = 11

        num_steps_prior = AttrDict(
            anneal='exp',
            init=1.,
            final=1e-5,
            steps_div=1e4,
            steps=1e5,
            hold_init=1e3,
            analytic=True
        )

        what_prior = AttrDict(loc=0., scale=1.)
        where_scale_prior = AttrDict(loc=0., scale=1.)
        where_shift_prior = AttrDict(loc=0., scale=1.)

        imgs = tf.placeholder(tf.float32, (None, batch_size,) + img_size, name='inpt')
        nums = tf.placeholder(tf.float32, (None, batch_size, 1), name='nums')

        modules = make_modules()
        air = SeqAIRModel(imgs, n_steps_per_image, crop_size, n_what, **modules)
        outputs = AttrDict({k: getattr(air, k) for k in air.cell.output_names})
        print 'Constructed model'

        train_step = air.train_step(learning_rate, 1e-4, what_prior, where_scale_prior,
                                    where_shift_prior, num_steps_prior, nums=nums)
        loss = air.loss.value
        print 'Computed gradients'

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        xx = np.random.rand(n_timesteps, *imgs.get_shape().as_list()[1:])
        res, l = sess.run([outputs, loss], {imgs: xx})

        print 'outputs:'
        for k, v in res.iteritems():
            print k, v.shape

        print 'loss = {}'.format(l)

        print 'Done'